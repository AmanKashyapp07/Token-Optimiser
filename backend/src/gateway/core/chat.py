from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gateway.core.budgeting import AdaptiveTokenBudgeter
from gateway.core.config import get_settings
from gateway.core.providers import Generation, Provider
from gateway.core.resilience import get_circuit_breaker
from gateway.core.store import Store
from gateway.features.cache.api import _lookup
from gateway.features.cache.coalescing import get_request_coalescer
from gateway.features.compression.service import get_service as get_compression_service
from gateway.features.routing.service import get_service as get_routing_service
from gateway.pricing import tier_cost_usd

# Mount endpoints at both /v1 (standard OpenAI SDK compatibility) and /api/v1 (internal gateway convention)
router = APIRouter(tags=["unified-inference-api"])

# Global bounded LRU cache of recent 200 execution traces for sub-millisecond retrieval
_RECENT_TRACES: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _record_trace_in_memory(trace_id: str, request_id: str, trace_data: dict[str, Any]) -> None:
    """Stores trace with FIFO/LRU eviction capped strictly at 200 items."""
    while len(_RECENT_TRACES) >= 200:
        _RECENT_TRACES.popitem(last=False)
    _RECENT_TRACES[trace_id] = trace_data
    _RECENT_TRACES[request_id] = trace_data


# ==============================================================================
# OpenAI-Compatible Schemas
# ==============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "auto"
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream: bool = False
    # Gateway Optimization Directives
    policy: Literal["economy", "balanced", "quality"] = "balanced"
    format_constraint: Literal["none", "3 bullets", "JSON only", "no preamble"] = "none"
    compression_budget: float = Field(default=0.6, ge=0.1, le=1.0)
    compression_method: Literal["heuristic", "crf"] = "heuristic"
    use_cache: bool = True
    simulate_provider_failure: bool = False  # For chaos & resilience testing


class ChatChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StageSpan(BaseModel):
    stage: str
    status: str
    latency_ms: float
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    trace_id: str
    request_id: str
    timestamp: str
    total_duration_ms: float
    stages: list[StageSpan]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
    optimisation: dict[str, Any]
    trace: ExecutionTrace


# ==============================================================================
# Unified Inference Execution Engine
# ==============================================================================

async def execute_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    response: Response,
    cache_scope: str = "public",
) -> ChatCompletionResponse:
    start_time = time.perf_counter()
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    settings = get_settings()
    provider = Provider(settings)
    store = Store(settings.database_path)
    stages: list[StageSpan] = []

    # 1. Preserve full conversation history & system prompt
    user_messages = [m for m in payload.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=422, detail="At least one user message is required.")
    raw_prompt = user_messages[-1].content

    system_messages = [m.content for m in payload.messages if m.role == "system"]
    system_instruction = "\n".join(system_messages) if system_messages else None

    # Multi-turn context preservation
    if len(payload.messages) > 1:
        context_turns = []
        for m in payload.messages:
            context_turns.append(f"{m.role.capitalize()}: {m.content}")
        compiled_query = "\n".join(context_turns)
    else:
        compiled_query = raw_prompt

    # --------------------------------------------------------------------------
    # STAGE 1: Dual-Tier Semantic Cache
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()
    cache_hit = False
    cached_entry = None

    if payload.use_cache:
        cached_entry = _lookup(
            raw_prompt,
            language="auto",
            scope_id=cache_scope,
            system_prompt=system_instruction,
            format_constraint=payload.format_constraint,
            model=payload.model,
        )
        if cached_entry and cached_entry.get("hit"):
            cache_hit = True

    cache_lat = round((time.perf_counter() - t_stage) * 1000, 2)
    stages.append(
        StageSpan(
            stage="semantic_cache",
            status="hit" if cache_hit else "miss",
            latency_ms=cache_lat,
            details={
                "match_type": cached_entry.get("match_type") if cached_entry else "none",
                "similarity": cached_entry.get("similarity", 0.0) if cached_entry else 0.0,
            },
        )
    )

    if cache_hit and cached_entry:
        total_lat = round((time.perf_counter() - start_time) * 1000, 2)
        answer_text = cached_entry.get("answer", "")
        if payload.format_constraint == "JSON only" and not answer_text.strip().startswith("{"):
            brief = answer_text.split(".")[0] if "." in answer_text else answer_text[:80]
            answer_text = json.dumps({"status": "success", "answer": brief.strip()})
        elif payload.format_constraint == "3 bullets" and not answer_text.strip().startswith(("•", "-", "*")):
            parts = [s.strip() for s in answer_text.replace("\n", " ").split(".") if s.strip()]
            if len(parts) >= 3:
                answer_text = f"• {parts[0]}.\n• {parts[1]}.\n• {parts[2]}."
            elif parts:
                answer_text = "\n".join(f"• {p}." for p in parts)

        finish_reason = "stop"
        if payload.max_tokens is not None and payload.max_tokens > 0:
            answer_text, was_capped = provider._truncate_to_max_tokens(answer_text, payload.max_tokens)
            if was_capped:
                finish_reason = "length"

        comp_tokens = provider.counter.count(answer_text)

        trace = ExecutionTrace(
            trace_id=trace_id,
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_duration_ms=total_lat,
            stages=stages,
        )
        _record_trace_in_memory(trace_id, request_id, trace.model_dump())

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Execution-Trace-Id"] = trace_id
        response.headers["X-Stage-Latencies"] = f"cache={cache_lat}ms; total={total_lat}ms"

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=settings.cheap_model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatChoiceMessage(content=answer_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=ChatUsage(prompt_tokens=0, completion_tokens=comp_tokens, total_tokens=comp_tokens),
            optimisation={
                "cache_hit": True,
                "compressed": False,
                "cost_usd": 0.0,
                "cost_saved_usd": round(tier_cost_usd("cheap", provider.counter.count(raw_prompt), comp_tokens), 6),
                "answer_source": f"cache:{cached_entry.get('match_type')}",
            },
            trace=trace,
        )

    # --------------------------------------------------------------------------
    # STAGE 2: Lexical & CRF Prompt Compression
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()
    comp_svc = get_compression_service()
    comp_res = comp_svc.analyze(
        compiled_query,
        method=payload.compression_method,
        budget=payload.compression_budget,
    )
    comp_lat = round((time.perf_counter() - t_stage) * 1000, 2)
    compressed_prompt = comp_res.get("compressed", compiled_query)
    stages.append(
        StageSpan(
            stage="prompt_compression",
            status="applied",
            latency_ms=comp_lat,
            details={
                "method": payload.compression_method,
                "original_tokens": comp_res.get("original_tokens"),
                "compressed_tokens": comp_res.get("compressed_tokens"),
                "savings_ratio": comp_res.get("savings_ratio"),
            },
        )
    )

    # --------------------------------------------------------------------------
    # STAGE 3: Model Routing
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()
    router_svc = get_routing_service()
    route_res = router_svc.predict(raw_prompt, payload.policy, compressed_prompt)
    routing_lat = round((time.perf_counter() - t_stage) * 1000, 2)

    selected_tier = "premium" if route_res.get("selected_tier") in {"strong", "quality"} else "cheap"
    chosen_model = (
        payload.model
        if payload.model and payload.model != "auto"
        else route_res.get("model", settings.cheap_model)
    )

    stages.append(
        StageSpan(
            stage="model_routing",
            status="routed",
            latency_ms=routing_lat,
            details={
                "selected_tier": route_res.get("selected_tier"),
                "selected_model": chosen_model,
                "policy": payload.policy,
            },
        )
    )

    # --------------------------------------------------------------------------
    # STAGE 4: ML-Powered Adaptive Token Budgeting
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()
    budgeter = AdaptiveTokenBudgeter()
    budget_res = budgeter.allocate(
        prompt=raw_prompt,
        format_constraint=payload.format_constraint,
        user_max_tokens=payload.max_tokens,
    )
    budget_lat = round((time.perf_counter() - t_stage) * 1000, 2)
    effective_max_tokens = budget_res.allocated_budget

    stages.append(
        StageSpan(
            stage="adaptive_token_budgeting",
            status="allocated",
            latency_ms=budget_lat,
            details={
                "strategy": budget_res.strategy,
                "allocated_budget": effective_max_tokens,
                "predicted_tokens": budget_res.predicted_tokens,
                "prediction_interval_margin": budget_res.prediction_interval_margin,
                "truncation_risk_warning": budget_res.truncation_risk_warning,
                "ridge_r2_score": budget_res.details.get("ridge_r2_score"),
            },
        )
    )

    # --------------------------------------------------------------------------
    # STAGE 5: Output Capping & Provider Inference Behind Circuit Breaker
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()
    circuit_breaker = get_circuit_breaker()

    # Prepend system instruction and format directives to prompt
    augmented_prompt = compressed_prompt
    if system_instruction:
        augmented_prompt = f"System: {system_instruction}\n\n{augmented_prompt}"
    if payload.format_constraint and payload.format_constraint != "none":
        augmented_prompt = f"{augmented_prompt} (Format requirement: {payload.format_constraint})"

    # Singleflight Coalescing Key for non-streaming calls
    coalescer = get_request_coalescer()
    coalesce_key = f"infer:{chosen_model}:{effective_max_tokens}:{payload.format_constraint}:{hash(augmented_prompt)}"

    async def _do_inference() -> tuple[Generation, bool, dict[str, Any]]:
        async def primary_provider_call():
            if payload.simulate_provider_failure:
                raise RuntimeError("Upstream frontier model 503 Service Unavailable")
            return await provider.generate(
                text=augmented_prompt,
                model=chosen_model,
                max_tokens=effective_max_tokens,
                temperature=payload.temperature,
            )

        async def fallback_provider_call():
            # Gracefully degrade to safe local fallback on upstream provider failure
            out = provider.answer_fixture(augmented_prompt, max_tokens=effective_max_tokens)
            out.model = settings.cheap_model
            return out

        # Circuit breaker directly wraps provider inference
        return await circuit_breaker.async_execute(primary_provider_call, fallback_provider_call)

    # Execute with singleflight coalescing
    (generation, fallback_applied, cb_telemetry), was_coalesced = await coalescer.do(
        coalesce_key, _do_inference
    )
    inference_lat = round((time.perf_counter() - t_stage) * 1000, 2)

    stages.append(
        StageSpan(
            stage="provider_inference",
            status="fallback" if fallback_applied else "generated",
            latency_ms=inference_lat,
            details={
                "model": chosen_model if not fallback_applied else settings.cheap_model,
                "mode": generation.mode,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "finish_reason": generation.finish_reason,
                "fallback_applied": fallback_applied,
                "circuit_state": cb_telemetry.get("circuit_state"),
                "coalesced": was_coalesced,
            },
        )
    )

    # --------------------------------------------------------------------------
    # STAGE 5: Cost Accounting & Financial Telemetry
    # --------------------------------------------------------------------------
    active_tier = "cheap" if fallback_applied else selected_tier
    cost_actual = tier_cost_usd(active_tier, generation.prompt_tokens, generation.completion_tokens)
    unconstrained_expected_tokens = max(105, generation.completion_tokens)
    baseline_cost = tier_cost_usd(
        "premium",
        comp_res.get("original_tokens", generation.prompt_tokens),
        unconstrained_expected_tokens,
    )
    cost_saved = max(0.0, baseline_cost - cost_actual)

    total_duration = round((time.perf_counter() - start_time) * 1000, 2)
    trace = ExecutionTrace(
        trace_id=trace_id,
        request_id=request_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_duration_ms=total_duration,
        stages=stages,
    )
    _record_trace_in_memory(trace_id, request_id, trace.model_dump())

    # Persist in SQLite transaction store
    store.record_request(
        text=raw_prompt,
        language="en",
        result={
            "trace_id": trace_id,
            "request_id": request_id,
            "cost_usd": cost_actual,
            "baseline_cost_usd": baseline_cost,
            "cost_saved_usd": cost_saved,
            "cache_hit": False,
            "model": chosen_model if not fallback_applied else settings.cheap_model,
            "route": active_tier,
            "input_tokens": generation.prompt_tokens,
            "output_tokens": generation.completion_tokens,
            "total_tokens": generation.total_tokens,
            "format_constraint": payload.format_constraint,
            "max_tokens": payload.max_tokens,
            "duration_ms": total_duration,
            "coalesced": was_coalesced,
            "fallback_applied": fallback_applied,
            "trace": trace.model_dump(),
        },
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Execution-Trace-Id"] = trace_id
    response.headers["X-Stage-Latencies"] = (
        f"cache={cache_lat}ms; comp={comp_lat}ms; route={routing_lat}ms; infer={inference_lat}ms; total={total_duration}ms"
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=chosen_model if not fallback_applied else settings.cheap_model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatChoiceMessage(content=generation.text),
                finish_reason=generation.finish_reason,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
        ),
        optimisation={
            "cache_hit": False,
            "compressed": comp_res.get("compressed_tokens", 0) < comp_res.get("original_tokens", 0),
            "original_prompt_tokens": comp_res.get("original_tokens"),
            "compressed_prompt_tokens": comp_res.get("compressed_tokens"),
            "selected_tier": active_tier,
            "cost_usd": round(cost_actual, 8),
            "baseline_cost_usd": round(baseline_cost, 8),
            "cost_saved_usd": round(cost_saved, 8),
            "cost_reduction_pct": round((cost_saved / max(1e-6, baseline_cost)) * 100, 1),
            "allocated_max_tokens": effective_max_tokens,
            "budget_strategy": budget_res.strategy,
            "coalesced": was_coalesced,
            "fallback_applied": fallback_applied,
        },
        trace=trace,
    )


async def stream_chat_completion(
    payload: ChatCompletionRequest,
    cache_scope: str = "public",
) -> AsyncIterator[str]:
    """Streams chat completion response using Server-Sent Events (SSE)."""
    settings = get_settings()
    provider = Provider(settings)
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # Extract user prompt
    user_prompts = [m.content for m in payload.messages if m.role == "user"]
    raw_prompt = user_prompts[-1] if user_prompts else ""
    chosen_model = payload.model if payload.model and payload.model != "auto" else settings.cheap_model

    # Stream chunks
    async for chunk in provider.stream(raw_prompt, chosen_model, max_tokens=payload.max_tokens, temperature=payload.temperature):
        chunk_data = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": chosen_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"

    # End with finish_reason and [DONE]
    final_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": chosen_model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ==============================================================================
# HTTP Route Declarations
# ==============================================================================

@router.post("/v1/chat/completions")
@router.post("/api/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    response: Response,
    cache_scope: str = Header(default="public", alias="X-Cache-Scope"),
):
    """OpenAI-compatible unified inference endpoint integrating the full 4-stage optimization pipeline."""
    if payload.stream:
        return StreamingResponse(
            stream_chat_completion(payload, cache_scope),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Request-ID": getattr(request.state, "request_id", uuid.uuid4().hex)},
        )
    return await execute_chat_completion(payload, request, response, cache_scope)


@router.get("/api/v1/traces/{trace_or_request_id}")
def get_trace(trace_or_request_id: str) -> dict[str, Any]:
    """Retrieve full stage execution trace for any request ID or trace ID."""
    if trace_or_request_id in _RECENT_TRACES:
        return _RECENT_TRACES[trace_or_request_id]

    # Search in SQLite store
    settings = get_settings()
    store = Store(settings.database_path)
    with store.connect() as db:
        rows = db.execute("select result_json from requests order by id desc limit 200").fetchall()
        for row in rows:
            try:
                data = json.loads(row[0])
                if data.get("trace_id") == trace_or_request_id or data.get("request_id") == trace_or_request_id:
                    if "trace" in data:
                        return data["trace"]
            except Exception:
                continue

    raise HTTPException(status_code=404, detail=f"Execution trace not found for identifier '{trace_or_request_id}'.")
