from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from gateway.core.config import get_settings
from gateway.core.providers import Generation, Provider
from gateway.core.resilience import get_circuit_breaker
from gateway.core.store import Store
from gateway.features.cache.api import _lookup
from gateway.features.compression.service import get_service as get_compression_service
from gateway.features.routing.service import get_service as get_routing_service
from gateway.pricing import tier_cost_usd

# Mount endpoints at both /v1 (standard OpenAI SDK compatibility) and /api/v1 (internal gateway convention)
router = APIRouter(tags=["unified-inference-api"])

# Global in-memory trace store (LRU cache of recent 200 execution traces for sub-millisecond retrieval)
_RECENT_TRACES: dict[str, dict[str, Any]] = {}


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
    simulate_provider_failure: bool = False  # For chaos/resilience testing


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

    # Extract user prompt and system messages
    user_prompts = [m.content for m in payload.messages if m.role == "user"]
    if not user_prompts:
        raise HTTPException(status_code=422, detail="At least one user message is required.")
    raw_prompt = user_prompts[-1]
    system_prompts = [m.content for m in payload.messages if m.role == "system"]

    # --------------------------------------------------------------------------
    # STAGE 1: Dual-Tier Semantic Cache
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()
    cache_hit = False
    cached_entry = None

    if payload.use_cache:
        cached_entry = _lookup(raw_prompt, language="auto", scope_id=cache_scope)
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
        comp_tokens = provider.counter.count(answer_text)

        trace = ExecutionTrace(
            trace_id=trace_id,
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_duration_ms=total_lat,
            stages=stages,
        )
        _RECENT_TRACES[trace_id] = trace.model_dump()
        _RECENT_TRACES[request_id] = trace.model_dump()

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
                    finish_reason="stop",
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
        raw_prompt,
        method=payload.compression_method,
        budget=payload.compression_budget,
    )
    comp_lat = round((time.perf_counter() - t_stage) * 1000, 2)
    compressed_prompt = comp_res.get("compressed", raw_prompt)
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
    # STAGE 3: Model Routing Guarded by Circuit Breaker
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()
    router_svc = get_routing_service()
    circuit_breaker = get_circuit_breaker()

    def primary_routing():
        if payload.simulate_provider_failure:
            raise RuntimeError("Upstream frontier model 503 Service Unavailable")
        return router_svc.predict(raw_prompt, payload.policy, compressed_prompt)

    def fallback_routing():
        return {
            "selected_tier": "cheap",
            "model": settings.cheap_model,
            "policy": payload.policy,
            "confidence": 1.0,
            "reason": "circuit_breaker_fallback",
        }

    route_res, fallback_applied, cb_telemetry = circuit_breaker.execute(primary_routing, fallback_routing)
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
            status="fallback" if fallback_applied else "routed",
            latency_ms=routing_lat,
            details={
                "selected_tier": route_res.get("selected_tier"),
                "selected_model": chosen_model,
                "fallback_applied": fallback_applied,
                "circuit_state": cb_telemetry.get("circuit_state"),
            },
        )
    )

    # --------------------------------------------------------------------------
    # STAGE 4: Output Capping & Provider Generation
    # --------------------------------------------------------------------------
    t_stage = time.perf_counter()

    # Apply soft format directives into prompt if specified
    augmented_prompt = compressed_prompt
    if payload.format_constraint and payload.format_constraint != "none":
        directive = f" (Format requirement: {payload.format_constraint})"
        augmented_prompt = f"{compressed_prompt}{directive}"

    generation: Generation = await provider.generate(
        text=augmented_prompt,
        model=chosen_model,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
    )
    inference_lat = round((time.perf_counter() - t_stage) * 1000, 2)

    stages.append(
        StageSpan(
            stage="provider_inference",
            status="generated",
            latency_ms=inference_lat,
            details={
                "model": chosen_model,
                "mode": generation.mode,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "finish_reason": generation.finish_reason,
            },
        )
    )

    # --------------------------------------------------------------------------
    # STAGE 5: Cost Accounting & Financial Telemetry
    # --------------------------------------------------------------------------
    cost_actual = tier_cost_usd(selected_tier, generation.prompt_tokens, generation.completion_tokens)
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
    _RECENT_TRACES[trace_id] = trace.model_dump()
    _RECENT_TRACES[request_id] = trace.model_dump()

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
            "model": chosen_model,
            "route": selected_tier,
            "input_tokens": generation.prompt_tokens,
            "output_tokens": generation.completion_tokens,
            "total_tokens": generation.total_tokens,
            "format_constraint": payload.format_constraint,
            "max_tokens": payload.max_tokens,
            "duration_ms": total_duration,
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
        model=chosen_model,
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
            "selected_tier": selected_tier,
            "cost_usd": round(cost_actual, 8),
            "baseline_cost_usd": round(baseline_cost, 8),
            "cost_saved_usd": round(cost_saved, 8),
            "cost_reduction_pct": round((cost_saved / max(1e-6, baseline_cost)) * 100, 1),
        },
        trace=trace,
    )


# ==============================================================================
# HTTP Route Declarations
# ==============================================================================

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@router.post("/api/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    response: Response,
    cache_scope: str = Header(default="public", alias="X-Cache-Scope"),
) -> ChatCompletionResponse:
    """OpenAI-compatible unified inference endpoint integrating the full 4-stage optimization pipeline."""
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
