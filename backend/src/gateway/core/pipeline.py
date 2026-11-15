from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Header, Request, Response
from pydantic import BaseModel, Field

from gateway.core.config import get_settings
from gateway.core.resilience import get_circuit_breaker
from gateway.core.store import Store
from gateway.features.cache.api import _lookup
from gateway.features.compression.service import get_service as get_compression_service
from gateway.features.insights.models import predict_output_length
from gateway.features.routing.service import get_service as get_routing_service
from gateway.modules.m1_pipeline.hinglish import code_mix_ratio
from gateway.pricing import tier_cost_usd

router = APIRouter(prefix="/api/v1/gateway", tags=["gateway-pipeline"])


class GatewayProcessRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    language: str = Field(default="en", max_length=16)
    policy: Literal["economy", "balanced", "quality"] = "balanced"
    format_constraint: Literal["none", "3 bullets", "JSON only", "no preamble"] = "none"
    max_tokens: int = Field(default=50, ge=10, le=500)
    compression_method: Literal["heuristic", "crf"] = "heuristic"
    compression_budget: float = Field(default=0.6, ge=0.1, le=1.0)
    protected_phrases: list[str] = Field(default_factory=list, max_length=20)
    simulate_upstream_failure: bool = False  # For chaos & resilience testing


@router.post("/process")
def process_request(
    payload: GatewayProcessRequest,
    request: Request,
    response: Response,
    cache_scope: str = Header(default="public", alias="X-Cache-Scope"),
) -> dict[str, Any]:
    """Unified, production-grade 4-stage optimization pipeline.

    Orchestrates:
    1. Semantic Cache Lookup (Exact & Embedding Cosine Search)
    2. Prompt Compression with protected entity preservation
    3. Difficulty-Aware Model Routing guarded by a Circuit Breaker
    4. Output Capping & Length Prediction & Cost Estimation
    5. ACID-compliant SQLite Transaction Logging
    """
    total_start = time.perf_counter()
    request_id = getattr(request.state, "request_id", "req-unknown")
    settings = get_settings()
    store = Store(settings.database_path)

    # -------------------------------------------------------------
    # STAGE 1: Dual-Tier Semantic Cache
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    cache_res = _lookup(payload.text, payload.language, cache_scope)
    cache_ms = (time.perf_counter() - t0) * 1000

    if cache_res and cache_res.get("hit"):
        total_ms = (time.perf_counter() - total_start) * 1000
        stage_latencies = {
            "cache_ms": round(cache_ms, 2),
            "compression_ms": 0.0,
            "routing_ms": 0.0,
            "capping_ms": 0.0,
            "total_ms": round(total_ms, 2),
        }
        response.headers["X-Stage-Latencies"] = (
            f"cache={cache_ms:.1f}ms; compression=0ms; routing=0ms; capping=0ms; total={total_ms:.1f}ms"
        )

        return {
            "request_id": request_id,
            "stage_latencies_ms": stage_latencies,
            "status": "completed",
            "cache_hit": True,
            "answer": cache_res.get("answer"),
            "answer_source": f"cache:{cache_res.get('match_type')}",
            "tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "cost": {
                "optimized_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "cost_saved_usd": 0.0,
                "cost_reduction_pct": 100.0,
            },
            "circuit_breaker": {
                "state": get_circuit_breaker().state.value,
                "fallback_applied": False,
            },
        }

    # -------------------------------------------------------------
    # STAGE 2: Lexical & CRF Prompt Compression
    # -------------------------------------------------------------
    t1 = time.perf_counter()
    comp_svc = get_compression_service()
    comp_res = comp_svc.analyze(
        payload.text,
        payload.compression_method,
        payload.compression_budget,
        payload.protected_phrases,
    )
    compression_ms = (time.perf_counter() - t1) * 1000
    compressed_text = comp_res.get("compressed", payload.text)

    # -------------------------------------------------------------
    # STAGE 3: Complexity Routing with Circuit Breaker
    # -------------------------------------------------------------
    t2 = time.perf_counter()
    router_svc = get_routing_service()
    circuit_breaker = get_circuit_breaker()

    def primary_routing():
        if payload.simulate_upstream_failure:
            raise RuntimeError("Upstream frontier model provider 503 Service Unavailable")
        return router_svc.predict(payload.text, payload.policy, compressed_text)

    def fallback_routing():
        # Fallback to safe local Economy tier
        return {
            "selected_tier": "cheap",
            "model": settings.cheap_model,
            "policy": payload.policy,
            "confidence": 1.0,
            "reason": "circuit_breaker_economy_fallback",
        }

    route_res, fallback_applied, cb_telemetry = circuit_breaker.execute(
        primary_routing, fallback_routing
    )
    routing_ms = (time.perf_counter() - t2) * 1000

    selected_tier = "premium" if route_res.get("selected_tier") in {"strong", "quality"} else "cheap"

    # -------------------------------------------------------------
    # STAGE 4: Output Capping, Length Prediction & Cost Engine
    # -------------------------------------------------------------
    t3 = time.perf_counter()
    input_tokens = comp_res.get("compressed_tokens", comp_res.get("original_tokens", 10))

    pred_res = predict_output_length(
        input_tokens=input_tokens,
        language=payload.language,
        format_constraint=payload.format_constraint,
        max_tokens=payload.max_tokens,
    )

    unconstrained_expected = 105  # Standard baseline completion length
    capped_tokens = min(pred_res.get("predicted_tokens", 40), payload.max_tokens)

    cost_opt = tier_cost_usd(selected_tier, input_tokens, capped_tokens)
    baseline_cost = tier_cost_usd(selected_tier, comp_res.get("original_tokens", input_tokens), unconstrained_expected)

    tokens_saved = max(0, unconstrained_expected - capped_tokens)
    cost_saved = max(0.0, baseline_cost - cost_opt)
    capping_ms = (time.perf_counter() - t3) * 1000

    # -------------------------------------------------------------
    # STAGE 5: Transaction Persistence & Telemetry
    # -------------------------------------------------------------
    total_ms = (time.perf_counter() - total_start) * 1000
    stage_latencies = {
        "cache_ms": round(cache_ms, 2),
        "compression_ms": round(compression_ms, 2),
        "routing_ms": round(routing_ms, 2),
        "capping_ms": round(capping_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    response.headers["X-Stage-Latencies"] = (
        f"cache={cache_ms:.1f}ms; compression={compression_ms:.1f}ms; "
        f"routing={routing_ms:.1f}ms; capping={capping_ms:.1f}ms; total={total_ms:.1f}ms"
    )

    result_data = {
        "request_id": request_id,
        "stage_latencies_ms": stage_latencies,
        "status": "completed",
        "cache_hit": False,
        "compression": {
            "method": payload.compression_method,
            "original_tokens": comp_res.get("original_tokens"),
            "compressed_tokens": comp_res.get("compressed_tokens"),
            "savings_ratio": comp_res.get("savings_ratio"),
            "compressed_text": compressed_text,
        },
        "routing": {
            "selected_tier": route_res.get("selected_tier"),
            "model": route_res.get("model", settings.cheap_model),
            "policy": payload.policy,
            "fallback_applied": fallback_applied,
        },
        "circuit_breaker": cb_telemetry,
        "output_capping": {
            "format_constraint": payload.format_constraint,
            "max_tokens_bound": payload.max_tokens,
            "predicted_output_tokens": capped_tokens,
            "tokens_saved_vs_baseline": tokens_saved,
            "token_reduction_pct": round((tokens_saved / unconstrained_expected) * 100, 1),
        },
        "cost": {
            "optimized_cost_usd": round(cost_opt, 8),
            "baseline_cost_usd": round(baseline_cost, 8),
            "cost_saved_usd": round(cost_saved, 8),
            "cost_reduction_pct": round((cost_saved / max(1e-6, baseline_cost)) * 100, 1),
        },
    }

    # Asynchronously record transaction in SQLite
    store.record_request(
        text=payload.text,
        language=payload.language,
        result={
            **result_data,
            "code_mix_ratio": round(code_mix_ratio(payload.text), 3),
        },
    )

    return result_data
