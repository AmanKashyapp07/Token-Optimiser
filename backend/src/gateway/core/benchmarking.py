from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from gateway.core.chat import ChatCompletionRequest, ChatMessage, execute_chat_completion
from gateway.core.config import get_settings

router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmarking"])

BENCHMARK_PROMPTS = [
    {"text": "How can I renew a library book?", "lang": "en", "fmt": "3 bullets"},
    {"text": "How do I connect to campus Wi-Fi?", "lang": "en", "fmt": "none"},
    {"text": "Where can I find the assignment deadline?", "lang": "en", "fmt": "JSON only"},
    {"text": "Meri library book kaise renew hogi online?", "lang": "hinglish", "fmt": "3 bullets"},
    {"text": "Campus Wi-Fi connect nahi ho raha hai phone me, please help.", "lang": "hinglish", "fmt": "none"},
    {"text": "पुस्तकालय की किताब ऑनलाइन कैसे नवीनीकृत करें?", "lang": "hi", "fmt": "none"},
    {"text": "असाइनमेंट सबमिशन की अंतिम तिथि क्या है?", "lang": "hi", "fmt": "no preamble"},
    {"text": "What is the computational complexity of merge sort vs quicksort?", "lang": "en", "fmt": "3 bullets"},
    {"text": "Can I get an extension for assignment 2 submission due to medical leave?", "lang": "en", "fmt": "none"},
    {"text": "Hostel mess menu and breakfast timings kya hain?", "lang": "hinglish", "fmt": "none"},
]


class BenchmarkRequest(BaseModel):
    num_requests: int = Field(default=20, ge=5, le=200)
    concurrency: int = Field(default=4, ge=1, le=20)
    use_cache: bool = True
    policy: Literal["economy", "balanced", "quality"] = "balanced"


class LatencyMetrics(BaseModel):
    min_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class QualityMetrics(BaseModel):
    non_empty_rate_pct: float
    format_adherence_pct: float
    error_count: int


class BenchmarkResult(BaseModel):
    total_requests: int
    concurrency: int
    wall_clock_time_s: float
    throughput_rps: float
    cache_hit_ratio: float
    cache_hits: int
    cache_misses: int
    latency: LatencyMetrics
    quality: QualityMetrics
    tokens: dict[str, int]
    economics: dict[str, Any]


async def run_gateway_benchmark(
    num_requests: int = 20,
    concurrency: int = 4,
    use_cache: bool = True,
    policy: str = "balanced",
) -> BenchmarkResult:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    cache_hits = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    baseline_cost = 0.0
    valid_format_count = 0
    errors = 0

    class DummyState:
        request_id = "bench-req"

    class DummyRequest:
        state = DummyState()
        headers = {}

    class DummyResponse:
        headers = {}

    async def single_worker(idx: int):
        nonlocal cache_hits, total_prompt_tokens, total_completion_tokens
        nonlocal total_cost, baseline_cost, valid_format_count, errors

        item = BENCHMARK_PROMPTS[idx % len(BENCHMARK_PROMPTS)]
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=item["text"])],
            policy=policy,  # type: ignore
            format_constraint=item["fmt"],  # type: ignore
            use_cache=use_cache,
            max_tokens=50,
        )

        async with semaphore:
            t0 = time.perf_counter()
            try:
                res = await execute_chat_completion(req, DummyRequest(), DummyResponse())
                dur_ms = (time.perf_counter() - t0) * 1000
                latencies.append(dur_ms)

                if res.optimisation.get("cache_hit"):
                    cache_hits += 1

                total_prompt_tokens += res.usage.prompt_tokens
                total_completion_tokens += res.usage.completion_tokens
                total_cost += res.optimisation.get("cost_usd", 0.0)
                baseline_cost += res.optimisation.get("baseline_cost_usd", 0.0)

                content = res.choices[0].message.content
                if content and content.strip():
                    fmt = item["fmt"]
                    if fmt == "JSON only":
                        clean_json = content.strip()
                        if clean_json.startswith("```json"):
                            clean_json = clean_json[7:]
                        if clean_json.startswith("```"):
                            clean_json = clean_json[3:]
                        if clean_json.endswith("```"):
                            clean_json = clean_json[:-3]
                        try:
                            json.loads(clean_json.strip())
                            valid_format_count += 1
                        except Exception:
                            pass  # Invalid JSON, do NOT increment
                    elif fmt == "3 bullets":
                        bullet_count = sum(
                            1 for line in content.splitlines()
                            if line.strip().startswith(("•", "-", "*"))
                            or (len(line.strip()) > 2 and line.strip()[0].isdigit() and line.strip()[1] in {".", ")"})
                        )
                        if bullet_count >= 2:
                            valid_format_count += 1
                    elif fmt == "no preamble":
                        low = content.lower().strip()
                        has_preamble = any(low.startswith(p) for p in ["sure,", "certainly,", "here is", "here are"])
                        if not has_preamble:
                            valid_format_count += 1
                    elif fmt == "none":
                        valid_format_count += 1
            except Exception:
                errors += 1

    t_start = time.perf_counter()
    tasks = [single_worker(i) for i in range(num_requests)]
    await asyncio.gather(*tasks)
    wall_clock = time.perf_counter() - t_start

    arr = np.array(latencies if latencies else [1.0])
    completed_reqs = len(latencies)

    latency_metrics = LatencyMetrics(
        min_ms=round(float(np.min(arr)), 2),
        mean_ms=round(float(np.mean(arr)), 2),
        p50_ms=round(float(np.percentile(arr, 50)), 2),
        p90_ms=round(float(np.percentile(arr, 90)), 2),
        p95_ms=round(float(np.percentile(arr, 95)), 2),
        p99_ms=round(float(np.percentile(arr, 99)), 2),
        max_ms=round(float(np.max(arr)), 2),
    )

    quality_metrics = QualityMetrics(
        non_empty_rate_pct=round((completed_reqs / max(1, num_requests)) * 100, 1),
        format_adherence_pct=round((valid_format_count / max(1, completed_reqs)) * 100, 1),
        error_count=errors,
    )

    cost_saved = max(0.0, baseline_cost - total_cost)
    pct_saved = round((cost_saved / max(1e-6, baseline_cost)) * 100, 1)

    return BenchmarkResult(
        total_requests=num_requests,
        concurrency=concurrency,
        wall_clock_time_s=round(wall_clock, 2),
        throughput_rps=round(completed_reqs / max(0.001, wall_clock), 1),
        cache_hit_ratio=round(cache_hits / max(1, completed_reqs), 3),
        cache_hits=cache_hits,
        cache_misses=completed_reqs - cache_hits,
        latency=latency_metrics,
        quality=quality_metrics,
        tokens={
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        economics={
            "optimized_cost_usd": round(total_cost, 6),
            "baseline_cost_usd": round(baseline_cost, 6),
            "total_saved_usd": round(cost_saved, 6),
            "cost_reduction_pct": pct_saved,
        },
    )


@router.post("/run", response_model=BenchmarkResult)
async def run_benchmark(payload: BenchmarkRequest = BenchmarkRequest()) -> BenchmarkResult:
    """Executes a live load benchmark measuring throughput, p50/p95/p99 latency, cache hit ratio, and cost."""
    return await run_gateway_benchmark(
        num_requests=payload.num_requests,
        concurrency=payload.concurrency,
        use_cache=payload.use_cache,
        policy=payload.policy,
    )


@router.get("/summary", response_model=BenchmarkResult)
async def benchmark_summary() -> BenchmarkResult:
    """Fast, pre-packaged reference benchmark run for quick telemetry queries."""
    return await run_gateway_benchmark(num_requests=10, concurrency=2, use_cache=True)
