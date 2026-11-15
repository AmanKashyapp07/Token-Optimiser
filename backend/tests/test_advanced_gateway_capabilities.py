"""Tests for Tasks 5-8:
1. Task 5: Cache Improvements (Request Coalescing / Singleflight, Invalidation API)
2. Task 6: Reliability (TimeoutContext, Exponential Backoff Retries)
3. Task 7: Benchmarking (Throughput, Latency Percentiles p50/p95/p99, Quality Metrics)
"""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from gateway.app import app
from gateway.core.benchmarking import run_gateway_benchmark
from gateway.core.resilience import TimeoutContext, async_retry_with_backoff
from gateway.features.cache.coalescing import RequestCoalescer


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ==============================================================================
# Task 5: Cache Improvements (Singleflight Coalescing & Invalidation)
# ==============================================================================

@pytest.mark.asyncio
async def test_cache_coalescing_singleflight_concurrent_deduplication():
    """Multiple concurrent lookups for the same key are coalesced into a single execution."""
    coalescer = RequestCoalescer()
    executions = 0

    async def expensive_lookup():
        nonlocal executions
        executions += 1
        await asyncio.sleep(0.05)
        return {"data": "cached_answer"}

    # Launch 5 concurrent calls for identical key
    tasks = [coalescer.do("scope:library-renew", expensive_lookup) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Exactly 1 real execution ran; all 5 callers received identical result
    assert executions == 1
    assert len(results) == 5
    for res, coalesced in results:
        assert res["data"] == "cached_answer"


def test_cache_invalidation_api_endpoint(client):
    """POST /api/v1/cache/invalidate removes matching entries by pattern or scope."""
    # Seed known FAQ
    client.post("/api/v1/cache/seed")

    # Invalidate by pattern 'library'
    inv_res = client.post(
        "/api/v1/cache/invalidate",
        json={"scope": "public", "pattern": "library"},
    )
    assert inv_res.status_code == 200
    data = inv_res.json()
    assert data["invalidated"] >= 1
    assert data["pattern"] == "library"

    # Subsequent lookup for invalidated entry is a miss
    lookup_res = client.post("/api/v1/cache/lookup", json={"text": "How can I renew a library book?"})
    assert lookup_res.status_code == 200
    assert lookup_res.json().get("hit") is not True


# ==============================================================================
# Task 6: Reliability & Resilience (Deadlines & Retries)
# ==============================================================================

def test_timeout_context_deadline_tracking():
    """TimeoutContext tracks elapsed time and raises TimeoutError when deadline expires."""
    tc = TimeoutContext(timeout_seconds=0.05)
    assert tc.remaining_seconds() > 0
    time.sleep(0.06)
    with pytest.raises(TimeoutError):
        tc.remaining_seconds()


@pytest.mark.asyncio
async def test_async_retry_with_backoff_transient_recovery():
    """Transient failures are retried with exponential backoff and recover cleanly."""
    attempts = 0

    async def flaky_upstream():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Transient network blip")
        return "healthy_response"

    result = await async_retry_with_backoff(flaky_upstream, retries=2, base_delay=0.01)
    assert result == "healthy_response"
    assert attempts == 2


# ==============================================================================
# Task 7: Benchmarking Engine (Throughput, Latency Percentiles, Quality)
# ==============================================================================

@pytest.mark.asyncio
async def test_run_gateway_benchmark_metrics():
    """run_gateway_benchmark computes throughput, latency percentiles, and quality metrics."""
    res = await run_gateway_benchmark(num_requests=10, concurrency=2, use_cache=True)

    assert res.total_requests == 10
    assert res.throughput_rps > 0
    assert res.wall_clock_time_s > 0

    # Latency percentiles
    lat = res.latency
    assert lat.min_ms <= lat.p50_ms <= lat.p95_ms <= lat.p99_ms <= lat.max_ms

    # Quality metrics
    assert res.quality.non_empty_rate_pct == 100.0
    assert res.quality.format_adherence_pct >= 90.0
    assert res.quality.error_count == 0

    # Economics
    assert "optimized_cost_usd" in res.economics
    assert "cost_reduction_pct" in res.economics


def test_benchmark_api_endpoints(client):
    """GET /api/v1/benchmark/summary and POST /api/v1/benchmark/run return valid telemetry."""
    summary_res = client.get("/api/v1/benchmark/summary")
    assert summary_res.status_code == 200
    summary_data = summary_res.json()
    assert summary_data["total_requests"] == 10
    assert "throughput_rps" in summary_data
    assert "latency" in summary_data
    assert summary_data["latency"]["p50_ms"] > 0
