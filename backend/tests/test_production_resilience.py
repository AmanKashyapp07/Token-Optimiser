"""Tests for enterprise production features:
1. Distributed Request Tracing (X-Request-ID & X-Response-Time-Ms)
2. Sliding-Window Rate Limiting (HTTP 429 & Retry-After)
3. Circuit Breaker & Automatic Fallback Resilience
4. Granular Stage Latency Telemetry & End-to-End Gateway Processing
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from gateway.app import app
from gateway.core.resilience import CircuitBreaker, CircuitState, get_circuit_breaker


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ==============================================================================
# 1. Distributed Request Tracing & Observability
# ==============================================================================

def test_request_id_generation_and_propagation(client):
    """Every request must carry a unique X-Request-ID and response time header."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    assert len(res.headers["X-Request-ID"]) >= 8
    assert "X-Response-Time-Ms" in res.headers
    assert float(res.headers["X-Response-Time-Ms"]) >= 0.0


def test_custom_client_request_id_preserved(client):
    """If client sends an X-Request-ID, the gateway must preserve it for correlation."""
    custom_trace_id = "trace-corp-analytics-998877"
    res = client.get("/api/v1/health", headers={"X-Request-ID": custom_trace_id})
    assert res.status_code == 200
    assert res.headers["X-Request-ID"] == custom_trace_id


# ==============================================================================
# 2. Sliding-Window Rate Limiting
# ==============================================================================

def test_rate_limiter_allows_normal_traffic(client):
    """Normal traffic within quota receives X-RateLimit headers."""
    res = client.get("/api/v1/routing/models")
    assert res.status_code == 200
    assert "X-RateLimit-Limit" in res.headers
    assert "X-RateLimit-Remaining" in res.headers
    assert int(res.headers["X-RateLimit-Remaining"]) >= 0


def test_rate_limiter_blocks_burst_with_http_429(client):
    """Bursts exceeding the client quota are rejected with HTTP 429 and Retry-After."""
    headers = {"X-API-Key": "test-burst-tenant", "X-Test-Rate-Limit": "3"}

    # First 3 requests should pass
    for _ in range(3):
        res = client.get("/api/v1/routing/models", headers=headers)
        assert res.status_code == 200

    # 4th request must be rate-limited
    blocked_res = client.get("/api/v1/routing/models", headers=headers)
    assert blocked_res.status_code == 429
    assert blocked_res.headers.get("Retry-After") is not None
    data = blocked_res.json()
    assert "Rate limit exceeded" in data["detail"]
    assert data["retry_after_seconds"] >= 1


# ==============================================================================
# 3. Circuit Breaker & Resilience Pattern
# ==============================================================================

def test_circuit_breaker_state_transitions():
    """Circuit breaker trips to OPEN on threshold failures, and degrades gracefully."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.2)
    assert cb.state == CircuitState.CLOSED

    def failing_primary():
        raise RuntimeError("Provider 503")

    def safe_fallback():
        return {"tier": "economy_fallback"}

    # 1st failure
    res, fallback, meta = cb.execute(failing_primary, safe_fallback)
    assert fallback is True
    assert res["tier"] == "economy_fallback"
    assert cb.state == CircuitState.CLOSED  # 1 of 2 failures

    # 2nd failure -> trips circuit to OPEN
    res2, fallback2, meta2 = cb.execute(failing_primary, safe_fallback)
    assert fallback2 is True
    assert cb.state == CircuitState.OPEN

    # While OPEN, primary function is not even attempted
    called = []

    def should_not_run():
        called.append(True)
        raise RuntimeError("Should never be called")

    res3, fallback3, meta3 = cb.execute(should_not_run, safe_fallback)
    assert fallback3 is True
    assert len(called) == 0  # Bypassed upstream completely!
    assert meta3["fallback_reason"] == "circuit_breaker_open_upstream_unhealthy"

    # After recovery timeout, transitions to HALF_OPEN
    time.sleep(0.25)

    def healthy_primary():
        return {"tier": "recovered"}

    res4, fallback4, _ = cb.execute(healthy_primary, safe_fallback)
    assert fallback4 is False
    assert res4["tier"] == "recovered"
    assert cb.state == CircuitState.CLOSED


# ==============================================================================
# 4. Unified Gateway Processing Pipeline (POST /api/v1/gateway/process)
# ==============================================================================

def test_gateway_process_full_pipeline_normal(client):
    """End-to-end execution across Cache -> Compression -> Routing -> Output Capping."""
    get_circuit_breaker().reset()

    payload = {
        "text": "Dear Admin, could you please provide step by step instructions to access library journal archives online?",
        "language": "en",
        "policy": "balanced",
        "format_constraint": "3 bullets",
        "max_tokens": 40,
        "compression_method": "heuristic",
        "compression_budget": 0.6,
    }

    res = client.post("/api/v1/gateway/process", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Observability & Request Tracing
    assert "request_id" in data
    assert "X-Stage-Latencies" in res.headers
    assert "cache=" in res.headers["X-Stage-Latencies"]

    # Granular Stage Latencies
    latencies = data["stage_latencies_ms"]
    assert "cache_ms" in latencies
    assert "compression_ms" in latencies
    assert "routing_ms" in latencies
    assert "capping_ms" in latencies
    assert "total_ms" in latencies
    assert latencies["total_ms"] > 0

    # Pipeline Output Contracts
    assert data["cache_hit"] is False
    assert data["compression"]["compressed_tokens"] <= data["compression"]["original_tokens"]
    assert data["routing"]["selected_tier"] in {"cheap", "strong", "quality", "economy", "balanced"}
    assert data["output_capping"]["max_tokens_bound"] == 40
    assert data["output_capping"]["token_reduction_pct"] > 50.0
    assert data["cost"]["optimized_cost_usd"] < data["cost"]["baseline_cost_usd"]
    assert data["cost"]["cost_reduction_pct"] > 50.0


def test_gateway_process_circuit_breaker_resilience(client):
    """When upstream provider fails, circuit breaker protects system and falls back to cheap tier."""
    get_circuit_breaker().reset()

    payload = {
        "text": "What is the computational complexity of Dijkstra algorithm?",
        "language": "en",
        "policy": "quality",
        "format_constraint": "none",
        "max_tokens": 50,
        "simulate_upstream_failure": True,  # Injects upstream provider 503
    }

    res = client.post("/api/v1/gateway/process", json=payload)
    assert res.status_code == 200
    data = res.json()

    # System did not crash; fallback was gracefully applied
    assert data["routing"]["fallback_applied"] is True
    assert data["routing"]["selected_tier"] == "cheap"
    assert data["circuit_breaker"]["failures_logged"] >= 1
