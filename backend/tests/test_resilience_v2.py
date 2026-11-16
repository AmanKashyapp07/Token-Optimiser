from __future__ import annotations

import asyncio
import pytest
from fastapi.testclient import TestClient

from gateway.app import app
from gateway.core.config import Settings
from gateway.core.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from gateway.features.cache.coalescing import (
    LeaderExecutionError,
    RequestCoalescer,
)
from gateway.features.cache.distributed_singleflight import (
    DistributedRequestCoalescer,
    MockRedisClient,
)


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. Singleflight Leader-Follower Exception & Cancellation Concurrency Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_singleflight_concurrent_followers_leader_failure():
    """When leader raises an error, all 50 concurrent followers receive LeaderExecutionError without deadlock."""
    coalescer = RequestCoalescer()
    fail_barrier = asyncio.Event()

    async def failing_leader():
        await fail_barrier.wait()
        raise RuntimeError("Simulated Upstream 503 Provider Gateway Timeout")

    # Launch 1 leader + 49 followers
    leader_task = asyncio.create_task(coalescer.do("test-leader-fail-key", failing_leader))
    await asyncio.sleep(0.01)  # Ensure leader registers in-flight

    follower_tasks = [
        asyncio.create_task(coalescer.do("test-leader-fail-key", failing_leader))
        for _ in range(49)
    ]

    assert coalescer.active_in_flight == 1

    # Trigger leader execution failure
    fail_barrier.set()

    # Verify leader receives raw exception
    with pytest.raises(RuntimeError, match="Simulated Upstream 503"):
        await leader_task

    # Verify all 49 followers receive structured LeaderExecutionError
    follower_results = await asyncio.gather(*follower_tasks, return_exceptions=True)
    assert len(follower_results) == 49

    for res in follower_results:
        assert isinstance(res, LeaderExecutionError)
        assert res.leader_failed is True
        assert res.key == "test-leader-fail-key"
        assert "Simulated Upstream 503" in str(res.original_error)

    # In-flight map is cleanly evacuated
    assert coalescer.active_in_flight == 0


@pytest.mark.asyncio
async def test_singleflight_follower_cancellation_does_not_abort_leader():
    """Cancelling a follower task does not cancel the in-flight leader or other followers."""
    coalescer = RequestCoalescer()
    start_gate = asyncio.Event()
    finish_gate = asyncio.Event()

    async def slow_leader():
        start_gate.set()
        await finish_gate.wait()
        return {"data": "leader_success"}

    # Start leader
    leader_task = asyncio.create_task(coalescer.do("key-cancel-test", slow_leader))
    await start_gate.wait()

    # Start 2 followers
    follower_1 = asyncio.create_task(coalescer.do("key-cancel-test", slow_leader))
    follower_2 = asyncio.create_task(coalescer.do("key-cancel-test", slow_leader))
    await asyncio.sleep(0.01)

    # Cancel follower 1
    follower_1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower_1

    # Release leader
    finish_gate.set()

    leader_res, leader_coalesced = await leader_task
    follower_2_res, follower_2_coalesced = await follower_2

    assert leader_res == {"data": "leader_success"}
    assert leader_coalesced is False
    assert follower_2_res == {"data": "leader_success"}
    assert follower_2_coalesced is True
    assert coalescer.active_in_flight == 0


@pytest.mark.asyncio
async def test_singleflight_telemetry_metrics_accounting():
    """RequestCoalescer accurately tracks total requests, coalesced hits, executions, and hit ratio."""
    coalescer = RequestCoalescer()

    async def fast_fn():
        await asyncio.sleep(0.02)
        return "ok"

    # 1 leader + 3 followers
    tasks = [asyncio.create_task(coalescer.do("metrics-key", fast_fn)) for _ in range(4)]
    await asyncio.gather(*tasks)

    metrics = coalescer.get_metrics()
    assert metrics["total_requests"] == 4
    assert metrics["coalesced_requests"] == 3
    assert metrics["leader_executions"] == 1
    assert metrics["leader_failures"] == 0
    assert metrics["coalesce_hit_ratio"] == 0.75

    coalescer.reset_metrics()
    reset_m = coalescer.get_metrics()
    assert reset_m["total_requests"] == 0
    assert reset_m["coalesced_requests"] == 0


# ==============================================================================
# 2. Circuit Breaker Hardening & Unhandled Failure Isolation
# ==============================================================================

def test_circuit_breaker_without_fallback_raises_errors_and_trips():
    """When no fallback is provided, CircuitBreaker raises original error, then CircuitBreakerOpenError."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    def failing_fn():
        raise ConnectionResetError("Provider TCP reset")

    # 1st failure raises ConnectionResetError
    with pytest.raises(ConnectionResetError):
        cb.execute(failing_fn, fallback_fn=None)
    assert cb.state == CircuitState.CLOSED

    # 2nd failure raises ConnectionResetError and trips circuit to OPEN
    with pytest.raises(ConnectionResetError):
        cb.execute(failing_fn, fallback_fn=None)
    assert cb.state == CircuitState.OPEN

    # 3rd attempt raises CircuitBreakerOpenError immediately without calling failing_fn
    with pytest.raises(CircuitBreakerOpenError, match="Circuit breaker is OPEN"):
        cb.execute(failing_fn, fallback_fn=None)


# ==============================================================================
# 3. Production Failure Isolation: No Silent Fake Fixtures in Production
# ==============================================================================

def test_production_environment_rejects_silent_fixtures_with_http_503(client, monkeypatch):
    """In production mode, upstream provider failures return HTTP 503 instead of silent canned fixtures."""
    prod_settings = Settings(environment="production", mode="live")
    monkeypatch.setattr("gateway.core.chat.get_settings", lambda: prod_settings)

    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "simulate_provider_failure": True,
        "use_cache": False,
    }

    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 503
    data = res.json()
    assert "upstream inference provider unavailable" in data["detail"].lower()


def test_demo_environment_allows_fixture_with_x_gateway_fallback_header(client, monkeypatch):
    """In demo mode, provider failure gracefully uses fixture and sets X-Gateway-Fallback: true."""
    demo_settings = Settings(environment="demo", mode="demo")
    monkeypatch.setattr("gateway.core.chat.get_settings", lambda: demo_settings)

    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "simulate_provider_failure": True,
        "use_cache": False,
    }

    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert res.headers.get("X-Gateway-Fallback") == "true"

    data = res.json()
    assert data["optimisation"]["fallback_applied"] is True
    assert data["optimisation"]["answer_source"] == "fallback_fixture"


def test_successful_request_sets_x_gateway_fallback_false(client):
    """Successful normal completions explicitly declare X-Gateway-Fallback: false."""
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Say hello"},
        ],
        "use_cache": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert res.headers.get("X-Gateway-Fallback") == "false"
    data = res.json()
    assert data["optimisation"]["fallback_applied"] is False


# ==============================================================================
# 4. Distributed Multi-Instance Architecture Blueprint Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_distributed_singleflight_local_fallback():
    """DistributedRequestCoalescer falls back cleanly to local singleflight when Redis is unconfigured."""
    dist_coalescer = DistributedRequestCoalescer(redis_client=None)

    async def compute():
        return {"result": 42}

    res, coalesced = await dist_coalescer.do("test-local-fallback", compute)
    assert res == {"result": 42}
    assert coalesced is False


@pytest.mark.asyncio
async def test_distributed_singleflight_multi_worker_cluster_coalescing():
    """Simulates 5 distributed gateway workers: exactly 1 leader executes, 4 followers receive via pub/sub."""
    mock_redis = MockRedisClient()
    workers = [
        DistributedRequestCoalescer(redis_client=mock_redis, worker_id=f"worker-{i}")
        for i in range(5)
    ]

    execution_count = 0

    async def expensive_llm_inference():
        nonlocal execution_count
        execution_count += 1
        await asyncio.sleep(0.05)
        return {"answer": "Paris is the capital of France", "tokens": 7}

    # All 5 workers execute simultaneously on identical query
    tasks = [
        asyncio.create_task(w.do("cluster-query-france", expensive_llm_inference))
        for w in workers
    ]

    results = await asyncio.gather(*tasks)

    # 1. Exactly 1 execution occurred across the entire cluster!
    assert execution_count == 1

    # 2. All 5 workers received the identical output
    for res, _ in results:
        assert res == {"answer": "Paris is the capital of France", "tokens": 7}

    # 3. Exactly 1 leader and 4 coalesced followers
    leaders = [res for res, was_coalesced in results if not was_coalesced]
    followers = [res for res, was_coalesced in results if was_coalesced]
    assert len(leaders) == 1
    assert len(followers) == 4


@pytest.mark.asyncio
async def test_distributed_singleflight_leader_failure_broadcast():
    """When distributed leader fails, followers receive error over pub/sub and do not hang."""
    mock_redis = MockRedisClient()
    leader = DistributedRequestCoalescer(redis_client=mock_redis, worker_id="worker-leader")
    follower = DistributedRequestCoalescer(redis_client=mock_redis, worker_id="worker-follower")

    fail_barrier = asyncio.Event()

    async def failing_inference():
        await fail_barrier.wait()
        raise TimeoutError("Upstream Provider Gateway SLA Exceeded 10.0s")

    leader_task = asyncio.create_task(leader.do("cluster-fail-key", failing_inference))
    await asyncio.sleep(0.01)

    follower_task = asyncio.create_task(follower.do("cluster-fail-key", failing_inference))

    # Trigger failure
    fail_barrier.set()

    with pytest.raises(TimeoutError, match="Upstream Provider Gateway SLA"):
        await leader_task

    with pytest.raises(LeaderExecutionError, match="Upstream Provider Gateway SLA"):
        await follower_task
