"""Tests for Tasks 1-4:
1. Correctness (Token capping, bounds, cross-validation metrics)
2. Unified Inference API (/v1/chat/completions OpenAI compatibility)
3. Provider Architecture (Timeouts, retries, standardized exceptions, token counting)
4. Observability (Request IDs, Execution Traces, Latency Spans, and GET /api/v1/traces/{id})
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.app import app
from gateway.core.resilience import get_circuit_breaker


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ==============================================================================
# Task 2: Unified Inference API (/v1/chat/completions)
# ==============================================================================

def test_chat_completions_openai_compatible_schema(client):
    """The /v1/chat/completions endpoint conforms to the OpenAI specification."""
    get_circuit_breaker().reset()
    payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "You are a concise academic assistant."},
            {"role": "user", "content": "How do I renew my library book online before the due date?"},
        ],
        "policy": "balanced",
        "format_constraint": "3 bullets",
        "max_tokens": 50,
    }

    # Test both /v1 and /api/v1 routes
    for endpoint in ["/v1/chat/completions", "/api/v1/chat/completions"]:
        res = client.post(endpoint, json=payload)
        assert res.status_code == 200
        data = res.json()

        # OpenAI standard fields
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["choices"][0]["finish_reason"] in {"stop", "length"}

        # Standard token usage
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]

        # Gateway optimization extensions
        assert "optimisation" in data
        assert "cost_usd" in data["optimisation"]
        assert "baseline_cost_usd" in data["optimisation"]


def test_chat_completions_semantic_cache_hit(client):
    """When a query matches the semantic cache, it resolves with $0 cost and cache metadata."""
    client.post("/api/v1/cache/seed")
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "How can I renew a library book?"},
        ],
        "use_cache": True,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["optimisation"]["cache_hit"] is True
    assert data["optimisation"]["cost_usd"] == 0.0
    assert data["usage"]["prompt_tokens"] == 0  # Zero upstream prompt compute
    assert "cache" in data["optimisation"]["answer_source"]


# ==============================================================================
# Task 1 & 3: Correctness, Output Capping & Provider Truncation
# ==============================================================================

def test_chat_completions_strict_output_capping(client):
    """Enforcing a tight max_tokens boundary truncates completion and sets finish_reason='length'."""
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Explain the full history of computer operating systems in detail."},
        ],
        "max_tokens": 15,  # Very tight token cap
        "use_cache": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["usage"]["completion_tokens"] <= 15
    assert data["choices"][0]["finish_reason"] == "length"


def test_chat_completions_resilience_circuit_breaker(client):
    """Simulated provider failure automatically triggers circuit breaker fallback."""
    get_circuit_breaker().reset()
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "What is the time complexity of QuickSort?"},
        ],
        "simulate_provider_failure": True,
        "use_cache": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Request succeeds via fallback
    assert len(data["choices"][0]["message"]["content"]) > 0
    trace = data["trace"]
    inference_stage = next(s for s in trace["stages"] if s["stage"] == "provider_inference")
    assert inference_stage["status"] == "fallback"
    assert inference_stage["details"]["fallback_applied"] is True


# ==============================================================================
# Task 4: Observability, Execution Traces & Retrieval
# ==============================================================================

def test_execution_trace_generation_and_retrieval(client):
    """Every completion generates a structured trace retrievable via /api/v1/traces/{id}."""
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Where can I check course syllabus and assignment submission links?"},
        ],
        "format_constraint": "none",
        "use_cache": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Verify Trace object in response
    trace = data["trace"]
    trace_id = trace["trace_id"]
    assert trace_id.startswith("trace-")
    assert trace["total_duration_ms"] > 0
    assert len(trace["stages"]) >= 4

    stage_names = [s["stage"] for s in trace["stages"]]
    assert "semantic_cache" in stage_names
    assert "prompt_compression" in stage_names
    assert "model_routing" in stage_names
    assert "provider_inference" in stage_names

    # Verify Response Headers
    assert res.headers.get("X-Execution-Trace-Id") == trace_id
    assert "X-Stage-Latencies" in res.headers
    assert "cache=" in res.headers["X-Stage-Latencies"]

    # Retrieve Trace from GET /api/v1/traces/{trace_id}
    trace_res = client.get(f"/api/v1/traces/{trace_id}")
    assert trace_res.status_code == 200
    fetched_trace = trace_res.json()
    assert fetched_trace["trace_id"] == trace_id
    assert fetched_trace["total_duration_ms"] == trace["total_duration_ms"]


def test_trace_not_found_returns_404(client):
    """Querying a non-existent trace returns a clean 404."""
    res = client.get("/api/v1/traces/non-existent-trace-id-12345")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_chat_completions_sse_streaming(client):
    """When stream=True, the gateway returns an SSE event stream with data: [DONE]."""
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Tell me about course registration."},
        ],
        "stream": True,
        "use_cache": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")

    body = res.text
    assert "data: " in body
    assert "data: [DONE]" in body
    assert "chat.completion.chunk" in body


def test_chat_completions_multi_turn_history_and_system_prompt(client):
    """Multi-turn conversation history and system instructions are preserved in execution trace."""
    payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "You are a concise academic tutor."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."},
            {"role": "user", "content": "What are its key benefits?"},
        ],
        "use_cache": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["choices"][0]["message"]["content"]) > 0
    trace = data["trace"]
    comp_stage = next(s for s in trace["stages"] if s["stage"] == "prompt_compression")
    assert comp_stage["status"] == "applied"

