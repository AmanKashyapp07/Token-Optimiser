from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.app import app
from gateway.core.budgeting import AdaptiveTokenBudgeter, TokenBudgetResult


@pytest.fixture
def client():
    return TestClient(app)


def test_adaptive_token_budgeter_allocation():
    """Verify that AdaptiveTokenBudgeter computes hat_y, margin, and allocated budget."""
    budgeter = AdaptiveTokenBudgeter(floor_tokens=35, ceiling_tokens=250)
    res = budgeter.allocate(
        prompt="Explain the difference between TCP and UDP transport protocols.",
        language="en",
        format_constraint="none",
    )

    assert isinstance(res, TokenBudgetResult)
    assert res.predicted_tokens > 0
    assert res.prediction_interval_margin > 0
    assert res.allocated_budget >= 35
    assert res.allocated_budget <= 250
    assert res.strategy == "ml_adaptive"
    assert res.truncation_risk_warning is False


def test_adaptive_budgeter_with_format_constraints():
    """Verify that format constraints dynamically shift the allocated budget."""
    budgeter = AdaptiveTokenBudgeter()
    res_none = budgeter.allocate("Where is the assignment submission link?", format_constraint="none")
    res_bullets = budgeter.allocate("Where is the assignment submission link?", format_constraint="3 bullets")

    assert res_none.strategy == "ml_adaptive"
    assert res_bullets.strategy == "ml_adaptive"
    assert res_bullets.details["format_constraint"] == "3 bullets"


def test_user_override_truncation_warning():
    """When a user specifies a static cap far below the predicted requirement, flag risk."""
    budgeter = AdaptiveTokenBudgeter()
    # Complex query with tiny 10-token cap
    res = budgeter.allocate(
        prompt="Explain modern operating system virtual memory paging, page tables, and TLB misses.",
        user_max_tokens=10,
    )
    assert res.strategy == "user_override"
    assert res.allocated_budget == 10
    assert res.truncation_risk_warning is True


def test_chat_completions_adaptive_budgeting_stage(client: TestClient):
    """Calling /v1/chat/completions without max_tokens activates adaptive token budgeting."""
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "How does dynamic programming memoization work?"},
        ],
        "use_cache": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Verify stage in execution trace
    trace = data["trace"]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert "adaptive_token_budgeting" in stage_names

    budget_stage = next(s for s in trace["stages"] if s["stage"] == "adaptive_token_budgeting")
    assert budget_stage["status"] == "allocated"
    assert budget_stage["details"]["strategy"] == "ml_adaptive"
    assert budget_stage["details"]["allocated_budget"] >= 35

    # Verify response optimization telemetry
    optim = data["optimisation"]
    assert optim["budget_strategy"] == "ml_adaptive"
    assert optim["allocated_max_tokens"] >= 35


def test_heuristic_vs_ml_budgeter_methods():
    """Verify static heuristic and uniform budget calculation baselines."""
    budgeter = AdaptiveTokenBudgeter()
    h_budget = budgeter.static_heuristic_budget(input_tokens=20, factor=1.5, floor=40)
    assert h_budget == 40  # floor applied

    h_budget_large = budgeter.static_heuristic_budget(input_tokens=100, factor=1.5, floor=40)
    assert h_budget_large == 150

    u_budget = budgeter.static_uniform_budget(cap=60)
    assert u_budget == 60
