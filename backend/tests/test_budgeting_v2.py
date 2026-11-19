from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.core.budgeting import AdaptiveTokenBudgeter, TokenBudgetResult


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prompt", "language", "floor", "ceiling"),
    [
        ("Explain the difference between TCP and UDP transport protocols.", "en", 35, 250),
        ("Bhai mujhe hostel room allot kaise karwana hai?", "hi", 25, 300),
        ("def quicksort(arr): return arr", "code", 30, 200),
    ],
)
def test_adaptive_token_budgeter_allocation(prompt: str, language: str, floor: int, ceiling: int):
    """Verify that AdaptiveTokenBudgeter computes hat_y, margin, and allocated budget within bounds."""
    budgeter = AdaptiveTokenBudgeter(floor_tokens=floor, ceiling_tokens=ceiling)
    res = budgeter.allocate(
        prompt=prompt,
        language=language,
        format_constraint="none",
    )

    assert isinstance(res, TokenBudgetResult)
    assert res.predicted_tokens > 0
    assert res.prediction_interval_margin > 0
    assert res.allocated_budget >= floor
    assert res.allocated_budget <= ceiling
    assert res.strategy == "ml_adaptive"
    assert res.truncation_risk_warning is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("format_constraint", "expected_detail"),
    [
        ("none", "none"),
        ("1 sentence", "1 sentence"),
        ("3 bullets", "3 bullets"),
        ("code only", "code only"),
        ("json", "json"),
    ],
)
def test_adaptive_budgeter_with_format_constraints(format_constraint: str, expected_detail: str):
    """Verify that format constraints dynamically shift the allocated budget metadata."""
    budgeter = AdaptiveTokenBudgeter()
    res = budgeter.allocate(
        "Where is the assignment submission link?",
        format_constraint=format_constraint,
    )

    assert res.strategy == "ml_adaptive"
    assert res.details["format_constraint"] == expected_detail
    assert res.allocated_budget >= budgeter.floor_tokens


@pytest.mark.unit
@pytest.mark.parametrize(
    ("user_cap", "expect_warning"),
    [
        (10, True),  # Starvation cap triggers truncation risk warning
        (500, False),  # Ample cap proceeds safely
    ],
)
def test_user_override_truncation_warning(user_cap: int, expect_warning: bool):
    """When a user specifies a static cap far below the predicted requirement, flag risk."""
    budgeter = AdaptiveTokenBudgeter()
    res = budgeter.allocate(
        prompt="Explain modern operating system virtual memory paging, page tables, and TLB misses.",
        user_max_tokens=user_cap,
    )
    assert res.strategy == "user_override"
    assert res.truncation_risk_warning is expect_warning
    if expect_warning:
        assert res.allocated_budget == user_cap
    else:
        assert res.allocated_budget <= user_cap


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_tokens", "factor", "floor", "expected"),
    [
        (20, 1.5, 40, 40),  # Floor clamp
        (100, 1.5, 40, 150),  # Linear scaling
        (0, 2.0, 50, 50),  # Zero tokens floor
        (300, 1.0, 20, 300),  # Exact product
    ],
)
def test_heuristic_vs_ml_budgeter_methods(input_tokens: int, factor: float, floor: int, expected: int):
    """Verify static heuristic and uniform budget calculation baselines."""
    budgeter = AdaptiveTokenBudgeter()
    h_budget = budgeter.static_heuristic_budget(input_tokens=input_tokens, factor=factor, floor=floor)
    assert h_budget == expected

    u_budget = budgeter.static_uniform_budget(cap=60)
    assert u_budget == 60


@pytest.mark.integration
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
