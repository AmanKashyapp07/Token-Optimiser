"""Unit and integration tests for Model Complexity Router Quadrant Benchmark."""

import json
from pathlib import Path

import pytest

from gateway.features.routing.benchmark import evaluate_routing_policy
from gateway.features.routing.service import RoutingService

pytestmark = [pytest.mark.unit]


def test_routing_benchmark_dataset_integrity():
    dataset_file = (
        Path(__file__).resolve().parents[2] / "backend" / "data" / "eval" / "routing_benchmark_200.jsonl"
    )
    assert dataset_file.exists(), f"Missing dataset: {dataset_file}"

    records = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 200, f"Expected 200 queries, found {len(records)}"

    quadrants = {r["quadrant"] for r in records}
    assert quadrants == {"short_hard", "long_easy", "standard_routine", "standard_complex"}

    # Each quadrant must contain exactly 50 items
    for quad in quadrants:
        items = [r for r in records if r["quadrant"] == quad]
        assert len(items) == 50, f"Quadrant {quad} has {len(items)} items, expected 50"


def test_short_hard_routing_behavior():
    router = RoutingService()
    # High-complexity mathematical proofs and algorithmic analysis must route to strong
    sh_query = "Prove that sqrt(2) is irrational."
    pred = router.predict(sh_query, policy="balanced")
    assert pred["selected_tier"] == "strong"
    assert pred["score"] >= pred["threshold"]


def test_standard_complex_routing_behavior():
    router = RoutingService()
    sc_query = (
        "Design a distributed, highly available transaction coordinator supporting two-phase commit "
        "with Paxos consensus and partition recovery."
    )
    pred = router.predict(sc_query, policy="balanced")
    assert pred["selected_tier"] == "strong"
    assert pred["score"] > 0.70


def test_policy_monotonicity_and_under_routing():
    dataset_file = (
        Path(__file__).resolve().parents[2] / "backend" / "data" / "eval" / "routing_benchmark_200.jsonl"
    )
    items = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    router = RoutingService()
    res_economy = evaluate_routing_policy(router, items, policy="economy")
    res_balanced = evaluate_routing_policy(router, items, policy="balanced")
    res_quality = evaluate_routing_policy(router, items, policy="quality")

    # Quality policy must have lowest under-routing risk on complex queries
    assert res_balanced["under_routing_rate_pct"] <= res_economy["under_routing_rate_pct"]
    assert res_quality["under_routing_rate_pct"] <= res_balanced["under_routing_rate_pct"]

    # Economy policy must have lowest cost / 10k
    assert (
        res_economy["cost_comparison"]["router_cost_10k_usd"]
        <= res_quality["cost_comparison"]["router_cost_10k_usd"]
    )

    # Router must provide cost savings compared to Always-Quality
    assert res_economy["cost_comparison"]["savings_vs_always_quality_pct"] > 25.0
