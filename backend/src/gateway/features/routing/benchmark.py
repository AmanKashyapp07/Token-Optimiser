"""Benchmark evaluation helpers for Model Complexity Router across 4 Quadrants."""

from __future__ import annotations

from typing import Any

from gateway.features.routing.service import RoutingService

# Pricing rates ($ / MTok)
CHEAP_IN_RATE = 0.075
CHEAP_OUT_RATE = 0.30
PREMIUM_IN_RATE = 0.30
PREMIUM_OUT_RATE = 1.20


def evaluate_routing_policy(
    router: RoutingService, items: list[dict[str, Any]], policy: str
) -> dict[str, Any]:
    quadrant_stats = {
        "short_hard": {"total": 0, "correct": 0, "under_routed": 0, "over_routed": 0},
        "long_easy": {"total": 0, "correct": 0, "under_routed": 0, "over_routed": 0},
        "standard_routine": {"total": 0, "correct": 0, "under_routed": 0, "over_routed": 0},
        "standard_complex": {"total": 0, "correct": 0, "under_routed": 0, "over_routed": 0},
    }

    routed_counts = {"cheap": 0, "strong": 0}
    total_items = len(items)
    correct_count = 0

    total_cheap_truth = 0
    total_strong_truth = 0
    false_cheap_count = 0
    false_strong_count = 0

    for item in items:
        quad = item["quadrant"]
        target = item["target_tier"]
        prompt = item["prompt"]

        if target == "cheap":
            total_cheap_truth += 1
        else:
            total_strong_truth += 1

        quadrant_stats[quad]["total"] += 1

        pred = router.predict(prompt, policy=policy)
        pred_tier = pred["selected_tier"]
        routed_counts[pred_tier] += 1

        if pred_tier == target:
            correct_count += 1
            quadrant_stats[quad]["correct"] += 1
        elif target == "strong" and pred_tier == "cheap":
            false_cheap_count += 1
            quadrant_stats[quad]["under_routed"] += 1
        elif target == "cheap" and pred_tier == "strong":
            false_strong_count += 1
            quadrant_stats[quad]["over_routed"] += 1

    overall_accuracy = (correct_count / total_items) * 100
    under_routing_rate = (false_cheap_count / total_strong_truth) * 100 if total_strong_truth else 0.0
    over_routing_rate = (false_strong_count / total_cheap_truth) * 100 if total_cheap_truth else 0.0

    avg_in_tok = 40
    avg_out_tok = 80
    cheap_cost_single = (avg_in_tok * CHEAP_IN_RATE + avg_out_tok * CHEAP_OUT_RATE) / 1_000_000
    premium_cost_single = (avg_in_tok * PREMIUM_IN_RATE + avg_out_tok * PREMIUM_OUT_RATE) / 1_000_000

    always_cheap_cost_10k = cheap_cost_single * 10_000
    always_quality_cost_10k = premium_cost_single * 10_000

    router_cost_total = (
        routed_counts["cheap"] * cheap_cost_single + routed_counts["strong"] * premium_cost_single
    )
    router_cost_10k = (router_cost_total / total_items) * 10_000

    cost_savings_vs_quality_pct = max(
        0.0, ((always_quality_cost_10k - router_cost_10k) / always_quality_cost_10k) * 100
    )

    quad_summary = {}
    for q_name, stats in quadrant_stats.items():
        q_tot = stats["total"]
        q_acc = (stats["correct"] / q_tot) * 100 if q_tot else 0.0
        quad_summary[q_name] = {
            "total": q_tot,
            "correct": stats["correct"],
            "accuracy_pct": round(q_acc, 1),
            "under_routed": stats["under_routed"],
            "over_routed": stats["over_routed"],
        }

    return {
        "policy": policy,
        "sample_size": total_items,
        "correct_predictions": correct_count,
        "accuracy_pct": round(overall_accuracy, 1),
        "under_routing_rate_pct": round(under_routing_rate, 1),
        "over_routing_rate_pct": round(over_routing_rate, 1),
        "routed_counts": routed_counts,
        "quadrants": quad_summary,
        "cost_comparison": {
            "always_cheap_cost_10k_usd": round(always_cheap_cost_10k, 4),
            "always_quality_cost_10k_usd": round(always_quality_cost_10k, 4),
            "router_cost_10k_usd": round(router_cost_10k, 4),
            "savings_vs_always_quality_pct": round(cost_savings_vs_quality_pct, 1),
        },
    }
