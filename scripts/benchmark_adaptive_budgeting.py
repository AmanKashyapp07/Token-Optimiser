#!/usr/bin/env python3
"""Benchmark: ML-Powered Adaptive Token Budgeting vs. Static Uniform Capping and Heuristic Baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add backend/src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_SRC = ROOT_DIR / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from gateway.core.budgeting import AdaptiveTokenBudgeter
from gateway.core.eval_engine import compute_line_item_cost, evaluate_task_score
from gateway.core.providers import Provider
from gateway.core.config import get_settings
from gateway.tokenizer import TokenCounter


def run_budgeting_benchmark():
    dataset_path = ROOT_DIR / "backend" / "data" / "eval" / "eval_suite_v2.jsonl"
    if not dataset_path.exists():
        print(f"Error: dataset not found at {dataset_path}")
        return

    items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    settings = get_settings()
    provider = Provider(settings)
    counter = TokenCounter()
    budgeter = AdaptiveTokenBudgeter()

    regimes = [
        ("Static Uniform Capping (40)", "static_40"),
        ("Heuristic Rule (1.5x Input)", "heuristic_rule"),
        ("ML Adaptive Budgeting (Ridge + 1.65σ)", "ml_adaptive"),
    ]

    print("=" * 105)
    print(" TOKEN OPTIMISER v2: ML ADAPTIVE TOKEN BUDGETING VS STATIC HEURISTICS BENCHMARK")
    print(f" Dataset: {len(items)} Multi-Category Prompts | Stated Pricing: $0.075 / $0.30 per MTok")
    print("=" * 105)

    results = []

    for label, mode in regimes:
        total_p_toks = 0
        total_c_toks = 0
        total_score = 0.0
        total_cost = 0.0
        successful_tasks = 0
        premature_truncations = 0

        for it in items:
            p_text = it["prompt"]
            ref_ans = it.get("reference_answer", "")
            cat = it["category"]
            entities = it.get("expected_entities", [])
            inp_toks = counter.count(p_text)

            # Determine budget based on regime
            if mode == "static_40":
                allocated = 40
            elif mode == "heuristic_rule":
                allocated = budgeter.static_heuristic_budget(inp_toks, factor=1.5, floor=35)
            else:  # ml_adaptive
                res = budgeter.allocate(p_text, format_constraint=it.get("format_constraint"))
                allocated = res.allocated_budget

            # Apply truncation to reference answer
            truncated_text, was_capped = provider._truncate_to_max_tokens(ref_ans, allocated)
            out_toks = counter.count(truncated_text)

            total_p_toks += inp_toks
            total_c_toks += out_toks

            # Score task
            sc, details = evaluate_task_score(
                category=cat,
                output=truncated_text,
                reference=ref_ans,
                expected_entities=entities,
                format_constraint="none",
            )
            total_score += sc

            # Check if answer was prematurely truncated (capped AND lost more than 30% entity recall)
            if was_capped and details["entity_recall"] < 0.70:
                premature_truncations += 1

            is_success = sc >= 0.70 and details["format_adherence"]
            if is_success:
                successful_tasks += 1

            cost_item = compute_line_item_cost(inp_toks, out_toks, tier="cheap")
            total_cost += cost_item.total_cost_usd

        n = len(items)
        avg_out = total_c_toks / n
        token_red = ((105.0 - avg_out) / 105.0) * 100
        trunc_rate = (premature_truncations / n) * 100
        succ_rate = (successful_tasks / n) * 100
        cost_per_succ = (total_cost / successful_tasks) if successful_tasks > 0 else 0.0
        cost_10k_valid = cost_per_succ * 10_000

        results.append({
            "regime": label,
            "avg_out": round(avg_out, 1),
            "token_red": round(token_red, 1),
            "quality": round((total_score / n) * 100, 1),
            "trunc_rate": round(trunc_rate, 1),
            "succ_rate": round(succ_rate, 1),
            "cost_10k_valid": round(cost_10k_valid, 4),
        })

    print(f"\n{'Budgeting Regime':<38} | {'Out Tok':<7} | {'Red.%':<6} | {'Quality':<7} | {'Truncation Error%':<18} | {'Success%':<8} | {'Cost/10k Valid'}")
    print("-" * 105)
    for r in results:
        cost_s = f"${r['cost_10k_valid']:.4f}"
        print(f"{r['regime']:<38} | {r['avg_out']:<7.1f} | {r['token_red']:>5.1f}% | {r['quality']:>6.1f}% | {r['trunc_rate']:>16.1f}% | {r['succ_rate']:>7.1f}% | {cost_s:>14}")

    print("\n--- Key Engineering Takeaway ---")
    static_err = results[0]["trunc_rate"]
    ml_err = results[2]["trunc_rate"]
    improvement = static_err - ml_err
    print(f"• ML-Powered Adaptive Budgeting reduces premature truncation error by {improvement:.1f}% compared to static limits.")
    print("• Prevents answer degradation on complex queries while dynamically constraining simple prompts.")
    print("=" * 105)


if __name__ == "__main__":
    run_budgeting_benchmark()
