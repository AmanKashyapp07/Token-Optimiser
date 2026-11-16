#!/usr/bin/env python3
"""CLI runner for Token Optimiser v2: Quality-Adjusted Cost Evaluation Benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add backend/src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_SRC = ROOT_DIR / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from gateway.core.eval_engine import QualityEvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="Run Token Optimiser v2 Quality & Cost Evaluation Benchmark")
    parser.add_argument("--threshold", type=float, default=0.70, help="Task success acceptance threshold (default: 0.70)")
    parser.add_argument("--dataset", type=str, default=None, help="Custom dataset path")
    args = parser.parse_args()

    runner = QualityEvaluationRunner(dataset_path=args.dataset)
    print("=" * 95)
    print(" TOKEN OPTIMISER v2: QUALITY-ADJUSTED COST EVALUATION BENCHMARK")
    print(f" Dataset: 250 Multilingual Prompts across 7 Categories | Success Threshold: {args.threshold * 100:.0f}%")
    print("=" * 95)

    strategies = [
        ("Unconstrained Baseline", "unconstrained"),
        ("Hard Ceiling (max=50)", "hard_ceiling"),
        ("Concise Bullets (3 bullets)", "concise_bullets"),
        ("Structured Output (JSON)", "structured_json"),
        ("Maximum Density (no preamble)", "max_density"),
    ]

    summary_rows = []
    category_breakdowns = {}

    for label, strat_key in strategies:
        report = runner.run_benchmark(strategy=strat_key, success_threshold=args.threshold)
        summary_rows.append((label, report))
        category_breakdowns[label] = report.categories

    print("\n--- 1. Multi-Strategy Comparative Cost & Quality Table ---")
    print(f"{'Strategy':<30} | {'Tokens':<6} | {'Red.%':<6} | {'Quality':<7} | {'Success%':<8} | {'Cost/10k Valid':<14} | {'Cost Red.%'}")
    print("-" * 95)
    for label, rep in summary_rows:
        cost_str = f"${rep.cost_per_10k_tasks_usd:.4f}"
        print(f"{label:<30} | {rep.avg_completion_tokens:<6.1f} | {rep.token_reduction_pct:>5.1f}% | {rep.quality_retention_pct:>6.1f}% | {rep.task_success_rate_pct:>7.1f}% | {cost_str:>14} | {rep.cost_reduction_pct:>9.1f}%")

    # Print Category Breakdown for Concise Bullets (Default Gateway Strategy)
    default_report = summary_rows[2][1]
    print(f"\n--- 2. Category-Level Performance Breakdown (Concise Bullets Strategy) ---")
    print(f"{'Category':<22} | {'Prompts':<7} | {'In Tok':<6} | {'Out Tok':<7} | {'Quality':<7} | {'Success%':<8} | {'Cost/Succ Task'}")
    print("-" * 95)
    for cat in default_report.categories:
        cost_unit = f"${cat.cost_per_successful_task_usd:.6f}"
        print(f"{cat.category:<22} | {cat.count:<7} | {cat.avg_prompt_tokens:<6.1f} | {cat.avg_completion_tokens:<7.1f} | {cat.avg_task_score * 100:>6.1f}% | {cat.task_success_rate_pct:>7.1f}% | {cost_unit:>14}")

    print("\n--- 3. Transparent Cost Accounting Line Items ---")
    print("• Stated Pricing Catalog: 20B Tier = $0.075/MTok Input, $0.30/MTok Output.")
    print("• Mathematical Formula: Total Cost = (PromptTokens / 1M) * 0.075 + (CompletionTokens / 1M) * 0.30")
    print("• Quality-Adjusted Metric: Cost Per Successful Task = Total Cost / Successful Tasks (Score >= 70%)")
    print("=" * 95)


if __name__ == "__main__":
    main()
