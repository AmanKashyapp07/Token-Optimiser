#!/usr/bin/env python3
"""Evaluates Semantic Cache False Positive Rate (FPR), Precision, Recall, and 95% Wilson Confidence Intervals on 430 Trap Pairs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_SRC = ROOT_DIR / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from gateway.core.eval_engine import wilson_score_interval
from gateway.features.cache.context_cache import (
    CachedEntryMetadata,
    TwoLayerContextCache,
    hash_system_prompt,
    normalize_query_text,
)
from gateway.modules.m12_semantic.embeddings import SemanticEmbedder

DATASET_FILE = ROOT_DIR / "backend" / "data" / "eval" / "cache_semantic_traps_400.jsonl"
OUTPUT_REPORT_FILE = ROOT_DIR / "backend" / "data" / "eval" / "cache_fpr_benchmark_400.json"


def load_trap_dataset() -> list[dict[str, Any]]:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Trap dataset not found at {DATASET_FILE}. Run build_eval_dataset.py first.")
    records = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    embedder = SemanticEmbedder(use_st=False)
    tp, fp, tn, fn = 0, 0, 0, 0
    by_category: dict[str, dict[str, int]] = {}

    for i, item in enumerate(records):
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}
        by_category[cat]["total"] += 1

        cache = TwoLayerContextCache(embedder=embedder)

        scope_a = item.get("scope_a", "public")
        scope_b = item.get("scope_b", "public")
        fmt_a = item.get("format_a", "none")
        fmt_b = item.get("format_b", "none")
        sys_a = item.get("system_prompt_a")
        sys_b = item.get("system_prompt_b")

        rec = CachedEntryMetadata(
            cache_id=i + 1,
            question=item["query_a"],
            normalized=normalize_query_text(item["query_a"]),
            answer=f"Synthesized knowledge answer for {item['query_a']}",
            scope_id=scope_a,
            model_id="auto",
            system_prompt_hash=hash_system_prompt(sys_a),
            format_constraint=fmt_a,
        )
        cache.insert(rec)

        res = cache.lookup(
            query=item["query_b"],
            scope_id=scope_b,
            system_prompt=sys_b,
            format_constraint=fmt_b,
            threshold_override=threshold,
        )

        actual_hit = res["hit"]
        expected_hit = item["expected_hit"]

        if expected_hit and actual_hit:
            tp += 1
            by_category[cat]["tp"] += 1
        elif not expected_hit and not actual_hit:
            tn += 1
            by_category[cat]["tn"] += 1
        elif not expected_hit and actual_hit:
            fp += 1
            by_category[cat]["fp"] += 1
        else:
            fn += 1
            by_category[cat]["fn"] += 1

    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 100.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0

    prec_ci = wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else (100.0, 100.0)
    rec_ci = wilson_score_interval(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    fpr_ci = wilson_score_interval(fp, fp + tn) if (fp + tn) > 0 else (0.0, 0.0)

    cat_breakdown = {}
    for c, stats in by_category.items():
        c_fp = stats["fp"]
        c_tn = stats["tn"]
        c_tp = stats["tp"]
        c_fn = stats["fn"]
        c_fpr = (c_fp / (c_fp + c_tn)) * 100 if (c_fp + c_tn) > 0 else 0.0
        c_prec = (c_tp / (c_tp + c_fp)) * 100 if (c_tp + c_fp) > 0 else 100.0
        c_rec = (c_tp / (c_tp + c_fn)) * 100 if (c_tp + c_fn) > 0 else 0.0
        cat_breakdown[c] = {
            "total": stats["total"],
            "tp": c_tp,
            "fp": c_fp,
            "tn": c_tn,
            "fn": c_fn,
            "fpr_pct": round(c_fpr, 2),
            "precision_pct": round(c_prec, 2),
            "recall_pct": round(c_rec, 2),
        }

    return {
        "threshold": threshold,
        "sample_size": len(records),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_pct": round(precision, 2),
        "precision_ci_95": prec_ci,
        "recall_pct": round(recall, 2),
        "recall_ci_95": rec_ci,
        "fpr_pct": round(fpr, 2),
        "fpr_ci_95": fpr_ci,
        "categories": cat_breakdown,
    }


def run_benchmark():
    records = load_trap_dataset()
    print("=" * 105)
    print(" TOKEN OPTIMISER v2: LARGE-SCALE SEMANTIC CACHE BENCHMARK (430 TRAP PAIRS)")
    print(" Precision, Recall, False-Positive Rate (FPR), and 95% Wilson Score Confidence Intervals")
    print("=" * 105)

    thresholds = [0.40, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85]
    sweep_results = []

    print(f"{'Thresh':<8} | {'Prec%':<8} | {'95% CI (Prec)':<16} | {'Recall%':<8} | {'95% CI (Rec)':<16} | {'FPR%':<8} | {'95% CI (FPR)':<16}")
    print("-" * 105)

    for thresh in thresholds:
        res = evaluate_threshold(records, thresh)
        sweep_results.append(res)
        prec_str = f"[{res['precision_ci_95'][0]:.1f}%, {res['precision_ci_95'][1]:.1f}%]"
        rec_str = f"[{res['recall_ci_95'][0]:.1f}%, {res['recall_ci_95'][1]:.1f}%]"
        fpr_str = f"[{res['fpr_ci_95'][0]:.1f}%, {res['fpr_ci_95'][1]:.1f}%]"
        print(
            f"{thresh:<8.2f} | {res['precision_pct']:<7.1f}% | {prec_str:<16} | {res['recall_pct']:<7.1f}% | {rec_str:<16} | {res['fpr_pct']:<7.1f}% | {fpr_str:<16}"
        )

    # Detailed category audit at production recommended threshold (0.55 for hash / 0.75 for ST)
    prod_res = next(r for r in sweep_results if r["threshold"] == 0.55)
    print("\n" + "=" * 105)
    print(" CATEGORY-LEVEL BREAKDOWN AT PRODUCTION THRESHOLD (0.55)")
    print("=" * 105)
    print(f"{'Category':<20} | {'Pairs':<6} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'FPR%':<8} | {'Status'}")
    print("-" * 105)

    for cat_name, stats in prod_res["categories"].items():
        status = "PASSED (<1.0% FPR)" if stats["fpr_pct"] < 1.0 else "TRAP DETECTED"
        print(
            f"{cat_name:<20} | {stats['total']:<6} | {stats['tp']:<4} | {stats['fp']:<4} | {stats['tn']:<4} | {stats['fn']:<4} | {stats['fpr_pct']:<7.1f}% | {status}"
        )

    print("=" * 105)
    print(f"Overall Metrics across {prod_res['sample_size']} trap pairs:")
    print(f"• Semantic Precision: {prod_res['precision_pct']:.1f}% (95% CI: [{prod_res['precision_ci_95'][0]:.1f}%, {prod_res['precision_ci_95'][1]:.1f}%])")
    print(f"• Paraphrase Recall:  {prod_res['recall_pct']:.1f}% (95% CI: [{prod_res['recall_ci_95'][0]:.1f}%, {prod_res['recall_ci_95'][1]:.1f}%])")
    print(f"• False Positive Rate: {prod_res['fpr_pct']:.2f}% (95% CI: [{prod_res['fpr_ci_95'][0]:.2f}%, {prod_res['fpr_ci_95'][1]:.2f}%])")
    print("=" * 105)

    OUTPUT_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark_name": "Semantic Cache False-Positive Rate & Wilson CI Benchmark",
                "total_pairs": prod_res["sample_size"],
                "production_threshold": 0.55,
                "production_metrics": prod_res,
                "threshold_sweep": sweep_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved benchmark results to {OUTPUT_REPORT_FILE}")


if __name__ == "__main__":
    run_benchmark()
