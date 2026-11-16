#!/usr/bin/env python3
"""Evaluates Semantic Cache False Positive Rate (FPR), Precision, and Context Gate Rejections."""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend/src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_SRC = ROOT_DIR / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from gateway.features.cache.context_cache import (
    CachedEntryMetadata,
    ContextAwareCacheKey,
    ContextCompatibilityGate,
    TwoLayerContextCache,
    hash_system_prompt,
    normalize_query_text,
)
from gateway.modules.m12_semantic.embeddings import SemanticEmbedder


def run_fpr_benchmark():
    embedder = SemanticEmbedder(use_st=False)
    cache = TwoLayerContextCache(embedder=embedder)

    # Seed knowledge base entries
    seeded_records = [
        CachedEntryMetadata(
            cache_id=1,
            question="How do I renew a library book online before due date?",
            normalized=normalize_query_text("How do I renew a library book online before due date?"),
            answer="Log in to the library portal, open issued items, and click Renew.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        ),
        CachedEntryMetadata(
            cache_id=2,
            question="Explain binary search in Python with code example.",
            normalized=normalize_query_text("Explain binary search in Python with code example."),
            answer="def binary_search(arr, target): low, high = 0, len(arr)-1 ...",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        ),
        CachedEntryMetadata(
            cache_id=3,
            question="What is the stock price of NVIDIA today?",
            normalized=normalize_query_text("What is the stock price of NVIDIA today?"),
            answer="NVIDIA stock price today is trading at $142.50.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        ),
        CachedEntryMetadata(
            cache_id=4,
            question="Where do I submit assignment 1 before the deadline?",
            normalized=normalize_query_text("Where do I submit assignment 1 before the deadline?"),
            answer="Submit assignment 1 on the course canvas portal under Assignment 1 tab.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        ),
        CachedEntryMetadata(
            cache_id=5,
            question="Summarize the French Revolution causes and timeline.",
            normalized=normalize_query_text("Summarize the French Revolution causes and timeline."),
            answer='{"event": "French Revolution", "year": 1789, "causes": ["economic crisis", "social inequality"]}',
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="JSON only",
        ),
        CachedEntryMetadata(
            cache_id=6,
            question="Explain recursion with simple examples for beginners.",
            normalized=normalize_query_text("Explain recursion with simple examples for beginners."),
            answer="Recursion is like looking into two facing mirrors. A function calls itself with a smaller base case.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash=hash_system_prompt("You are an elementary school teacher explaining concepts simply."),
            format_constraint="none",
        ),
        CachedEntryMetadata(
            cache_id=7,
            question="Confidential student grade records for CS101.",
            normalized=normalize_query_text("Confidential student grade records for CS101."),
            answer="Alice: A, Bob: B+",
            scope_id="tenant-alpha",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        ),
    ]

    for rec in seeded_records:
        cache.insert(rec)

    # Test cases: (query, sys_prompt, format, scope, should_match, description)
    test_cases = [
        # True Positives (Valid paraphrases with compatible context)
        ("How can I renew my library book online before the due date?", None, "none", "public", True, "Valid Paraphrase: Library renewal"),
        ("Explain binary search implementation in Python", None, "none", "public", True, "Valid Paraphrase: Binary search Python"),
        ("What is today's current stock price of NVIDIA?", None, "none", "public", True, "Valid Paraphrase: NVIDIA today stock"),

        # Tricky False Positive Traps (MUST BE REJECTED)
        ("Explain binary search in Java with code example.", None, "none", "public", False, "Language Divergence Trap: Python vs Java"),
        ("What was the stock price of NVIDIA yesterday?", None, "none", "public", False, "Temporal Divergence Trap: Today vs Yesterday"),
        ("Where do I submit assignment 2 before the deadline?", None, "none", "public", False, "Assignment ID Trap: Assignment 1 vs 2"),
        ("Summarize the French Revolution causes and timeline.", None, "3 bullets", "public", False, "Format Divergence Trap: JSON vs 3 bullets"),
        ("Explain recursion with simple examples for beginners.", "You are a PhD theoretical computer scientist.", "none", "public", False, "System Prompt Trap: Elementary vs PhD"),
        ("Confidential student grade records for CS101.", None, "none", "public", False, "Tenant Isolation Trap: Public query vs tenant-alpha"),
        ("Where is the library located on campus?", None, "none", "public", False, "Keyword Overlap Trap: Library location vs renewal"),
    ]

    print("=" * 90)
    print(" TOKEN OPTIMISER v2: SEMANTIC CACHE FALSE-POSITIVE & CONTEXT GATE BENCHMARK")
    print("=" * 90)
    print(f"{'Test Description':<40} | {'Expected':<10} | {'Actual':<10} | {'Layer':<6} | {'Status'}")
    print("-" * 90)

    tp, fp, tn, fn = 0, 0, 0, 0

    for query, sys_prompt, fmt, scope, expected_match, desc in test_cases:
        res = cache.lookup(
            query=query,
            scope_id=scope,
            system_prompt=sys_prompt,
            format_constraint=fmt,
            threshold_override=0.55,
        )

        actual_hit = res["hit"]
        layer_used = f"L{res.get('layer', '-')}"

        if expected_match and actual_hit:
            tp += 1
            verdict = "PASS (True Positive)"
        elif not expected_match and not actual_hit:
            tn += 1
            verdict = "PASS (Correct Rejection)"
        elif not expected_match and actual_hit:
            fp += 1
            verdict = "FAIL (False Positive Trap!)"
        else:
            fn += 1
            verdict = "FAIL (False Negative)"

        exp_str = "HIT" if expected_match else "MISS"
        act_str = "HIT" if actual_hit else "MISS"
        print(f"{desc:<40} | {exp_str:<10} | {act_str:<10} | {layer_used:<6} | {verdict}")

    total_tests = len(test_cases)
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0

    print("-" * 90)
    print(f"Summary Metrics: Total Tests: {total_tests} | True Positives: {tp} | True Negatives: {tn}")
    print(f"• False Positive Rate (FPR): {fpr:.1f}% (Hard target: < 1.0%)")
    print(f"• Semantic Precision:        {precision:.1f}%")
    print(f"• Paraphrase Recall:        {recall:.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    run_fpr_benchmark()
