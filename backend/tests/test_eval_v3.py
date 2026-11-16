"""Unit and integration tests for v3 Phase 1: Ground-truth task correctness and 400+ cache trap benchmark."""

import json
from pathlib import Path

from gateway.core.eval_engine import (
    CodeCorrectnessEvaluator,
    FactualRubricEvaluator,
    JSONCorrectnessEvaluator,
    wilson_score_interval,
)
from gateway.features.cache.context_cache import (
    CachedEntryMetadata,
    TwoLayerContextCache,
    normalize_query_text,
)
from gateway.modules.m12_semantic.embeddings import SemanticEmbedder


# ==============================================================================
# 1. Ground-Truth Code Evaluator Tests
# ==============================================================================

def test_code_evaluator_valid_execution():
    code = """
def is_palindrome(s):
    c = ''.join(ch.lower() for ch in s if ch.isalnum())
    return c == c[::-1]
"""
    test_code = """
assert is_palindrome('racecar') is True
assert is_palindrome('hello') is False
assert is_palindrome('A man a plan a canal Panama') is True
"""
    res = CodeCorrectnessEvaluator.evaluate(code, test_code)
    assert res.is_correct is True
    assert res.syntax_valid is True
    assert res.assertion_passed is True
    assert res.score == 1.0
    assert res.error is None


def test_code_evaluator_assertion_failure():
    # Buggy palindrome implementation
    code = """
def is_palindrome(s):
    return False
"""
    test_code = "assert is_palindrome('racecar') is True"
    res = CodeCorrectnessEvaluator.evaluate(code, test_code)
    assert res.is_correct is False
    assert res.syntax_valid is True
    assert res.assertion_passed is False
    assert res.score == 0.5
    assert "AssertionError" in str(res.error)


def test_code_evaluator_syntax_error():
    bad_code = "def is_palindrome(s): return s == "
    res = CodeCorrectnessEvaluator.evaluate(bad_code, "assert True")
    assert res.is_correct is False
    assert res.syntax_valid is False
    assert res.score == 0.0
    assert "SyntaxError" in str(res.error)


def test_code_evaluator_timeout_sandbox():
    infinite_loop = """
def is_palindrome(s):
    while True:
        pass
    return True
"""
    test_code = "assert is_palindrome('test') is True"
    res = CodeCorrectnessEvaluator.evaluate(infinite_loop, test_code, timeout_sec=0.2)
    assert res.is_correct is False
    assert "TimeoutError" in str(res.error)


# ==============================================================================
# 2. Ground-Truth JSON Schema Evaluator Tests
# ==============================================================================

def test_json_evaluator_valid_schema():
    raw_json = '{"student_name": "Alice", "course_code": "CS101", "status": "enrolled", "credits": 4}'
    schema = {"student_name": "str", "course_code": "str", "status": "str", "credits": "number"}
    res = JSONCorrectnessEvaluator.evaluate(raw_json, schema)
    assert res.is_correct is True
    assert res.json_valid is True
    assert res.schema_valid is True
    assert res.score == 1.0
    assert not res.missing_keys


def test_json_evaluator_missing_required_keys():
    raw_json = '{"student_name": "Alice"}'
    schema = {"student_name": "str", "course_code": "str", "status": "str"}
    res = JSONCorrectnessEvaluator.evaluate(raw_json, schema)
    assert res.is_correct is False
    assert res.json_valid is True
    assert res.schema_valid is False
    assert "course_code" in res.missing_keys
    assert "status" in res.missing_keys


def test_json_evaluator_markdown_fence_stripping():
    fenced = "```json\n{\"hostname\": \"node-01\", \"status\": \"healthy\"}\n```"
    schema = {"hostname": "str", "status": "str"}
    res = JSONCorrectnessEvaluator.evaluate(fenced, schema)
    assert res.is_correct is True
    assert res.json_valid is True


def test_json_evaluator_malformed_syntax():
    malformed = '{"hostname": "node-01", status: healthy}'
    res = JSONCorrectnessEvaluator.evaluate(malformed)
    assert res.is_correct is False
    assert res.json_valid is False
    assert "JSONDecodeError" in str(res.error)


# ==============================================================================
# 3. Ground-Truth Factual Rubric Evaluator Tests
# ==============================================================================

def test_factual_rubric_evaluator_all_present():
    answer = "Albert Einstein published the general theory of relativity in 1915."
    critical_facts = ["Einstein", "1915"]
    res = FactualRubricEvaluator.evaluate(answer, critical_facts)
    assert res.is_correct is True
    assert res.score == 1.0
    assert not res.missing_facts


def test_factual_rubric_evaluator_missing_fact():
    answer = "Albert Einstein was a prominent theoretical physicist who won the Nobel Prize."
    critical_facts = ["Einstein", "1915"]
    res = FactualRubricEvaluator.evaluate(answer, critical_facts)
    assert res.is_correct is False
    assert res.score == 0.5
    assert "1915" in res.missing_facts


# ==============================================================================
# 4. Statistical Wilson Score Confidence Interval Tests
# ==============================================================================

def test_wilson_score_interval_properties():
    # 0 out of 100
    low, high = wilson_score_interval(0, 100)
    assert low == 0.0
    assert 0.0 < high <= 3.8

    # 100 out of 100
    low, high = wilson_score_interval(100, 100)
    assert 96.0 <= low < 100.0
    assert high == 100.0

    # 50 out of 100
    low, high = wilson_score_interval(50, 100)
    assert 40.0 < low < 50.0
    assert 50.0 < high < 60.0


# ==============================================================================
# 5. Semantic Cache Trap Rejection Tests (430 Trap Dataset Verification)
# ==============================================================================

def test_cache_traps_dataset_structure():
    trap_file = Path(__file__).resolve().parents[2] / "backend" / "data" / "eval" / "cache_semantic_traps_400.jsonl"
    assert trap_file.exists()
    records = []
    with open(trap_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    assert len(records) >= 400

    categories = {r["category"] for r in records}
    assert "negation" in categories
    assert "numerical" in categories
    assert "temporal" in categories
    assert "language_api" in categories
    assert "context_gate" in categories
    assert "paraphrase_hit" in categories


def test_cache_context_gate_negation_trap_rejection():
    cache = TwoLayerContextCache(embedder=SemanticEmbedder(use_st=False))
    cache.insert(
        CachedEntryMetadata(
            cache_id=1,
            question="Is transaction TXN-100 successful?",
            normalized=normalize_query_text("Is transaction TXN-100 successful?"),
            answer="Yes, transaction TXN-100 has completed successfully.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        )
    )

    # Opposing polarity MUST be rejected
    res = cache.lookup(
        query="Is transaction TXN-100 unsuccessful?",
        threshold_override=0.55,
    )
    assert res["hit"] is False
    assert any("divergence" in d.lower() for d in res.get("rejection_details", []))


def test_cache_context_gate_numerical_trap_rejection():
    cache = TwoLayerContextCache(embedder=SemanticEmbedder(use_st=False))
    cache.insert(
        CachedEntryMetadata(
            cache_id=1,
            question="What is 10% of 200?",
            normalized=normalize_query_text("What is 10% of 200?"),
            answer="10% of 200 is 20.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        )
    )

    # Numerical difference 200 vs 500 MUST be rejected
    res = cache.lookup(
        query="What is 10% of 500?",
        threshold_override=0.55,
    )
    assert res["hit"] is False
    assert any("numerical divergence" in d.lower() for d in res.get("rejection_details", []))


def test_cache_context_gate_true_paraphrase_hit():
    cache = TwoLayerContextCache(embedder=SemanticEmbedder(use_st=False))
    cache.insert(
        CachedEntryMetadata(
            cache_id=1,
            question="How do I renew my library book online before the due date?",
            normalized=normalize_query_text("How do I renew my library book online before the due date?"),
            answer="Sign into the university library portal and click Renew under Issued Books.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        )
    )

    # Valid paraphrase with identical semantic meaning
    res = cache.lookup(
        query="What is the procedure to renew a library book online prior to expiry?",
        threshold_override=0.50,
    )
    assert res["hit"] is True
    assert res["answer"] is not None
