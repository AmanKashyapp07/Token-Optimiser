from __future__ import annotations

import json
from pathlib import Path
import pytest

from gateway.core.eval_engine import (
    QualityCostReport,
    QualityEvaluationRunner,
    compute_line_item_cost,
    evaluate_task_score,
    score_coding,
    score_general_knowledge,
    score_json,
    score_reasoning,
    score_summarization,
)

EVAL_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "eval_suite_v2.jsonl"


def test_eval_suite_dataset_integrity():
    """Verify that the 250-prompt evaluation dataset exists and adheres to schema."""
    assert EVAL_DATASET_PATH.exists(), f"Missing dataset at {EVAL_DATASET_PATH}"
    items = []
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    assert len(items) == 250
    categories = {it["category"] for it in items}
    expected_categories = {
        "general_knowledge",
        "coding",
        "reasoning",
        "summarization",
        "json_generation",
        "indic_hinglish",
        "complex_qa",
    }
    assert categories == expected_categories

    for it in items:
        assert "prompt" in it and len(it["prompt"]) > 5
        assert "expected_entities" in it and isinstance(it["expected_entities"], list)
        assert "reference_answer" in it and len(it["reference_answer"]) > 0


def test_category_scoring_functions():
    """Test individual category scoring mechanics for accuracy and syntax."""
    # General knowledge
    gk_score = score_general_knowledge("Paris is the capital of France on the Seine river.", ["Paris", "Seine"])
    assert gk_score == 1.0

    # Coding (valid AST + keyword)
    code_pass = "def is_palindrome(s):\n    return s == s[::-1]"
    code_score = score_coding(code_pass, ["def is_palindrome", "return"])
    assert code_score == 1.0

    # Coding (invalid syntax gets penalized)
    code_fail = "def is_palindrome(s) return s =="
    assert score_coding(code_fail, ["def is_palindrome"]) < 1.0

    # Reasoning
    reas_score = score_reasoning("The ball costs $0.05 because x + (x + 1.00) = 1.10.", ["ball costs $0.05", "1.10"])
    assert reas_score == 1.0

    # Summarization
    ref = "CRISPR-Cas9 uses a guide RNA to create a targeted double-strand break in DNA."
    hyp = "CRISPR uses guide RNA to make a double-strand break."
    summ_score = score_summarization(hyp, ref, ["guide RNA", "double-strand break"])
    assert summ_score > 0.60

    # JSON syntax parsing
    valid_json = '{"student_name": "Alice", "course": "CS101"}'
    invalid_json = '{"student_name": "Alice", course: CS101}'
    assert score_json(valid_json, ["Alice"]) >= 0.8
    assert score_json(invalid_json, ["Alice"]) == 0.0

    # Unified task evaluation
    task_sc, details = evaluate_task_score("general_knowledge", "Paris is in France.", "Paris, France", ["Paris", "France"], format_constraint="none")
    assert task_sc == 1.0
    assert details["format_adherence"] is True


def test_line_item_cost_accounting():
    """Verify exact line-item mathematical accounting without hidden costs."""
    # 10,000 requests with 30 prompt tokens and 40 completion tokens
    p_tokens = 30 * 10_000  # 300,000 prompt tokens
    c_tokens = 40 * 10_000  # 400,000 completion tokens
    e_tokens = 30 * 10_000  # 300,000 embedding tokens

    cost = compute_line_item_cost(p_tokens, c_tokens, tier="cheap", embedding_tokens=e_tokens)
    # Stated rates: $0.075 / MTok input, $0.30 / MTok output, $0.02 / MTok embedding
    expected_input = (300_000 / 1_000_000) * 0.075    # $0.0225
    expected_output = (400_000 / 1_000_000) * 0.30     # $0.1200
    expected_embedding = (300_000 / 1_000_000) * 0.02  # $0.0060
    expected_total = expected_input + expected_output + expected_embedding  # $0.1485

    assert pytest.approx(cost.input_cost_usd, rel=1e-5) == expected_input
    assert pytest.approx(cost.output_cost_usd, rel=1e-5) == expected_output
    assert pytest.approx(cost.embedding_cost_usd, rel=1e-5) == expected_embedding
    assert pytest.approx(cost.total_cost_usd, rel=1e-5) == expected_total


def test_quality_evaluation_runner():
    """Execute evaluation benchmark and verify QualityCostReport metrics."""
    runner = QualityEvaluationRunner(dataset_path=EVAL_DATASET_PATH)
    report = runner.run_benchmark(strategy="concise_bullets", success_threshold=0.60)

    assert isinstance(report, QualityCostReport)
    assert report.total_prompts == 250
    assert report.successful_tasks > 0
    assert report.task_success_rate_pct > 0.0
    assert report.avg_task_score > 0.50
    assert report.cost_per_successful_task_usd > 0.0
    assert report.cost_per_10k_tasks_usd > 0.0
    assert len(report.categories) == 7
    assert "capping_savings_usd" in report.savings_breakdown
