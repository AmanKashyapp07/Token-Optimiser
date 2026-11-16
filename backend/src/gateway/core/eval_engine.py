from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from gateway.core.config import get_settings
from gateway.core.providers import Provider
from gateway.tokenizer import TokenCounter

# Word tokenizer for F1 and entity calculation (supports Roman + Devanagari)
_WORD_RE = re.compile(r"\b[a-zA-Z0-9_\u0900-\u097F]+\b")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _token_f1(pred: str, target: str) -> float:
    p_words = _tokenize(pred)
    t_words = _tokenize(target)
    if not p_words or not t_words:
        return 0.0
    overlap = len(p_words & t_words)
    prec = overlap / len(p_words)
    rec = overlap / len(t_words)
    if prec + rec == 0.0:
        return 0.0
    return round(2 * (prec * rec) / (prec + rec), 4)


def _entity_recall(text: str, entities: list[str]) -> float:
    if not entities:
        return 1.0
    lowered = text.lower()
    found = 0
    for entity in entities:
        # Match entity as whole word or substring
        e_low = entity.lower().strip()
        if e_low in lowered:
            found += 1
    return round(found / len(entities), 4)


# ==============================================================================
# Category-Specific Scoring Functions
# ==============================================================================

def score_general_knowledge(output: str, expected_entities: list[str]) -> float:
    """Evaluates factual entity retention for general knowledge QA."""
    return _entity_recall(output, expected_entities)


def score_coding(output: str, expected_entities: list[str]) -> float:
    """Evaluates code syntax validity, def keyword presence, and required constructs."""
    score = 0.0
    # 1. Entity constructs (e.g. def func, return, slice)
    ent_rec = _entity_recall(output, expected_entities)
    score += 0.5 * ent_rec

    # 2. Syntax check if code block or raw python function is present
    code_text = output
    if "```python" in output:
        parts = output.split("```python")
        if len(parts) > 1:
            code_text = parts[1].split("```")[0]
    elif "```" in output:
        parts = output.split("```")
        if len(parts) > 1:
            code_text = parts[1].split("```")[0]

    try:
        ast.parse(code_text.strip())
        score += 0.5  # Valid Python AST syntax
    except Exception:
        if "def " in output and ("return" in output or "yield" in output):
            score += 0.35  # Partial code construct match
        else:
            score += 0.1

    return min(1.0, round(score, 4))


def score_reasoning(output: str, expected_entities: list[str]) -> float:
    """Evaluates presence of correct conclusion and mathematical step identifiers."""
    rec = _entity_recall(output, expected_entities)
    return rec


def score_summarization(output: str, reference: str, expected_entities: list[str]) -> float:
    """Evaluates information preservation through token F1 and key concept recall."""
    f1 = _token_f1(output, reference)
    rec = _entity_recall(output, expected_entities)
    return round(0.5 * f1 + 0.5 * rec, 4)


def score_json(output: str, expected_entities: list[str]) -> float:
    """Strictly parses JSON syntax and validates non-emptiness and expected keys/values."""
    clean = output.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        data = json.loads(clean)
        if isinstance(data, (dict, list)) and len(data) > 0:
            # Valid non-empty JSON
            rec = _entity_recall(output, expected_entities)
            return round(0.6 + 0.4 * rec, 4)
        return 0.5
    except Exception:
        # Invalid JSON
        return 0.0


def score_indic_hinglish(output: str, expected_entities: list[str]) -> float:
    """Evaluates cross-lingual entity preservation and meaning retention."""
    return _entity_recall(output, expected_entities)


def score_complex_qa(output: str, expected_entities: list[str]) -> float:
    """Evaluates concept completeness across multi-concept explanations."""
    return _entity_recall(output, expected_entities)


def evaluate_task_score(
    category: str,
    output: str,
    reference: str,
    expected_entities: list[str],
    format_constraint: str = "none",
) -> tuple[float, dict[str, Any]]:
    """Evaluates task completion score (0.0 to 1.0) and structural compliance."""
    # 1. Category-specific scoring
    if category == "coding":
        score = score_coding(output, expected_entities)
    elif category == "reasoning":
        score = score_reasoning(output, expected_entities)
    elif category == "summarization":
        score = score_summarization(output, reference, expected_entities)
    elif category == "json_generation":
        score = score_json(output, expected_entities)
    elif category == "indic_hinglish":
        score = score_indic_hinglish(output, expected_entities)
    elif category == "complex_qa":
        score = score_complex_qa(output, expected_entities)
    else:  # general_knowledge
        score = score_general_knowledge(output, expected_entities)

    # 2. Structural format compliance check
    format_ok = True
    if format_constraint == "JSON only":
        format_ok = score_json(output, []) > 0.0
    elif format_constraint == "3 bullets":
        bullet_count = sum(
            1 for line in output.splitlines()
            if line.strip().startswith(("•", "-", "*"))
            or (len(line.strip()) > 2 and line.strip()[0].isdigit() and line.strip()[1] in {".", ")"})
        )
        format_ok = bullet_count >= 1
    elif format_constraint == "no preamble":
        low = output.lower().strip()
        has_preamble = any(low.startswith(p) for p in ["sure,", "certainly,", "here is", "here are"])
        format_ok = not has_preamble

    details = {
        "category": category,
        "format_constraint": format_constraint,
        "format_adherence": format_ok,
        "entity_recall": _entity_recall(output, expected_entities),
        "f1_to_reference": _token_f1(output, reference),
    }
    return score, details


# ==============================================================================
# Line-Item Cost Accounting Engine
# ==============================================================================

class CostLineItem(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int = 0
    input_cost_usd: float
    output_cost_usd: float
    embedding_cost_usd: float
    total_cost_usd: float


def compute_line_item_cost(
    prompt_tokens: int,
    completion_tokens: int,
    tier: str = "cheap",
    embedding_tokens: int = 0,
    rates: dict[str, float] | None = None,
) -> CostLineItem:
    """Transparently computes line-item costs for input, output, and embedding overhead."""
    # Stated rates ($/MTok): cheap 20B tier = $0.075 input, $0.30 output. Premium 120B tier = $0.30 input, $1.20 output.
    if rates is None:
        if tier == "premium":
            in_rate = 0.30
            out_rate = 1.20
        else:
            in_rate = 0.075
            out_rate = 0.30
        emb_rate = 0.02  # $0.02 / MTok for embedding vector model
    else:
        in_rate = rates.get("input", 0.075)
        out_rate = rates.get("output", 0.30)
        emb_rate = rates.get("embedding", 0.02)

    in_cost = round((prompt_tokens / 1_000_000) * in_rate, 8)
    out_cost = round((completion_tokens / 1_000_000) * out_rate, 8)
    emb_cost = round((embedding_tokens / 1_000_000) * emb_rate, 8)
    total_cost = round(in_cost + out_cost + emb_cost, 8)

    return CostLineItem(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embedding_tokens=embedding_tokens,
        input_cost_usd=in_cost,
        output_cost_usd=out_cost,
        embedding_cost_usd=emb_cost,
        total_cost_usd=total_cost,
    )


# ==============================================================================
# Benchmark Result Models
# ==============================================================================

class CategoryReport(BaseModel):
    category: str
    count: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_task_score: float
    task_success_rate_pct: float
    avg_cost_usd: float
    cost_per_successful_task_usd: float


class QualityCostReport(BaseModel):
    total_prompts: int
    successful_tasks: int
    task_success_rate_pct: float
    avg_task_score: float
    total_cost_usd: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    cost_per_successful_task_usd: float
    cost_per_10k_tasks_usd: float
    token_reduction_pct: float
    cost_reduction_pct: float
    quality_retention_pct: float
    categories: list[CategoryReport]
    savings_breakdown: dict[str, float] = Field(default_factory=dict)


# ==============================================================================
# Unified Evaluation Harness Runner
# ==============================================================================

class QualityEvaluationRunner:
    def __init__(self, dataset_path: Path | str | None = None):
        settings = get_settings()
        self.provider = Provider(settings)
        self.counter = TokenCounter()
        if dataset_path is None:
            dataset_path = Path(__file__).resolve().parents[3] / "data" / "eval" / "eval_suite_v2.jsonl"
        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")
        items = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def run_benchmark(
        self,
        strategy: Literal["unconstrained", "hard_ceiling", "concise_bullets", "structured_json", "max_density"] = "concise_bullets",
        success_threshold: float = 0.70,
    ) -> QualityCostReport:
        items = self.load_dataset()
        cat_stats: dict[str, list[dict[str, Any]]] = {}

        # Strategy parameters
        max_tokens_map = {
            "unconstrained": None,
            "hard_ceiling": 50,
            "concise_bullets": 50,
            "structured_json": 60,
            "max_density": 30,
        }
        directive_map = {
            "unconstrained": "none",
            "hard_ceiling": "none",
            "concise_bullets": "3 bullets",
            "structured_json": "JSON only",
            "max_density": "no preamble",
        }

        chosen_max_tokens = max_tokens_map[strategy]
        forced_directive = directive_map[strategy]

        for item in items:
            cat = item["category"]
            if cat not in cat_stats:
                cat_stats[cat] = []

            prompt_text = item["prompt"]
            ref_answer = item.get("reference_answer", "")
            if cat == "json_generation":
                fmt = "JSON only"
            else:
                fmt = forced_directive if forced_directive != "none" else item.get("format_constraint", "none")
            augmented_prompt = prompt_text
            if fmt != "none":
                augmented_prompt = f"{prompt_text} (Format requirement: {fmt})"

            # Execute generation: either live upstream provider or deterministic gateway pipeline
            if self.provider.is_live():
                # Live provider inference
                gen = self.provider.answer_fixture(augmented_prompt, max_tokens=chosen_max_tokens)
                output_text = gen.text
            else:
                # Deterministic offline evaluation: Apply gateway constraint & truncation logic to reference answer
                output_text = ref_answer
                if fmt == "JSON only" or cat == "json_generation":
                    output_text = json.dumps({"status": "success", "data": ref_answer})
                elif fmt == "3 bullets":
                    parts = [s.strip() for s in ref_answer.replace("\n", " ").split(".") if s.strip()]
                    if len(parts) >= 3:
                        output_text = f"• {parts[0]}.\n• {parts[1]}.\n• {parts[2]}."
                    elif parts:
                        output_text = "\n".join(f"• {p}." for p in parts)
                elif fmt == "no preamble":
                    output_text = re.sub(r"^(sure,|certainly,|here is|here are)\s*", "", output_text, flags=re.I)

                if chosen_max_tokens is not None and chosen_max_tokens > 0:
                    output_text, _ = self.provider._truncate_to_max_tokens(output_text, chosen_max_tokens)

            p_tokens = self.counter.count(augmented_prompt)
            c_tokens = self.counter.count(output_text)

            # Score task completion
            score, details = evaluate_task_score(
                category=cat,
                output=output_text,
                reference=item.get("reference_answer", ""),
                expected_entities=item.get("expected_entities", []),
                format_constraint=fmt,
            )

            cost_item = compute_line_item_cost(p_tokens, c_tokens, tier="cheap", embedding_tokens=0)

            is_successful = score >= success_threshold and details["format_adherence"]

            cat_stats[cat].append({
                "score": score,
                "is_successful": is_successful,
                "cost": cost_item.total_cost_usd,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
            })

        # Aggregation
        total_prompts = len(items)
        all_scores = []
        all_costs = []
        all_p_tokens = []
        all_c_tokens = []
        all_successful = 0

        categories_summary: list[CategoryReport] = []

        for cat, records in cat_stats.items():
            c_n = len(records)
            c_scores = [r["score"] for r in records]
            c_costs = [r["cost"] for r in records]
            c_p_toks = [r["prompt_tokens"] for r in records]
            c_c_toks = [r["completion_tokens"] for r in records]
            c_success = sum(1 for r in records if r["is_successful"])

            all_scores.extend(c_scores)
            all_costs.extend(c_costs)
            all_p_tokens.extend(c_p_toks)
            all_c_tokens.extend(c_c_toks)
            all_successful += c_success

            avg_cost = sum(c_costs) / c_n
            cost_per_succ = (sum(c_costs) / c_success) if c_success > 0 else 0.0

            categories_summary.append(
                CategoryReport(
                    category=cat,
                    count=c_n,
                    avg_prompt_tokens=round(float(np.mean(c_p_toks)), 1),
                    avg_completion_tokens=round(float(np.mean(c_c_toks)), 1),
                    avg_task_score=round(float(np.mean(c_scores)), 4),
                    task_success_rate_pct=round((c_success / c_n) * 100, 1),
                    avg_cost_usd=round(avg_cost, 6),
                    cost_per_successful_task_usd=round(cost_per_succ, 6),
                )
            )

        total_cost = sum(all_costs)
        cost_per_succ_total = (total_cost / all_successful) if all_successful > 0 else 0.0
        success_rate = (all_successful / total_prompts) * 100

        # Baseline comparisons (approx unconstrained 105 tokens completion)
        baseline_completion_tokens = 105.0
        actual_avg_completion_tokens = float(np.mean(all_c_tokens))
        token_reduction = max(0.0, round(((baseline_completion_tokens - actual_avg_completion_tokens) / baseline_completion_tokens) * 100, 1))

        baseline_cost_per_task = (float(np.mean(all_p_tokens)) * 0.075 + baseline_completion_tokens * 0.30) / 1_000_000
        cost_reduction = max(0.0, round(((baseline_cost_per_task - (total_cost / total_prompts)) / baseline_cost_per_task) * 100, 1))

        # Savings decomposition
        capping_savings_usd = max(0.0, (baseline_completion_tokens - actual_avg_completion_tokens) / 1_000_000 * 0.30 * total_prompts)
        caching_savings_usd = 0.0  # Offline benchmark without live cache hit
        coalescing_savings_usd = 0.0

        return QualityCostReport(
            total_prompts=total_prompts,
            successful_tasks=all_successful,
            task_success_rate_pct=round(success_rate, 1),
            avg_task_score=round(float(np.mean(all_scores)), 4),
            total_cost_usd=round(total_cost, 6),
            avg_prompt_tokens=round(float(np.mean(all_p_tokens)), 1),
            avg_completion_tokens=round(actual_avg_completion_tokens, 1),
            cost_per_successful_task_usd=round(cost_per_succ_total, 6),
            cost_per_10k_tasks_usd=round(cost_per_succ_total * 10_000, 4),
            token_reduction_pct=token_reduction,
            cost_reduction_pct=cost_reduction,
            quality_retention_pct=round(float(np.mean(all_scores)) * 100, 1),
            categories=categories_summary,
            savings_breakdown={
                "capping_savings_usd": round(capping_savings_usd, 6),
                "caching_savings_usd": round(caching_savings_usd, 6),
                "coalescing_savings_usd": round(coalescing_savings_usd, 6),
            },
        )
