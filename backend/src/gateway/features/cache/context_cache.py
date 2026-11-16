from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from gateway.modules.m12_semantic.embeddings import SemanticEmbedder

# Regexes for entity / context divergence gating
_PROG_LANGS = {"python", "java", "c++", "cpp", "c#", "csharp", "javascript", "typescript", "golang", "go", "rust", "ruby", "php"}
_PRESENT_TEMPORALS = {"today", "current", "latest", "now"}
_PAST_TEMPORALS = {"yesterday", "previous", "last week", "last year", "past"}
_FUTURE_TEMPORALS = {"tomorrow", "next week", "next year", "upcoming"}
_NUMBER_OR_ID_RE = re.compile(r"\b(?:assignment|cs|hw|lab|quiz|exam|part|phase|v|version)\s*([0-9]+|[a-z])\b", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+\b")


def normalize_query_text(text: str) -> str:
    """Canonical string normalization: NFKC unicode, lowercase, collapse whitespace, drop punctuation."""
    norm = unicodedata.normalize("NFKC", text or "").lower()
    norm = re.sub(r"[^\w\s\u0900-\u097F]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def hash_system_prompt(system_prompt: str | None) -> str:
    """Hashes system instructions to a deterministic 12-char hex string."""
    if not system_prompt or not system_prompt.strip():
        return "none"
    norm = normalize_query_text(system_prompt)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ContextAwareCacheKey:
    """Compound multi-attribute cache key guaranteeing tenant and context isolation."""
    scope_id: str = "public"
    model_id: str = "auto"
    system_prompt_hash: str = "none"
    format_constraint: str = "none"
    normalized_query: str = ""

    def compound_key(self) -> str:
        """Deterministic SHA256 composite hash for Layer 1 exact lookup."""
        raw = f"{self.scope_id}::{self.model_id}::{self.system_prompt_hash}::{self.format_constraint}::{self.normalized_query}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CachedEntryMetadata:
    cache_id: int
    question: str
    normalized: str
    answer: str
    scope_id: str
    model_id: str
    system_prompt_hash: str
    format_constraint: str
    source: str = "cache"
    embedding: list[float] | None = None
    created_at: str = ""


class ContextCompatibilityGate:
    """Validates whether a candidate semantic cache entry is safe to return.

    Rejects candidates if:
    1. System instructions differ.
    2. Format constraints differ (e.g. JSON vs bullets).
    3. Programming languages differ (e.g. Python vs Java).
    4. Temporal references differ (e.g. today vs yesterday).
    5. Numeric assignment / course IDs differ (e.g. assignment 1 vs assignment 2).
    """

    @staticmethod
    def extract_critical_entities(text: str) -> dict[str, set[str]]:
        lowered = text.lower()
        words = set(re.findall(r"\b[a-zA-Z0-9+#_]+\b", lowered))

        langs = words & _PROG_LANGS
        # Normalize cpp / csharp aliases
        if "cpp" in langs:
            langs.add("c++")
        if "csharp" in langs:
            langs.add("c#")

        temporal_buckets = set()
        if any(t in lowered for t in _PRESENT_TEMPORALS):
            temporal_buckets.add("present")
        if any(t in lowered for t in _PAST_TEMPORALS):
            temporal_buckets.add("past")
        if any(t in lowered for t in _FUTURE_TEMPORALS):
            temporal_buckets.add("future")

        id_matches = set(_NUMBER_OR_ID_RE.findall(lowered))
        numbers = set(_NUMERIC_TOKEN_RE.findall(lowered))

        return {
            "langs": langs,
            "temporals": temporal_buckets,
            "ids": id_matches,
            "numbers": numbers,
        }

    @classmethod
    def is_compatible(
        cls,
        req_key: ContextAwareCacheKey,
        req_text: str,
        candidate: CachedEntryMetadata,
    ) -> tuple[bool, str]:
        # 1. System Prompt Isolation Gate
        if req_key.system_prompt_hash != candidate.system_prompt_hash:
            return False, f"System prompt mismatch: '{req_key.system_prompt_hash}' vs '{candidate.system_prompt_hash}'"

        # 2. Format Constraint Gate
        if req_key.format_constraint != candidate.format_constraint:
            return False, f"Format constraint mismatch: '{req_key.format_constraint}' vs '{candidate.format_constraint}'"

        # 3. Model Compatibility Gate (if non-auto)
        if req_key.model_id != "auto" and candidate.model_id != "auto" and req_key.model_id != candidate.model_id:
            return False, f"Model mismatch: '{req_key.model_id}' vs '{candidate.model_id}'"

        # 4. Critical Entity & Divergence Gate
        req_ent = cls.extract_critical_entities(req_text)
        cand_ent = cls.extract_critical_entities(candidate.question)

        # Check programming language divergence
        if req_ent["langs"] or cand_ent["langs"]:
            if req_ent["langs"] != cand_ent["langs"]:
                return False, f"Programming language divergence: {req_ent['langs']} vs {cand_ent['langs']}"

        # Check temporal marker divergence (e.g. today vs yesterday)
        if req_ent["temporals"] or cand_ent["temporals"]:
            if req_ent["temporals"] != cand_ent["temporals"]:
                return False, f"Temporal reference divergence: {req_ent['temporals']} vs {cand_ent['temporals']}"

        # Check specific assignment / course ID divergence (assignment 1 vs 2)
        if req_ent["ids"] or cand_ent["ids"]:
            if req_ent["ids"] != cand_ent["ids"]:
                return False, f"Identifier divergence: {req_ent['ids']} vs {cand_ent['ids']}"

        return True, "compatible"


class TwoLayerContextCache:
    """Production Two-Layer Cache:
    Layer 1: O(1) Exact Compound Hash Match (High Confidence)
    Layer 2: Vector Cosine Similarity + Context Compatibility Gate (Validation Required)
    """

    def __init__(self, embedder: SemanticEmbedder | None = None):
        self.embedder = embedder or SemanticEmbedder(use_st=False)
        self.exact_index: dict[str, CachedEntryMetadata] = {}  # compound_key -> entry
        self.entries_by_scope: dict[str, list[CachedEntryMetadata]] = {}

    def insert(self, entry: CachedEntryMetadata) -> str:
        """Registers a cache entry in both exact compound index and semantic scope pool."""
        c_key = ContextAwareCacheKey(
            scope_id=entry.scope_id,
            model_id=entry.model_id,
            system_prompt_hash=entry.system_prompt_hash,
            format_constraint=entry.format_constraint,
            normalized_query=entry.normalized,
        ).compound_key()

        self.exact_index[c_key] = entry
        if entry.scope_id not in self.entries_by_scope:
            self.entries_by_scope[entry.scope_id] = []
        self.entries_by_scope[entry.scope_id].append(entry)
        return c_key

    def lookup(
        self,
        query: str,
        scope_id: str = "public",
        system_prompt: str | None = None,
        format_constraint: str = "none",
        model_id: str = "auto",
        threshold_override: float | None = None,
    ) -> dict[str, Any]:
        """Dual-tier context-aware lookup."""
        normalized = normalize_query_text(query)
        sys_hash = hash_system_prompt(system_prompt)

        req_key = ContextAwareCacheKey(
            scope_id=scope_id,
            model_id=model_id,
            system_prompt_hash=sys_hash,
            format_constraint=format_constraint,
            normalized_query=normalized,
        )
        compound_k = req_key.compound_key()

        # ----------------------------------------------------------------------
        # LAYER 1: Exact Compound Match (O(1), 100% Confidence, < 1ms)
        # ----------------------------------------------------------------------
        if compound_k in self.exact_index:
            hit = self.exact_index[compound_k]
            return {
                "hit": True,
                "layer": 1,
                "match_type": "exact",
                "similarity": 1.0,
                "matched_question": hit.question,
                "answer": hit.answer,
                "cache_id": hit.cache_id,
                "scope_id": hit.scope_id,
                "system_prompt_hash": hit.system_prompt_hash,
                "format_constraint": hit.format_constraint,
                "note": "Exact compound key match across query, system prompt, format, and tenant scope.",
            }

        # ----------------------------------------------------------------------
        # LAYER 2: Semantic Similarity + Context Compatibility Gate
        # ----------------------------------------------------------------------
        scope_entries = self.entries_by_scope.get(scope_id, [])
        if not scope_entries:
            return {
                "hit": False,
                "layer": 2,
                "match_type": "none",
                "similarity": 0.0,
                "matched_question": None,
                "answer": None,
                "note": "No entries in tenant scope.",
            }

        default_thresh = 0.75 if self.embedder.backend == "st" else 0.60
        threshold = threshold_override if threshold_override is not None else default_thresh

        best_entry: CachedEntryMetadata | None = None
        best_sim = 0.0
        rejection_reasons = []

        for cand in scope_entries:
            sim = float(self.embedder.similarity(query, cand.question))
            if sim > best_sim:
                best_sim = sim
                best_entry = cand

            if sim >= threshold:
                is_compat, reason = ContextCompatibilityGate.is_compatible(req_key, query, cand)
                if is_compat:
                    # Semantic Hit passes all context gates
                    return {
                        "hit": True,
                        "layer": 2,
                        "match_type": "semantic",
                        "similarity": round(sim, 4),
                        "matched_question": cand.question,
                        "answer": cand.answer,
                        "cache_id": cand.cache_id,
                        "scope_id": cand.scope_id,
                        "system_prompt_hash": cand.system_prompt_hash,
                        "format_constraint": cand.format_constraint,
                        "threshold": threshold,
                        "note": f"Semantic hit verified by Context Compatibility Gate (similarity: {sim:.3f} >= {threshold}).",
                    }
                else:
                    rejection_reasons.append(f"Rejected candidate '{cand.question}': {reason}")

        # Cache Miss
        return {
            "hit": False,
            "layer": 2,
            "match_type": "none",
            "similarity": round(best_sim, 4),
            "matched_question": best_entry.question if best_entry else None,
            "answer": None,
            "threshold": threshold,
            "note": "No semantic match met threshold and passed context compatibility gates.",
            "rejection_details": rejection_reasons,
        }
