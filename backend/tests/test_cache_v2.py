from __future__ import annotations

import pytest

from gateway.features.cache.context_cache import (
    CachedEntryMetadata,
    ContextAwareCacheKey,
    TwoLayerContextCache,
    hash_system_prompt,
    normalize_query_text,
)
from gateway.modules.m12_semantic.embeddings import SemanticEmbedder


@pytest.fixture
def cache():
    embedder = SemanticEmbedder(use_st=False)
    two_layer = TwoLayerContextCache(embedder=embedder)

    # Seed knowledge base
    two_layer.insert(
        CachedEntryMetadata(
            cache_id=101,
            question="How do I renew my library book online before due date?",
            normalized=normalize_query_text("How do I renew my library book online before due date?"),
            answer="Sign in to the library portal, navigate to Issued Items, and select Renew.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        )
    )
    two_layer.insert(
        CachedEntryMetadata(
            cache_id=102,
            question="Write binary search in Python with code.",
            normalized=normalize_query_text("Write binary search in Python with code."),
            answer="def binary_search(arr, target): ...",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        )
    )
    two_layer.insert(
        CachedEntryMetadata(
            cache_id=103,
            question="What is the stock price of NVIDIA today?",
            normalized=normalize_query_text("What is the stock price of NVIDIA today?"),
            answer="NVIDIA stock price today is $142.50.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        )
    )
    two_layer.insert(
        CachedEntryMetadata(
            cache_id=104,
            question="Explain recursion for young children.",
            normalized=normalize_query_text("Explain recursion for young children."),
            answer="Recursion is like Russian nesting dolls.",
            scope_id="public",
            model_id="auto",
            system_prompt_hash=hash_system_prompt("You are a kindergarten teacher."),
            format_constraint="none",
        )
    )
    two_layer.insert(
        CachedEntryMetadata(
            cache_id=105,
            question="Confidential payroll data for executive staff.",
            normalized=normalize_query_text("Confidential payroll data for executive staff."),
            answer="CONFIDENTIAL_SALARY_DATA",
            scope_id="tenant-secure",
            model_id="auto",
            system_prompt_hash="none",
            format_constraint="none",
        )
    )
    return two_layer


@pytest.mark.unit
def test_context_aware_cache_key_generation():
    """Verify deterministic SHA256 composite compound key calculation."""
    key1 = ContextAwareCacheKey(
        scope_id="public",
        model_id="auto",
        system_prompt_hash="sys123",
        format_constraint="3 bullets",
        normalized_query="how do i renew library book",
    )
    key2 = ContextAwareCacheKey(
        scope_id="public",
        model_id="auto",
        system_prompt_hash="sys123",
        format_constraint="3 bullets",
        normalized_query="how do i renew library book",
    )
    key3 = ContextAwareCacheKey(
        scope_id="public",
        model_id="auto",
        system_prompt_hash="sys999",  # Different system prompt
        format_constraint="3 bullets",
        normalized_query="how do i renew library book",
    )

    assert key1.compound_key() == key2.compound_key()
    assert key1.compound_key() != key3.compound_key()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("divergent_system_prompt", "should_hit"),
    [
        ("You are a theoretical computer science professor.", False),
        ("Act as an expert Linux kernel developer.", False),
        ("You are a kindergarten teacher.", True),
    ],
)
def test_system_prompt_isolation(cache: TwoLayerContextCache, divergent_system_prompt: str, should_hit: bool):
    """Identical query with different system instructions must not produce a false cache hit."""
    res = cache.lookup(
        query="Explain recursion for young children.",
        system_prompt=divergent_system_prompt,
        scope_id="public",
    )
    assert res["hit"] is should_hit
    if should_hit:
        assert res["layer"] == 1
    else:
        assert res["match_type"] == "none"


@pytest.mark.unit
@pytest.mark.parametrize(
    "divergent_language",
    ["Java", "Rust", "Go", "C++"],
)
def test_programming_language_divergence_rejection(cache: TwoLayerContextCache, divergent_language: str):
    """Python vs alternate language queries must be rejected even with high lexical similarity."""
    # Knowledge base has 'Write binary search in Python with code.'
    res = cache.lookup(
        query=f"Write binary search in {divergent_language} with code.",
        scope_id="public",
        threshold_override=0.50,
    )
    assert res["hit"] is False
    assert res["match_type"] == "none"


@pytest.mark.unit
@pytest.mark.parametrize(
    "divergent_temporal_query",
    [
        "What was the stock price of NVIDIA yesterday?",
        "What will be the stock price of NVIDIA tomorrow?",
        "What was the stock price of NVIDIA last month?",
    ],
)
def test_temporal_divergence_rejection(cache: TwoLayerContextCache, divergent_temporal_query: str):
    """Today vs divergent temporal queries must be rejected by temporal compatibility gate."""
    # Knowledge base has 'What is the stock price of NVIDIA today?'
    res = cache.lookup(
        query=divergent_temporal_query,
        scope_id="public",
        threshold_override=0.50,
    )
    assert res["hit"] is False
    assert res["match_type"] == "none"


@pytest.mark.unit
def test_tenant_isolation_boundary(cache: TwoLayerContextCache):
    """Tenant-secure entries must never be accessible from public or distinct tenant scopes."""
    res_public = cache.lookup(
        query="Confidential payroll data for executive staff.",
        scope_id="public",
    )
    assert res_public["hit"] is False

    res_other_tenant = cache.lookup(
        query="Confidential payroll data for executive staff.",
        scope_id="tenant-finance",
    )
    assert res_other_tenant["hit"] is False

    res_authorized = cache.lookup(
        query="Confidential payroll data for executive staff.",
        scope_id="tenant-secure",
    )
    assert res_authorized["hit"] is True
    assert res_authorized["answer"] == "CONFIDENTIAL_SALARY_DATA"


@pytest.mark.unit
def test_layer1_exact_vs_layer2_semantic_routing(cache: TwoLayerContextCache):
    """Layer 1 resolves in O(1) compound lookup; Layer 2 resolves near paraphrases."""
    # Layer 1: Identical normalized string
    exact_res = cache.lookup(
        query="How do I renew my library book online before due date?",
        scope_id="public",
    )
    assert exact_res["hit"] is True
    assert exact_res["layer"] == 1
    assert exact_res["match_type"] == "exact"

    # Layer 2: Paraphrase meeting similarity threshold & passing compatibility gate
    semantic_res = cache.lookup(
        query="Can you explain how to renew my library book online before due date?",
        scope_id="public",
        threshold_override=0.50,
    )
    assert semantic_res["hit"] is True
    assert semantic_res["layer"] == 2
    assert semantic_res["match_type"] == "semantic"
