"""Feature extraction tests. Labels elsewhere are authored fixtures, not model correctness."""

from __future__ import annotations

import math
import pytest

from gateway.features.routing.features import MODEL_FEATURE_NAMES, extract_features


@pytest.mark.unit
@pytest.mark.parametrize("blank_text", ["", "   \n\t ", "\r\n  "])
def test_empty_and_whitespace_only_are_safe(blank_text: str):
    features = extract_features(blank_text)
    assert features.language_group == "unknown"
    assert len(features.vector()) == len(MODEL_FEATURE_NAMES)
    assert all(math.isfinite(value) for value in features.vector())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prompt", "expected_lang"),
    [
        ("Hello", "english"),
        ("How are you doing today?", "english"),
    ],
)
def test_plain_english_sentence(prompt: str, expected_lang: str):
    features = extract_features(prompt)
    assert features.language_group == expected_lang
    assert features.code_mix_ratio == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "hinglish_prompt",
    [
        "bhai kal chutti hai kya",
        "yaar please assignment ka submission link bhej do",
    ],
)
def test_romanized_hinglish_sentence(hinglish_prompt: str):
    # The imported helper returns "hinglish" for this input.
    assert extract_features(hinglish_prompt).language_group == "hinglish"


@pytest.mark.unit
def test_devanagari_sentence_is_hindi():
    features = extract_features("मेरे खाते का पैसा नहीं भेजना")
    assert features.language_group == "hindi"
    assert features.devanagari_ratio > 0.9


@pytest.mark.unit
def test_mixed_devanagari_and_latin_has_code_mix():
    assert extract_features("मेरा wifi काम नहीं कर रहा").code_mix_ratio > 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected_word_count"),
    [
        ("user@example.com", 1),
        ("admin@university.edu", 1),
    ],
)
def test_email_counts_as_one_word_and_sets_flag(text: str, expected_word_count: int):
    features = extract_features(text)
    assert features.word_count == expected_word_count
    assert features.has_email_or_amount == 1


@pytest.mark.unit
def test_vector_shape_determinism_and_finiteness():
    text = "yaar mera email user@example.com par bhejo na, deadline kal hai"
    first = extract_features(text)
    second = extract_features(text)
    assert len(first.vector()) == len(MODEL_FEATURE_NAMES)
    assert first.vector() == second.vector()
    assert first.as_dict() == second.as_dict()
    assert all(isinstance(value, float) for value in first.vector())
    assert all(math.isfinite(value) for value in first.vector())


@pytest.mark.unit
def test_long_input_is_truncated_without_raising():
    features = extract_features("a " * 15_000)  # 30,000 characters
    assert features.truncated is True
    assert features.char_length == 30_000
    assert len(features.vector()) == len(MODEL_FEATURE_NAMES)
