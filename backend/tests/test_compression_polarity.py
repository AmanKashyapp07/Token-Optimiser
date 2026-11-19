"""Unit and regression tests for the Immutable Polarity and Tense Invariant in Indic Compaction."""

from __future__ import annotations

import pytest

from gateway.lexicons import (
    _IMMUTABLE_POLARITY_TOKENS,
    _IMMUTABLE_TENSE_TOKENS,
    prune,
    verify_polarity_invariant,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "required_token", "verb_token"),
    [
        ("Bhai suno mujhe hostel room me nahi jaana hai", "nahi", "jaana"),
        ("Arrey yaar ye button press mat karna", "mat", "karna"),
        ("Please do not submit the incomplete form", "not", "submit"),
    ],
)
def test_immutable_polarity_tokens_preserved(text: str, required_token: str, verb_token: str):
    """Immutable negation tokens must strictly remain in pruned text and drop records."""
    pruned, drops = prune(text)
    assert required_token in pruned.lower()
    assert verb_token in pruned.lower()
    assert verify_polarity_invariant(text, pruned) is True
    assert not any(required_token in d.token.lower() for d in drops)


@pytest.mark.unit
def test_prohibitive_na_vs_conversational_na():
    # Prohibitive 'na' followed by verb: must NOT be dropped
    prohibitive = "Bhai please wahan na jao"
    pruned_proh, _ = prune(prohibitive)
    assert "na" in pruned_proh.lower()
    assert "jao" in pruned_proh.lower()

    # Conversational 'na' as sentence-final tag: safe to drop
    conversational = "Email bhejo na"
    pruned_conv, drops_conv = prune(conversational)
    assert any("na" == d.token.lower() for d in drops_conv)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected_tense_marker"),
    [
        ("Main kal library gaya tha", "tha"),
        ("Main abhi library ja raha hoon", "raha"),
        ("Main kal library jaunga aur book renew hogi", "hogi"),
    ],
)
def test_tense_preservation(text: str, expected_tense_marker: str):
    pruned, _ = prune(text)
    assert expected_tense_marker in pruned.lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected_digits"),
    [
        ("Bhai room 302 ka AC repair kar do port 8080 par", ["302", "8080"]),
        ("Server error HTTP 503 on cluster 99", ["503", "99"]),
        ("Roll number 2024CS001 passed", ["2024", "001"]),
    ],
)
def test_numeric_digit_preservation(text: str, expected_digits: list[str]):
    pruned, _ = prune(text)
    for digit in expected_digits:
        assert digit in pruned
    assert verify_polarity_invariant(text, pruned) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("original", "mutated", "expected_valid"),
    [
        ("Mujhe nahi jaana hai", "Mujhe jaana hai", False),
        ("Order ORD-500 is confirmed", "Order ORD-200 is confirmed", False),
        ("Aap mat aana", "Aap mat aana", True),
        ("Meeting at 4pm", "Meeting at 4pm", True),
    ],
)
def test_verify_polarity_invariant_detects_loss(original: str, mutated: str, expected_valid: bool):
    assert verify_polarity_invariant(original, mutated) is expected_valid


@pytest.mark.unit
def test_immutable_sets_coverage():
    assert "nahi" in _IMMUTABLE_POLARITY_TOKENS
    assert "mat" in _IMMUTABLE_POLARITY_TOKENS
    assert "not" in _IMMUTABLE_POLARITY_TOKENS
    assert "tha" in _IMMUTABLE_TENSE_TOKENS
    assert "hai" in _IMMUTABLE_TENSE_TOKENS
    assert "hogi" in _IMMUTABLE_TENSE_TOKENS
