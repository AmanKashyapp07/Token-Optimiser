"""Unit and regression tests for the Immutable Polarity and Tense Invariant in Indic Compaction."""

from gateway.lexicons import (
    _IMMUTABLE_POLARITY_TOKENS,
    _IMMUTABLE_TENSE_TOKENS,
    prune,
    verify_polarity_invariant,
)


def test_immutable_polarity_nahi_preserved():
    text = "Bhai suno mujhe hostel room me nahi jaana hai"
    pruned, drops = prune(text)
    # Vocative 'bhai' and attention getter 'suno' may drop, but 'nahi' MUST remain
    assert "nahi" in pruned.lower()
    assert "jaana" in pruned.lower()
    assert verify_polarity_invariant(text, pruned) is True
    # Ensure dropped records do not contain 'nahi'
    assert not any("nahi" in d.token.lower() for d in drops)


def test_immutable_polarity_mat_preserved():
    text = "Arrey yaar ye button press mat karna"
    pruned, drops = prune(text)
    assert "mat" in pruned.lower()
    assert "karna" in pruned.lower()
    assert verify_polarity_invariant(text, pruned) is True
    assert not any("mat" in d.token.lower() for d in drops)


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


def test_tense_preservation():
    past = "Main kal library gaya tha"
    present = "Main abhi library ja raha hoon"
    future = "Main kal library jaunga aur book renew hogi"

    p_past, _ = prune(past)
    assert "tha" in p_past.lower()

    p_pres, _ = prune(present)
    assert "raha" in p_pres.lower()

    p_fut, _ = prune(future)
    assert "hogi" in p_fut.lower()


def test_numeric_digit_preservation():
    text = "Bhai room 302 ka AC repair kar do port 8080 par"
    pruned, _ = prune(text)
    assert "302" in pruned
    assert "8080" in pruned
    assert verify_polarity_invariant(text, pruned) is True


def test_verify_polarity_invariant_detects_loss():
    original = "Mujhe nahi jaana hai"
    mutated = "Mujhe jaana hai"  # 'nahi' dropped
    assert verify_polarity_invariant(original, mutated) is False

    original_num = "Order ORD-500 is confirmed"
    mutated_num = "Order ORD-200 is confirmed"  # 500 lost
    assert verify_polarity_invariant(original_num, mutated_num) is False


def test_immutable_sets_coverage():
    assert "nahi" in _IMMUTABLE_POLARITY_TOKENS
    assert "mat" in _IMMUTABLE_POLARITY_TOKENS
    assert "not" in _IMMUTABLE_POLARITY_TOKENS
    assert "tha" in _IMMUTABLE_TENSE_TOKENS
    assert "hai" in _IMMUTABLE_TENSE_TOKENS
    assert "hogi" in _IMMUTABLE_TENSE_TOKENS
