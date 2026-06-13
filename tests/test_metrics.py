"""Tests for seer.eval.metrics — quality + RT metrics (RTSS pivot)."""
from __future__ import annotations

import pytest

from seer.eval.metrics import (
    LatencyStats,
    bound_pessimism,
    exact_match,
    f1_score,
    miss_ratio,
    normalize,
    substring_match,
    tpot_percentile,
)

# ---------------------------------------------------------------------------
#  Quality metrics (carried over from NeurIPS-era)
# ---------------------------------------------------------------------------

def test_normalize():
    assert normalize("  Hello, World! ") == "hello world"
    assert normalize("The Quick Brown FOX.") == "quick brown fox"


def test_exact_match():
    assert exact_match("Hello", "hello") == 1.0
    assert exact_match("12345", "12345") == 1.0
    assert exact_match("answer is 12345", "12345") == 0.0


def test_f1_partial_match():
    f1 = f1_score("the quick brown fox", "the quick red fox")
    assert 0.0 < f1 < 1.0


def test_f1_identical_is_one():
    assert f1_score("hello world", "hello world") == 1.0


def test_substring_match():
    assert substring_match("the password is 12345", "12345") == 1.0
    assert substring_match("no match here", "12345") == 0.0


def test_empty_inputs():
    assert f1_score("", "") == 1.0
    assert f1_score("abc", "") == 0.0
    assert exact_match(None, "abc") == 0.0


# ---------------------------------------------------------------------------
#  Regression: P0-7 (SQuAD F1 must match standard evaluate-v1.1 semantics)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewers #1 & #3): the previous set-based
# intersection over-counted repeated tokens. These tests pin the
# multiset semantics so a model that fills gibberish with one correct
# token does not score 1.0.

def test_f1_garbage_prediction_with_correct_token_in_padding():
    """Reviewer-flagged case: pred is mostly garbage with the
    correct answer repeated once → F1 must be far below 1.0."""
    f1 = f1_score("garbage tokens 1978 more garbage", "1978")
    # Standard SQuAD: 1 token in common, precision=1/5, recall=1/1
    # → f1 = 2 · 0.2 · 1.0 / 1.2 ≈ 0.333
    assert f1 == pytest.approx(2 * (1 / 5) * 1.0 / (1 / 5 + 1.0), abs=1e-6)
    assert f1 < 0.4


def test_f1_repeated_correct_token_does_not_inflate_score():
    """Before P0-7, set-intersection gave F1=1.0 for repeated tokens.
    With multiset semantics, repetitions cap at the reference count."""
    # pred: "1978 1978 1978", ref: "1978"
    f1 = f1_score("1978 1978 1978", "1978")
    # Standard SQuAD: 1 token in common, precision=1/3, recall=1/1
    # → f1 = 2 · (1/3) · 1.0 / (1/3 + 1.0) = 0.5
    assert f1 == pytest.approx(0.5, abs=1e-6), (
        f"set-based intersection would give 1.0; multiset gives 0.5 "
        f"(got {f1})"
    )


def test_f1_the_the_the_garbage_against_short_answer():
    """Reviewer-flagged exact case: pred='The The The ...' (which
    normalises to empty after article removal) vs ref='1978' → F1=0."""
    pred = "The The The The The The The The The The"
    f1 = f1_score(pred, "1978")
    # 'the' is an article and stripped by normalize → pred normalises
    # to empty string → F1 = 0 (empty pred against non-empty ref).
    assert f1 == 0.0


def test_f1_matches_squad_v11_reference_on_canonical_pairs():
    """Spot-check several pairs against the SQuAD evaluate-v1.1
    reference results (computed by hand from the public formula)."""
    cases = [
        # (pred, ref, expected_f1, tolerance)
        ("Manchester United", "Manchester United", 1.0, 1e-9),
        ("Manchester", "Manchester United", 2 * 1.0 * 0.5 / 1.5, 1e-6),
        ("United Manchester", "Manchester United", 1.0, 1e-9),  # bag-of-words
        ("Manchester City Football Club", "Manchester United Football Club",
         2 * 0.75 * 0.75 / 1.5, 1e-6),
        # pred="12345" vs ref="answer is 12345" — 'is' is not an
        # article so ref normalises to 3 tokens; common=1 → f1=0.5
        ("12345", "answer is 12345", 0.5, 1e-6),
        # pred="12345" vs ref="answer 12345" — 2-token ref;
        # common=1 → precision=1, recall=0.5 → f1=2/3
        ("12345", "answer 12345", 2 * 1.0 * 0.5 / 1.5, 1e-6),
    ]
    for pred, ref, expected, tol in cases:
        got = f1_score(pred, ref)
        assert got == pytest.approx(expected, abs=tol), (
            f"f1_score({pred!r}, {ref!r}) = {got}, expected {expected}"
        )


def test_f1_normalisation_strips_articles_and_punctuation():
    """SQuAD-style normalisation: lowercase, strip articles, strip
    punctuation, collapse whitespace."""
    f1 = f1_score("The answer.", "answer")
    assert f1 == 1.0
    f1 = f1_score("ANSWER", "answer")
    assert f1 == 1.0
    f1 = f1_score("answer,", "answer.")
    assert f1 == 1.0


# ---------------------------------------------------------------------------
#  RT metrics (RTSS pivot)
# ---------------------------------------------------------------------------

def test_tpot_percentile_basic():
    xs = list(range(1, 101))
    assert tpot_percentile(xs, 50.0) == pytest.approx(50.5)
    assert tpot_percentile(xs, 99.0) == pytest.approx(99.01)


def test_tpot_percentile_empty_returns_zero():
    assert tpot_percentile([], 50.0) == 0.0


def test_tpot_percentile_invalid_pct():
    with pytest.raises(ValueError):
        tpot_percentile([1, 2, 3], 150.0)


def test_miss_ratio_basic():
    xs = [10.0, 20.0, 30.0, 40.0, 100.0]
    assert miss_ratio(xs, deadline_us=25.0) == pytest.approx(0.6)
    assert miss_ratio([], 50.0) == 0.0
    assert miss_ratio(xs, deadline_us=200.0) == 0.0


def test_bound_pessimism_corners():
    assert bound_pessimism(0.05, 0.025) == pytest.approx(2.0)
    assert bound_pessimism(0.0, 0.0) == 1.0
    assert bound_pessimism(0.05, 0.0) == float("inf")


def test_latency_stats_summary():
    xs = [10.0, 20.0, 30.0, 40.0, 1000.0]
    stats = LatencyStats.from_vector(xs, deadline_us=50.0)
    assert stats.n == 5
    assert stats.miss_count == 1
    assert stats.miss_ratio == pytest.approx(0.2)
    assert stats.max_us == pytest.approx(1000.0)
    d = stats.to_dict()
    assert "p99_us" in d
    assert "miss_ratio" in d
