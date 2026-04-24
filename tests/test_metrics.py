"""Tests for seer.eval.metrics."""
from __future__ import annotations

from seer.eval.metrics import exact_match, f1_score, normalize, substring_match


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
