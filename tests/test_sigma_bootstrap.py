"""Tests for the σ bootstrap-CI helper (P3-1).

Reviewer round (2026-05, reviewers #1 + #3): the eC calibrated bound
depends on σ as a point estimate from ~600 pilot steps. The bound is
exponential in 1/σ², so an order-of-magnitude swing in the bound
can come from a ~10% swing in σ. These tests pin the
``sigma_bootstrap_ci`` and ``sigma_cross_trace_table`` contract.
"""
from __future__ import annotations

import random

import pytest

from seer.timing.sigma_estimate import (
    sigma_bootstrap_ci,
    sigma_cross_trace_table,
)


def _gaussian_pilot(n=600, sigma=1200.0, mu=33000.0, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def test_bootstrap_ci_shape():
    """Bootstrap CI returns the expected keys."""
    pilot = _gaussian_pilot()
    ci = sigma_bootstrap_ci(pilot, n_resamples=200)
    for k in (
        "sigma_point_us", "sigma_median_us", "sigma_lo_us",
        "sigma_hi_us", "confidence", "n_samples", "n_resamples",
    ):
        assert k in ci, f"missing key {k}"
    assert ci["n_samples"] == 600
    assert ci["n_resamples"] == 200
    assert ci["confidence"] == pytest.approx(0.95)


def test_bootstrap_ci_brackets_point_estimate():
    """The CI bounds must straddle the median of the resampled σs."""
    pilot = _gaussian_pilot()
    ci = sigma_bootstrap_ci(pilot, n_resamples=500)
    assert ci["sigma_lo_us"] <= ci["sigma_median_us"] <= ci["sigma_hi_us"]
    # The point estimate σ from the full sample sits near the median.
    assert abs(ci["sigma_point_us"] - ci["sigma_median_us"]) < ci["sigma_point_us"] * 0.2


def test_bootstrap_ci_narrows_with_larger_n():
    """CI width should shrink as the input sample grows."""
    small = sigma_bootstrap_ci(_gaussian_pilot(n=100, seed=1), n_resamples=300)
    large = sigma_bootstrap_ci(_gaussian_pilot(n=5000, seed=1), n_resamples=300)
    small_width = small["sigma_hi_us"] - small["sigma_lo_us"]
    large_width = large["sigma_hi_us"] - large["sigma_lo_us"]
    assert large_width < small_width


def test_bootstrap_ci_rejects_tiny_samples():
    with pytest.raises(ValueError, match="at least 20"):
        sigma_bootstrap_ci([1.0, 2.0, 3.0])


def test_cross_trace_table_returns_one_row_per_trace():
    pilots = {
        "mooncake":   _gaussian_pilot(n=600, sigma=1200.0, seed=10),
        "squad150":   _gaussian_pilot(n=600, sigma=4500.0, seed=20),
        "qwen":       _gaussian_pilot(n=600, sigma=3800.0, seed=30),
    }
    rows = sigma_cross_trace_table(pilots, n_resamples=300)
    by_name = {r["trace"]: r for r in rows}
    assert set(by_name) == {"mooncake", "squad150", "qwen"}
    # Cross-workload diagnosis: SQuAD's σ-CI must NOT overlap Mooncake's
    # (the bound's calibration on Mooncake doesn't transfer).
    moo = by_name["mooncake"]
    sq = by_name["squad150"]
    assert moo["sigma_hi_us"] < sq["sigma_lo_us"], (
        f"Mooncake σ_hi ({moo['sigma_hi_us']:.0f} µs) must be below "
        f"SQuAD σ_lo ({sq['sigma_lo_us']:.0f} µs) for the cross-trace "
        f"diagnosis to fire."
    )


def test_cross_trace_table_handles_too_few_samples_gracefully():
    pilots = {
        "tiny": [1.0, 2.0, 3.0, 4.0],
        "ok":   _gaussian_pilot(n=100, seed=42),
    }
    rows = sigma_cross_trace_table(pilots, n_resamples=100)
    tiny_row = next(r for r in rows if r["trace"] == "tiny")
    assert "error" in tiny_row, "tiny pilot must yield an error key"
    ok_row = next(r for r in rows if r["trace"] == "ok")
    assert "error" not in ok_row, "100-sample pilot should succeed"
