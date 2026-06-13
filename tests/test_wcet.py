"""Tests for seer.lap.wcet — CPU-side smoke test that doesn't need a GPU."""
from __future__ import annotations

import pytest

from seer.lap.model import build_model
from seer.lap.train import wcet_aware_score
from seer.lap.wcet import _percentile, measure_torch_wcet


def test_percentile_basic():
    xs = list(range(1, 101))
    assert _percentile(xs, 50.0) == 50.5
    assert _percentile(xs, 99.0) == 99.01


def test_percentile_empty():
    assert _percentile([], 50.0) == 0.0


def test_measure_torch_wcet_cpu():
    """End-to-end smoke test: a Tiny-MLP runs on CPU without crashing."""
    model = build_model("tiny_mlp", input_dim=37, n_horizons=4, history_n=32)
    result = measure_torch_wcet(
        model=model,
        batch_size=32,
        feat_dim=37,
        device="cpu",
        n_reps=20,
        n_warmup=5,
    )
    assert result.n_reps == 20
    assert result.feat_dim == 37
    assert result.batch_size == 32
    assert result.p50_us > 0
    assert result.max_us >= result.p99_us >= result.p50_us
    # Tiny MLP on CPU at batch=32 should finish in well under 100 ms
    assert result.p99_us < 100_000.0


# ---------------------------------------------------------------------------
#  Regression: P0-3 (WCET-aware Pareto-filter score formula)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewer #2): paper §4.1 originally claimed a
# back-propagated loss ``L_focal + beta * max(0, measured - W)^2``, but
# the implementation only used the penalty for epoch-end checkpoint
# selection (``score = mean_auc - 1e-3 * wcet_penalty``). The paper has
# been rewritten to describe the selection-score Pareto filter; this
# test pins the exact formula so future edits cannot silently drift
# from the published equation (eq:wcet-score in paper/sections/04_method.tex).

def test_wcet_score_in_budget_is_unpenalised():
    """Measured below budget → penalty is zero → score == AUC."""
    s = wcet_aware_score(mean_auc=0.94, measured_us=33.8, budget_us=200.0, beta=1.0)
    assert s == pytest.approx(0.94, abs=1e-9)


def test_wcet_score_at_budget_boundary_is_unpenalised():
    """Measured exactly at budget → penalty still zero (hinge)."""
    s = wcet_aware_score(mean_auc=0.90, measured_us=200.0, budget_us=200.0, beta=1.0)
    assert s == pytest.approx(0.90, abs=1e-9)


def test_wcet_score_over_budget_is_penalised_quadratically():
    """Penalty scales as (measured - budget)² * beta * 1e-3."""
    # measured 1000µs, budget 200µs → excess 800µs → 800^2 * 1.0 * 1e-3 = 640.0
    s = wcet_aware_score(mean_auc=0.92, measured_us=1000.0, budget_us=200.0, beta=1.0)
    assert s == pytest.approx(0.92 - 640.0, abs=1e-9)


def test_wcet_score_beta_zero_disables_filter():
    """When beta=0 (warmup epochs) the WCET term contributes nothing
    even at gross over-budget."""
    s = wcet_aware_score(mean_auc=0.85, measured_us=5000.0, budget_us=200.0, beta=0.0)
    assert s == pytest.approx(0.85, abs=1e-9)


def test_wcet_score_disqualifies_blockrnn_against_tinymlp():
    """Paper claim: BlockRNN (1.87ms P99.9) is disqualified by the
    WCET-aware filter against TinyMLP (33.8µs P99.9) at the same AUC.
    This test reproduces the §6.E ablation outcome in one line."""
    tiny = wcet_aware_score(mean_auc=0.941, measured_us=33.8, budget_us=200.0, beta=1.0)
    rnn = wcet_aware_score(mean_auc=0.941, measured_us=1870.0, budget_us=200.0, beta=1.0)
    assert tiny > rnn, (
        f"TinyMLP must out-rank BlockRNN by the WCET filter "
        f"(got tiny={tiny:.3f} vs rnn={rnn:.3f})"
    )
