"""Tests for the seer.timing module — SLO parsing, deadline derivation,
PI controller, and schedulability bounds."""
from __future__ import annotations

import pytest

from seer.timing import (
    LambdaController,
    SLOClass,
    derive_deadline_us,
    lemma1_lateness_bound,
    lemma2_bernstein_miss_prob_bound,
    lemma2_bernstein_mixture_bound,
    lemma2_miss_prob_bound,
    lemma3_heuristic_gap,
    min_hbm_budget_for_slo,
    min_hbm_budget_for_slo_measured,
    parse_slo,
)

# ---------------------------------------------------------------------------
#  SLO parsing
# ---------------------------------------------------------------------------

def test_parse_preset():
    s = parse_slo("chat-50ms")
    assert s.kind == "TPOT"
    assert s.threshold_ms == pytest.approx(50.0)
    assert s.percentile == pytest.approx(99.0)


def test_parse_p99_50ms():
    s = parse_slo("P99=50ms")
    assert s.kind == "TPOT"
    assert s.threshold_ms == pytest.approx(50.0)
    assert s.percentile == pytest.approx(99.0)
    assert s.miss_target == pytest.approx(0.01, abs=1e-6)


def test_parse_ttft_p99_1s():
    s = parse_slo("TTFT-P99=1s")
    assert s.kind == "TTFT"
    assert s.threshold_ms == pytest.approx(1000.0)


def test_parse_p999():
    s = parse_slo("P99.9=50ms")
    assert s.percentile == pytest.approx(99.9)
    assert s.miss_target == pytest.approx(0.001, abs=1e-6)


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_slo("not-a-slo")


# ---------------------------------------------------------------------------
#  Deadline derivation
# ---------------------------------------------------------------------------

def test_tpot_deadline_us():
    s = parse_slo("P99=50ms")
    assert derive_deadline_us(s) == pytest.approx(50_000.0)


def test_ttft_deadline_us():
    s = parse_slo("TTFT-P99=1s")
    # No prefill given → returns the full threshold in µs
    assert derive_deadline_us(s) == pytest.approx(1_000_000.0)
    # With prefill given → subtracts it
    assert derive_deadline_us(s, prefill_us=400_000.0) == pytest.approx(600_000.0)


# ---------------------------------------------------------------------------
#  PI controller
# ---------------------------------------------------------------------------

def test_lambda_controller_drops_when_over_budget():
    c = LambdaController(target_us=50_000.0, kp=1e-3, ki=0.0, initial=1.0)
    for _ in range(100):
        c.observe(60_000.0)  # always over target
    assert c.current_lambda() < 1.0


def test_lambda_controller_rises_when_under_budget():
    c = LambdaController(target_us=50_000.0, kp=1e-3, ki=0.0, initial=0.5)
    for _ in range(100):
        c.observe(40_000.0)
    assert c.current_lambda() > 0.5


def test_lambda_controller_clamps():
    c = LambdaController(target_us=1.0, kp=1.0, ki=0.0,
                         initial=1.0, lambda_min=0.0, lambda_max=2.0)
    for _ in range(50):
        c.observe(1_000_000.0)
    assert c.current_lambda() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
#  Lemmas
# ---------------------------------------------------------------------------

def test_lemma1_linearity():
    # E[L_t] = eps * B_t * ell_bar
    assert lemma1_lateness_bound(0.1, 100, 200) == pytest.approx(2000.0)
    assert lemma1_lateness_bound(0.0, 100, 200) == 0.0
    assert lemma1_lateness_bound(0.5, 100, 200) == pytest.approx(10000.0)


def test_lemma1_invalid_epsilon():
    with pytest.raises(ValueError):
        lemma1_lateness_bound(1.5, 100, 200)


def test_lemma2_zero_when_deadline_far():
    # Far enough past mean → bound near 0
    bound = lemma2_miss_prob_bound(
        epsilon=0.1, ell_bar_us=200, B_t=64,
        deadline_us=1_000_000, sigma_residual_us=100,
        base_cost_us=1500,
    )
    assert 0.0 <= bound < 1e-3


def test_lemma2_one_when_deadline_close_to_mean():
    bound = lemma2_miss_prob_bound(
        epsilon=0.5, ell_bar_us=200, B_t=64,
        deadline_us=1500.0, sigma_residual_us=100,
        base_cost_us=1500.0,
    )
    # Deadline ≤ mean → bound is vacuous (1.0)
    assert bound == 1.0


def test_lemma2_monotonic_in_deadline():
    a = lemma2_miss_prob_bound(0.1, 200, 64, 5_000, 100, base_cost_us=1500)
    b = lemma2_miss_prob_bound(0.1, 200, 64, 50_000, 100, base_cost_us=1500)
    assert b <= a  # bigger deadline → tighter (smaller) miss bound


def test_lemma2_monotonic_in_epsilon():
    a = lemma2_miss_prob_bound(0.05, 200, 64, 50_000, 100, base_cost_us=1500)
    b = lemma2_miss_prob_bound(0.30, 200, 64, 50_000, 100, base_cost_us=1500)
    assert b >= a  # bigger epsilon → bigger miss bound


def test_lemma3_gap_grows_with_shift():
    h_low, lap_low = lemma3_heuristic_gap(0.1, 0.1, 64, 200)
    h_hi, lap_hi = lemma3_heuristic_gap(0.5, 0.1, 64, 200)
    assert h_hi > h_low
    # LAP unchanged because it's parameterized only on epsilon
    assert lap_low == lap_hi


def test_min_hbm_budget_inverts():
    # Find smallest budget φ ≤ 1 keeping bound ≤ ρ. Should be in (0, 1].
    phi = min_hbm_budget_for_slo(
        epsilon=0.1, ell_bar_us=200, B_full=64,
        deadline_us=50_000, sigma_residual_us=100,
        miss_target=0.01, base_cost_us=1500,
    )
    assert 0.0 < phi <= 1.0
    # And monotonic: tighter miss target → bigger phi (need more HBM)
    phi_tight = min_hbm_budget_for_slo(
        epsilon=0.1, ell_bar_us=200, B_full=64,
        deadline_us=50_000, sigma_residual_us=100,
        miss_target=0.001, base_cost_us=1500,
    )
    assert phi_tight >= phi - 1e-6


# ---------------------------------------------------------------------------
#  Bernstein + Bernstein-mixture variants of Lemma 2
# ---------------------------------------------------------------------------

def test_lemma2_variants_agree_on_vacuous_regime():
    # When D < μ all three forms must return 1.0 (vacuous).
    args = dict(ell_bar_us=200, B_t=64, deadline_us=10_000,
                sigma_residual_us=100, base_cost_us=20_000)
    assert lemma2_miss_prob_bound(epsilon=0.1, **args) == 1.0
    assert lemma2_bernstein_miss_prob_bound(epsilon=0.1, **args) == 1.0
    args_mix = dict(args)
    args_mix.pop("base_cost_us", None)
    assert lemma2_bernstein_mixture_bound(
        epsilon_mean=0.1, epsilon_var=0.01,
        base_cost_us=20_000, **{k: v for k, v in args.items() if k != "base_cost_us"}
    ) == 1.0


def test_lemma2_variants_in_unit_interval():
    # All forms must return values in [0, 1] across a sweep.
    for D_ms in (35, 50, 75, 100):
        h = lemma2_miss_prob_bound(
            epsilon=0.07, ell_bar_us=200, B_t=12.8,
            deadline_us=D_ms * 1000, sigma_residual_us=1500,
            base_cost_us=22_000)
        b = lemma2_bernstein_miss_prob_bound(
            epsilon=0.07, ell_bar_us=200, B_t=12.8,
            deadline_us=D_ms * 1000, sigma_residual_us=1500,
            base_cost_us=22_000)
        m = lemma2_bernstein_mixture_bound(
            epsilon_mean=0.07, epsilon_var=0.005,
            ell_bar_us=200, B_t=12.8,
            deadline_us=D_ms * 1000, sigma_residual_us=1500,
            base_cost_us=22_000)
        for v in (h, b, m):
            assert 0.0 <= v <= 1.0, (D_ms, v)


def test_bernstein_mixture_eps_var_bounds():
    # epsilon_var must lie in [0, ε̄(1-ε̄)].
    with pytest.raises(ValueError):
        lemma2_bernstein_mixture_bound(
            epsilon_mean=0.1, epsilon_var=0.5,  # >> 0.1*0.9 = 0.09
            ell_bar_us=200, B_t=64,
            deadline_us=50_000, sigma_residual_us=100,
            base_cost_us=20_000)


def test_bernstein_mixture_dominates_homogeneous_at_max_var():
    # At v_eps = ε̄(1-ε̄) (max-spread mixture), the per-block variance
    # collapses to 0 in the bound's ν, leaving only σ². The mixture
    # bound should NOT be looser than the homogeneous Bernstein at
    # any operating point we test (it's strictly tighter when v_eps>0).
    common = dict(ell_bar_us=200, B_t=12.8,
                  deadline_us=40_000, sigma_residual_us=1500,
                  base_cost_us=22_000)
    homo = lemma2_bernstein_miss_prob_bound(epsilon=0.07, **common)
    mix_low_var = lemma2_bernstein_mixture_bound(
        epsilon_mean=0.07, epsilon_var=0.0, **common)
    mix_max_var = lemma2_bernstein_mixture_bound(
        epsilon_mean=0.07, epsilon_var=0.07 * 0.93, **common)
    # v_eps=0 should match (or essentially match) homogeneous Bernstein
    assert mix_low_var <= homo + 1e-12
    # v_eps>0 should be at least as tight as v_eps=0
    assert mix_max_var <= mix_low_var + 1e-12


# ---------------------------------------------------------------------------
#  Measured ε(φ) curve sizing
# ---------------------------------------------------------------------------

def test_min_hbm_budget_measured_basic():
    # A monotone-decreasing ε(φ) curve: more HBM → lower miss rate.
    eps_curve = [(0.1, 0.30), (0.2, 0.18), (0.4, 0.10), (0.8, 0.06), (1.0, 0.05)]
    phi = min_hbm_budget_for_slo_measured(
        eps_curve=eps_curve, ell_bar_us=200, B_full=64,
        deadline_us=100_000, sigma_residual_us=1500,
        miss_target=0.01, base_cost_us=22_000)
    assert 0.0 < phi <= 1.0
    # Tighter miss target → at least as much HBM
    phi_tight = min_hbm_budget_for_slo_measured(
        eps_curve=eps_curve, ell_bar_us=200, B_full=64,
        deadline_us=100_000, sigma_residual_us=1500,
        miss_target=0.001, base_cost_us=22_000)
    assert phi_tight >= phi - 1e-6


def test_min_hbm_budget_measured_handles_unsorted():
    # Should sort knots itself.
    unordered = [(0.4, 0.10), (1.0, 0.05), (0.1, 0.30), (0.8, 0.06), (0.2, 0.18)]
    phi = min_hbm_budget_for_slo_measured(
        eps_curve=unordered, ell_bar_us=200, B_full=64,
        deadline_us=100_000, sigma_residual_us=1500,
        miss_target=0.01, base_cost_us=22_000)
    assert 0.0 < phi <= 1.0


def test_min_hbm_budget_measured_empty_raises():
    with pytest.raises(ValueError):
        min_hbm_budget_for_slo_measured(
            eps_curve=[], ell_bar_us=200, B_full=64,
            deadline_us=100_000, sigma_residual_us=1500,
            miss_target=0.01)


# ---------------------------------------------------------------------------
#  Regression: B4 (monotone-cleaning must lift low-φ ε up, not push down)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewer #2): the previous two-pass running-min
# cleaning collapsed the ε(φ) curve to its global minimum, leaking
# high-budget ε all the way down to the lowest-φ knot. These tests pin
# the corrected right→left running-max projection so the sizing rule
# stays conservative under measurement noise.

def test_monotone_clean_preserves_already_monotone_curve():
    """A strictly monotone-decreasing input passes through unchanged
    (up to clamping)."""
    from seer.timing.schedulability import _monotone_clean_eps_curve
    inp = [(0.1, 0.60), (0.2, 0.30), (0.4, 0.10)]
    out = _monotone_clean_eps_curve(inp)
    assert out == [(0.1, 0.60), (0.2, 0.30), (0.4, 0.10)]


def test_monotone_clean_old_bug_would_collapse_to_global_min():
    """B4 regression: under the OLD two-pass running-min code, this
    monotone input would be flattened to [(0.1, 0.1), (0.2, 0.1),
    (0.4, 0.1)] — ε(0.1)=0.6 leaked down to 0.1. The fix keeps the
    knot values intact."""
    from seer.timing.schedulability import _monotone_clean_eps_curve
    inp = [(0.1, 0.60), (0.2, 0.30), (0.4, 0.10)]
    out = _monotone_clean_eps_curve(inp)
    # ε(0.1) must NOT collapse to 0.1 (the high-budget value)
    eps_at_01 = next(e for phi, e in out if phi == 0.1)
    assert eps_at_01 == 0.60, (
        f"ε(0.1) must stay at 0.60, not collapse to global-min "
        f"(got {eps_at_01})"
    )


def test_monotone_clean_lifts_low_phi_violation_upward():
    """When a low-φ knot is below a higher-φ knot (noise), the
    cleaned curve lifts the low-φ knot up to match the higher-φ
    knot — conservative for sizing inversion."""
    from seer.timing.schedulability import _monotone_clean_eps_curve
    # Noisy: ε(0.1) = 0.3 should be at least ε(0.2) = 0.5 to be monotone
    inp = [(0.1, 0.30), (0.2, 0.50), (0.4, 0.10)]
    out = _monotone_clean_eps_curve(inp)
    eps_at_01 = next(e for phi, e in out if phi == 0.1)
    eps_at_02 = next(e for phi, e in out if phi == 0.2)
    assert eps_at_01 >= eps_at_02, (
        f"low-φ ε must dominate high-φ ε (got 0.1:{eps_at_01}, "
        f"0.2:{eps_at_02})"
    )
    assert eps_at_01 == 0.50, (
        f"lifted low-φ ε must match the higher-φ violator "
        f"(expected 0.50, got {eps_at_01})"
    )


def test_monotone_clean_resolves_internal_bump():
    """An internal non-monotone bump (high ε at mid-φ) lifts the
    neighboring lower-φ knots."""
    from seer.timing.schedulability import _monotone_clean_eps_curve
    inp = [(0.1, 0.60), (0.2, 0.30), (0.3, 0.50), (0.4, 0.20)]
    out = _monotone_clean_eps_curve(inp)
    expected = {0.1: 0.60, 0.2: 0.50, 0.3: 0.50, 0.4: 0.20}
    for phi, eps in out:
        assert eps == pytest.approx(expected[phi]), (
            f"at φ={phi}, expected ε={expected[phi]}, got {eps}"
        )


def test_monotone_clean_clamps_to_unit_interval():
    """ε values outside [0, 1] are clipped before monotone projection."""
    from seer.timing.schedulability import _monotone_clean_eps_curve
    inp = [(0.1, 1.5), (0.4, -0.2), (0.7, 0.05)]
    out = _monotone_clean_eps_curve(inp)
    for _phi, eps in out:
        assert 0.0 <= eps <= 1.0


# ---------------------------------------------------------------------------
#  Regression: B10 (ε estimator must match deployed selection policy)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewer #3): the previous estimator
# computed (1 - recall) at a 0.5 LAP-score threshold; the deployed
# policy is rank-based (sink + window + greedy top-K by score). These
# tests pin the policy-equivalent selection contract.

def test_policy_equivalent_fn_keeps_sink_and_window():
    """B10: at budget=1.0, sink+window cover the entire kept set
    (10 blocks, sink=4, window=4 → 8 forced + 2 by score = 10).
    No false-negative is possible when every block is kept."""
    from seer.timing.schedulability import policy_equivalent_false_negatives
    block_ids = list(range(10))
    scores = [0.0] * 10
    truth = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 5 positives
    fn, pos = policy_equivalent_false_negatives(
        block_ids, scores, truth, hbm_budget=1.0,
    )
    assert fn == 0
    assert pos == 5


def test_policy_equivalent_fn_at_low_budget():
    """B10: at budget=0.4 on 10 blocks → budget_k=4, sink=4, window=4
    overflow → kept set picks window then sink (top-K-by-score never
    fires). Truth positives outside forced set count as FN."""
    from seer.timing.schedulability import policy_equivalent_false_negatives
    block_ids = list(range(10))
    scores = [0.0] * 10
    # Positives in the MIDDLE (5, 6) — neither sink nor window
    truth = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    fn, pos = policy_equivalent_false_negatives(
        block_ids, scores, truth, hbm_budget=0.4, sink=4, window=4,
    )
    # budget_k=4 < sink+window=8 → kept = window(6..9)+sink(0..3)
    # truncated to budget=4. Positives 5, 6:
    #   - block 6: depends on order; with window-then-sink truncation
    #     kept includes block 6 (it's in [-4:][::-1] = [9,8,7,6])
    #   - block 5: NOT in kept (window covers 6-9, sink covers 0-3)
    # So 1 FN.
    assert pos == 2
    assert fn == 1


def test_policy_equivalent_fn_uses_lap_score_for_greedy():
    """B10: at intermediate budget, the greedy fill must use LAP
    scores. Truth-positive blocks with HIGH score get kept; those
    with LOW score lose to non-positive blocks with higher score."""
    from seer.timing.schedulability import policy_equivalent_false_negatives
    block_ids = list(range(20))
    # sink=4 → blocks 0..3 forced, window=4 → blocks 16..19 forced.
    # budget=0.5 × 20 = 10, so 2 more slots filled by score.
    # Truth positives: block 8 (high score, kept) and block 9
    # (low score, OUT-RANKED by non-positives 5, 6, 7 with higher scores).
    scores = [0.0] * 20
    scores[5] = 0.8
    scores[6] = 0.7
    scores[7] = 0.6
    scores[8] = 0.95  # truth-positive, high score → kept
    scores[9] = 0.05  # truth-positive, low score → out-ranked, evicted
    truth = [0] * 20
    truth[8] = 1
    truth[9] = 1
    fn, pos = policy_equivalent_false_negatives(
        block_ids, scores, truth, hbm_budget=0.5, sink=4, window=4,
    )
    assert pos == 2
    # Top-2 candidates by score among non-forced: block 8 (0.95), 5 (0.8)
    # → kept = {0,1,2,3,8,5,16,17,18,19}; block 9 is missed.
    assert fn == 1


def test_policy_equivalent_fn_handles_empty_group():
    """B10: a group with no positives returns (0, 0)."""
    from seer.timing.schedulability import policy_equivalent_false_negatives
    fn, pos = policy_equivalent_false_negatives(
        block_ids=[1, 2, 3], lap_scores=[0.5, 0.5, 0.5],
        truth_top_k=[0, 0, 0], hbm_budget=0.5,
    )
    assert fn == 0
    assert pos == 0


def test_policy_equivalent_fn_never_exceeds_budget():
    """T1-C (May 2026 reviewer round R1-1.1#3 / R2-#4): the kept-set
    chosen by the estimator must NEVER exceed ``budget_k =
    ceil(hbm_budget * n)`` — otherwise it claims more coverage than
    the deployed policy actually has, under-estimating ε at the
    sizing-rule floor budgets.

    The previous estimator unioned sink + window unconditionally and
    could end up with len(kept) > budget_k when budget_k < sink+window.
    """
    # 20 blocks, hbm_budget ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.80}
    # → budget_k ∈ {1, 2, 3, 4, 6, 8, 16}.  sink+window=8, so any
    # budget < 0.40 triggers the truncation path.
    import numpy as np

    from seer.timing.schedulability import policy_equivalent_false_negatives
    rng = np.random.default_rng(42)
    n = 20
    for hbm_budget in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.80, 1.0]:
        scores = rng.uniform(0.0, 1.0, n).tolist()
        truth = rng.integers(0, 2, n).tolist()
        block_ids = list(range(n))
        fn, pos = policy_equivalent_false_negatives(
            block_ids, scores, truth, hbm_budget=hbm_budget, sink=4, window=4,
        )
        budget_k = max(1, int(round(hbm_budget * n)))
        # If kept ever exceeded budget_k, fn would be artificially low.
        # We can't observe kept directly here, but the inner assert
        # in the function guarantees this property; this test makes
        # the assertion path actually fire across the relevant grid.
        assert 0 <= fn <= pos


def test_policy_equivalent_fn_matches_deployed_seerpolicy():
    """T1-C: pointwise parity check between the estimator's kept-set
    and the deployed SEERPolicy.select_to_keep output. This is the
    contract the estimator promises to honour."""
    import numpy as np

    from seer.policy.seer import SEERPolicy
    from seer.timing.schedulability import policy_equivalent_false_negatives

    rng = np.random.default_rng(0)
    n = 16
    block_ids = list(range(n))
    scores = rng.uniform(0.0, 1.0, n)
    # Build a fake LAP that returns the pre-computed scores when
    # called on each block's feature row. SEERPolicy._features stuffs
    # the score into a known slot; here we just stub the predictor
    # to read it back from ``block_stats[bid]["lap_score"]`` via an
    # ID->score lookup keyed on a sentinel feature column.
    block_stats = {
        bid: {"lap_score": float(scores[i]), "io_cost": 0.0,
              "last_attention": float(scores[i]),
              "cumulative_attention": float(scores[i]),
              "recency": float(i)}
        for i, bid in enumerate(block_ids)
    }

    # Fake LAP: ignores features, returns the scores in the order
    # SEERPolicy iterates bids (which is the insertion order of
    # ``block_stats``). Same scoring as the estimator's ``lap_scores``
    # argument so the kept-set should match by construction.
    score_iter = {bid: float(scores[i]) for i, bid in enumerate(block_ids)}
    def fake_lap(X):
        # SEERPolicy passes one feature row per bid in block_stats
        # order; we just return the corresponding score per row.
        return np.array(
            [score_iter[bid] for bid in block_ids][: X.shape[0]],
            dtype=np.float32,
        ).reshape(-1, 1)

    for hbm_budget in [0.10, 0.20, 0.40, 0.80, 1.0]:
        budget_k = max(1, int(round(hbm_budget * n)))
        pol = SEERPolicy(lap_predictor=fake_lap, sink=4, window=4,
                         lam_io=0.0)
        kept_deployed = pol.select_to_keep(block_stats, budget=budget_k, step=0)

        # Build truth_top_k that marks the deployed-kept blocks as
        # positives. The estimator must then return fn == 0 because
        # the deployed selection is, by construction, perfect against
        # this synthetic truth.
        truth = [1 if bid in kept_deployed else 0 for bid in block_ids]
        fn, pos = policy_equivalent_false_negatives(
            block_ids, scores.tolist(), truth,
            hbm_budget=hbm_budget, sink=4, window=4,
        )
        assert pos == len(kept_deployed)
        assert fn == 0, (
            f"estimator's kept-set ≠ SEERPolicy's at hbm_budget={hbm_budget} "
            f"(budget_k={budget_k}); fn={fn} but should be 0"
        )


def test_policy_equivalent_fn_diverges_from_threshold_recall():
    """B10: build a setup where the 0.5-threshold-recall estimator
    gives a very different ε from the policy-equivalent estimator,
    proving the two are not interchangeable.

    Scenario: 20 blocks, 6 positives with score = 0.55 (all clear
    the 0.5 threshold). Threshold-recall would say ε ≈ 0. But at
    budget=0.30 (= 6 blocks) with sink=4 + window=4 = 8 forced
    slots filling the whole budget, the score-based greedy never
    fires; positives outside sink/window are missed.
    """
    from seer.timing.schedulability import policy_equivalent_false_negatives
    block_ids = list(range(20))
    scores = [0.0] * 20
    # Positives in mid-range (5..10), all with score 0.55 (above 0.5)
    truth = [0] * 20
    for j in [5, 6, 7, 8, 9, 10]:
        truth[j] = 1
        scores[j] = 0.55
    fn, pos = policy_equivalent_false_negatives(
        block_ids, scores, truth, hbm_budget=0.30, sink=4, window=4,
    )
    assert pos == 6
    # budget=6 = sink(4) + window(4) overflow → kept set is window
    # blocks (16..19) + sink blocks (0..3) truncated to budget=6.
    # None of blocks 5..10 are in kept → 6 FN.
    assert fn == 6, (
        f"policy-equivalent must miss all 6 positives outside "
        f"sink/window (got fn={fn})"
    )


def test_monotone_clean_sizing_rule_does_not_silently_undersize():
    """Behavioural integration: on a noisy curve, the cleaned ε(φ)
    fed into the sizing rule must be conservative — i.e.\\ the
    returned φ_min must never be smaller than what the same sizing
    rule returns on the explicitly-cleaned curve (defensive against
    a future regression that drops or reverses the projection)."""
    from seer.timing.schedulability import _monotone_clean_eps_curve

    noisy = [(0.1, 0.05), (0.2, 0.40), (0.4, 0.20), (1.0, 0.06)]
    cleaned = _monotone_clean_eps_curve(noisy)

    common_kwargs = dict(
        ell_bar_us=200, B_full=64, deadline_us=50_000,
        sigma_residual_us=1200, miss_target=0.01, base_cost_us=30_000,
    )
    phi_noisy = min_hbm_budget_for_slo_measured(
        eps_curve=noisy, **common_kwargs)
    phi_pre = min_hbm_budget_for_slo_measured(
        eps_curve=cleaned, **common_kwargs)
    # The function must apply its own cleaning, so feeding it the
    # already-cleaned curve must yield the same answer.
    assert phi_noisy == pytest.approx(phi_pre, abs=1e-6), (
        f"sizing rule must be idempotent under monotone cleaning "
        f"(got φ_noisy={phi_noisy}, φ_pre-cleaned={phi_pre})"
    )


# ---------------------------------------------------------------------------
#  Frozen SLOClass dataclass
# ---------------------------------------------------------------------------

def test_sloclass_is_immutable():
    s = SLOClass(name="t", kind="TPOT", percentile=99.0, threshold_ms=50.0)
    with pytest.raises((AttributeError, TypeError)):
        s.threshold_ms = 100.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
#  R17: per-step conditional tail bound (Lemma 2''')
# ---------------------------------------------------------------------------

def test_per_step_conditional_bound_is_tighter_than_union():
    """Per-step conditional bound replaces the B_t * q_base union with
    q_step alone. At calibrated q_base=0.0067 and B_t=32 this is a 32x
    tightening of the heavy-tail contribution."""
    from seer.timing import (
        lemma2_per_step_conditional_bound,
        lemma2_truncated_heavy_tail_bound,
    )
    common = dict(
        epsilon_mean=0.06, epsilon_var=0.01, ell_bar_us=200.0,
        B_t=32.0, deadline_us=50_000.0, sigma_residual_us=1200.0,
        base_cost_us=33_800.0, rho_bar=0.08,
    )
    union = lemma2_truncated_heavy_tail_bound(escape_mass=0.0067, **common)
    per_step = lemma2_per_step_conditional_bound(q_step=0.0067, **common)
    assert per_step <= union + 1e-12
    assert per_step < union


def test_per_step_bound_validates_q_step_range():
    from seer.timing import lemma2_per_step_conditional_bound
    common = dict(
        epsilon_mean=0.06, epsilon_var=0.01, ell_bar_us=200.0,
        B_t=32.0, deadline_us=50_000.0, sigma_residual_us=1200.0,
        base_cost_us=0.0,
    )
    with pytest.raises(ValueError, match="q_step"):
        lemma2_per_step_conditional_bound(q_step=-0.1, **common)
    with pytest.raises(ValueError, match="q_step"):
        lemma2_per_step_conditional_bound(q_step=1.5, **common)


def test_estimate_per_step_burst_rate_matches_calibration_band():
    """Estimator returns per-step burst frequency at mu+3sigma threshold."""
    from seer.timing import estimate_per_step_burst_rate_from_trace

    iofree = [200.0 + 5.0 * ((i % 7) - 3) for i in range(1000)]
    bursts = [1200.0] * 7
    q = estimate_per_step_burst_rate_from_trace(iofree + bursts)
    assert 0.005 <= q <= 0.012


def test_per_step_bound_recovers_union_at_pessimistic_q():
    """If q_step is set to B_t*q_base, per-step recovers union exactly."""
    from seer.timing import (
        lemma2_per_step_conditional_bound,
        lemma2_truncated_heavy_tail_bound,
    )
    q_base = 0.005
    B_t = 32.0
    common = dict(
        epsilon_mean=0.05, epsilon_var=0.008, ell_bar_us=200.0,
        B_t=B_t, deadline_us=50_000.0, sigma_residual_us=1200.0,
        base_cost_us=33_800.0, rho_bar=0.08,
    )
    union = lemma2_truncated_heavy_tail_bound(escape_mass=q_base, **common)
    per_step_pess = lemma2_per_step_conditional_bound(
        q_step=B_t * q_base, **common)
    assert per_step_pess == pytest.approx(union, abs=1e-9)

