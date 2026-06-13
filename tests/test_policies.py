"""Tests for the KV policies — both the new seer.policy home and the
deprecated seer.eval.policies shim."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# Canonical (RTSS pivot) imports
from seer.policy import (
    H2OPolicy,
    SafeFallbackPolicy,
    SEERPolicy,
    build_policy,
)


def _fake_block_stats(n_blocks: int = 10) -> dict[int, dict]:
    """Construct a synthetic block_stats dict that all policies can consume."""
    out = {}
    for i in range(n_blocks):
        out[i] = {
            "attn_score_now": float((i + 1) / n_blocks),   # later blocks score higher
            "attn_history": [float((i + 1) / n_blocks)] * 8,
            "position": i * 32,
            "position_norm": i / max(1, n_blocks - 1),
            "last_top_k_step": i,
            "steps_since_top_k": max(0, n_blocks - i - 1),
            "persistence": 0.1,
            "layer_scalar": 0.5,
            "head_scalar": 0.5,
            "io_cost": 0.0,
        }
    return out


# ---------------------------------------------------------------------------
#  Baseline policies
# ---------------------------------------------------------------------------

def test_full_cache_keeps_everything():
    pol = build_policy("full")
    stats = _fake_block_stats(10)
    assert pol.select_to_keep(stats, budget=3, step=0) == set(range(10))


def test_streaming_respects_budget_and_includes_sink():
    pol = build_policy("streaming", sink=2, window=2)
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=5, step=3)
    assert len(out) <= 5
    assert any(b in out for b in [0, 1])
    assert any(b in out for b in [8, 9])


def test_h2o_respects_budget():
    pol = build_policy("h2o", hh_frac=0.5)
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=4, step=5)
    assert len(out) == 4


def test_quest_includes_recent_floor():
    pol = build_policy("quest", recent_floor=2)
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=4, step=3)
    assert {8, 9}.issubset(out)


def test_recency_picks_latest():
    pol = build_policy("recency")
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=3, step=0)
    assert out == {7, 8, 9}


def test_random_is_deterministic():
    p1 = build_policy("random", seed=7)
    p2 = build_policy("random", seed=7)
    stats = _fake_block_stats(10)
    assert p1.select_to_keep(stats, 5, 0) == p2.select_to_keep(stats, 5, 0)


def test_infinigen_keeps_recency_floor_and_proxy_top_k():
    """P1-7: InfiniGen baseline must (a) force-keep the recency floor
    (last ``n_recent`` blocks) and (b) fill the rest with the top
    proxy-scoring blocks."""
    from seer.policy.infinigen import InfiniGenPolicy

    pol = InfiniGenPolicy(n_recent=2, history_k=3)
    stats = _fake_block_stats(10)
    # Make block 1 the hottest in proxy-history but NOT recent.
    stats[1]["attn_history"] = [0.99, 0.99, 0.99]
    # Make block 5 medium-hot.
    stats[5]["attn_history"] = [0.5, 0.5, 0.5]
    out = pol.select_to_keep(stats, budget=4, step=5)
    # Recency floor (n_recent=2) → must include blocks 8, 9
    assert {8, 9}.issubset(out)
    # Top proxy among non-recent → must include block 1
    assert 1 in out, f"InfiniGen must include hot block 1 (got {sorted(out)})"


def test_infinigen_respects_budget():
    """P1-7: InfiniGen must not exceed the active-mask budget."""
    from seer.policy.infinigen import InfiniGenPolicy

    pol = InfiniGenPolicy(n_recent=2, history_k=4)
    stats = _fake_block_stats(20)
    out = pol.select_to_keep(stats, budget=6, step=10)
    assert len(out) == 6


def test_infinigen_built_via_registry():
    """P1-7: ``build_policy('infinigen')`` returns an InfiniGenPolicy."""
    pol = build_policy("infinigen", n_recent=4, history_k=2)
    from seer.policy.infinigen import InfiniGenPolicy
    assert isinstance(pol, InfiniGenPolicy)
    assert pol.n_recent == 4
    assert pol.history_k == 2


def test_runner_exposes_infinigen_in_policy_choices():
    """T2-E (May 2026 reviewer round R1-1.1#4, R2-#8): the runner's
    ``--policy`` argparse choices must include ``infinigen`` so the
    related-work-closest-prior can actually be invoked as a baseline.
    Without this entry the policy registry contains InfiniGenPolicy
    but the CLI rejects the flag, which is the dead-code state both
    reviewers flagged."""
    from seer.eval.runner import _build_argparser
    ap = _build_argparser()
    policy_action = next(a for a in ap._actions if a.dest == "policy")
    assert "infinigen" in policy_action.choices, (
        f"runner --policy choices must include 'infinigen'; got "
        f"{policy_action.choices}"
    )


def test_snapkv_is_frozen_after_prefill():
    pol = build_policy("snapkv")
    stats = _fake_block_stats(10)
    first = pol.select_to_keep(stats, budget=3, step=0)
    stats[0]["attn_score_now"] = 999.0
    later = pol.select_to_keep(stats, budget=3, step=5)
    assert first.issubset(later)


# ---------------------------------------------------------------------------
#  SEER + RTSS additions
# ---------------------------------------------------------------------------

def _fake_lap(X: np.ndarray) -> np.ndarray:
    h0 = X[:, 0]
    return np.stack([h0, h0, h0, h0], axis=1)


def test_seer_policy_shape_roundtrip():
    pol = build_policy("seer", lap_predictor=_fake_lap, sink=1, window=1)
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=5, step=3)
    assert len(out) <= 5
    assert 0 in out
    assert 9 in out


def test_seer_with_slo_uses_pi_controller():
    """When constructed with an SLO, SEERPolicy should set up a PI
    controller that observes step latencies."""
    from seer.timing import parse_slo
    pol = SEERPolicy(
        lap_predictor=_fake_lap,
        sink=1, window=1,
        slo=parse_slo("P99=50ms"),
    )
    assert pol.controller is not None
    initial_lambda = pol.current_lambda()
    # Feed many over-deadline observations: lambda should drop
    for _ in range(200):
        pol.on_step_end(step=1, step_latency_us=200_000)
    assert pol.current_lambda() < initial_lambda


def test_seer_confidence_tracking():
    pol = SEERPolicy(lap_predictor=_fake_lap, sink=1, window=1)
    stats = _fake_block_stats(10)
    pol.select_to_keep(stats, budget=5, step=3)
    # confidence should reflect at least one observation now
    assert 0.0 <= pol.confidence() <= 1.0


def test_safe_fallback_routes_to_fallback_when_low_confidence():
    primary = SEERPolicy(
        lap_predictor=lambda X: np.zeros((len(X), 4)),  # confidence == 0
        sink=1, window=1,
    )
    fallback = H2OPolicy()
    safe = SafeFallbackPolicy(primary=primary, fallback=fallback,
                              conf_threshold=0.5)
    stats = _fake_block_stats(10)
    # First call goes to primary regardless (no signal yet); subsequent
    # calls flip to fallback because primary confidence is 0.
    safe.select_to_keep(stats, budget=4, step=0)  # warmup
    safe.select_to_keep(stats, budget=4, step=1)
    safe.select_to_keep(stats, budget=4, step=2)
    s = safe.stats()
    assert s["fallback_calls"] >= 2


# ---------------------------------------------------------------------------
#  Deprecation shim still works
# ---------------------------------------------------------------------------

def test_legacy_import_path_still_works():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from seer.eval.policies import build_policy as legacy_build  # noqa: PLC0415
    pol = legacy_build("h2o")
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=3, step=0)
    assert len(out) == 3


# ---------------------------------------------------------------------------
#  Regression: B1 (sink/window must count blocks, not token positions)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewer #2): SEERPolicy and the speculative
# warmup wrapper compared ``position`` (= block_id * 32) against the
# sink/window counts, so sink=4 actually kept only the block at
# position 0 instead of the first four blocks. These tests assert the
# corrected block-id semantics and must stay green.

def _const_lap(value: float = 0.5):
    """LAP stub whose horizon scores are a fixed constant so the
    sink/window force-keep decides the kept set entirely."""

    def _call(X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 4), value, dtype=np.float32)

    return _call


def _middle_heavy_lap(n_blocks: int = 20):
    """LAP stub that gives middle blocks high scores and edge blocks
    low scores. Used to verify the sink/window FORCE-keep path: if
    the bug returns (i.e. only block 0 and block n-1 are forced),
    the utility-greedy fill will prefer the middle blocks and the
    edges (sink/window) will be missing — exactly what B1 caused."""

    def _call(X: np.ndarray) -> np.ndarray:
        # X is (n_blocks, 37). Use position_norm (column 33) to derive
        # scores that peak at the middle.
        # The feature vector order is: [32 history] + [recency, persistence,
        # position_norm, layer_scalar, head_scalar].
        pos_norm = X[:, 32 + 2]  # position_norm
        # Triangular peak at 0.5; edges (sink/window) get score ~0.
        scores = 1.0 - 2.0 * np.abs(pos_norm - 0.5)
        return np.stack([scores] * 4, axis=1).astype(np.float32)

    return _call


def test_seer_sink_keeps_first_n_blocks():
    """B1 regression: sink=4 must keep blocks 0..3 even when LAP
    scores prefer the middle. Under the original buggy code only
    block 0 was force-kept; the utility-greedy fill would then pick
    middle blocks and {1, 2, 3} would NOT appear in ``kept``.
    """
    pol = SEERPolicy(lap_predictor=_middle_heavy_lap(20), sink=4, window=4)
    stats = _fake_block_stats(20)
    kept = pol.select_to_keep(stats, budget=12, step=10)
    assert {0, 1, 2, 3}.issubset(kept), (
        f"sink=4 must force-keep blocks 0..3 against a middle-heavy "
        f"LAP (got {sorted(kept)})"
    )


def test_seer_window_keeps_last_n_blocks():
    """B1 regression: window=4 must keep blocks n-4..n-1 against
    a middle-heavy LAP."""
    pol = SEERPolicy(lap_predictor=_middle_heavy_lap(20), sink=4, window=4)
    stats = _fake_block_stats(20)
    kept = pol.select_to_keep(stats, budget=12, step=10)
    assert {16, 17, 18, 19}.issubset(kept), (
        f"window=4 must force-keep blocks 16..19 against a "
        f"middle-heavy LAP (got {sorted(kept)})"
    )


def test_seer_sink_window_combined_exact_budget():
    """B1 regression: with budget == sink + window, the kept set is
    exactly the union of sink and window — no room for utility-driven
    middle blocks. Under the buggy code only {0, n-1} were forced and
    the remaining (budget - 2) slots would be filled by the
    middle-heavy LAP, so the test would fail."""
    pol = SEERPolicy(lap_predictor=_middle_heavy_lap(20), sink=4, window=4)
    stats = _fake_block_stats(20)
    kept = pol.select_to_keep(stats, budget=8, step=10)
    forced = {0, 1, 2, 3, 16, 17, 18, 19}
    assert kept == forced, (
        f"budget=sink+window must yield exactly the forced set "
        f"(expected {sorted(forced)}, got {sorted(kept)})"
    )


def test_seer_sink_window_overlap_when_few_blocks():
    """B1 regression: if sink + window > n_blocks, the union (not the
    sum) is the force-keep set, and budget can fully cover it."""
    pol = SEERPolicy(lap_predictor=_const_lap(0.5), sink=4, window=4)
    stats = _fake_block_stats(6)  # 6 blocks, sink=4 + window=4 overlap
    kept = pol.select_to_keep(stats, budget=6, step=3)
    # sorted_bids = [0,1,2,3,4,5]; sink first-4 = {0,1,2,3};
    # window last-4 = {2,3,4,5}; union = {0,1,2,3,4,5}
    assert kept == set(range(6))


def test_seer_budget_smaller_than_sink_plus_window_prefers_window():
    """B1 regression: when budget < sink + window, the kept set
    prefers window over sink (recency dominates next-token attention).
    """
    pol = SEERPolicy(lap_predictor=_const_lap(0.5), sink=3, window=3)
    stats = _fake_block_stats(20)
    # budget=4 < 3 + 3 = 6 → window-then-sink ordering selects 19,18,17,0
    kept = pol.select_to_keep(stats, budget=4, step=5)
    assert len(kept) == 4
    assert {17, 18, 19}.issubset(kept), (
        f"window blocks must be preserved first (got {sorted(kept)})"
    )


def test_speculative_warmup_sink_window_block_count():
    """B1 regression: SpeculativeWarmupPolicy._speculative_select had
    the same buggy sink/window logic. After the fix, warmup-mode
    selection must also force-keep the right blocks."""
    from seer.policy.speculative_warmup import SpeculativeWarmupPolicy

    primary = SEERPolicy(lap_predictor=_const_lap(0.5), sink=4, window=4)
    wrapper = SpeculativeWarmupPolicy(
        primary=primary,
        trigger_threshold=10.0,  # always trigger
        trigger_miss_rate=-1.0,
        cooldown_steps=0,
    )
    stats = _fake_block_stats(20)
    # First call triggers warmup mode (confidence < threshold).
    kept = wrapper.select_to_keep(stats, budget=12, step=1)
    forced = {0, 1, 2, 3, 16, 17, 18, 19}
    assert forced.issubset(kept), (
        f"warmup-mode sink+window must keep forced blocks "
        f"(missing {sorted(forced - kept)})"
    )


# ---------------------------------------------------------------------------
#  Regression: B2 (speculative warmup must enlarge the kept set)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewer #2 + #3): the warmup wrapper
# charged a fixed ``warmup_size * ell_bar / period`` overhead but the
# kept set returned to the simulator was still only ``budget`` blocks,
# so the speculative prefetch had no effect on the next step's attention
# misses. These tests assert that warmup steps yield a kept set of size
# ``budget + warmup_size`` and non-warmup steps stay at ``budget``.

def test_speculative_warmup_enlarges_kept_set_on_warmup_step():
    """B2 regression: warmup step must return budget + warmup_size
    blocks (the whole point of speculative pre-warm)."""
    from seer.policy.speculative_warmup import SpeculativeWarmupPolicy

    primary = SEERPolicy(lap_predictor=_const_lap(0.5), sink=4, window=4)
    wrapper = SpeculativeWarmupPolicy(
        primary=primary,
        trigger_threshold=10.0,
        trigger_miss_rate=-1.0,
        warmup_size=12,
        cooldown_steps=0,
    )
    # n=60 ≫ budget+warmup_size=22, so the enlargement is not clipped.
    stats = _fake_block_stats(60)
    kept = wrapper.select_to_keep(stats, budget=10, step=1)
    assert len(kept) == 10 + 12, (
        f"warmup must enlarge kept set to budget+warmup_size=22, "
        f"got |kept|={len(kept)}"
    )


def test_speculative_warmup_non_warmup_step_respects_budget():
    """B2 regression: non-warmup steps must still respect ``budget``;
    the wrapper must not silently leak the enlarged set into normal
    operation."""
    from seer.policy.speculative_warmup import SpeculativeWarmupPolicy

    primary = SEERPolicy(lap_predictor=_const_lap(0.5), sink=4, window=4)
    wrapper = SpeculativeWarmupPolicy(
        primary=primary,
        trigger_threshold=-1.0,  # never trigger (confidence >= -1)
        trigger_miss_rate=10.0,  # never trigger
        warmup_size=12,
        cooldown_steps=0,
    )
    stats = _fake_block_stats(60)
    kept = wrapper.select_to_keep(stats, budget=10, step=1)
    assert len(kept) == 10, (
        f"non-warmup step must respect budget=10, got |kept|={len(kept)}"
    )


def test_speculative_warmup_clipped_to_total_blocks():
    """B2 regression: when budget + warmup_size > n_total, the kept
    set is clipped to n_total (cannot fabricate non-existent blocks)."""
    from seer.policy.speculative_warmup import SpeculativeWarmupPolicy

    primary = SEERPolicy(lap_predictor=_const_lap(0.5), sink=4, window=4)
    wrapper = SpeculativeWarmupPolicy(
        primary=primary,
        trigger_threshold=10.0,
        trigger_miss_rate=-1.0,
        warmup_size=50,
        cooldown_steps=0,
    )
    stats = _fake_block_stats(20)  # n=20 < budget+warmup_size=60
    kept = wrapper.select_to_keep(stats, budget=10, step=1)
    assert len(kept) == 20, (
        f"clipped warmup must return all {20} available blocks, "
        f"got |kept|={len(kept)}"
    )


def test_speculative_warmup_overhead_is_charged():
    """B2 regression: even though the kept set is now enlarged, the
    documented per-step warmup_overhead_us() must still report the
    prefetch IO cost so the simulator can account for it."""
    from seer.policy.speculative_warmup import SpeculativeWarmupPolicy

    primary = SEERPolicy(lap_predictor=_const_lap(0.5), sink=4, window=4)
    wrapper = SpeculativeWarmupPolicy(
        primary=primary,
        trigger_threshold=10.0,
        trigger_miss_rate=-1.0,
        warmup_size=12,
        cooldown_steps=0,
    )
    stats = _fake_block_stats(60)
    wrapper.select_to_keep(stats, budget=10, step=1)
    # warmup_size=12, ell_bar=200µs, decision_period=8 → 300 µs charge
    assert wrapper.warmup_overhead_us() == pytest.approx(300.0, abs=1e-6)
