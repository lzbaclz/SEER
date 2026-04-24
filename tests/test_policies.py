"""Tests for the KV policies in seer.eval.policies."""
from __future__ import annotations

import numpy as np

from seer.eval.policies import build_policy


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


def test_full_cache_keeps_everything():
    pol = build_policy("full")
    stats = _fake_block_stats(10)
    assert pol.select_to_keep(stats, budget=3, step=0) == set(range(10))


def test_streaming_respects_budget_and_includes_sink():
    pol = build_policy("streaming", sink=2, window=2)
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=5, step=3)
    assert len(out) <= 5
    # Sink present:
    assert any(b in out for b in [0, 1])
    # Window present:
    assert any(b in out for b in [8, 9])


def test_h2o_respects_budget():
    pol = build_policy("h2o", hh_frac=0.5)
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=4, step=5)
    assert len(out) == 4


def test_recency_picks_latest():
    pol = build_policy("recency")
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=3, step=0)
    assert out == {7, 8, 9}  # last 3 by position


def test_random_is_deterministic():
    p1 = build_policy("random", seed=7)
    p2 = build_policy("random", seed=7)
    stats = _fake_block_stats(10)
    assert p1.select_to_keep(stats, 5, 0) == p2.select_to_keep(stats, 5, 0)


def test_snapkv_is_frozen_after_prefill():
    pol = build_policy("snapkv")
    stats = _fake_block_stats(10)
    first = pol.select_to_keep(stats, budget=3, step=0)
    # Mutate one score; SnapKV should not re-rank
    stats[0]["attn_score_now"] = 999.0
    later = pol.select_to_keep(stats, budget=3, step=5)
    assert first.issubset(later)


def test_seer_policy_shape_roundtrip():
    # Stub LAP predictor: return probabilities matching attn_score_now
    def fake_lap(X: np.ndarray) -> np.ndarray:
        # Just use hist_0 (first feature column) as the "probability" for all horizons
        h0 = X[:, 0]
        return np.stack([h0, h0, h0, h0], axis=1)

    pol = build_policy("seer", lap_predictor=fake_lap, sink=1, window=1)
    stats = _fake_block_stats(10)
    out = pol.select_to_keep(stats, budget=5, step=3)
    assert len(out) <= 5
    # Sink (block 0) and last (block 9) must be in kept set
    assert 0 in out
    assert 9 in out
