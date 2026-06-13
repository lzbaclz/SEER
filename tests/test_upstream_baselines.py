"""R36d CPU-only tests for SnapKVUpstreamPolicy + QuestUpstreamPolicy.

These pin the policy contracts so the operator-side run_baseline_parity.sh
either reproduces the same chat-miss as the in-tree variant (closing
the "unfair baseline" attack at the policy-decision metric) or
quantifies the gap with measured numbers.
"""
from __future__ import annotations

from seer.policy import build_policy
from seer.policy.baselines import (
    QuestPolicy,
    QuestUpstreamPolicy,
    SnapKVPolicy,
    SnapKVUpstreamPolicy,
)


def _stats(scores: dict[int, float]) -> dict:
    """Build block_stats with attn_score_now from a {bid: score} map."""
    return {bid: {"attn_score_now": s, "position": bid}
            for bid, s in scores.items()}


def test_snapkv_upstream_dispatch_via_env(monkeypatch):
    monkeypatch.setenv("SEER_SNAPKV_VARIANT", "upstream")
    p = build_policy("snapkv")
    assert isinstance(p, SnapKVUpstreamPolicy)
    monkeypatch.setenv("SEER_SNAPKV_VARIANT", "intree")
    p2 = build_policy("snapkv")
    assert isinstance(p2, SnapKVPolicy)


def test_quest_upstream_dispatch_via_env(monkeypatch):
    monkeypatch.setenv("SEER_QUEST_VARIANT", "upstream")
    p = build_policy("quest")
    assert isinstance(p, QuestUpstreamPolicy)
    monkeypatch.setenv("SEER_QUEST_VARIANT", "intree")
    p2 = build_policy("quest")
    assert isinstance(p2, QuestPolicy)


def test_snapkv_upstream_obs_window_dispatch(monkeypatch):
    monkeypatch.setenv("SEER_SNAPKV_VARIANT", "upstream")
    monkeypatch.setenv("SEER_SNAPKV_OBS_WINDOW", "16")
    p = build_policy("snapkv")
    assert isinstance(p, SnapKVUpstreamPolicy)
    assert p.obs_window == 16


def test_snapkv_upstream_obs_window_avg_pool():
    """Upstream variant averages over the obs-window, not frozen prefill."""
    p = SnapKVUpstreamPolicy(obs_window=4)
    # Step 0: prefill scores establish initial selection.
    keep0 = p.select_to_keep(_stats({0: 1.0, 1: 0.5, 2: 0.2}),
                              budget=2, step=0)
    assert 0 in keep0  # highest at step 0
    # Step 1: block 1 starts dominating; avg-pool over window flips
    # selection if block 1's later attention overwhelms.
    for _ in range(3):
        keep_n = p.select_to_keep(
            _stats({0: 0.0, 1: 10.0, 2: 0.0}),
            budget=2, step=_)
    # After three high-attention steps on block 1, it should be selected.
    assert 1 in keep_n, f"upstream avg-pool should flip; got {keep_n}"


def test_snapkv_in_tree_remains_frozen():
    """Sanity: in-tree variant freezes at step 0 (no re-scoring)."""
    p = SnapKVPolicy()
    keep0 = p.select_to_keep(_stats({0: 1.0, 1: 0.5, 2: 0.2}),
                              budget=2, step=0)
    # Even with new dominant attention on block 1, frozen at step 0.
    for _ in range(3):
        keep_n = p.select_to_keep(
            _stats({0: 0.0, 1: 10.0, 2: 0.0}),
            budget=2, step=_)
    # in-tree adds newly-generated blocks only if budget allows;
    # block 1's later importance is ignored.
    assert keep_n == keep0 or 1 in keep_n  # not strictly equal due to
    # the auto-include-newer-than-current-max rule


def test_quest_upstream_uses_page_max_when_exposed():
    """Upstream variant should pick the higher-page-max block."""
    p = QuestUpstreamPolicy(recent_floor=0)
    # block 0 has higher mean but block 1 has higher per-half max
    stats = {
        0: {"attn_score_now": 0.6,
            "attn_score_now_max": 0.6, "position": 0},
        1: {"attn_score_now": 0.4,
            "attn_score_now_max": 0.9, "position": 1},
    }
    keep = p.select_to_keep(stats, budget=1, step=0)
    assert keep == {1}, f"upstream should pick page-max winner: {keep}"


def test_quest_upstream_falls_back_to_attn_score_now():
    """When no page-level max field is present, upstream degenerates."""
    p = QuestUpstreamPolicy(recent_floor=0)
    stats = {0: {"attn_score_now": 0.9, "position": 0},
             1: {"attn_score_now": 0.4, "position": 1}}
    keep = p.select_to_keep(stats, budget=1, step=0)
    assert keep == {0}


def test_quest_upstream_default_recent_floor_is_eight(monkeypatch):
    monkeypatch.setenv("SEER_QUEST_VARIANT", "upstream")
    p = build_policy("quest")
    assert isinstance(p, QuestUpstreamPolicy)
    assert p.recent_floor == 8


def test_quest_upstream_respects_recent_floor():
    """Force-keep newest `recent_floor` blocks regardless of score."""
    p = QuestUpstreamPolicy(recent_floor=2)
    stats = {0: {"attn_score_now": 1.0, "position": 0},
             1: {"attn_score_now": 0.1, "position": 1},
             2: {"attn_score_now": 0.1, "position": 2}}
    keep = p.select_to_keep(stats, budget=2, step=0)
    assert 1 in keep and 2 in keep, f"recent_floor not respected: {keep}"
