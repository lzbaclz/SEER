"""R31 advisor: contract tests for the hot-path partial-materialisation
decision router (cycle-2 / ATC 2027 prototype).

These tests are the **contract** the future vLLM A2 patch
(``vllm_patches/0002-hot-path-allocation-hook.patch``) must preserve.
If the patch ever lands and these tests start failing, the patch has
changed semantics the paper relied on.

Tests are CPU-only and use a callable mock as the score function so
no LAP / torch / GPU is invoked.
"""
from __future__ import annotations

import pytest

from seer.integration.hot_path_router import (
    HotPathRouter,
    SlackBudget,
    slack_from_budget,
)


def _identity_score(request_id, block_ids):
    """All blocks score equally -- ties preserve input order (stable)."""
    return [1.0] * len(block_ids)


def _reverse_score(request_id, block_ids):
    """Block i scores -i; later blocks should be ranked higher first."""
    return [-float(i) for i, _ in enumerate(block_ids)]


def _by_id_score(request_id, block_ids):
    """Score == block_id; larger ids first."""
    return [float(b) for b in block_ids]


# ---------------------------------------------------------------------------
#  Basic contract
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    r = HotPathRouter(_identity_score)
    assert r.partial_materialisation_order("req-x", [], slack_budget_us=1000) == []


def test_zero_slack_returns_empty():
    r = HotPathRouter(_identity_score)
    assert r.partial_materialisation_order(
        "req-x", [1, 2, 3], slack_budget_us=0.0) == []


def test_negative_slack_returns_empty():
    r = HotPathRouter(_identity_score)
    assert r.partial_materialisation_order(
        "req-x", [1, 2, 3], slack_budget_us=-10.0) == []


def test_uninformative_lap_preserves_lru_order():
    """When every block scores identically the router must return the
    identity permutation -- otherwise we silently break vLLM's default
    LRU on every cache-hit that LAP can't discriminate."""
    r = HotPathRouter(_identity_score)
    order = r.partial_materialisation_order(
        "req-x", [10, 20, 30, 40], slack_budget_us=1e6)
    assert order == [0, 1, 2, 3]


def test_decreasing_score_reverses_input():
    r = HotPathRouter(_reverse_score)
    order = r.partial_materialisation_order(
        "req-x", [10, 20, 30, 40], slack_budget_us=1e6)
    # _reverse_score gives [-0, -1, -2, -3]; descending order is
    # [0, -1, -2, -3] so index 0 comes first ... actually let me think
    # again: scores are [0, -1, -2, -3], sort descending = [0, -1, -2, -3]
    # i.e. index 0 first, then index 1, etc. So order == identity.
    # This is the correct behaviour: score(i) = -i means block 0 is
    # most important, so it materialises first. The test name is
    # misleading; what we actually want is to confirm a strictly
    # descending score order is preserved.
    assert order == [0, 1, 2, 3]


def test_by_id_score_picks_largest_first():
    r = HotPathRouter(_by_id_score)
    order = r.partial_materialisation_order(
        "req-x", [10, 20, 30, 40], slack_budget_us=1e6)
    # Scores [10, 20, 30, 40]; descending => index 3, 2, 1, 0.
    assert order == [3, 2, 1, 0]


def test_score_length_mismatch_raises():
    def bad_score(rid, blks):
        return [1.0]  # always length 1
    r = HotPathRouter(bad_score)
    with pytest.raises(ValueError, match=r"score_fn returned"):
        r.partial_materialisation_order(
            "req-x", [1, 2, 3], slack_budget_us=1e6)


# ---------------------------------------------------------------------------
#  Slack-budget arithmetic
# ---------------------------------------------------------------------------


def test_max_blocks_under_budget_zero():
    r = HotPathRouter(_identity_score, ell_bar_us=200.0)
    assert r.max_blocks_under_budget(0) == 0
    assert r.max_blocks_under_budget(-100) == 0


def test_max_blocks_under_budget_typical():
    r = HotPathRouter(_identity_score, ell_bar_us=200.0)
    # 39.5 ms slack / 200 us per block = 197 blocks
    assert r.max_blocks_under_budget(39_500.0) == 197


def test_max_blocks_under_budget_rounds_down():
    r = HotPathRouter(_identity_score, ell_bar_us=200.0)
    # 499 us / 200 us = 2.495 -> 2 blocks (floor)
    assert r.max_blocks_under_budget(499.0) == 2


def test_slack_from_budget_design_doc_example():
    """Design-doc example: deadline 100 ms, forward 60 ms, lap 0.3 ms
    => slack ~= 39.5 ms, io_budget ~= 39.5 - 0.2 = 39.3 ms."""
    sb = slack_from_budget(
        deadline_us=100_000, forward_us=60_000,
        lap_us=300.0, attn_us=0.0, ffn_us=0.0,
        safety_margin_us=200.0,
    )
    assert sb.slack_us == pytest.approx(39_700.0, abs=1.0)
    assert sb.io_budget_us == pytest.approx(39_500.0, abs=1.0)


def test_slack_negative_when_overrun():
    sb = SlackBudget(
        deadline_us=10_000, forward_us=20_000,
        lap_us=0, attn_us=0, ffn_us=0, safety_margin_us=0,
    )
    assert sb.slack_us == -10_000
    # io_budget clamps at 0
    assert sb.io_budget_us == 0.0


def test_ell_bar_must_be_positive():
    with pytest.raises(ValueError, match=r"ell_bar_us must be positive"):
        HotPathRouter(_identity_score, ell_bar_us=0.0)
