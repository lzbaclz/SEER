"""Baseline KV-cache eviction policies — consolidated for the RTSS pivot.

Each class here mirrors the published method as closely as a *policy-only*
implementation can. The simulator (:mod:`seer.eval.sim`) drives them with
a uniform ``block_stats`` dict, so swapping baselines is a one-line change.

Implementations
---------------
* :class:`StreamingPolicy` — Xiao et al. ICLR 2024
* :class:`H2OPolicy` — Zhang et al. NeurIPS 2023
* :class:`SnapKVPolicy` — Li et al. NeurIPS 2024
* :class:`QuestPolicy` — Tang et al. ICML 2024 (page-level top-k)
* :class:`RecencyPolicy` — sanity baseline
* :class:`RandomPolicy` — sanity lower bound

These were six separate files in the NeurIPS-era layout
(``seer/eval/policies/{streaming,h2o,snapkv,recency,random_policy}.py``);
they are merged here because they share so much surface that splitting
adds no organizational value.
"""
from __future__ import annotations

import random

from seer.policy.base import KVPolicy

# ---------------------------------------------------------------------------
#  StreamingLLM
# ---------------------------------------------------------------------------

class StreamingPolicy(KVPolicy):
    """Attention sink + sliding window. Position heuristic, no attention."""
    name = "streaming"

    def __init__(self, sink: int = 4, window: int = 64):
        self.sink = sink
        self.window = window

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        ids_sorted = sorted(block_stats.keys())
        keep: set[int] = set(ids_sorted[: self.sink])
        keep.update(ids_sorted[-self.window:])
        if len(keep) > budget:
            order = ids_sorted[-self.window:][::-1] + ids_sorted[: self.sink]
            keep = set(order[:budget])
        return keep


# ---------------------------------------------------------------------------
#  H2O
# ---------------------------------------------------------------------------

class H2OPolicy(KVPolicy):
    """Heavy hitters (cumulative attention) + recent."""
    name = "h2o"

    def __init__(self, hh_frac: float = 0.5):
        assert 0.0 <= hh_frac <= 1.0
        self.hh_frac = hh_frac

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        n_hh = int(round(budget * self.hh_frac))

        def _cumulative(stats: dict) -> float:
            hist = stats.get("attn_history")
            if hist:
                return float(sum(hist))
            return float(stats.get("attn_score_now", 0.0))

        ranked_hh = sorted(block_stats.items(), key=lambda kv: -_cumulative(kv[1]))
        keep: set[int] = {bid for bid, _ in ranked_hh[:n_hh]}
        ranked_recent = sorted(
            block_stats.items(), key=lambda kv: -kv[1].get("position", 0)
        )
        for bid, _ in ranked_recent:
            if len(keep) >= budget:
                break
            keep.add(bid)
        return keep


class H2OCumulativePolicy(KVPolicy):
    """H2O parity variant (R7-US-W5): cumulative attention from step 0.

    Upstream FMInference/H2O accumulates per-block attention from the very
    first step of the request, whereas :class:`H2OPolicy` sums over the
    rolling-window history exposed in `attn_history`. This class maintains
    an internal running sum keyed by block id so its `n` block ranking
    matches upstream H2O within a per-token-vs-per-block constant. The
    hh_frac / recent split mirrors :class:`H2OPolicy`.

    The implementation is intentionally minimal: kept blocks accumulate
    `attn_score_now`, evicted blocks are zeroed on rebuild. This is the
    in-session bracket for the parity gap flagged in
    `docs/baseline_parity_audit.md` and is used by §6's reactive-lag
    H2O-parity row.
    """
    name = "h2o_cumulative"

    def __init__(self, hh_frac: float = 0.5):
        assert 0.0 <= hh_frac <= 1.0
        self.hh_frac = hh_frac
        self._running: dict[int, float] = {}

    def reset(self):
        self._running.clear()

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        # Accumulate cumulative attention per block id.
        for bid, st in block_stats.items():
            attn_now = float(st.get("attn_score_now", 0.0))
            self._running[bid] = self._running.get(bid, 0.0) + attn_now

        n_hh = int(round(budget * self.hh_frac))
        ranked_hh = sorted(block_stats.items(),
                           key=lambda kv: -self._running.get(kv[0], 0.0))
        keep: set[int] = {bid for bid, _ in ranked_hh[:n_hh]}
        ranked_recent = sorted(
            block_stats.items(), key=lambda kv: -kv[1].get("position", 0)
        )
        for bid, _ in ranked_recent:
            if len(keep) >= budget:
                break
            keep.add(bid)
        # Zero out cumulative state for blocks that fell out (mirrors
        # upstream behaviour: an evicted block restarts at 0 if reloaded).
        for bid in list(self._running.keys()):
            if bid not in block_stats:
                del self._running[bid]
        return keep


# ---------------------------------------------------------------------------
#  SnapKV
# ---------------------------------------------------------------------------

class SnapKVPolicy(KVPolicy):
    """Prefill-time compression frozen through decode."""
    name = "snapkv"

    def __init__(self) -> None:
        self._selection: set[int] | None = None

    def reset(self) -> None:
        self._selection = None

    def select_to_keep(self, block_stats, budget, step):
        if self._selection is None or step == 0:
            ranked = sorted(
                block_stats.items(),
                key=lambda kv: -kv[1].get("attn_score_now", 0.0),
            )
            self._selection = {bid for bid, _ in ranked[:budget]}
        # Auto-include newly generated blocks (max id) up to budget.
        if self._selection is not None:
            current_max = max(self._selection) if self._selection else -1
            for bid in block_stats.keys():
                if bid not in self._selection and len(self._selection) < budget and bid > current_max:
                    self._selection.add(bid)
        return self._selection or set()


class SnapKVUpstreamPolicy(KVPolicy):
    """Upstream FasterDecoding/SnapKV parity variant (R36d).

    Compared to the in-tree :class:`SnapKVPolicy` (prefill-frozen
    selection), the upstream SnapKV (Li et al. NeurIPS 2024)
    maintains a sliding ``observation window`` of the most-recent
    `obs_window` tokens (default 32 in the upstream codebase) that
    re-scores the prefill top-K against current attention via an
    average-pool aggregation. The selection is therefore not frozen
    at prefill but adapts slowly during decode.

    Parity audit per ``docs/baseline_parity_audit.md`` (SnapKV
    section): in-tree freezes; upstream re-scores. This class is the
    in-session bracket gated by ``SEER_SNAPKV_VARIANT=upstream``.

    Note: the upstream's pooling is over attention heads at the
    token level; our `block_stats` is already head-aggregated by the
    LAP feature extractor so we re-score per-block via
    ``attn_score_now`` averaged over the obs-window of step-relative
    scores. The result is upstream-faithful on the policy-decision
    metric (selection ordering) within the within-block
    head-aggregation constant.
    """
    name = "snapkv_upstream"

    def __init__(self, obs_window: int = 32) -> None:
        self.obs_window = obs_window
        self._selection: set[int] | None = None
        self._history: dict[int, list[float]] = {}

    def reset(self) -> None:
        self._selection = None
        self._history.clear()

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        # Accumulate a sliding window of `attn_score_now` per block.
        for bid, st in block_stats.items():
            buf = self._history.setdefault(bid, [])
            buf.append(float(st.get("attn_score_now", 0.0)))
            if len(buf) > self.obs_window:
                del buf[:-self.obs_window]
        # Avg-pool over the obs-window (upstream's average-pool).
        scored = {
            bid: (sum(self._history[bid]) / max(1, len(self._history[bid])))
            for bid in block_stats.keys()
        }
        ranked = sorted(block_stats.items(),
                        key=lambda kv: -scored[kv[0]])
        self._selection = {bid for bid, _ in ranked[:budget]}
        # GC dropped blocks.
        for bid in list(self._history.keys()):
            if bid not in block_stats:
                del self._history[bid]
        return self._selection


# ---------------------------------------------------------------------------
#  Quest
# ---------------------------------------------------------------------------

class QuestPolicy(KVPolicy):
    """Page-level top-k attention estimate (Tang et al., ICML 2024).

    Quest groups KV tokens into pages and uses the *max* attention within
    a page as the page's score. We simulate the per-page score with the
    block's own score (one block ≈ one page in our 32-token blocking).
    """
    name = "quest"

    def __init__(self, recent_floor: int = 4):
        self.recent_floor = recent_floor

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        # Force-keep newest `recent_floor` blocks
        ids_sorted = sorted(block_stats.keys())
        keep: set[int] = set(ids_sorted[-self.recent_floor:])
        # Fill budget by current attention score (page-level top-k proxy)
        ranked = sorted(
            block_stats.items(),
            key=lambda kv: -kv[1].get("attn_score_now", 0.0),
        )
        for bid, _ in ranked:
            if len(keep) >= budget:
                break
            keep.add(bid)
        return keep


class QuestUpstreamPolicy(KVPolicy):
    """Upstream Quest parity variant (R36d).

    Compared to in-tree :class:`QuestPolicy` (one-block-per-page at
    32-token blocks), the upstream Quest (Tang et al. ICML 2024) uses
    a smaller 16-token page size with per-page max-attention as the
    score. We bracket that by sub-dividing each 32-token block into
    two synthetic pages and taking the max of the two halves' scores
    when stat fields are available; otherwise we fall back to the
    block's own score (consistent with the in-tree variant).

    The recent_floor upstream default is $8$ (we used $4$). This
    class accepts both via `recent_floor=8`.

    Parity audit per ``docs/baseline_parity_audit.md`` (Quest
    section): in-tree is $2\\times$ coarser at the same budget; the
    upstream variant has slightly higher recall on the heavy-hitter
    page in long-decode workloads. The ordering vs other
    attention-tracking baselines is preserved.
    """
    name = "quest_upstream"

    def __init__(self, recent_floor: int = 8) -> None:
        self.recent_floor = recent_floor

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        ids_sorted = sorted(block_stats.keys())
        # recent_floor=0 must mean no force-keep (not "keep everything"
        # which is the Python `lst[-0:]` slicing surprise).
        if self.recent_floor > 0:
            keep: set[int] = set(ids_sorted[-self.recent_floor:])
        else:
            keep = set()
        # Page-level max-attention proxy: if `attn_score_now_max` or
        # per-half page scores are exposed, use them; otherwise the
        # block's own score (degenerate equivalent of in-tree Quest).
        def page_max(st: dict) -> float:
            if "attn_score_now_max" in st:
                return float(st["attn_score_now_max"])
            halves = st.get("attn_score_now_halves")
            if isinstance(halves, (list, tuple)) and halves:
                return float(max(halves))
            return float(st.get("attn_score_now", 0.0))

        ranked = sorted(
            block_stats.items(), key=lambda kv: -page_max(kv[1])
        )
        for bid, _ in ranked:
            if len(keep) >= budget:
                break
            keep.add(bid)
        return keep


# ---------------------------------------------------------------------------
#  Recency / Random — sanity baselines
# ---------------------------------------------------------------------------

class RecencyPolicy(KVPolicy):
    """Keep the N most recent blocks (LRU by position)."""
    name = "recency"

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        ranked = sorted(block_stats.items(), key=lambda kv: -kv[1].get("position", 0))
        return {bid for bid, _ in ranked[:budget]}


class RandomPolicy(KVPolicy):
    """Random eviction — sanity lower bound."""
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select_to_keep(self, block_stats, budget, step):
        keys = list(block_stats.keys())
        self.rng.shuffle(keys)
        return set(keys[:budget])
