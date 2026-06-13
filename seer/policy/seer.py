"""SEER joint policy with adaptive λ and schedulability awareness.

This is the main contribution of the paper. The decision algorithm at
each policy invocation:

  1. Build the LAP feature vector for every candidate block.
  2. Score blocks: ``p̂ = LAP(features)`` for the configured horizon.
  3. Compute utility ``U_i = p̂_i − λ_t · IO_cost(tier_i)``.
     ``λ_t`` comes from a PI controller that tracks observed P99 TPOT
     against the SLO target (see :class:`seer.timing.LambdaController`).
  4. Force-keep the sink (first few blocks) and the sliding window
     (latest few blocks) as a deadline-safety floor.
  5. Greedy-fill the remaining budget by ``U_i`` descending.

The PI controller is *optional* — pass ``slo=None`` to fall back to a
fixed ``lam_io`` and recover the NeurIPS-era behavior.

Confidence statistics (mean p̂_max over the last ``conf_window`` decisions)
are exposed via :meth:`confidence` so :class:`seer.policy.fallback.SafeFallbackPolicy`
can wrap us with a hard "below 0.4 → switch to H2O" cutout.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

from seer.lap.features import HISTORY_N
from seer.policy.base import KVPolicy
from seer.timing.slo import LambdaController, SLOClass, derive_deadline_us


class SEERPolicy(KVPolicy):
    """Schedulability-aware learned eviction-prefetch policy."""

    name = "seer"

    def __init__(
        self,
        lap_predictor: Callable[[np.ndarray], np.ndarray],
        sink: int = 4,
        window: int = 4,
        horizon_idx: int = 1,  # 0->h1, 1->h4, 2->h16, 3->h64
        lam_io: float = 0.5,
        history_n: int = HISTORY_N,
        slo: SLOClass | None = None,
        controller: LambdaController | None = None,
        conf_window: int = 100,
    ):
        self.lap = lap_predictor
        self.sink = sink
        self.window = window
        self.horizon_idx = horizon_idx
        self.history_n = history_n
        self.slo = slo
        self.lam_io_fixed = float(lam_io)

        if controller is not None:
            self.controller = controller
        elif slo is not None:
            target_us = derive_deadline_us(slo) * 0.95   # PI tracks 95% of deadline
            self.controller = LambdaController(target_us=target_us, initial=lam_io)
        else:
            self.controller = None  # use fixed lam_io

        self._conf_window = deque(maxlen=conf_window)

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if self.controller is not None:
            self.controller.reset()
        self._conf_window.clear()

    def on_step_end(self, step: int, step_latency_us: float | None = None) -> None:
        if self.controller is not None and step_latency_us is not None:
            self.controller.observe(step_latency_us)

    # ------------------------------------------------------------------
    #  Selection
    # ------------------------------------------------------------------

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()

        bids = list(block_stats.keys())
        X = np.stack(
            [self._features(block_stats[b]) for b in bids],
            axis=0,
        ).astype(np.float32)
        probs = self.lap(X)
        if probs.ndim == 1:
            probs = probs[:, None]
        scores = probs[:, min(self.horizon_idx, probs.shape[1] - 1)]

        # update confidence rolling mean
        self._conf_window.append(float(scores.max()) if scores.size > 0 else 0.0)

        io_costs = np.array(
            [block_stats[b].get("io_cost", 0.0) for b in bids],
            dtype=np.float32,
        )
        lam = (
            float(self.controller.current_lambda())
            if self.controller is not None
            else self.lam_io_fixed
        )
        utility = scores - lam * io_costs

        # ---- Force-keep sink + sliding window
        # B1 fix (reviewer round May 2026): ``sink`` / ``window`` are
        # counts of *blocks*, not token positions. The previous code
        # compared ``position`` (= block_id * BLOCK_SIZE, typically 32)
        # against the sink count (e.g. 4), so sink=4 only ever kept the
        # block at position 0. We now sort by block_id and take the
        # first ``sink`` and last ``window`` entries — mirrors
        # :class:`seer.policy.baselines.StreamingPolicy`.
        sorted_bids = sorted(bids)
        forced: set[int] = set(sorted_bids[: self.sink])
        forced.update(sorted_bids[-self.window :])

        remaining = budget - len(forced)
        if remaining <= 0:
            # Budget cannot hold both sink and window: prefer the
            # sliding window (most-recent blocks dominate next-token
            # attention), then backfill with sink. Deterministic order
            # matches StreamingPolicy's fallback path.
            window_then_sink = (
                sorted_bids[-self.window :][::-1] + sorted_bids[: self.sink]
            )
            seen: set[int] = set()
            ordered: list[int] = []
            for b in window_then_sink:
                if b not in seen:
                    seen.add(b)
                    ordered.append(b)
            return set(ordered[:budget])

        cand = [(b, float(u)) for b, u in zip(bids, utility, strict=False) if b not in forced]
        cand.sort(key=lambda kv: -kv[1])
        picked = {b for b, _ in cand[:remaining]}
        return forced | picked

    # ------------------------------------------------------------------
    #  Confidence (used by SafeFallbackPolicy)
    # ------------------------------------------------------------------

    def confidence(self) -> float:
        """Rolling mean of LAP's max prob over the last `conf_window`
        decisions. ~1.0 = LAP is confident; ~0.5 = uncertain.
        """
        if not self._conf_window:
            return 1.0
        return float(sum(self._conf_window) / len(self._conf_window))

    def current_lambda(self) -> float:
        if self.controller is not None:
            return float(self.controller.current_lambda())
        return self.lam_io_fixed

    # ------------------------------------------------------------------
    #  Feature packing (mirrors seer.lap.features column order)
    # ------------------------------------------------------------------

    def _features(self, stats: dict) -> np.ndarray:
        history = list(stats.get("attn_history", []))[-self.history_n:]
        hist_arr = np.zeros(self.history_n, dtype=np.float32)
        if history:
            rev = list(reversed(history))
            for i, v in enumerate(rev[: self.history_n]):
                hist_arr[i] = float(v)

        recency = stats.get("steps_since_top_k", self.history_n + 1)
        recency_log = float(np.log1p(min(recency, self.history_n + 1)))
        persistence = float(stats.get("persistence", 0.0))
        position_norm = float(stats.get("position_norm", 0.0))
        layer_scalar = float(stats.get("layer_scalar", 0.0))
        head_scalar = float(stats.get("head_scalar", 0.0))

        return np.concatenate([
            hist_arr,
            np.array(
                [recency_log, persistence, position_norm, layer_scalar, head_scalar],
                dtype=np.float32,
            ),
        ])
