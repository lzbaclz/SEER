"""SEER: the proposed learned policy.

At each decision point we:
  1. Build the LAP feature vector for every candidate block.
  2. Score blocks with the LAP (per-block future-top-k probability).
  3. Optionally subtract an IO-cost term  λ · io_cost(tier).
  4. Force-keep sink + sliding window.
  5. Fill remaining budget with greedy top-utility selection.

The LAP predictor is passed in as any callable
    P : np.ndarray[N, D] -> np.ndarray[N, n_horizons]
(see `seer.lap.infer.LAPPredictor`). This lets us swap between ONNX and
torch backends transparently.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from seer.eval.policies.base import KVPolicy
from seer.lap.features import HISTORY_N


class SEERPolicy(KVPolicy):
    name = "seer"

    def __init__(
        self,
        lap_predictor: Callable[[np.ndarray], np.ndarray],
        sink: int = 4,
        window: int = 4,
        horizon_idx: int = 1,  # 0->h1, 1->h4, 2->h16, 3->h64
        lam_io: float = 0.0,
        history_n: int = HISTORY_N,
    ):
        self.lap = lap_predictor
        self.sink = sink
        self.window = window
        self.horizon_idx = horizon_idx
        self.lam_io = lam_io
        self.history_n = history_n

    # ------------------------------------------------------------------
    #  Selection
    # ------------------------------------------------------------------

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()

        bids = list(block_stats.keys())
        X = np.stack([self._features(block_stats[b]) for b in bids], axis=0).astype(np.float32)
        probs = self.lap(X)  # [N, n_horizons]
        if probs.ndim == 1:
            probs = probs[:, None]
        scores = probs[:, min(self.horizon_idx, probs.shape[1] - 1)]

        io_costs = np.array(
            [block_stats[b].get("io_cost", 0.0) for b in bids],
            dtype=np.float32,
        )
        utility = scores - self.lam_io * io_costs

        # ---- Force-keep sink + sliding window
        max_pos = max(block_stats[b].get("position", 0) for b in bids)
        min_pos = min(block_stats[b].get("position", 0) for b in bids)
        forced: set[int] = set()
        for b in bids:
            pos = block_stats[b].get("position", 0)
            if pos <= min_pos + self.sink:
                forced.add(b)
            if pos >= max_pos - self.window:
                forced.add(b)

        remaining = budget - len(forced)
        if remaining <= 0:
            return set(list(forced)[:budget])

        # ---- Greedy top-remaining by utility
        cand = [(b, float(u)) for b, u in zip(bids, utility) if b not in forced]
        cand.sort(key=lambda kv: -kv[1])
        picked = {b for b, _ in cand[:remaining]}
        return forced | picked

    # ------------------------------------------------------------------
    #  Feature packing (mirrors seer.lap.features)
    # ------------------------------------------------------------------

    def _features(self, stats: dict) -> np.ndarray:
        history = list(stats.get("attn_history", []))[-self.history_n:]
        hist_arr = np.zeros(self.history_n, dtype=np.float32)
        if history:
            # features.py shifts by 1..history_n, so hist_arr[0] = attn at t-1
            # i.e. the MOST RECENT past score comes first.
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
