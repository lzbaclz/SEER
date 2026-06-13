"""Per-block statistics accumulator used by the offline simulator.

This is the bridge between raw attention traces and the policy interface:
as the simulator replays a trace / generates, it keeps a running view of
every block's attention history, recency of top-k membership, persistence,
etc., and hands that dict to `policy.select_to_keep()` each step.
"""
from __future__ import annotations

from collections import defaultdict

from seer.lap.features import HISTORY_N


class BlockStatsBuffer:
    """Accumulates per-block state for one (layer, head_group) stream.

    Keeps only the trailing HISTORY_N attention scores per block, so
    memory is O(n_blocks × HISTORY_N) regardless of decode length.
    """

    def __init__(self, history_n: int = HISTORY_N):
        self.history_n = history_n
        self.attn_history: dict[int, list[float]] = defaultdict(list)
        self.last_top_k_step: dict[int, int] = {}
        self.persistence_window: dict[int, list[int]] = defaultdict(list)
        self.position: dict[int, int] = {}
        self.current_step: int = 0

    # ------------------------------------------------------------------

    def update_step(
        self,
        step: int,
        per_block_attn: dict[int, float],
        per_block_is_top_k: dict[int, int],
    ) -> None:
        """Ingest one step of attention scores + top-k labels."""
        self.current_step = step
        for bid, score in per_block_attn.items():
            buf = self.attn_history[bid]
            buf.append(float(score))
            if len(buf) > self.history_n:
                del buf[0]
            if per_block_is_top_k.get(bid, 0):
                self.last_top_k_step[bid] = step
            pw = self.persistence_window[bid]
            pw.append(int(per_block_is_top_k.get(bid, 0)))
            if len(pw) > self.history_n:
                del pw[0]

    def set_position(self, bid: int, pos: int) -> None:
        self.position[bid] = pos

    # ------------------------------------------------------------------

    def snapshot(
        self,
        layer_scalar: float,
        head_scalar: float,
    ) -> dict[int, dict]:
        """Return the dict-of-stats consumable by KVPolicy.select_to_keep()."""
        max_pos = max(self.position.values()) if self.position else 1
        max_pos = max(max_pos, 1)
        out: dict[int, dict] = {}
        for bid in self.attn_history.keys():
            hist = self.attn_history[bid]
            pw = self.persistence_window[bid]
            last_tk = self.last_top_k_step.get(bid, -1)
            steps_since = (
                (self.current_step - last_tk) if last_tk >= 0 else self.history_n + 1
            )
            out[bid] = {
                "attn_score_now": hist[-1] if hist else 0.0,
                "attn_history": list(hist),
                "position": self.position.get(bid, bid),
                "position_norm": self.position.get(bid, bid) / max_pos,
                "last_top_k_step": last_tk,
                "steps_since_top_k": steps_since,
                "persistence": (sum(pw) / len(pw)) if pw else 0.0,
                "layer_scalar": layer_scalar,
                "head_scalar": head_scalar,
                "io_cost": 0.0,  # filled in by backend if tier-aware
            }
        return out
