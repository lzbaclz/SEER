"""H2O (Zhang et al., NeurIPS 2023): heavy-hitter + recent.

Splits the HBM budget between:
  1. "Heavy hitters" — blocks with the highest *cumulative* attention score
     so far (captures Attention Sinks and long-lived important tokens).
  2. "Recent" — the newest blocks (local attention).

Classic split is 50/50. Heavy hitters use the running sum of past attention
weights as the importance proxy (we expose this through
`block_stats[bid]["attn_history"]`).
"""
from __future__ import annotations

from seer.eval.policies.base import KVPolicy


class H2OPolicy(KVPolicy):
    name = "h2o"

    def __init__(self, hh_frac: float = 0.5):
        """hh_frac : fraction of budget reserved for heavy hitters."""
        assert 0.0 <= hh_frac <= 1.0
        self.hh_frac = hh_frac

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()

        n_hh = int(round(budget * self.hh_frac))
        n_recent = budget - n_hh

        # Heavy hitters: top n_hh by sum of attn_history (or attn_score_now fallback)
        def _cumulative(stats: dict) -> float:
            hist = stats.get("attn_history")
            if hist:
                return float(sum(hist))
            return float(stats.get("attn_score_now", 0.0))

        ranked_hh = sorted(
            block_stats.items(),
            key=lambda kv: -_cumulative(kv[1]),
        )
        keep: set[int] = {bid for bid, _ in ranked_hh[:n_hh]}

        # Recent: top n_recent by position
        ranked_recent = sorted(
            block_stats.items(),
            key=lambda kv: -kv[1].get("position", 0),
        )
        for bid, _ in ranked_recent:
            if len(keep) >= budget:
                break
            keep.add(bid)
        return keep
