"""Recency-only baseline: keep the N most recent blocks (LRU by position)."""
from __future__ import annotations

from seer.eval.policies.base import KVPolicy


class RecencyPolicy(KVPolicy):
    name = "recency"

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        ranked = sorted(
            block_stats.items(),
            key=lambda kv: -kv[1].get("position", 0),
        )
        return {bid for bid, _ in ranked[:budget]}
