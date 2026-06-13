"""StreamingLLM (Xiao et al., ICLR 2024): attention sink + sliding window.

Keeps the first `sink` blocks (attention sink) and the last `window` blocks
(recent tokens). Everything else is evicted. Pure position heuristic — does
not look at attention scores at all.
"""
from __future__ import annotations

from seer.eval.policies.base import KVPolicy


class StreamingPolicy(KVPolicy):
    name = "streaming"

    def __init__(self, sink: int = 4, window: int = 64):
        """
        sink   : number of initial blocks to always keep
        window : number of trailing blocks to always keep (blocks, not tokens)
        """
        self.sink = sink
        self.window = window

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()
        ids_sorted = sorted(block_stats.keys())  # by block_id (== position order)
        keep: set[int] = set(ids_sorted[: self.sink])
        keep.update(ids_sorted[-self.window:])
        if len(keep) > budget:
            # If user-given sink+window exceeds budget, trim the middle-most
            # by preferring recent over sink (to match the paper's behavior
            # under tight budgets).
            order = ids_sorted[-self.window:][::-1] + ids_sorted[: self.sink]
            keep = set(order[:budget])
        return keep
