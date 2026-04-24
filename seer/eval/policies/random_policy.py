"""Random eviction — sanity baseline to sit at the bottom of any table."""
from __future__ import annotations

import random

from seer.eval.policies.base import KVPolicy


class RandomPolicy(KVPolicy):
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select_to_keep(self, block_stats, budget, step):
        keys = list(block_stats.keys())
        self.rng.shuffle(keys)
        return set(keys[:budget])
