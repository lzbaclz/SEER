"""SnapKV (Li et al., NeurIPS 2024): prefill-time compression, frozen through decode.

At the end of prefill (step == 0), SnapKV ranks blocks by the attention
received from the last `obs_window` query tokens and keeps the top-B.
The selection is then frozen for the rest of the decode — no further
re-ranking. This is essentially a "one-shot" baseline against which our
continuous LAP-driven policy is compared.
"""
from __future__ import annotations

from seer.eval.policies.base import KVPolicy


class SnapKVPolicy(KVPolicy):
    name = "snapkv"

    def __init__(self):
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
        # Ensure any NEW blocks (tokens generated during decode) are added —
        # SnapKV in practice keeps newly-generated tokens on-GPU.
        for bid in block_stats.keys():
            if bid not in self._selection and len(self._selection) < budget:
                # we treat "max block_id" as newly-generated
                if bid >= max(self._selection) if self._selection else True:
                    self._selection.add(bid)
        return self._selection
