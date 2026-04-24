"""Abstract KV-cache policy interface.

A policy observes, at each decode step, per-block statistics for every KV
block in the current request. It returns the set of block_ids that should
be kept in HBM. Blocks not in the returned set are demoted to the next
storage tier.

The interface is backend-agnostic: we use it both in the offline simulator
(`seer.eval.sim`) and, post-integration, as the brain behind the vLLM
block manager hook. Concrete policies are stateless between requests —
any per-request state is passed in through `block_stats`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class KVPolicy(ABC):
    name: str = "base"

    @abstractmethod
    def select_to_keep(
        self,
        block_stats: dict[int, dict],
        budget: int,
        step: int,
    ) -> set[int]:
        """Return block_ids to keep in HBM.

        Parameters
        ----------
        block_stats : {block_id: stats_dict}
            Each value dict may contain (all optional):
              "attn_score_now"      : float  — attn score at current step
              "attn_history"        : list[float] — past attn scores (newest last)
              "position"            : int    — block_start_token
              "position_norm"       : float  — block_start / max_position
              "last_top_k_step"     : int    — step of most recent is_top_k=1
              "steps_since_top_k"   : int
              "persistence"         : float  — fraction of past N steps in top-k
              "layer_scalar"        : float
              "head_scalar"         : float
              "io_cost"             : float  — cost of fetching from current tier
        budget : int
            Maximum number of blocks that may reside in HBM.
        step : int
            Current decode step (0 == end of prefill).

        Returns
        -------
        set[int]
            block_ids to keep (size ≤ budget).
        """

    def reset(self) -> None:
        """Called at the start of each new request."""

    def on_step_end(self, step: int) -> None:
        """Optional hook called after select_to_keep at each step."""


# ---------------------------------------------------------------------------
#  Trivial full-cache baseline
# ---------------------------------------------------------------------------

class FullCachePolicy(KVPolicy):
    """Keeps every block (quality upper bound, throughput lower bound)."""
    name = "full"

    def select_to_keep(self, block_stats, budget, step):
        return set(block_stats.keys())
