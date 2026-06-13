"""InfiniGen-style baseline policy (OSDI'24, Lee et al.) -- DEPRECATED
SURROGATE.

R31 (advisor 2-week sprint): the paper's quantitative claims have been
purged of any InfiniGen comparison because this module is a
simulator-internal surrogate, not a faithful reproduction of the
upstream artifact. **Runs of ``--policy infinigen`` must not be
interpreted as a head-to-head comparison against upstream InfiniGen.**
The module is retained for traceability of prior submission attempts
and for in-repo ablation against SEER's own LAP, but is excluded from
all paper-headline tables. A faithful port (with the upstream proxy
forward and the cost model from the OSDI'24 paper) is on the
ATC 2027 roadmap; see ``todo_atc.md``.

Faithful re-implementation caveat
---------------------------------
We do not have an end-to-end InfiniGen artifact to reuse, so we
implement its core mechanism rather than a bit-exact replica. The
two faithful invariants:

  1. **Speculative attention proxy.** InfiniGen estimates next-step
     attention from a smaller proxy (a few heads of the attention
     layer or a smaller transformer). Our proxy uses the per-block
     mean of the **last-K attention scores** of the same model (no
     small surrogate model loaded; this gives InfiniGen the same
     simulator-level information that SEER's LAP sees). This
     under-favours InfiniGen against a real-proxy implementation by
     about the cost of one small-model forward pass per decision
     period — discussed in the paper as a deliberately
     pessimistic-against-InfiniGen choice.

  2. **Speculative prefetch.** The kept set is the union of (a) a
     recency floor of N_recent=8 blocks, and (b) the top-K blocks by
     the proxy score, where K = budget - n_recent. No SLO-aware λ
     control; no schedulability test. This matches InfiniGen's
     paper-described policy.

References
----------
* Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim.
  "InfiniGen: Efficient Generative Inference of Large Language
  Models with Dynamic KV Cache Management." OSDI 2024.
"""
from __future__ import annotations

from seer.policy.base import KVPolicy


class InfiniGenPolicy(KVPolicy):
    """InfiniGen-style speculative-prefetch baseline.

    Parameters
    ----------
    n_recent : int
        Hard recency floor — these blocks are always kept regardless
        of the proxy score. Default 8 (matches InfiniGen's window).
    history_k : int
        Number of past attention scores averaged into the proxy.
        Default 4 — small enough to be cheap, large enough to smooth
        single-step noise.
    """

    name = "infinigen"

    def __init__(self, n_recent: int = 8, history_k: int = 4) -> None:
        self.n_recent = int(n_recent)
        self.history_k = int(history_k)

    def select_to_keep(self, block_stats, budget, step):
        if not block_stats:
            return set()

        sorted_bids = sorted(block_stats.keys())

        # 1) Recency floor — top n_recent latest blocks.
        kept: set[int] = set(sorted_bids[-self.n_recent:])

        # 2) Speculative proxy score — mean of the last
        #    ``history_k`` attention observations per block.
        def proxy(stats: dict) -> float:
            hist = stats.get("attn_history") or []
            tail = list(hist)[-self.history_k:]
            if not tail:
                return float(stats.get("attn_score_now", 0.0))
            return float(sum(tail) / len(tail))

        remaining = budget - len(kept)
        if remaining <= 0:
            # Budget cannot hold the full recency floor — clip.
            return set(sorted_bids[-budget:]) if budget > 0 else set()

        # Rank by proxy score descending; pick top-(remaining) blocks
        # not already in the recency floor.
        candidates = [
            (bid, proxy(stats))
            for bid, stats in block_stats.items()
            if bid not in kept
        ]
        candidates.sort(key=lambda kv: -kv[1])
        for bid, _ in candidates[:remaining]:
            kept.add(bid)
        return kept
