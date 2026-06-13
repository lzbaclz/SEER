"""R31 advisor: hot-path partial-materialisation decision router (A2).

This is the Python-side of the cycle-2 (ATC 2027) hot-path binding
described in ``docs/hot_path_prototype_design.md``. The future vLLM
patch (``vllm_patches/0002-hot-path-allocation-hook.patch``) will
invoke :meth:`HotPathRouter.partial_materialisation_order` from inside
``KVCacheManager.get_computed_blocks`` on a prefix-cache hit, allowing
SEER to choose **which** subset of matched prefix blocks to actually
materialise to HBM under the per-step slack budget.

**Status: prototype.** No vLLM hook calls this module yet; the RTSS
paper continues to claim "diagnosed non-binding" for the patched fork.
The router is unit-test-only until the cycle-2 patch ships.

Design contract (must be preserved when the vLLM patch lands):

- ``partial_materialisation_order`` returns a list of indices into
  ``matched_blocks``; entries earlier in the list will be materialised
  first. vLLM is responsible for truncating at the point where
  cumulative materialisation cost would exceed the slack budget.
- The router is CPU-only and pure-Python (no torch on the hot path
  in the prototype). LAP forward is invoked via the existing
  :class:`seer.lap.LAP` adapter; in tests we replace it with a
  callable mock.
- Score convention: **higher score == more important to materialise
  first**. The vLLM hook may either truncate the suffix or schedule
  the suffix for cold-tier fetch.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class _ScoreFn(Protocol):
    """Minimal protocol the router needs from a LAP-like scorer."""

    def __call__(self, request_id: str,
                 block_ids: Sequence[int]) -> list[float]:
        ...


@dataclass(frozen=True)
class SlackBudget:
    """Per-step slack budget breakdown (microseconds).

    The router consumes only ``io_budget_us``; the other fields are
    retained for diagnostics so the future vLLM hook can log them on
    a per-step basis without re-deriving them.
    """
    deadline_us: float
    forward_us: float
    lap_us: float
    attn_us: float
    ffn_us: float
    safety_margin_us: float

    @property
    def slack_us(self) -> float:
        return (self.deadline_us
                - self.forward_us
                - self.lap_us
                - self.attn_us
                - self.ffn_us)

    @property
    def io_budget_us(self) -> float:
        return max(0.0, self.slack_us - self.safety_margin_us)


class HotPathRouter:
    """SEER-side decision router for vLLM hot-path A2 binding.

    See ``docs/hot_path_prototype_design.md`` for the full contract.
    """

    def __init__(
        self,
        score_fn: _ScoreFn | Callable[[str, Sequence[int]], list[float]],
        ell_bar_us: float = 200.0,
    ) -> None:
        if ell_bar_us <= 0:
            raise ValueError(f"ell_bar_us must be positive; got {ell_bar_us}")
        self._score_fn = score_fn
        self._ell_bar_us = float(ell_bar_us)

    def partial_materialisation_order(
        self,
        request_id: str,
        matched_blocks: Sequence[int],
        slack_budget_us: float,
    ) -> list[int]:
        """Return SEER-ranked indices into ``matched_blocks``.

        Indices earlier in the returned list will be materialised first.
        vLLM is responsible for truncating at the materialisation
        budget; the router only returns the **order**.
        """
        n = len(matched_blocks)
        if n == 0:
            return []
        if slack_budget_us <= 0:
            return []  # caller will truncate to zero anyway; be explicit
        scores = self._score_fn(request_id, list(matched_blocks))
        if len(scores) != n:
            raise ValueError(
                f"score_fn returned {len(scores)} scores for "
                f"{n} blocks; lengths must match"
            )
        # Stable sort by descending score; ties preserve vLLM's default
        # LRU input order (= "preserves LRU when LAP is uninformative").
        order = sorted(range(n), key=lambda i: -scores[i])
        return order

    def max_blocks_under_budget(
        self, slack_budget_us: float,
    ) -> int:
        """How many blocks fit under the slack budget.

        Diagnostic helper for the vLLM hook -- not on the critical path.
        """
        if slack_budget_us <= 0:
            return 0
        return int(math.floor(slack_budget_us / self._ell_bar_us))


def slack_from_budget(
    deadline_us: float,
    forward_us: float,
    lap_us: float = 33.8,
    attn_us: float = 0.0,
    ffn_us: float = 0.0,
    safety_margin_us: float = 200.0,
) -> SlackBudget:
    """Convenience constructor matching the design-doc derivation.

    ``lap_us`` default is the paper-headline TinyMLP P99.9 on A100;
    ``safety_margin_us`` default is the Lemma 2 ell_max truncation
    budget. The vLLM patch is expected to compute ``forward_us`` /
    ``attn_us`` / ``ffn_us`` from its own profiling counters.
    """
    return SlackBudget(
        deadline_us=deadline_us,
        forward_us=forward_us,
        lap_us=lap_us,
        attn_us=attn_us,
        ffn_us=ffn_us,
        safety_margin_us=safety_margin_us,
    )


__all__ = ("HotPathRouter", "SlackBudget", "slack_from_budget")
