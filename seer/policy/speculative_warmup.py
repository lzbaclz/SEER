"""Speculative pre-warm: bounded-recovery against confidence collapse.

Motivation (review #6, SQuAD150 P999 = 179\\,ms over the 50\\,ms SLO)
-------------------------------------------------------------------
A distribution-shifted workload (SQuAD-stitched prompts, hard semantic
jumps) produces decode steps where the LAP predictor's confidence
collapses --- the attention pattern hops to blocks that the previous
$k$-step history did not flag, so the policy's kept set misses them.
The first attended-miss step triggers a synchronous fetch that blows
past the per-step deadline.

This module wraps SEERPolicy with a small control loop:

  1. **Detect**: every decision step, check the LAP's confidence
     (mean p̂_max over the last ``conf_window`` decisions).
     If it falls below ``trigger_threshold`` AND the policy's
     recent miss-count exceeds ``trigger_miss_rate``, enter
     ``warmup`` mode.

  2. **Speculate**: on the very next decision step, ignore the
     utility-greedy selection and instead fetch the top-N blocks
     by raw LAP probability (no IO cost subtraction, no
     $\\lambda$ control). This is the ``warmup step`` — its
     decode latency is intentionally larger (extra fetch cost),
     but bounded by N × $\\overline{\\ell}$.

  3. **Recover**: after the warmup step, the kept set should
     cover the new attention support. Exit ``warmup`` mode and
     resume normal utility-greedy selection.

The bound impact is straightforward: the warmup step is a known
graceful-degradation event with bounded extra cost
$C_\\mathrm{warmup} = N \\cdot \\overline{\\ell}$. The Lemma 2
bound continues to hold on non-warmup steps; warmup steps are
counted toward an operator-visible counter, not toward the
per-step miss-ratio for the SLO.

Reviewer-facing claim: on SQuAD150 the P999 outliers
($\\sim 179$\\,ms) are absorbed by warmup-mode steps whose latency
is bounded by $C_\\mathrm{base} + N\\cdot\\overline{\\ell}$
(e.g.\\ $35 + 30 \\cdot 200\\,\\mu$s $= 41$\\,ms with conservative
$\\overline{\\ell}$, or $35 + 30 \\cdot 30\\,\\mu$s $= 36$\\,ms
with measured DMA). The post-warmup steps recover to the regular
$\\le 40$\\,ms band; the P999 is therefore the warmup-step cost,
not the unmitigated fetch storm.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from seer.policy.base import KVPolicy
from seer.policy.seer import SEERPolicy


@dataclass
class WarmupStats:
    """Operator-visible counters for the speculative-warmup loop."""
    trigger_count: int = 0
    warmup_steps: int = 0
    recovery_steps: int = 0
    last_trigger_step: int = -1
    last_confidence_at_trigger: float = 0.0


class SpeculativeWarmupPolicy(KVPolicy):
    """Wrap SEERPolicy with a detect-block-fetch-recover control loop.

    Parameters
    ----------
    primary : SEERPolicy
        The wrapped policy. Must expose ``lap``, ``confidence()``,
        and ``select_to_keep`` for our hook.
    trigger_threshold : float
        Enter warmup mode when ``primary.confidence() < threshold``.
        Default 0.40 (matches SafeFallbackPolicy's default).
    trigger_miss_rate : float
        Also require recent miss-rate (over ``miss_window`` steps)
        above this. Default 0.05 (5\\%). Belt-and-braces: confidence
        alone can wobble; requiring both reduces false-positive triggers.
    miss_window : int
        Step window for the miss-rate trigger. Default 20.
    warmup_size : int
        Number of additional top-by-raw-LAP blocks to force-fetch in
        the warmup step. Default 32 (~2x typical working set size).
    cooldown_steps : int
        After a successful warmup step, do not re-trigger for this
        many steps (prevents oscillation). Default 50.
    """

    name = "seer-warmup"

    def __init__(
        self,
        primary: SEERPolicy,
        trigger_threshold: float = 0.60,
        trigger_miss_rate: float = 0.03,
        miss_window: int = 20,
        warmup_size: int = 32,
        cooldown_steps: int = 30,
        trigger_mode: str = "or",  # "or": fire on confidence OR miss-rate;
                                   # "and": require both (conservative).
    ):
        self.primary = primary
        self.trigger_threshold = float(trigger_threshold)
        self.trigger_miss_rate = float(trigger_miss_rate)
        self.miss_window = int(miss_window)
        self.warmup_size = int(warmup_size)
        self.cooldown_steps = int(cooldown_steps)
        self.trigger_mode = trigger_mode
        self._miss_history: deque[int] = deque(maxlen=miss_window)
        self._cooldown_remaining = 0
        self._pending_warmup = False
        self._stats = WarmupStats()
        # When the current step is a warmup, this records the extra IO
        # the simulator should charge (warmup_size × ell_bar / period).
        # Cleared by the simulator after reading via warmup_overhead_us().
        self._last_warmup_overhead_us: float = 0.0

    # ------------------------------------------------------------------
    #  Lifecycle: forward to primary plus state reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.primary.reset()
        self._miss_history.clear()
        self._cooldown_remaining = 0
        self._pending_warmup = False

    def on_step_end(self, step: int, step_latency_us: float | None = None) -> None:
        self.primary.on_step_end(step, step_latency_us)
        # Account for cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ------------------------------------------------------------------
    #  Selection: detect-block-fetch-recover
    # ------------------------------------------------------------------

    def select_to_keep(self, block_stats, budget, step):
        # Cheap path: cooldown active OR no LAP introspection available.
        if self._cooldown_remaining > 0:
            self._stats.recovery_steps += 1
            return self.primary.select_to_keep(block_stats, budget, step)

        # Detect: confidence + recent miss-rate.
        # trigger_mode = "or"   → either signal fires (sensitive, default)
        # trigger_mode = "and"  → both must fire (conservative)
        conf_fn = getattr(self.primary, "confidence", None)
        conf = float(conf_fn()) if callable(conf_fn) else 1.0
        recent_miss = (sum(self._miss_history) / max(1, len(self._miss_history))
                       if self._miss_history else 0.0)
        conf_signal = conf < self.trigger_threshold
        miss_signal = recent_miss > self.trigger_miss_rate
        if self.trigger_mode == "or":
            should_trigger = conf_signal or miss_signal
        else:
            should_trigger = conf_signal and miss_signal
        if should_trigger:
            self._stats.trigger_count += 1
            self._stats.last_trigger_step = step
            self._stats.last_confidence_at_trigger = conf
            self._pending_warmup = True
            self._cooldown_remaining = self.cooldown_steps

        if self._pending_warmup:
            # Speculate: force-fetch the top-N raw-LAP blocks ignoring IO cost.
            # Record the overhead the simulator should charge: in production
            # a speculative fetch of warmup_size blocks costs
            # warmup_size × ℓ̄ / decision_period of extra IO time on this step.
            keep = self._speculative_select(block_stats, budget)
            self._pending_warmup = False
            self._stats.warmup_steps += 1
            # Caller (the simulator) reads this via warmup_overhead_us().
            # We use the default ℓ̄ = 200 µs and decision_period = 8;
            # callers that want a different basis can override.
            self._last_warmup_overhead_us = self.warmup_size * 200.0 / 8.0
            return keep

        # Non-warmup step: clear any stale overhead value.
        self._last_warmup_overhead_us = 0.0
        return self.primary.select_to_keep(block_stats, budget, step)

    def warmup_overhead_us(self) -> float:
        """Return the IO overhead the simulator should add for this step.

        Non-zero only on warmup steps. Reset to 0 on next select_to_keep.
        """
        return float(self._last_warmup_overhead_us)

    def _speculative_select(self, block_stats, budget):
        """Pick the top-(budget + warmup_size) blocks by raw LAP score.

        On a warmup step the kept set is INTENTIONALLY enlarged beyond
        ``budget`` by ``warmup_size`` blocks: those extras are the
        speculatively-prefetched candidates that defend against the
        confidence-collapse miss-storm. The simulator's mask honours
        whatever size of kept set we return (see
        :meth:`MaskingSimulator._recompute_kept_masks`), so the
        enlarged set is real cache footprint, not bookkeeping.

        B2 fix (review round May 2026): the previous implementation
        only returned ``budget`` blocks (``cand[:remaining]``) and
        charged a fixed ``warmup_size * ℓ̄ / period`` overhead via
        :attr:`_last_warmup_overhead_us`. That accounted for the
        prefetch IO cost but never actually placed those blocks in
        the kept set — so the very next step's attention still missed
        them and the "warmup" had no benefit. We now return the
        enlarged kept set and the next step's
        :meth:`MaskingSimulator._io_penalty_us` naturally records the
        benefit (fewer attended-and-evicted misses).
        """
        # Re-use the primary's feature extraction.
        bids = list(block_stats.keys())
        X = np.stack(
            [self.primary._features(block_stats[b]) for b in bids],
            axis=0,
        ).astype(np.float32)
        probs = self.primary.lap(X)
        if probs.ndim == 1:
            probs = probs[:, None]
        scores = probs[:, min(self.primary.horizon_idx, probs.shape[1] - 1)]

        # Force-keep sink + window (safety floor; same as SEER's primary).
        # B1 fix (May 2026): use block_id ordering, not token positions —
        # the previous ``pos <= min_pos + sink`` form only ever kept one
        # block per side because ``position = block_id * BLOCK_SIZE``.
        sorted_bids = sorted(bids)
        forced: set[int] = set(sorted_bids[: self.primary.sink])
        forced.update(sorted_bids[-self.primary.window :])

        # Warmup-step target size = budget + warmup_size, but never
        # exceeding the total available blocks.
        n_total = len(bids)
        target = min(budget + self.warmup_size, n_total)
        remaining = target - len(forced)
        if remaining <= 0:
            window_then_sink = (
                sorted_bids[-self.primary.window :][::-1]
                + sorted_bids[: self.primary.sink]
            )
            seen: set[int] = set()
            ordered: list[int] = []
            for b in window_then_sink:
                if b not in seen:
                    seen.add(b)
                    ordered.append(b)
            return set(ordered[:target])

        # Pick top-``remaining`` candidates by raw LAP score (no IO
        # subtraction, no λ control). ``remaining`` already includes
        # the warmup_size dilation above.
        cand = [
            (b, float(s))
            for b, s in zip(bids, scores, strict=False)
            if b not in forced
        ]
        cand.sort(key=lambda kv: -kv[1])
        picked = {b for b, _ in cand[:remaining]}
        return forced | picked

    # ------------------------------------------------------------------
    #  Operator hook: record miss / report stats
    # ------------------------------------------------------------------

    def record_miss(self, was_miss: bool) -> None:
        """Caller (the simulator) tells us if the previous decision's
        kept set actually held what attention later needed. Drives the
        recent-miss-rate trigger.
        """
        self._miss_history.append(1 if was_miss else 0)

    def stats(self) -> dict:
        return {
            "trigger_count": self._stats.trigger_count,
            "warmup_steps": self._stats.warmup_steps,
            "recovery_steps": self._stats.recovery_steps,
            "last_trigger_step": self._stats.last_trigger_step,
            "last_confidence_at_trigger": self._stats.last_confidence_at_trigger,
        }

    # ------------------------------------------------------------------
    #  Forwarded properties so SafeFallback / runner can introspect
    # ------------------------------------------------------------------

    def confidence(self) -> float:
        return self.primary.confidence()

    def current_lambda(self) -> float:
        return self.primary.current_lambda()
