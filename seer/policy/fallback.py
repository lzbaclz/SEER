"""Safe-fallback wrapper: switch to a heuristic when the predictor is uncertain.

A LAP that is right 90% of the time still has 10% wrong decisions. Under
a tight SLO, those wrong decisions can cluster on a single critical
request and tank P99 / P999 TPOT. The :class:`SafeFallbackPolicy`
mitigates this by routing low-confidence decisions to a deterministic
heuristic baseline (typically H2O), so the worst case is bounded by the
heuristic's worst case rather than by the predictor's.

Usage::

    seer = SEERPolicy(lap_predictor=...)
    safe = SafeFallbackPolicy(primary=seer, fallback=H2OPolicy(), conf_threshold=0.4)
    runner.set_policy(safe)

The wrapper is transparent — it implements :class:`KVPolicy` and forwards
``reset()`` / ``on_step_end()`` to both children.
"""
from __future__ import annotations

from seer.policy.base import KVPolicy


class SafeFallbackPolicy(KVPolicy):
    """Wrap a primary policy; switch to fallback when confidence drops."""

    name = "seer-safe"

    def __init__(
        self,
        primary: KVPolicy,
        fallback: KVPolicy,
        conf_threshold: float = 0.4,
    ):
        self.primary = primary
        self.fallback = fallback
        self.conf_threshold = float(conf_threshold)
        self._fallback_count = 0
        self._primary_count = 0

    def reset(self) -> None:
        self.primary.reset()
        self.fallback.reset()
        self._fallback_count = 0
        self._primary_count = 0

    def on_step_end(self, step: int, step_latency_us: float | None = None) -> None:
        self.primary.on_step_end(step, step_latency_us)
        self.fallback.on_step_end(step, step_latency_us)

    def select_to_keep(self, block_stats, budget, step):
        # The primary's confidence is set the *previous* time it was
        # called. On the very first call we have no signal — defer to
        # primary. If primary doesn't expose a confidence we always
        # take the primary path.
        conf_fn = getattr(self.primary, "confidence", None)
        confidence = float(conf_fn()) if callable(conf_fn) else 1.0
        if step > 0 and confidence < self.conf_threshold:
            self._fallback_count += 1
            return self.fallback.select_to_keep(block_stats, budget, step)
        self._primary_count += 1
        return self.primary.select_to_keep(block_stats, budget, step)

    def stats(self) -> dict:
        total = self._primary_count + self._fallback_count
        return {
            "primary_calls": self._primary_count,
            "fallback_calls": self._fallback_count,
            "fallback_fraction": (self._fallback_count / total) if total else 0.0,
        }
