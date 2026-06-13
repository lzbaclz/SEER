"""Online σ / escape-mass tracker for adaptive Lemma 2 bound publication.

At deployment time the operator wants the bound to remain a strict upper
envelope across workload shifts (review point M1 / T10). This tracker
maintains a rolling window of per-step decode latencies and, every
``republish_every`` observations, re-computes both:

  * σ = (P99.5 - P50) / sqrt(2 ln 200)            (sub-Gaussian proxy)
  * q = Pr(per-block excess > ell_max_factor · ℓ̄) (heavy-tail escape mass)

and recomputes the Lemma 2 / 2'-truncated bound. If the empirical q
crosses a threshold (default 1e-4), the tracker flips to the
truncated-heavy-tail bound, otherwise stays on the Bernstein-mixture
form. The bound is exposed via ``current_bound(D)`` so an admission
controller can read it before every decision cycle.

This addresses the reviewer's "you can't claim strict upper envelope
across workloads" objection by making the bound's underlying σ / q
estimates live with the running workload.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from seer.timing.schedulability import (
    estimate_escape_mass_from_trace,
    lemma2_bernstein_mixture_bound,
    lemma2_truncated_heavy_tail_bound,
)


@dataclass
class OnlineBoundState:
    """Snapshot of the bound's parameters at one publication moment."""
    n_steps: int
    sigma_us: float
    q: float
    mu0_us: float
    using_truncated: bool


class OnlineBoundTracker:
    """Republish the Lemma 2 bound as the workload's distribution shifts."""

    def __init__(
        self,
        epsilon_mean: float,
        epsilon_var: float,
        ell_bar_us: float,
        B_t: float,
        window: int = 200,
        republish_every: int = 100,
        ell_max_factor: float = 4.0,
        q_threshold: float = 1e-4,
        warmup_skip: int = 1,
    ):
        self.epsilon_mean = epsilon_mean
        self.epsilon_var = epsilon_var
        self.ell_bar_us = ell_bar_us
        self.B_t = B_t
        self.window: deque[float] = deque(maxlen=window)
        self.republish_every = republish_every
        self.ell_max_factor = ell_max_factor
        self.q_threshold = q_threshold
        self.warmup_skip = warmup_skip
        self._n_obs = 0
        self._n_skipped = 0
        self._state: OnlineBoundState | None = None

    def observe(self, step_latency_us: float) -> None:
        """Push a per-step measurement; recomputes bound at the cadence."""
        if self._n_skipped < self.warmup_skip:
            self._n_skipped += 1
            return
        self.window.append(float(step_latency_us))
        self._n_obs += 1
        if self._n_obs >= self.republish_every and self._n_obs % self.republish_every == 0:
            self._republish()

    def _percentile(self, p: float) -> float:
        xs = sorted(self.window)
        if not xs:
            return float("nan")
        k = (len(xs) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(xs) - 1)
        frac = k - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    def _republish(self) -> None:
        n = len(self.window)
        if n < 20:
            return
        mu0 = self._percentile(50.0)
        p995 = self._percentile(99.5)
        factor = math.sqrt(2 * math.log(1 / 0.005))
        sigma = max(1e-3, (p995 - mu0) / factor)
        q = estimate_escape_mass_from_trace(
            list(self.window),
            B_t=self.B_t,
            p50_us=mu0,
            ell_bar_us=self.ell_bar_us,
            ell_max_factor=self.ell_max_factor,
        )
        self._state = OnlineBoundState(
            n_steps=n, sigma_us=sigma, q=q, mu0_us=mu0,
            using_truncated=(q > self.q_threshold),
        )

    @property
    def state(self) -> OnlineBoundState | None:
        return self._state

    def current_bound(self, deadline_us: float) -> float | None:
        """Return the current Pr(C_t > D) upper bound, or None if not yet
        republished. Switches between Bernstein-mixture and truncated
        heavy-tail based on whether the observed q exceeds the threshold.
        """
        s = self._state
        if s is None:
            return None
        if s.using_truncated:
            return lemma2_truncated_heavy_tail_bound(
                epsilon_mean=self.epsilon_mean,
                epsilon_var=self.epsilon_var,
                ell_bar_us=self.ell_bar_us,
                B_t=self.B_t,
                deadline_us=deadline_us,
                sigma_residual_us=s.sigma_us,
                escape_mass=s.q,
                base_cost_us=s.mu0_us,
            )
        return lemma2_bernstein_mixture_bound(
            epsilon_mean=self.epsilon_mean,
            epsilon_var=self.epsilon_var,
            ell_bar_us=self.ell_bar_us,
            B_t=self.B_t,
            deadline_us=deadline_us,
            sigma_residual_us=s.sigma_us,
            base_cost_us=s.mu0_us,
        )
