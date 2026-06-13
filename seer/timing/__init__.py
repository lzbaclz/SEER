"""Timing model + schedulability analysis for SEER.

This module is the *theoretical core* of the RTSS-track SEER paper.
It owns three concerns:

  1. SLO definitions and per-token deadline derivation (`slo.py`)
  2. The PI controller that adapts the joint-policy IO weight (`slo.py`)
  3. Closed-form schedulability bounds — Lemmas 1, 2, 3 (`schedulability.py`)

The CLI `python -m seer.timing.schedulability` is the headline tool:
given a LAP checkpoint, a workload trace and an SLO it reports both
the analytical bound from Lemma 2 and the empirical miss ratio from
trace replay, plus the gap between them (the *pessimism factor*
reported in §6.5 of the paper).
"""

from seer.timing.schedulability import (
    BoundReport,
    estimate_escape_mass_from_trace,
    estimate_per_step_burst_rate_from_trace,
    lemma1_lateness_bound,
    lemma2_bernstein_miss_prob_bound,
    lemma2_bernstein_mixture_bound,
    lemma2_miss_prob_bound,
    lemma2_per_step_conditional_bound,
    lemma2_truncated_heavy_tail_bound,
    lemma3_heuristic_gap,
    min_hbm_budget_for_slo,
    min_hbm_budget_for_slo_measured,
)
from seer.timing.slo import (
    LambdaController,
    SLOClass,
    derive_deadline_us,
    parse_slo,
)

__all__ = [
    "SLOClass",
    "parse_slo",
    "derive_deadline_us",
    "LambdaController",
    "BoundReport",
    "lemma1_lateness_bound",
    "lemma2_miss_prob_bound",
    "lemma2_bernstein_miss_prob_bound",
    "lemma2_bernstein_mixture_bound",
    "lemma2_truncated_heavy_tail_bound",
    "lemma2_per_step_conditional_bound",
    "estimate_escape_mass_from_trace",
    "estimate_per_step_burst_rate_from_trace",
    "lemma3_heuristic_gap",
    "min_hbm_budget_for_slo",
    "min_hbm_budget_for_slo_measured",
]
