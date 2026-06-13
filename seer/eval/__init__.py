"""Evaluation infrastructure: metrics + benchmark runner.

Policies have moved to :mod:`seer.policy` as part of the RTSS pivot;
:func:`build_policy` is re-exported from there for backwards compatibility.
"""

from seer.eval.metrics import (
    LatencyStats,
    bound_pessimism,
    exact_match,
    f1_score,
    miss_ratio,
    normalize,
    substring_match,
    tpot_percentile,
)
from seer.policy import build_policy

__all__ = [
    "build_policy",
    "exact_match",
    "f1_score",
    "normalize",
    "substring_match",
    "tpot_percentile",
    "miss_ratio",
    "bound_pessimism",
    "LatencyStats",
]
