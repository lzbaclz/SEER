"""Quality + real-time metrics for the SEER benchmark runner.

The quality functions (:func:`exact_match`, :func:`f1_score`,
:func:`substring_match`) follow the SQuAD v1.1 reference
implementation.

P0-7 fix (review round May 2026, reviewers #1 & #3): the previous
``f1_score`` used a *set* intersection between predicted and
reference tokens, which inflated the score whenever the model
repeated a correct answer token multiple times inside otherwise
gibberish output (e.g. ``"the 1978 the 1978" vs "1978"`` scored
1.0 instead of the standard-SQuAD 0.667). We now use a multiset
(``collections.Counter``) intersection — bit-exact to the
``compute_f1`` routine in ``evaluate-v1.1.py``.

The RTSS pivot adds:

* :func:`tpot_percentile` — a single percentile of a per-step latency vector
* :func:`miss_ratio` — fraction of steps where ``latency > deadline``
* :func:`bound_pessimism` — analytical / measured ratio
* :class:`LatencyStats` — bundled summary of a per-step latency vector
"""
from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
#  Quality (string) metrics
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lower + strip punctuation + collapse whitespace + drop leading articles."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, ref: str) -> float:
    if pred is None or ref is None:
        return 0.0
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def f1_score(pred: str, ref: str) -> float:
    """SQuAD v1.1 token-level F1 score (bit-exact to evaluate-v1.1.py).

    Uses multiset (``collections.Counter``) intersection so that
    repeated tokens are counted at most as many times as they
    appear in both prediction and reference. This rules out the
    inflation that the previous set-based implementation suffered
    when the model repeated a correct answer token inside
    gibberish output (see P0-7 note in module docstring).
    """
    if pred is None or ref is None:
        return 0.0
    p_tokens = normalize(pred).split()
    r_tokens = normalize(ref).split()
    if not p_tokens and not r_tokens:
        return 1.0
    if not p_tokens or not r_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(r_tokens)  # multiset intersection
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def substring_match(pred: str, ref: str) -> float:
    """Useful for RULER password-style needles — 'the answer is 12345' counts."""
    if not pred or not ref:
        return 0.0
    return 1.0 if normalize(ref) in normalize(pred) else 0.0


# ---------------------------------------------------------------------------
#  RT metrics — RTSS pivot
# ---------------------------------------------------------------------------

def tpot_percentile(latencies_us: Iterable[float], pct: float) -> float:
    """Percentile of a per-step latency vector (in microseconds).

    ``pct`` is in [0, 100]. Linear interpolation between adjacent ranks.
    Empty input returns 0.
    """
    xs = sorted(float(x) for x in latencies_us)
    if not xs:
        return 0.0
    if not (0.0 <= pct <= 100.0):
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    k = (len(xs) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def miss_ratio(latencies_us: Iterable[float], deadline_us: float) -> float:
    """Fraction of latencies exceeding the deadline."""
    xs = list(latencies_us)
    if not xs:
        return 0.0
    miss = sum(1 for x in xs if float(x) > float(deadline_us))
    return float(miss) / len(xs)


def bound_pessimism(bound: float, measured: float) -> float:
    """Analytical bound divided by measured value, with safe fallbacks."""
    if measured <= 0:
        return float("inf") if bound > 0 else 1.0
    return float(bound) / float(measured)


@dataclass
class LatencyStats:
    """Compact summary of a per-step latency vector for JSON dumps."""
    n: int
    mean_us: float
    p50_us: float
    p90_us: float
    p99_us: float
    p999_us: float
    max_us: float
    miss_count: int
    miss_ratio: float

    @classmethod
    def from_vector(cls, latencies_us: Iterable[float],
                    deadline_us: float) -> LatencyStats:
        xs = [float(x) for x in latencies_us]
        if not xs:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        miss = sum(1 for x in xs if x > deadline_us)
        return cls(
            n=len(xs),
            mean_us=sum(xs) / len(xs),
            p50_us=tpot_percentile(xs, 50.0),
            p90_us=tpot_percentile(xs, 90.0),
            p99_us=tpot_percentile(xs, 99.0),
            p999_us=tpot_percentile(xs, 99.9),
            max_us=max(xs),
            miss_count=miss,
            miss_ratio=miss / len(xs),
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_us": self.mean_us,
            "p50_us": self.p50_us,
            "p90_us": self.p90_us,
            "p99_us": self.p99_us,
            "p999_us": self.p999_us,
            "max_us": self.max_us,
            "miss_count": self.miss_count,
            "miss_ratio": self.miss_ratio,
        }
