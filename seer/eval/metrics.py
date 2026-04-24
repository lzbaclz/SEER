"""Simple, standard metrics for QA / needle tasks.

For LongBench we follow their official per-task metrics; for RULER the
output is exact-match on the needle. These helpers cover the common cases.
"""
from __future__ import annotations

import re
import string


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
    if pred is None or ref is None:
        return 0.0
    p_tokens = normalize(pred).split()
    r_tokens = normalize(ref).split()
    if not p_tokens and not r_tokens:
        return 1.0
    if not p_tokens or not r_tokens:
        return 0.0
    common = set(p_tokens) & set(r_tokens)
    if not common:
        return 0.0
    precision = sum(1 for t in p_tokens if t in common) / len(p_tokens)
    recall = sum(1 for t in r_tokens if t in common) / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def substring_match(pred: str, ref: str) -> float:
    """Useful for RULER password-style needles — 'the answer is 12345' counts."""
    if not pred or not ref:
        return 0.0
    return 1.0 if normalize(ref) in normalize(pred) else 0.0
