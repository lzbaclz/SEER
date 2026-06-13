"""Deprecation shim — the canonical home of policies is :mod:`seer.policy`.

This module remains importable so existing scripts (legacy
``experiments/e1_predictability/run.sh``, notebooks, etc.) keep working.
New code should ``from seer.policy import build_policy``.
"""
import warnings as _warnings

from seer.policy import (  # noqa: F401  (re-export)
    FullCachePolicy,
    H2OPolicy,
    KVPolicy,
    QuestPolicy,
    RandomPolicy,
    RecencyPolicy,
    SafeFallbackPolicy,
    SEERPolicy,
    SnapKVPolicy,
    StreamingPolicy,
    build_policy,
)

_warnings.warn(
    "seer.eval.policies has moved to seer.policy (RTSS pivot). "
    "Update imports to keep deprecation warnings out of test logs.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "KVPolicy",
    "FullCachePolicy",
    "StreamingPolicy",
    "H2OPolicy",
    "SnapKVPolicy",
    "QuestPolicy",
    "RecencyPolicy",
    "RandomPolicy",
    "SEERPolicy",
    "SafeFallbackPolicy",
    "build_policy",
]
