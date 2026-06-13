"""Learned Attention Predictor (LAP) — the core model of SEER.

RTSS pivot note: the canonical module name for the model definitions is
:mod:`seer.lap.model` (singular). The plural :mod:`seer.lap.models` is
kept as a deprecation shim for the remaining NeurIPS-era call sites.
"""

from seer.lap.features import HISTORY_N, build_features
from seer.lap.model import (
    BlockRNN,
    BlockTransformer,
    TinyMLP,
    build_model,
    count_params,
)

__all__ = [
    "HISTORY_N",
    "build_features",
    "TinyMLP",
    "BlockRNN",
    "BlockTransformer",
    "build_model",
    "count_params",
]
