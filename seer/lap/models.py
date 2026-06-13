"""Deprecated module path — use ``seer.lap.model`` (singular) instead.

This shim is preserved so existing imports keep working during the
RTSS pivot. New code should import from :mod:`seer.lap.model`.
"""
from __future__ import annotations

import warnings as _warnings

from seer.lap.model import (  # noqa: F401  (re-exported)
    BlockRNN,
    BlockTransformer,
    TinyMLP,
    build_model,
    count_params,
)

_warnings.warn(
    "seer.lap.models is deprecated; import from seer.lap.model (singular) instead.",
    DeprecationWarning,
    stacklevel=2,
)
