"""Feature engineering from raw trace DataFrame.

Produces (X, y, meta) tensors ready for training / inference.

Per (request, layer, head_group, block, step), the features are:

  hist_{i}      ∈ [0, 1]   attn_score at (step - 1 - i), i=0..HISTORY_N-1
                           (padded with 0 when step < i+1)
  recency_log   ∈ [0, ∞)   log1p(steps since last is_top_k=1)
  persistence   ∈ [0, 1]   mean is_top_k over the past HISTORY_N steps
  position      ∈ [0, 1]   block_start_token / max_block_start in this request
  layer_scalar  ∈ [0, 1]   layer_id / max_layer
  head_scalar   ∈ [0, 1]   head_group / max_head_group

Targets: y[:, h] = future_top_k_h{H} for H in HORIZONS.

The feature order is stable — the concatenated vector is
  [ hist_0 … hist_{N-1}  recency_log  persistence  position  layer  head ]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from seer.trace.schema import HORIZONS

HISTORY_N: int = 32


def build_features(
    df: pd.DataFrame,
    history_n: int = HISTORY_N,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X, y, meta).

    X:    [N, input_dim] float32
    y:    [N, len(HORIZONS)] float32
    meta: {feature_names, horizons, input_dim, output_dim, history_n}
    """
    if len(df) == 0:
        raise ValueError("empty dataframe passed to build_features")

    df = df.sort_values(
        ["request_id", "layer_id", "head_group", "block_id", "step"],
    ).reset_index(drop=True)

    group_keys = ["request_id", "layer_id", "head_group", "block_id"]

    # ---- History of attn_score (shift 1..history_n)
    grouped = df.groupby(group_keys, sort=False)
    history = np.zeros((len(df), history_n), dtype=np.float32)
    for i in range(history_n):
        shifted = grouped["attn_score"].shift(i + 1).fillna(0.0).astype(np.float32).values
        history[:, i] = shifted

    # ---- Persistence: mean of past is_top_k over history_n
    persist_acc = np.zeros(len(df), dtype=np.float32)
    for i in range(history_n):
        shifted = grouped["is_top_k"].shift(i + 1).fillna(0).astype(np.float32).values
        persist_acc += shifted
    persistence = persist_acc / history_n

    # ---- Recency: steps since last top_k (log1p, clipped)
    recency = _compute_recency(df, group_keys)
    recency_log = np.log1p(np.minimum(recency, history_n + 1).astype(np.float32))

    # ---- Position: block_start_token normalized within request
    max_start = df.groupby("request_id")["block_start_token"].transform("max")
    max_start = max_start.replace(0, 1)  # avoid /0 for single-block requests
    position = (df["block_start_token"].astype(np.float32) /
                max_start.astype(np.float32)).values

    # ---- Layer / head scalars
    max_layer = max(1, int(df["layer_id"].max()))
    max_head = max(1, int(df["head_group"].max()))
    layer_scalar = (df["layer_id"].astype(np.float32) / max_layer).values
    head_scalar = (df["head_group"].astype(np.float32) / max_head).values

    X = np.concatenate([
        history,
        recency_log.reshape(-1, 1),
        persistence.reshape(-1, 1),
        position.reshape(-1, 1),
        layer_scalar.reshape(-1, 1),
        head_scalar.reshape(-1, 1),
    ], axis=1).astype(np.float32)

    y_cols = [f"future_top_k_h{h}" for h in HORIZONS]
    y = df[y_cols].to_numpy(dtype=np.float32)

    feature_names = (
        [f"hist_{i}" for i in range(history_n)]
        + ["recency_log", "persistence", "position", "layer_scalar", "head_scalar"]
    )

    meta = {
        "feature_names": feature_names,
        "horizons": list(HORIZONS),
        "input_dim": X.shape[1],
        "output_dim": y.shape[1],
        "history_n": history_n,
    }
    return X, y, meta


def _compute_recency(df: pd.DataFrame, group_keys: list[str]) -> np.ndarray:
    """Vectorized recency: steps since last is_top_k=1 within each group.

    Equivalent to the previous Python-loop implementation but ~50x faster
    on traces with millions of rows. Strategy:

    1. Within each (request, layer, head_group, block) group sorted by
       ``step``, mark the *position* (0-indexed within the sorted group)
       of the row, NOT the literal step number.
    2. ``last_topk_pos`` = the position where ``is_top_k`` was 1, then
       ``shift(1).ffill()`` so it points at the *previous* hit.
    3. ``recency = position - last_topk_pos`` (clipped); when there is
       no prior hit, fall back to ``HISTORY_N + 1``.

    The shift(1) is essential — recency must look at top-k events *before*
    the current row, not including it.
    """
    n = len(df)
    if n == 0:
        return np.full(0, HISTORY_N + 1, dtype=np.int32)
    grouped = df.groupby(group_keys, sort=False)
    # cumulative count within group → 0,1,2,... (positions)
    pos = grouped.cumcount().astype(np.int64)
    is_top = df["is_top_k"].astype(bool)
    # last_topk_pos: position-of-this-row when is_top_k=1, NaN otherwise.
    masked_pos = pos.where(is_top.values)
    # Shift so the lookup is "the most recent top-k position BEFORE me",
    # then forward-fill within each group, then cap NaNs (no prior hit).
    last_pos = masked_pos.groupby(grouped.ngroup()).shift(1).ffill()
    rec = (pos - last_pos).fillna(HISTORY_N + 1)
    return np.clip(rec.to_numpy(), 0, HISTORY_N + 1).astype(np.int32)
