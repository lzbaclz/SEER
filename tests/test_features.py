"""Tests for seer.lap.features.build_features."""
from __future__ import annotations

import numpy as np

from seer.lap.features import HISTORY_N, build_features
from seer.trace.schema import HORIZONS


def test_output_shapes(toy_trace_df):
    X, y, meta = build_features(toy_trace_df)
    n_rows = len(toy_trace_df)
    assert X.shape == (n_rows, meta["input_dim"])
    assert y.shape == (n_rows, len(HORIZONS))
    assert meta["input_dim"] == HISTORY_N + 5  # history + 5 aux
    assert meta["output_dim"] == len(HORIZONS)


def test_feature_dtypes_are_float32(toy_trace_df):
    X, y, _ = build_features(toy_trace_df)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_history_is_causal(toy_trace_df):
    """hist_i at step=0 must be all zeros (no history yet)."""
    X, _, _ = build_features(toy_trace_df)
    # find the step==0 rows via the original df
    step0_mask = toy_trace_df.sort_values(
        ["request_id", "layer_id", "head_group", "block_id", "step"]
    ).reset_index(drop=True)["step"] == 0
    assert np.all(X[step0_mask, :HISTORY_N] == 0.0)


def test_position_is_normalized(toy_trace_df):
    X, _, meta = build_features(toy_trace_df)
    pos = X[:, meta["feature_names"].index("position")]
    assert pos.min() >= 0.0
    assert pos.max() <= 1.0


def test_recency_finite(toy_trace_df):
    X, _, meta = build_features(toy_trace_df)
    rec = X[:, meta["feature_names"].index("recency_log")]
    assert np.all(np.isfinite(rec))
    assert (rec >= 0).all()


def test_feature_names_stable(toy_trace_df):
    _, _, meta = build_features(toy_trace_df)
    expected_tail = ["recency_log", "persistence", "position", "layer_scalar", "head_scalar"]
    assert meta["feature_names"][-5:] == expected_tail
    assert meta["feature_names"][0] == "hist_0"
    assert meta["feature_names"][HISTORY_N - 1] == f"hist_{HISTORY_N-1}"
