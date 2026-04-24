"""Tests for seer.trace.schema and the loader."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa

from seer.trace.loader import load_traces, split_by_request
from seer.trace.schema import (
    HORIZONS,
    arrow_schema,
    compute_top_k,
    horizon_columns,
)


def test_compute_top_k_respects_min():
    assert compute_top_k(100) == 32           # min kicks in
    assert compute_top_k(10_000) == 1000      # 10% rule dominates


def test_horizon_columns_match_horizons():
    assert len(horizon_columns()) == len(HORIZONS)
    for h, name in zip(HORIZONS, horizon_columns()):
        assert name == f"future_top_k_h{h}"


def test_arrow_schema_fields():
    sch = arrow_schema()
    names = set(sch.names)
    assert {"request_id", "layer_id", "head_group", "block_id",
            "block_start_token", "step", "attn_score", "is_top_k"}.issubset(names)
    for h in HORIZONS:
        assert f"future_top_k_h{h}" in names


def test_loader_roundtrip(toy_trace_df):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "req_00.parquet"
        toy_trace_df.to_parquet(path, engine="pyarrow", index=False, schema=arrow_schema())
        df2 = load_traces(d)
        assert len(df2) == len(toy_trace_df)


def test_split_by_request_no_leakage(toy_trace_df):
    tr, va, te = split_by_request(toy_trace_df, train_frac=0.6, val_frac=0.2)
    tr_ids = set(tr["request_id"].unique())
    va_ids = set(va["request_id"].unique())
    te_ids = set(te["request_id"].unique())
    assert not (tr_ids & va_ids)
    assert not (tr_ids & te_ids)
    assert not (va_ids & te_ids)
