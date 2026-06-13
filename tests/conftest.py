"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_trace_df() -> pd.DataFrame:
    """Construct a small synthetic trace DataFrame matching the parquet schema.

    3 requests × 2 layers × 2 head_groups × 4 blocks × 10 steps = 480 rows.
    attn_score is sampled from a distribution where block 0 is always hot
    (so LAP should learn "block 0 = top-k") — useful for sanity tests.

    The implementation avoids ``DataFrame.groupby().apply()`` because its
    behavior around index/column round-tripping changed in pandas 2.2+ /
    pandas 3.0 and is brittle across versions. We compute ``is_top_k``
    and the future-horizon labels directly with vectorized numpy.
    """
    rng = np.random.default_rng(0)
    rows = []
    for req in range(3):
        for layer in range(2):
            for hg in range(2):
                for step in range(10):
                    for bid in range(4):
                        # block 0 is the "attention sink" — always high score
                        base = 0.9 if bid == 0 else 0.2
                        score = base + rng.normal(0, 0.05)
                        rows.append({
                            "request_id": req,
                            "layer_id": layer,
                            "head_group": hg,
                            "block_id": bid,
                            "block_start_token": bid * 32,
                            "step": step,
                            "attn_score": float(np.clip(score, 0, 1)),
                        })
    df = pd.DataFrame(rows)

    # ---- is_top_k: top-1 within each (layer, head_group, step) ----
    df["is_top_k"] = 0
    keys = ["layer_id", "head_group", "step"]
    # idxmax of attn_score per group → the rows that are top-1
    top_idx = df.groupby(keys, sort=False)["attn_score"].idxmax()
    df.loc[top_idx.values, "is_top_k"] = 1

    # ---- Future horizons: "any is_top_k in next H steps for this block" ----
    block_keys = ["request_id", "layer_id", "head_group", "block_id"]
    for h in (1, 4, 16, 64):
        col = f"future_top_k_h{h}"
        df[col] = 0
        # For each block-grouped contiguous (sorted-by-step) sequence,
        # set future_h[i] = any is_top_k in (i+1, i+H+1).
        df_sorted = df.sort_values(block_keys + ["step"]).reset_index()
        for _, idx in df_sorted.groupby(block_keys, sort=False).indices.items():
            ys = df_sorted.loc[idx, "is_top_k"].to_numpy()
            out = np.zeros_like(ys)
            for i in range(len(ys)):
                window_end = min(i + 1 + h, len(ys))
                out[i] = int(ys[i + 1: window_end].any()) if i + 1 < len(ys) else 0
            # df_sorted["index"] holds the original row label
            orig_rows = df_sorted.loc[idx, "index"].values
            df.loc[orig_rows, col] = out

    # ---- Timing breakdown (RTSS pivot) — zero-fill for the toy fixture ----
    for col in ("c_lap_us", "c_io_us", "c_attn_us", "c_ffn_us", "step_latency_us"):
        df[col] = 0.0

    cast = {
        "request_id": "int64", "layer_id": "int32", "head_group": "int32",
        "block_id": "int32", "block_start_token": "int32", "step": "int32",
        "attn_score": "float32", "is_top_k": "uint8",
        **{f"future_top_k_h{h}": "uint8" for h in (1, 4, 16, 64)},
        **{c: "float32" for c in
           ("c_lap_us", "c_io_us", "c_attn_us", "c_ffn_us", "step_latency_us")},
    }
    return df.astype(cast).reset_index(drop=True)
