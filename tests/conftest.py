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

    # Fill is_top_k: top-1 per (layer, head_group, step) (k=1 for this size)
    def _mark(g):
        g = g.copy()
        g["is_top_k"] = 0
        idx = g["attn_score"].idxmax()
        g.loc[idx, "is_top_k"] = 1
        return g
    df = df.groupby(["layer_id", "head_group", "step"], group_keys=False).apply(_mark)

    # Fill future horizons (very simple: "any is_top_k in next H steps for this block")
    for h in (1, 4, 16, 64):
        col = f"future_top_k_h{h}"
        df[col] = 0
        for key, group in df.groupby(["request_id", "layer_id", "head_group", "block_id"]):
            steps = group.sort_values("step")
            ys = steps["is_top_k"].values
            out = np.zeros_like(ys)
            for i in range(len(ys)):
                out[i] = int(ys[i + 1: i + 1 + h].any()) if i + 1 < len(ys) else 0
            df.loc[steps.index, col] = out
    cast = {
        "request_id": "int64", "layer_id": "int32", "head_group": "int32",
        "block_id": "int32", "block_start_token": "int32", "step": "int32",
        "attn_score": "float32", "is_top_k": "uint8",
        **{f"future_top_k_h{h}": "uint8" for h in (1, 4, 16, 64)},
    }
    return df.astype(cast).reset_index(drop=True)
