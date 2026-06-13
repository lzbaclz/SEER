"""Load parquet traces + train/val/test split by request_id."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_traces(path: str | Path, max_files: int | None = None) -> pd.DataFrame:
    """Load all `*.parquet` files under `path` and concatenate into one DataFrame.

    If `path` is a single file, load just that file.
    """
    p = Path(path)
    if p.is_file():
        return pd.read_parquet(p)
    files = sorted(p.glob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def split_by_request(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split trace rows into train / val / test by ``request_id``.

    The split is at the request level (not row level) to avoid
    within-request leakage. For tiny trace sets (smoke tests with N < 5)
    we override the requested fractions so that *every* split contains
    at least one request, allowing the trainer's per-epoch validation
    AUC step to succeed.
    """
    rng = np.random.default_rng(seed)
    req_ids = df["request_id"].unique().copy()
    rng.shuffle(req_ids)
    n = len(req_ids)
    if n == 0:
        empty = df.iloc[0:0].copy()
        return empty, empty.copy(), empty.copy()
    if n < 3:
        # Degenerate: not enough requests to do a 3-way split. Reuse the
        # only request across all three splits; better than crashing.
        sub = df[df["request_id"].isin(req_ids)].reset_index(drop=True)
        return sub, sub.copy(), sub.copy()
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    # Test gets whatever is left, but at least 1 request.
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_ids = set(req_ids[:n_train])
    val_ids = set(req_ids[n_train:n_train + n_val])
    test_ids = set(req_ids[n_train + n_val:])
    if not test_ids:  # belt-and-braces: keep the last id for test
        test_ids = {req_ids[-1]}
    return (
        df[df["request_id"].isin(train_ids)].reset_index(drop=True),
        df[df["request_id"].isin(val_ids)].reset_index(drop=True),
        df[df["request_id"].isin(test_ids)].reset_index(drop=True),
    )
