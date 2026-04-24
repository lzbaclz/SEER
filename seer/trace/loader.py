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
    """Split trace rows into train / val / test by `request_id` to avoid
    within-request leakage (past→future within a prompt shares a distribution)."""
    rng = np.random.default_rng(seed)
    req_ids = df["request_id"].unique()
    rng.shuffle(req_ids)
    n = len(req_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = set(req_ids[:n_train])
    val_ids = set(req_ids[n_train:n_train + n_val])
    test_ids = set(req_ids[n_train + n_val:])
    return (
        df[df["request_id"].isin(train_ids)].reset_index(drop=True),
        df[df["request_id"].isin(val_ids)].reset_index(drop=True),
        df[df["request_id"].isin(test_ids)].reset_index(drop=True),
    )
