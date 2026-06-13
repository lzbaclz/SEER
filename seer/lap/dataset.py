"""Thin PyTorch Dataset over numpy feature matrices (fits in RAM for < ~50M rows)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TraceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "feature/label length mismatch"
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
