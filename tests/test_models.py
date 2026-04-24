"""Tests for seer.lap.models."""
from __future__ import annotations

import torch

from seer.lap.models import build_model, count_params


INPUT_DIM = 32 + 5
N_HORIZONS = 4


def _assert_forward_shapes(model, batch: int = 8) -> None:
    x = torch.randn(batch, INPUT_DIM)
    y = model(x)
    assert y.shape == (batch, N_HORIZONS)
    assert y.dtype == torch.float32


def test_tiny_mlp_forward():
    m = build_model("tiny_mlp", input_dim=INPUT_DIM, n_horizons=N_HORIZONS)
    _assert_forward_shapes(m)
    assert count_params(m) < 200_000, "TinyMLP should stay well under 200K params"


def test_block_rnn_forward():
    m = build_model("block_rnn", input_dim=INPUT_DIM, n_horizons=N_HORIZONS)
    _assert_forward_shapes(m)


def test_block_transformer_forward():
    m = build_model("block_transformer", input_dim=INPUT_DIM, n_horizons=N_HORIZONS)
    _assert_forward_shapes(m)


def test_unknown_model_raises():
    import pytest
    with pytest.raises(ValueError):
        build_model("no_such_model", input_dim=INPUT_DIM, n_horizons=N_HORIZONS)


def test_model_gradients_flow():
    m = build_model("tiny_mlp", input_dim=INPUT_DIM, n_horizons=N_HORIZONS)
    x = torch.randn(4, INPUT_DIM, requires_grad=False)
    y = m(x).sum()
    y.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad, "No gradient flowed through TinyMLP"
