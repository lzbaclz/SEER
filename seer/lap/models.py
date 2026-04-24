"""LAP model architectures.

Three variants of increasing complexity:
  tiny_mlp          ~  50K params   (main target; < 0.1% FLOPs budget)
  block_rnn         ~ 100K params   (ablation: sequential encoding of history)
  block_transformer ~ 500K params   (ablation: self-attention over history)

All models take the concatenated feature vector as input and produce logits
of shape [batch, n_horizons] (apply sigmoid for probabilities).
"""
from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
#  1. Tiny MLP — main production model
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, n_horizons: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
#  2. Block-RNN — GRU over the history sequence + MLP head over aux features
# ---------------------------------------------------------------------------

class BlockRNN(nn.Module):
    def __init__(
        self,
        history_n: int,
        aux_dim: int,
        n_horizons: int,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_n = history_n
        self.aux_dim = aux_dim
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + aux_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, history_n + aux_dim]
        hist = x[:, : self.history_n].unsqueeze(-1)  # [B, T, 1]
        aux = x[:, self.history_n:]
        _, h = self.gru(hist)                        # h: [1, B, hidden]
        h = h.squeeze(0)                             # [B, hidden]
        z = torch.cat([h, aux], dim=-1)
        return self.head(z)


# ---------------------------------------------------------------------------
#  3. Block-Transformer — 1-layer self-attention over the history sequence
# ---------------------------------------------------------------------------

class BlockTransformer(nn.Module):
    def __init__(
        self,
        history_n: int,
        aux_dim: int,
        n_horizons: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_n = history_n
        self.aux_dim = aux_dim
        self.proj_in = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, history_n, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(d_model + aux_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hist = x[:, : self.history_n].unsqueeze(-1)  # [B, T, 1]
        aux = x[:, self.history_n:]
        z = self.proj_in(hist) + self.pos
        z = self.encoder(z)
        z = z.mean(dim=1)
        z = torch.cat([z, aux], dim=-1)
        return self.head(z)


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_model(
    name: str,
    input_dim: int,
    n_horizons: int,
    history_n: int = 32,
) -> nn.Module:
    aux_dim = input_dim - history_n
    if name == "tiny_mlp":
        return TinyMLP(input_dim=input_dim, n_horizons=n_horizons)
    if name == "block_rnn":
        return BlockRNN(history_n=history_n, aux_dim=aux_dim, n_horizons=n_horizons)
    if name == "block_transformer":
        return BlockTransformer(history_n=history_n, aux_dim=aux_dim, n_horizons=n_horizons)
    raise ValueError(f"unknown model: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
