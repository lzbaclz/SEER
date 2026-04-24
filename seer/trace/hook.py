"""Attention hooks for HuggingFace `transformers` generate().

Collects per-block attention statistics each time an attention module runs,
buffers them in memory, and flushes to parquet at end of each request.

The hook averages attention weights over:
  - the query dimension (decode usually Q=1; for prefill we mean-pool)
  - heads within a GQA head-group (defaults to num_key_value_heads)
  - tokens inside a 32-token block

This gives one scalar attn_score per (layer, head_group, block, step).
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch

from seer.trace.schema import BLOCK_SIZE, HORIZONS, arrow_schema, compute_top_k

if TYPE_CHECKING:
    import pandas as pd


class AttentionTracer:
    """Per-process trace collector.

    Usage
    -----
    >>> tracer = AttentionTracer(n_layers=32, n_head_groups=8)
    >>> tracer.attach(model)
    >>> tracer.begin_request(req_id)
    >>> model.generate(..., output_attentions=True, return_dict_in_generate=True)
    >>> tracer.finalize_request()
    >>> tracer.to_parquet("req_0001.parquet")
    >>> tracer.clear()
    """

    def __init__(
        self,
        n_layers: int,
        n_head_groups: int,
        block_size: int = BLOCK_SIZE,
    ):
        self.n_layers = n_layers
        self.n_head_groups = n_head_groups
        self.block_size = block_size

        self.current_request_id: int = 0
        self.current_step: int = 0
        self.records: list[dict] = []
        self._handles: list = []

    # ------------------------------------------------------------------
    #  Hook attach / detach
    # ------------------------------------------------------------------

    def attach(self, model) -> None:
        """Register forward hooks on every self-attention module."""
        for layer_id, layer in enumerate(self._find_attention_layers(model)):
            handle = layer.register_forward_hook(self._make_hook(layer_id))
            self._handles.append(handle)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @staticmethod
    def _find_attention_layers(model) -> list:
        """Heuristic: any module whose registered name ends in 'self_attn'.

        Works for Llama, Mistral, Qwen2, etc. Override this for other
        architectures (e.g. GPT-NeoX uses 'attention').
        """
        layers = []
        for name, module in model.named_modules():
            if name.endswith("self_attn"):
                layers.append(module)
        if not layers:  # fallback
            for name, module in model.named_modules():
                if name.endswith("attention"):
                    layers.append(module)
        return layers

    # ------------------------------------------------------------------
    #  Per-step control
    # ------------------------------------------------------------------

    def begin_request(self, request_id: int) -> None:
        self.current_request_id = request_id
        self.current_step = 0

    def advance_step(self) -> None:
        self.current_step += 1

    # ------------------------------------------------------------------
    #  The hook itself
    # ------------------------------------------------------------------

    def _make_hook(self, layer_id: int):
        def hook(module, inputs, outputs):
            attn_weights = self._extract_attn(outputs)
            if attn_weights is None:
                return
            with torch.no_grad():
                self._record(layer_id, attn_weights)
        return hook

    @staticmethod
    def _extract_attn(outputs):
        """HF attention modules return (attn_output, attn_weights, past_kv).
        We pick attn_weights if present and 4-D."""
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            w = outputs[1]
            if torch.is_tensor(w) and w.ndim == 4:
                return w
        return None

    def _record(self, layer_id: int, attn_weights: torch.Tensor) -> None:
        # attn_weights: [batch, n_heads, q_len, k_len]
        aw = attn_weights.detach().float()
        B, H, Q, K = aw.shape
        # mean over query (handles both prefill Q>1 and decode Q=1)
        aw = aw.mean(dim=2)                             # [B, H, K]
        # fold heads into head_groups (GQA-aware)
        group_size = max(1, H // self.n_head_groups)
        G = H // group_size
        aw = aw[:, : G * group_size].reshape(B, G, group_size, K).mean(dim=2)  # [B, G, K]
        # block-pool along K
        n_blocks = (K + self.block_size - 1) // self.block_size
        # efficient block mean
        pad_len = n_blocks * self.block_size - K
        if pad_len > 0:
            aw = torch.nn.functional.pad(aw, (0, pad_len))
        aw = aw.reshape(B, G, n_blocks, self.block_size).mean(dim=3)  # [B, G, n_blocks]
        aw = aw.cpu().numpy()

        for b in range(B):
            for g in range(G):
                scores = aw[b, g]  # [n_blocks]
                for block_id in range(n_blocks):
                    self.records.append({
                        "request_id": int(self.current_request_id),
                        "layer_id": int(layer_id),
                        "head_group": int(g),
                        "block_id": int(block_id),
                        "block_start_token": int(block_id * self.block_size),
                        "step": int(self.current_step),
                        "attn_score": float(scores[block_id]),
                    })

    # ------------------------------------------------------------------
    #  Label computation (top-k now + future horizons)
    # ------------------------------------------------------------------

    def finalize_request(self) -> None:
        """Compute is_top_k + future_top_k_h{H} columns for the current req."""
        req_recs = [r for r in self.records if r["request_id"] == self.current_request_id]
        if not req_recs:
            return

        # --- 1. is_top_k: per (layer, head_group, step)
        by_group = defaultdict(list)
        for r in req_recs:
            by_group[(r["layer_id"], r["head_group"], r["step"])].append(r)
        for recs in by_group.values():
            k = compute_top_k(len(recs))
            idx_sorted = sorted(range(len(recs)), key=lambda i: -recs[i]["attn_score"])
            top_set = set(idx_sorted[:k])
            for i, r in enumerate(recs):
                r["is_top_k"] = 1 if i in top_set else 0

        # --- 2. future_top_k_h{H}: per (layer, head_group, block)
        by_block = defaultdict(list)
        for r in req_recs:
            by_block[(r["layer_id"], r["head_group"], r["block_id"])].append(r)
        for recs in by_block.values():
            recs.sort(key=lambda r: r["step"])
            for i, r in enumerate(recs):
                for h in HORIZONS:
                    fut = 0
                    for j in range(i + 1, min(i + 1 + h, len(recs))):
                        if recs[j]["is_top_k"]:
                            fut = 1
                            break
                    r[f"future_top_k_h{h}"] = fut

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        df = pd.DataFrame(self.records)
        # Ensure all columns exist
        for h in HORIZONS:
            col = f"future_top_k_h{h}"
            if col not in df.columns:
                df[col] = 0
        if "is_top_k" not in df.columns:
            df["is_top_k"] = 0
        cast = {
            "request_id": "int64",
            "layer_id": "int32",
            "head_group": "int32",
            "block_id": "int32",
            "block_start_token": "int32",
            "step": "int32",
            "attn_score": "float32",
            "is_top_k": "uint8",
        }
        cast.update({f"future_top_k_h{h}": "uint8" for h in HORIZONS})
        return df.astype(cast)

    def to_parquet(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_parquet(path, engine="pyarrow", index=False, schema=arrow_schema())

    def clear(self) -> None:
        self.records.clear()
