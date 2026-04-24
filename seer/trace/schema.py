"""Attention trace schema.

Each row is one (request, layer, head_group, block, step) tuple with
attention statistics aggregated over heads in the group.

Columns (parquet):
  request_id:          int64   — which prompt
  layer_id:            int32   — transformer layer index
  head_group:          int32   — GQA KV-head group id (== head_id for MHA)
  block_id:            int32   — 32-token block index (aligned with OrchFS
                                  SSD block at d_head=128, fp16)
  block_start_token:   int32   — first token position inside this block
  step:                int32   — decode step (0 == end of prefill)
  attn_score:          float32 — mean attn weight within block, across heads
  is_top_k:            uint8   — 1 if this block is in top-k at this step
  future_top_k_h{1,4,16,64}:  uint8 — label for each horizon H
                                      (1 iff this block is in top-k in any
                                      of the next H decode steps)

Top-k at a step is computed per (layer, head_group, step) with
  k = max(MIN_TOP_K, TOP_K_FRACTION * total_blocks_that_step).
"""
from dataclasses import dataclass

import pyarrow as pa

# -- Hyper-parameters of the trace representation ------------------------------

BLOCK_SIZE: int = 32        # tokens per KV-block; 32KB at d_head=128, fp16
HORIZONS: tuple = (1, 4, 16, 64)
TOP_K_FRACTION: float = 0.10
MIN_TOP_K: int = 32


def compute_top_k(num_blocks: int) -> int:
    """How many blocks count as 'top-k' for a given total."""
    return max(MIN_TOP_K, int(num_blocks * TOP_K_FRACTION))


@dataclass
class TraceRow:
    """Python-level view of one trace row. Parquet uses the arrow schema."""
    request_id: int
    layer_id: int
    head_group: int
    block_id: int
    block_start_token: int
    step: int
    attn_score: float
    is_top_k: int = 0
    future_top_k_h1: int = 0
    future_top_k_h4: int = 0
    future_top_k_h16: int = 0
    future_top_k_h64: int = 0


def arrow_schema() -> pa.Schema:
    """Canonical Arrow schema for the trace parquet files."""
    fields = [
        pa.field("request_id", pa.int64()),
        pa.field("layer_id", pa.int32()),
        pa.field("head_group", pa.int32()),
        pa.field("block_id", pa.int32()),
        pa.field("block_start_token", pa.int32()),
        pa.field("step", pa.int32()),
        pa.field("attn_score", pa.float32()),
        pa.field("is_top_k", pa.uint8()),
    ]
    for h in HORIZONS:
        fields.append(pa.field(f"future_top_k_h{h}", pa.uint8()))
    return pa.schema(fields)


def horizon_columns() -> list[str]:
    return [f"future_top_k_h{h}" for h in HORIZONS]
