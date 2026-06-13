"""Smoke-test the Phase 2.5L correctness fix (second iteration, minimal-cost).

Tests that ``save_kv_layer``'s K-magnitude path:
  1. Updates the magnitude entry for the slot indicated by
     ``attn_metadata.slot_mapping.max() // block_size`` (the most
     recently-written block id), not the wrong ``n_blk-1`` slot
     of the static pool that the prior buggy code refreshed.
  2. Leaves all other slots' magnitudes at zero (we only update one
     slot per call by design — the minimum-cost path).
  3. Increments ``_n_attn_driven_decisions``.
  4. Builds a decision whose ``block_ids`` stays small (length 2,
     the same as the prior code) so vLLM's
     ``get_num_new_matched_tokens`` does not over-report
     cached tokens.

The first-iteration fix (commit-staged but rolled back) updated ALL
touched blocks per call and ranked by GPU topk — functionally correct
but added enough kernel launches + syncs that per-step P50 went from
31\\,ms to 80\\,ms. This second iteration trades coverage for cost:
one slot per call, eventual convergence over ~32 calls (one per
layer + reasonable step count).
"""
from __future__ import annotations

import os
import sys
import types

import torch

os.environ["SEER_DISABLE_XFER"] = "1"
os.environ["SEER_USE_KV_MAGNITUDE"] = "1"

from seer.integration.vllm_connector import (  # noqa: E402
    make_seer_connector_class,
)
from seer.integration.vllm_connector import _RequestState  # noqa: E402,F401


def _fake_lap():
    class _Stub:
        def __call__(self, feats):
            import numpy as np
            return np.zeros((feats.shape[0], 4), dtype="float32")
    return _Stub()


def _build_cfg():
    cfg = types.SimpleNamespace()
    cfg.parallel_config = types.SimpleNamespace(tensor_parallel_size=1)
    cfg.kv_transfer_config = types.SimpleNamespace(
        kv_connector_extra_config={
            "policy": "seer",
            "budget_blocks": 8,
            "horizon_steps": 4,
            "p_threshold": 0.0,
        }
    )
    return cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Conn = make_seer_connector_class()
    inst = Conn(
        vllm_config=_build_cfg(),
        role=None,
        kv_cache_config=None,
        lap=_fake_lap(),
        budget_blocks=8,
    )
    inst.planner.budget_blocks = 8

    n_blocks_total = 32
    block_size = 16
    n_kv_heads = 8
    head_dim = 64
    kv_layer = torch.zeros(
        (2, n_blocks_total, block_size, n_kv_heads, head_dim),
        dtype=torch.float16, device=device,
    )
    # Write distinct K data into block 21 (the largest touched).
    kv_layer[0, 21] = torch.randn(
        block_size, n_kv_heads, head_dim,
        dtype=torch.float16, device=device,
    ) * 5.0  # explicitly large so the magnitude is unambiguously > 0
    # Slot mapping covers block 21 (slots 336..351).
    slot_mapping = torch.tensor(
        [21 * block_size + j for j in range(block_size)],
        dtype=torch.int64, device=device,
    )
    attn_metadata = types.SimpleNamespace(slot_mapping=slot_mapping)

    inst.save_kv_layer("layer.0", kv_layer, attn_metadata)

    cache = getattr(inst, "_kv_mag_cache", None)
    assert cache is not None and "layer.0" in cache, "cache not init'd"
    entry = cache["layer.0"]
    print(f"cache entry: n_blocks={entry['n_blocks']}, "
          f"mag.shape={tuple(entry['mag'].shape)}, "
          f"touched_count={entry.get('touched_count')}")

    # Only block 21 should have a non-zero magnitude.
    for b in range(n_blocks_total):
        mag = float(entry["mag"][b].item())
        if b == 21:
            assert mag > 0, f"block 21 mag should be > 0, got {mag}"
        else:
            assert mag == 0.0, f"block {b} mag should be 0, got {mag}"
    print(f"block 21 mag = {float(entry['mag'][21].item()):.4f} (>0 ✓)")
    print(f"all other slots magnitude == 0 ✓")

    # Decision is small (2 entries from bids=[0,1]).
    state = inst._states.get("layer:layer.0")
    assert state is not None and state.last_decision is not None
    assert len(state.last_decision.block_ids) == 2, \
        f"decision should have 2 entries (bids=[0,1]), got "\
        f"{len(state.last_decision.block_ids)}"
    print(f"last_decision.block_ids = {state.last_decision.block_ids}")

    assert getattr(inst, "_n_attn_driven_decisions", 0) >= 1
    print("OK: minimum-cost Phase 2.5L slot-correct K-mag path verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
