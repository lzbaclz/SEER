"""Unit-level smoke test for the Phase 3 xfer round-trip.

Constructs a synthetic kv_layer with known K and V data, calls
``save_kv_layer`` with a slot-mapping that touches blocks {3, 7, 11},
then constructs a fake ``forward_context`` and a fake request state
whose ``last_decision.block_ids`` contains the same slot ids,
calls ``start_load_kv``, and asserts that vLLM's destination
``kv_cache`` tensor now contains the saved data (i.e. the save->load
round trip is functionally correct).

This is the unit-level certificate that the save_index + aligned-key
fix is correct in isolation. The integration-level vLLM sweep
remains at n_load=0 because the scheduler-side decision uses a
logical-block-index namespace [0, n_blocks_seen) that does NOT
match the worker-side save_index keys (real cache slot ids from
slot_mapping); bridging that gap requires either reading vLLM's
block_table from forward_context or a workload that exercises
prefix caching with cache-slot-id-aware decisions. That is the
remaining ATC follow-up scoped honestly in the paper.
"""
from __future__ import annotations

import os
import sys
import types

import torch

os.environ.pop("SEER_DISABLE_XFER", None)  # xfer ENABLED
os.environ["SEER_USE_KV_MAGNITUDE"] = "1"

from seer.integration.vllm_connector import (  # noqa: E402
    PrefetchDecision,
    _RequestState,
    make_seer_connector_class,
)


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
    # Source kv_layer with known per-block data: K[b] = b, V[b] = 100+b.
    kv_layer = torch.zeros(
        (2, n_blocks_total, block_size, n_kv_heads, head_dim),
        dtype=torch.float16, device=device,
    )
    touched = [3, 7, 11]
    for b in touched:
        kv_layer[0, b] = float(b)
        kv_layer[1, b] = 100.0 + float(b)

    slot_mapping = torch.tensor(
        [b * block_size + j for b in touched for j in range(block_size)],
        dtype=torch.int64, device=device,
    )
    attn_metadata = types.SimpleNamespace(slot_mapping=slot_mapping)

    inst.save_kv_layer("layer.0", kv_layer, attn_metadata)

    # Verify host_pool got the K and V batched buffers. We do NOT
    # read individual host_pool elements here because doing so
    # triggers a sync-on-pinned-CPU-read that interleaves with the
    # xfer stream and (on PyTorch 2.6 + CUDA 12.4) breaks the
    # FIRST subsequent ``non_blocking=True`` HTOD copy on the same
    # stream. We verify presence + correctness of the
    # ``_save_index`` mapping instead.
    assert ("layer.0", "K") in inst._host_pool, "K host not allocated"
    assert ("layer.0", "V") in inst._host_pool, "V host not allocated"
    save_idx = inst._save_index
    for b in touched:
        assert ("layer.0", b) in save_idx, \
            f"save_index missing entry for slot {b}"
    print(f"save_index entries: "
          f"{sorted([(b, save_idx[('layer.0', b)]) for b in touched])}")

    # Build a forward_context for start_load_kv. Destination kv_cache
    # is a fresh zero-tensor; we expect start_load_kv to fill the
    # rows for slots 3, 7, 11 with the saved K and V data.
    dst_kv_cache = torch.zeros_like(kv_layer)
    forward_context = types.SimpleNamespace(
        layer_name="layer.0",
        kv_cache=dst_kv_cache,
    )

    # Inject a request state with last_decision.block_ids = touched.
    # This is the scheduler-side decision content vLLM would otherwise
    # build; we set it directly to bypass the logical-block-index
    # namespace mismatch. We also remove the worker-side
    # ``layer:layer.0`` side-channel state that save_kv_layer's
    # planner-fire creates with synthetic block-ids [0, 1] (the K/V
    # split-index artefact, not real slot ids); those entries would
    # produce bookkeeping ``n_load_miss`` events on each
    # start_load_kv that are not load-bearing for the round-trip
    # verification here.
    inst._states = {}
    inst._states["test_req"] = _RequestState(
        request_id="test_req",
        last_decision=PrefetchDecision(
            block_ids=list(touched),
            horizon_steps=4,
            max_p=1.0,
        ),
    )

    # Drain save before load. The full sync prevents the first
    # post-save ``non_blocking=True`` HTOD copy on the xfer stream
    # from returning zeros (reproducer on PyTorch 2.6 + CUDA 12.4).
    if device == "cuda":
        torch.cuda.synchronize()
    inst.start_load_kv(forward_context)
    inst.wait_for_layer_load("layer.0")
    if device == "cuda":
        torch.cuda.synchronize()

    # Verify the destination has the saved K and V data for each
    # touched slot.
    for b in touched:
        k_actual = float(dst_kv_cache[0, b, 0, 0, 0].item())
        v_actual = float(dst_kv_cache[1, b, 0, 0, 0].item())
        k_expected = float(b)
        v_expected = 100.0 + float(b)
        assert k_actual == k_expected, \
            f"K slot {b}: expected {k_expected}, got {k_actual}"
        assert v_actual == v_expected, \
            f"V slot {b}: expected {v_expected}, got {v_actual}"
    print("round-trip verified: K and V data recovered for slots 3, 7, 11 ✓")

    stats = inst._xfer_stats
    print(f"n_save={stats['n_save']}, n_load={stats['n_load']}, "
          f"n_load_miss={stats['n_load_miss']}")
    # The save_kv_layer planner-fire path creates a side-channel
    # state keyed by ``"layer:layer.0"`` whose last_decision has
    # block_ids unrelated to the saved slots; those produce
    # bookkeeping ``n_load_miss`` events on each start_load_kv.
    # We only assert the test_req's 3 hits made it through.
    assert stats["n_save"] >= 3, f"expected >= 3 saves, got {stats['n_save']}"
    assert stats["n_load"] >= 3, f"expected >= 3 loads, got {stats['n_load']}"
    print("OK: Phase 3 xfer round-trip verified at unit level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
