# vLLM source patches

This directory contains the **real vLLM source patches** that SEER's framework
claims need at the runtime layer. Each patch is a unified `git format-patch`
diff against a pinned upstream vLLM commit and is meant to be (i) applied to a
local vLLM fork for end-to-end measurement, and (ii) submitted upstream as a
standalone PR proposing an extension to the `KVConnector_V1` protocol.

## 0001-add-eviction-order-hook.patch

**Target**: `vllm/v1/core/kv_cache_manager.py:KVCacheManager.free()`,
upstream commit `3015d56` (release tag `v0.8.5.post1`).

**Diff size**: 31 insertions, 2 deletions.

**Purpose**: extends `KVConnector_V1` with an optional `get_eviction_order`
hook. When a request is freed, the manager asks the registered connector
to return a policy-driven reordering of the request's blocks. The front of
the returned list becomes the next candidate for eviction (vLLM's free
queue is FIFO from the front); the back survives longer in the queue and
is more likely to be reused via prefix-cache hits by subsequent requests.

When no connector is registered, or the connector does not implement
`get_eviction_order`, or it raises, the fallback is vLLM's existing
default ordering (reverse-order under `enable_caching=True`).

**Why this is the right hook**:
- It is the smallest possible API surface that lets a learned attention-demand
  predictor influence vLLM's eviction order.
- It runs only at request termination (cheap: amortised across all decode
  steps that benefit from policy-driven cache reuse).
- It is fully backward-compatible: existing connectors without
  `get_eviction_order` see no behavioural change.
- It is the only existing per-request, per-block decision point in vLLM's
  v1 BlockPool that already accepts a caller-supplied ordering (see
  `BlockPool.free_blocks`'s docstring: "The blocks should be ordered by
  their eviction priority, where the first block will be evicted first.").

**SEER use site**: `seer/integration/vllm_connector.py:SeerKVConnector.get_eviction_order`
implements the hook by routing through the registered policy's
`planner.plan(scores, request_id)` so that SEER's LAP, H2O's raw-attention,
and Streaming's recency window each produce a policy-specific ordering.

## How to apply (local fork)

```bash
# 1. Clone vLLM at the pinned commit.
git clone --depth 1 --branch v0.8.5.post1 \
    https://github.com/vllm-project/vllm.git /path/to/vllm-seer-fork
cd /path/to/vllm-seer-fork
git checkout -b seer-eviction-hook

# 2. Apply the patch.
git am /path/to/SEER/seer/integration/vllm_patches/0001-add-eviction-order-hook.patch

# 3. Run SEER with PYTHONPATH override so the fork's python files take
#    precedence over the conda env's site-packages vllm (compiled .so
#    files are symlinked from site-packages into the fork; see
#    seer/integration/vllm_patches/install_fork.sh).
PYTHONPATH=/path/to/vllm-seer-fork:$PYTHONPATH \
    python -m experiments.eF_mixed_slo.driver ...
```

## How this would be submitted upstream

The patch is structured as a minimal, opt-in protocol extension: no existing
behaviour changes unless a connector implements the new hook. A real upstream
PR would:

1. Add `get_eviction_order(request, blocks)` to `KVConnectorBase_V1`'s
   abstract API (as an optional method, default `None`).
2. Apply this exact `KVCacheManager.free()` change.
3. Document the contract in the connector base class docstring.
4. Add a unit test that registers a stub connector returning a known order
   and verifies `BlockPool.free_blocks` receives the same order.

The patch in this directory is steps 2 only (the minimal runtime path).
Steps 1, 3, 4 are paperwork-only and would be added for an actual PR
submission.
