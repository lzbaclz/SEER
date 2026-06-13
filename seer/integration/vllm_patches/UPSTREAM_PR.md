# Upstream vLLM PR draft: `KVConnector_V1.get_eviction_order` hook

This document is the draft cover letter for submitting
`0001-add-eviction-order-hook.patch` upstream to `vllm-project/vllm`.

## Title

`[Feature][KVConnector] Add optional get_eviction_order hook for policy-driven free-queue ordering`

## TL;DR

A 31-line opt-in hook on `KVCacheManager.free()` that lets a registered
`KVConnector_V1` reorder a request's blocks before they enter
`BlockPool.free_block_queue`. Connectors that do not implement the hook
see vLLM's existing reverse-order heuristic; existing connectors
(`Mooncake`, `LMCache`, `NixlConnector`, `OffloadingConnector`, the
in-tree `SharedStorageConnector`) are unaffected.

## Motivation

`KVConnectorBase_V1` currently has hooks for KV-cache transfer
(`start_load_kv`, `save_kv_layer`, `wait_for_save`, etc.) but no hook
that lets a connector influence vLLM's *eviction* order at request
termination. `KVCacheManager.free()` orders the freed blocks with a
fixed reverse-order heuristic (when caching is enabled) before pushing
them to `BlockPool.free_block_queue`. The queue is FIFO from the
front; the order of blocks at `free()` time therefore decides which
blocks survive longest for prefix-cache hits by subsequent requests.

A connector backed by a learned attention-demand predictor or a
score-based scheme can predict which blocks are most likely to be
reused by later requests and place them at the back of the free
queue so they survive longer for prefix-cache hits. Without this
hook the connector can save blocks to its host pool and reload them
via `start_load_kv`, but cannot influence which physical HBM blocks
vLLM keeps allocated. Modifying `BlockPool._maybe_evict_cached_block`
or the `FreeKVCacheBlockQueue` priority requires deeper refactors and
breaks the "connector is opt-in" invariant; the eviction-order
decision at `free()` time is the minimum-touch insertion point.

## What the patch changes

```
vllm/v1/core/kv_cache_manager.py | 33 +++++++++++++++++++++++++++++++--
1 file changed, 31 insertions(+), 2 deletions(-)
```

1. `KVCacheManager.free()` materialises the existing reverse-ordered
   list into a `list` rather than an `Iterable` so the connector hook
   can iterate it without consuming it.
2. If `has_kv_transfer_group() and is_v1_kv_transfer_group()` and the
   registered connector exposes `get_eviction_order`, the manager
   calls `_conn.get_eviction_order(request, list(ordered_blocks))`
   and uses the returned ordering iff it is a permutation of the
   input (length-checked).
3. Any exception in the hook falls through `try/except` to the
   default ordering, so a buggy connector cannot break vLLM's `free()`
   path.

## Behavioural notes for reviewers

- The hook is purely a *reordering* of blocks the request is already
  freeing. It does not extend the block lifetime, does not allocate,
  and does not free anything not already being freed.
- The hook does not see blocks from other requests --- a connector
  cannot use it to globally rearrange the free queue, only to choose
  the eviction priority of one request's blocks at the moment those
  blocks become free.
- Backward compatibility: existing in-tree connectors do not implement
  `get_eviction_order` and see no behavioural change. The patch only
  adds a new dispatch path; nothing in the default code path is
  semantically modified.
- Front-of-list = evicted first, back-of-list = survives longest;
  this matches the existing reverse-order intent (tail blocks evicted
  first) so a no-op connector that returns the input untouched is
  equivalent to today's behaviour.

## Reference implementation

`SeerKVConnector.get_eviction_order`
(`seer/integration/vllm_connector.py`, ~120 LOC) routes the
per-request reordering through a policy planner (LAP, H2O, Streaming,
or empty) and caches the LAP decision per `request_id` so the hot
path is `O(1)` amortised. The planner branches:

- `SEER` → LAP attention-demand prediction; high-confidence blocks to
  the back.
- `H2O` → raw-attention magnitude score; low-magnitude to the front.
- `Streaming` → recency window; sink + window stays at the back.
- `full` → empty decision; falls back to vLLM's default ordering.

## Measured cost

On the SEER reference run (3-seed Llama-2-7B + 1×A100 SXM4-80GB,
50 prompts/cell, `--disable_prefix_cache`,
`gpu_memory_utilization=0.20`, `P99 TPOT=200ms` SLO), the hook fires
55 times/seed and reorders 1,466 (Streaming) / 1,470 (H2O) / 1,474
(SEER) blocks per seed on average. Per-policy routing is differentiated
(Streaming: 100% to back; H2O: 23% to back / 77% to front; SEER:
89% to back / 11% to front). No measurable per-step latency tax: the
patched-fork wall time matches the unpatched baseline within ±1s on
~60s runs.

## Tests

`tests/v1/core/test_kv_cache_manager_eviction_hook.py` (added in
commit 6ed6b84) covers the patch with 4 mock-based unit tests:

1. **Happy path** (`test_happy_path_hook_reorders_free_queue`): a
   stub connector returning a known ordering causes
   `BlockPool.free_blocks` to receive that exact ordering.
2. **Fault tolerance** (`test_hook_exception_falls_back_to_default`):
   a stub connector raising in `get_eviction_order` falls back to
   vLLM's default reverse-order without raising upward.
3. **Backward compat** (`test_no_hook_implementation_leaves_default_unchanged`):
   a connector without `get_eviction_order` sees the existing
   reverse-order behaviour unchanged.
4. **Length-check fallback** (`test_hook_returning_wrong_length_falls_back`):
   a connector returning an ordering of the wrong length falls back
   to default (length-checked in the patch).

All 4 tests pass on the fork; they run independently of CUDA (mock
KVCacheBlock + mock connector).

## Patch file and fork

- `0001-add-eviction-order-hook.patch` — apply with `git am`.
- Fork: `vllm-project/vllm@v0.8.5.post1` on branch
  `seer-eviction-hook`. The patch was authored against the
  `v0.8.5.post1` release tag and applies cleanly to that point.
- Reproducer: `bash install_fork.sh /path/to/fork /path/to/site-packages`.

## Status

This is a draft cover letter. The SEER paper cites this patch as
evidence that the framework's eviction-policy formalisation is wired
through to a real, upstream-submittable vLLM source change. The PR
will be opened against `vllm-project/vllm:main` once the camera-ready
deadline passes; until then this file documents the intended
upstream change.
