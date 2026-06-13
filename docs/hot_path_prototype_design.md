# SEER hot-path eviction prototype — design doc

> R31 advisor: cycle-2 (ATC 2027) prototype scaffold. RTSS 2026 submission
> remains "diagnostic-only"; this document is the design contract that
> the Python decision router + the future vLLM patch (`vllm_patches/
> 0002-hot-path-allocation-hook.patch`) implement against. **No prose
> in this file should be cited by the RTSS paper.** The paper's
> "diagnosed non-binding" framing remains correct until A2 lands.

## Why the current hook does not bind

The R10-era patch (`0001-add-eviction-order-hook.patch`) attaches at
`KVCacheManager.free` — i.e., when a request *terminates*. The hook
sees the request's blocks after the decode is finished and reorders
vLLM's free queue for use by future requests. This means:

1. The blocks the current decode step actually reads were allocated
   before any SEER decision could fire.
2. Re-ordering the free queue only matters for **subsequent**
   requests that hit the prefix cache; in the prefix-cache-off
   regime SEER evaluates under, the reorder is a no-op for end-to-end
   chat-miss.
3. The §VI.D 6-way chat-miss tie (`tab:multiturn-slo`) is therefore
   a **structural** outcome under this hook, not a SEER-policy
   defect.

## Three candidate hot-path bindings

The advisor's `todo_atc.md` enumerates A1 / A2 / A3:

- **A1** (`BlockPool.allocate` hook): preempt cold blocks at step `t`
  if the predictor says block `b` will not be hit at step `t+1`.
  Largest blast radius; breaks "connector is opt-in"; requires a
  deeper patch and a deferred-allocation path.
- **A2** (`get_num_new_matched_tokens` + partial-materialisation
  hook): on a cross-prompt prefix-cache hit, choose **which** subset
  of matched prefix blocks to actually materialise to HBM. Smallest
  patch that puts the decision on the hot path. ✓ **chosen for the
  prototype.**
- **A3** (disaggregated prefill/decode handoff): use the existing
  eviction-order hook to decide which blocks survive the prefill→decode
  handoff. Most aligned with the existing patch but requires a
  disaggregated deployment.

## A2 design contract

### vLLM-side patch (~80 LoC)

Hook location: `vllm/v1/core/kv_cache_manager.py`, function
`get_computed_blocks` / `get_num_new_matched_tokens`.

```
# pseudo-diff
def get_computed_blocks(self, request, ...):
    matched_blocks, n_tokens = self._compute_prefix_cache_hit(request)
    if self._seer_connector is not None and len(matched_blocks) > 0:
        order = self._seer_connector.partial_materialisation_order(
            request_id=request.request_id,
            matched_blocks=[b.block_id for b in matched_blocks],
            slack_budget_us=self._deadline_slack_us(request),
        )
        # Truncate to what the slack budget allows.
        matched_blocks = [matched_blocks[i] for i in order
                          if i < len(matched_blocks)]
    return matched_blocks, n_tokens
```

The Python decision router (see below) returns an ordered list
of indices into `matched_blocks`; vLLM truncates to fit within the
predicted slack budget. The truncated suffix is rebuilt from cold
storage on the decode path, which **is** the hot path — chat-miss
becomes a function of the SEER ordering instead of vLLM's default
LRU.

### Python-side decision router

`seer/integration/hot_path_router.py` defines:

```python
class HotPathRouter:
    def partial_materialisation_order(
        self,
        request_id: str,
        matched_blocks: list[int],
        slack_budget_us: float,
    ) -> list[int]:
        """Return the SEER-ranked indices into matched_blocks.

        Indices earlier in the list will be materialised first;
        vLLM truncates at the point where cumulative materialisation
        cost would exceed slack_budget_us.
        """
```

Backed by `seer.lap.LAP` for the next-step attention prediction and
`seer.timing.schedulability` for the slack budget conversion. CPU-
mockable for tests (no torch / GPU needed in unit tests).

### Slack budget derivation

```
slack_us = deadline_us - (forward_us + lap_us + attn_us + ffn_us)
io_budget_us = slack_us - safety_margin (default 200us == LAP-WCET)
```

Then `min(len(matched_blocks),
floor(io_budget_us / ell_bar_us))` blocks are kept (in SEER order),
the rest deferred to cold-tier with on-demand fetch.

## Decision-router unit-test contract

`tests/test_hot_path_router.py` (CPU-only):

1. With `slack_budget_us=0`, return empty.
2. With infinite slack, return identity permutation (no SEER
   reordering needed).
3. Given a fake LAP that maps block `i` → score `-i` (later blocks
   matter more), router output reverses the input order.
4. Given a fake LAP that maps every block to a constant score,
   router output preserves vLLM's default LRU order (stable sort).
5. Slack derivation: deadline `100ms`, forward `60ms`, lap `0.3ms`
   ⇒ `slack ≈ 39.5ms`; with `ell_bar=200us` the router keeps at
   most `~197` blocks.

These tests are the **contract** the future vLLM patch must
preserve.

## Pre-flight checklist for cycle-2 ATC submission

After the A2 patch lands and the decision router is wired:

- [ ] `make reproduce-full` regenerates ≥1 cell where SEER's
  chat-miss is ≥30% lower than the best of
  {H2O, SnapKV, Quest, Streaming} at $D \in \{50, 100, 200\}$ ms,
  `n_seeds≥5`, `p<0.01`.
- [ ] Mechanism counters confirm policy-distinct routing on the
  hot path (not just at request-termination free queue).
- [ ] `§VI.D` text retracts the "diagnosed non-binding" framing
  and reports the policy-distinct end-to-end delta.
- [ ] Patch upstreamed to vLLM (or at least: PR opened) before
  ATC submission.

## Out of scope (deferred to cycle-3+)

- A1 (full hot-path block-allocation rewrite).
- A3 (disaggregated prefill/decode).
- Real NVMe (cgroup + `io_uring`).
- 70B + TP-4.
- Multi-tenant production-trace replay.
