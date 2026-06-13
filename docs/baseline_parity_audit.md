# Baseline parity audit (R6-W3)

This document discloses the implementation deltas between the in-tree
baselines in `seer/policy/baselines.py` and the upstream reference
artefacts they are named for. The R6 reviewer correctly noted that
these are policy-only reimplementations and not faithful end-to-end
port of the original repositories; this document quantifies the
deltas so an operator can decide whether the head-to-head ordering
shown in §6 (Lemma 3 reactive lag, `tab:multiturn-mechanism`,
`tab:multiturn-slo`) is faithful enough for their decision.

A full public-reference parity audit (LMCache, CacheGen, the upstream
InfiniGen repository) is the ATC-track follow-up captured in
`todo_atc.md`.

## What our in-tree baselines DO replicate

All four attention-tracking baselines implement the published scoring
rule, on top of the same block-stats input that our LAP-driven
SEERPolicy consumes (so the framework's `tab:reactive-lag`
class-separation result is implementation-independent and stays
qualitatively the same under public-reference parity).

| Policy | Reference | Faithful scoring rule | File path |
| --- | --- | --- | --- |
| **StreamingLLM** | Xiao et al. ICLR 2024 | Yes — attention sink (4 blocks) + sliding window | `seer/policy/baselines.py:31` |
| **H2O** | Zhang et al. NeurIPS 2023 | Yes — cumulative attention "heavy hitter" + recent | `seer/policy/baselines.py:55` |
| **SnapKV** | Li et al. NeurIPS 2024 | Yes — prefill-time top-K frozen through decode | `seer/policy/baselines.py:90` |
| **Quest** | Tang et al. ICML 2024 | Approximate — block-level top-K (one block ≈ one page at 32-token block_size) | `seer/policy/baselines.py:120` |

## What our in-tree baselines DO NOT replicate

The following implementation aspects of the upstream repositories are
**not** replicated and the head-to-head ordering should be read with
this in mind:

### StreamingLLM (Xiao et al. 2024)
- **Position-encoded attention**: upstream uses a re-anchored RoPE
  positional encoding when blocks are evicted; in-tree uses raw
  block IDs without re-anchoring. This means `tab:multiturn-slo`
  chat-miss for streaming is upper-bounded by what the upstream
  fork would deliver; the relative class-separation
  `streaming τ=∞*` vs `seer τ=3` finding is robust to this delta.
- **Attention sink count**: in-tree default `sink=4`; upstream
  paper uses `sink=4` (matches).
- **Window size**: in-tree default `window=64`; upstream defaults
  vary by model (we use the LLaMA-2-7B default).

### H2O (Zhang et al. 2023)
- **Scoring**: in-tree uses `sum(attn_history)` over the rolling
  history window; upstream computes cumulative attention from
  step 0 to current step (longer horizon). For the 1000-step
  reactive-lag witness this is a measurable difference; for the
  multi-turn chat workload in `tab:multiturn-slo` the upstream
  formulation has slightly higher recall on heavy hitters.
- **Heavy-hitter fraction**: in-tree default `hh_frac=0.5`;
  upstream paper sweeps `{0.25, 0.5, 0.75}` and selects per-model.
  We use the median.
- **Eviction granularity**: upstream is per-token; ours is
  per-block (block_size=32 tokens). This makes our H2O reproduction
  coarser; the upstream would have higher fidelity at the same
  budget but the gap is consistent with the H2O persistence AUC
  $0.89$ on e1 ($+0.12$ vs raw-attn, $-0.05$ vs LAP).

### SnapKV (Li et al. 2024)
- **Local window**: in-tree freezes the prefill-time top-K and only
  appends newly generated blocks; upstream maintains a sliding
  "local observation window" of ~32 most-recent tokens that
  re-scores the prefill top-K. The in-tree variant is more
  conservative (no re-scoring), which makes it deterministically
  worse than upstream at the same budget on long-decode workloads
  but does not affect the qualitative `tab:reactive-lag`
  τ-ordering.
- **Pooling**: upstream uses average-pool / max-pool aggregation
  over heads to compute the prefill score; in-tree uses the
  `attn_score_now` from our trace, which is already
  head-aggregated by the LAP feature extractor.

### Quest (Tang et al. 2024)
- **Page-level top-K**: upstream uses per-page max-attention as
  the page's score; in-tree uses one-block-per-page at
  `block_size=32`. At 32-token blocks and the upstream's typical
  16-token page size, our reproduction is 2× coarser. The
  ordering vs other attention-tracking baselines is preserved.
- **`recent_floor`**: in-tree default $4$; upstream typically
  $8$--$32$.

### InfiniGen (Lee et al. 2024)
- **NOT a policy reimplementation** — InfiniGen appears only as a
  trace-level surrogate in `experiments/eD_adversarial/`,
  specifically a synthetic "miss-then-prefetch" workload that
  matches the public InfiniGen trace traces' attention drift
  profile. No part of the InfiniGen runtime (CPU-offload, MagicPIG
  signing, etc.) is reproduced. The InfiniGen comparison in §6
  is therefore not a system head-to-head and is flagged as
  surrogate-only in the paper text.

## Where the parity gap matters

The framework's load-bearing claims (C1: probabilistic deadline-miss
bound + sizing rule under deployed ε; C2: MBPET-bounded LAP at
P99.9 ≤ 33.8µs) do **not** depend on baseline-fidelity. They consume
measured ε(φ) curves (R6 operator-default: correlation-aware
Corollary 2'' at $\bar\rho_\mathrm{FN}=0.08$) and TRT-measured
forward-pass latencies. The framework's deployable envelope —
$3/20$ operator-actionable cells under LAP, $1/20$ under raw-attn —
is calibrated against measured LAP inputs, not against baseline
implementations.

The parity gap **does** affect:

1. The reactive-lag τ values in `tab:reactive-lag` for streaming,
   H2O, raw-attn (the *ordering* is robust: attention-agnostic
   class diverges; attention-tracking class is bounded).
2. The patched-fork end-to-end chat-miss comparison in
   `tab:multiturn-slo` (`6-way tie`): with public-reference H2O
   we expect H2O to perform similarly or slightly better on
   multi-turn cache reuse; the 6-way tie is robust because the
   patch does not bind decode-step latency in the prefix-cache-free
   regime exercised (independent of baseline implementation).

## Follow-up (todo_atc.md)

Public-reference baseline integration is a 4-6 week sprint per
baseline:

- **H2O public reference**: clone `https://github.com/FMInference/H2O`,
  port their attention-sink + sliding-window + heavy-hitter
  selector to our vLLM-fork connector path, validate
  `chat-miss(D)` parity within ±2% on the headline workload.
- **SnapKV public reference**: clone `https://github.com/FasterDecoding/SnapKV`,
  port their local-observation-window re-scoring; validate
  `multi-turn chat-miss` parity on ShareGPT.
- **Quest public reference**: clone `https://github.com/mit-han-lab/Quest`,
  port their per-page max-attention scoring at upstream
  page_size=16.
- **InfiniGen**: clone `https://github.com/snu-comparch/InfiniGen`,
  reproduce their CPU-DRAM offload substrate; cross-validate the
  trace-level surrogate currently used in eD.

These are tracked as `A4 (baseline-parity)` in
[`todo_atc.md`](../todo_atc.md).
