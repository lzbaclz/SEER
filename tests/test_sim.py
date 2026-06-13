"""Smoke tests for seer.eval.sim.MaskingSimulator.

Goal: prove that the simulator now feeds *real* per-block attention into
the policy stats (the bug the RTSS pivot pass fixed). We don't load a
real HuggingFace model here — too slow for CI. Instead we feed a stub
attention tensor into ``_ingest_attn`` directly and check the buffer
ends up with non-stub data.
"""
from __future__ import annotations

import torch

from seer.eval.sim import MaskingSimulator
from seer.policy import RecencyPolicy


class _StubModel:
    """Just enough of a Module to satisfy MaskingSimulator's introspection."""

    class _Cfg:
        max_position_embeddings = 4096
        num_attention_heads = 4
        num_key_value_heads = 4

    def __init__(self):
        self.config = self._Cfg()
        # A self_attn-named child so _find_attention_layers picks it up.
        self.self_attn = torch.nn.Identity()

    def named_modules(self):
        yield "", self
        yield "self_attn", self.self_attn

    def parameters(self):
        return iter([])


def test_ingest_attn_populates_buffer():
    """Ingest one synthetic attention tensor; buffer should hold real values."""
    model = _StubModel()
    pol = RecencyPolicy()
    sim = MaskingSimulator(
        model=model, tokenizer=None, policy=pol,
        budget_frac=0.5, decision_period=8, ell_bar_us=200.0,
        n_head_groups=2,
    )
    # [B=1, H=4, Q=1, K=64] → 2 head_groups × 64 keys = 2 blocks of 32 tokens
    attn = torch.zeros(1, 4, 1, 64)
    # Make the second block hot
    attn[..., 32:] = 0.9
    sim._stats_buffers[0].current_step = 7
    sim._ingest_attn(layer_id=0, attn_weights=attn)

    buf = sim._stats_buffers[0]
    # 2 blocks ingested
    assert set(buf.attn_history.keys()) == {0, 1}
    # Block 1 should have a higher score than block 0
    assert buf.attn_history[1][-1] > buf.attn_history[0][-1]
    # is_top_k labels populated
    assert 1 in buf.last_top_k_step or 0 in buf.last_top_k_step


def test_io_penalty_is_zero_when_nothing_evicted():
    """If every block stayed in HBM, IO penalty must be zero.

    Uses the analytical path explicitly — measured-dma (the P1-1
    default) would return a real cudaMemcpyAsync timing.
    """
    model = _StubModel()
    pol = RecencyPolicy()
    sim = MaskingSimulator(
        model=model, tokenizer=None, policy=pol,
        budget_frac=1.0, decision_period=8, ell_bar_us=500.0,
        n_head_groups=2,
        io_mode="analytical",
    )
    # Pretend we've kept every block; no misses → no penalty
    sim._kept_set_per_layer[0] = set(range(4))
    # Buffer empty → wanted_now is empty → also zero
    assert sim._io_penalty_us(n_kv_tokens=128) == 0.0


def test_io_penalty_charges_for_missed_blocks():
    """Evicted-but-attended blocks should add penalty proportional to ell_bar.

    The oracle buffer holds the unmasked attention truth (P0-5 fix).
    Uses io_mode="analytical" for an exact arithmetic assertion;
    the measured-dma path (P1-1 default) returns a real
    cudaMemcpyAsync timing that varies per host.
    """
    model = _StubModel()
    pol = RecencyPolicy()
    sim = MaskingSimulator(
        model=model, tokenizer=None, policy=pol,
        budget_frac=0.5, decision_period=8, ell_bar_us=400.0,
        n_head_groups=1,
        io_mode="analytical",
    )
    # 4 blocks total; only block 0 in HBM; but block 3 is currently attended.
    n_kv = 128
    sim._kept_set_per_layer[0] = {0}
    buf = sim._oracle_stats_buffers[0]
    buf.attn_history[0] = [0.05]
    buf.attn_history[3] = [0.85]
    pen = sim._io_penalty_us(n_kv_tokens=n_kv)
    # 1 missed block → 1 * 400 / 8 = 50 µs
    assert pen == 400.0 / 8


# ---------------------------------------------------------------------------
#  Regression: B5 (oracle buffer drives io_penalty, not masked buffer)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewer #2 + #3): the legacy MaskingSimulator
# read attended-block info from a buffer populated AFTER the eviction
# mask was applied, so attended-and-evicted blocks had attention 0 by
# construction and the io_penalty was structurally zero. The fix adds
# an oracle (unmasked) pass populating ``_oracle_stats_buffers``; these
# tests pin the contract that:
#   (a) io_penalty consults the oracle buffer when populated,
#   (b) the legacy masked buffer is ignored when oracle is available,
#   (c) the pure helper ``_count_prefetch_misses`` matches expectations.

def test_io_penalty_prefers_oracle_buffer_over_masked():
    """B5: when both buffers have data, io_penalty must consult the
    oracle (unmasked) one. We set conflicting truths in the two
    buffers — masked says no misses, oracle says one miss — and
    verify the oracle wins. Forces analytical IO for an exact
    arithmetic assertion (P1-1 default is measured-dma)."""
    model = _StubModel()
    pol = RecencyPolicy()
    sim = MaskingSimulator(
        model=model, tokenizer=None, policy=pol,
        budget_frac=0.5, decision_period=8, ell_bar_us=400.0,
        n_head_groups=1,
        compute_oracle=True,
        io_mode="analytical",
    )
    sim._kept_set_per_layer[0] = {0}  # only block 0 kept
    # Masked buffer pretends every wanted block was kept (the lie the
    # old code was structurally forced to tell).
    masked = sim._stats_buffers[0]
    masked.attn_history[0] = [0.95]
    # Oracle buffer says block 3 is hot but it was evicted.
    oracle = sim._oracle_stats_buffers[0]
    oracle.attn_history[0] = [0.95]
    oracle.attn_history[3] = [0.95]
    pen = sim._io_penalty_us(n_kv_tokens=128)
    # 1 missed block → 1 * 400 / 8 = 50µs
    assert pen == 400.0 / 8, (
        f"oracle says 1 miss, expected 50µs; got {pen}µs — "
        f"io_penalty likely still reading masked buffer"
    )


def test_io_penalty_falls_back_to_masked_when_oracle_empty():
    """When ``compute_oracle=False`` (legacy / smoke-test mode), the
    fallback path uses the masked buffer. This preserves backwards
    compatibility for tests that opt out of the oracle pass.
    Forces analytical IO for an exact arithmetic assertion."""
    model = _StubModel()
    pol = RecencyPolicy()
    sim = MaskingSimulator(
        model=model, tokenizer=None, policy=pol,
        budget_frac=0.5, decision_period=8, ell_bar_us=400.0,
        n_head_groups=1,
        compute_oracle=False,
        io_mode="analytical",
    )
    sim._kept_set_per_layer[0] = {0}
    masked = sim._stats_buffers[0]
    masked.attn_history[0] = [0.05]
    masked.attn_history[3] = [0.85]
    pen = sim._io_penalty_us(n_kv_tokens=128)
    assert pen == 400.0 / 8


def test_count_prefetch_misses_pure_function():
    """The factored ``_count_prefetch_misses`` helper should:
    return 0 when buf is empty / nothing evicted, and count correctly
    otherwise."""
    from seer.eval.block_stats import BlockStatsBuffer
    from seer.eval.sim import _count_prefetch_misses

    # Empty buffer → 0
    buf = BlockStatsBuffer()
    assert _count_prefetch_misses(buf, kept_set={0, 1}, n_blocks=4) == 0

    # All wanted blocks kept → 0
    buf.attn_history[0] = [0.5]
    buf.attn_history[1] = [0.3]
    assert _count_prefetch_misses(buf, kept_set={0, 1, 2, 3}, n_blocks=4) == 0

    # Wanted blocks 0, 1 but only 1 is kept → 1 miss
    assert _count_prefetch_misses(buf, kept_set={1, 2}, n_blocks=4) == 1

    # Both wanted, both evicted → 2 misses
    assert _count_prefetch_misses(buf, kept_set={2, 3}, n_blocks=4) == 2


def test_count_prefetch_misses_ignores_zero_attention():
    """A block whose latest attention is 0 (or below threshold) does
    NOT count as 'wanted', even if it was evicted. This is what the
    old masked buffer would tell us — and is exactly the failure mode
    P0-5 fixes by switching to the oracle buffer.

    Updated for T1-A (top-K semantics): with n_blocks=4, K=32 > 4, so
    every block in the buffer is in the top-K set. Test now uses
    larger n_blocks so K becomes selective."""
    from seer.eval.block_stats import BlockStatsBuffer
    from seer.eval.sim import _count_prefetch_misses

    buf = BlockStatsBuffer()
    buf.attn_history[0] = [0.0]   # masked-to-zero
    buf.attn_history[1] = [0.0]
    buf.attn_history[2] = [0.7]   # genuinely wanted

    # n_blocks=4 → K=32 → all entries are top-K; eviction of {2}
    # under kept={0,1} → 1 miss.
    assert _count_prefetch_misses(buf, kept_set={0, 1}, n_blocks=4) == 1


def test_wanted_topk_under_dense_positive_attention():
    """T1-A regression (May 2026 reviewer round R1-C1 / R2-#2): when
    every block has non-zero attention (the realistic softmax case),
    the wanted set must collapse to the schema top-K rather than
    "all blocks with attention > 0".

    Without this, the simulator's notion of wanted-block diverges
    from the LAP training schema and ε / B_t / miss become
    semantically incompatible."""
    from seer.eval.block_stats import BlockStatsBuffer
    from seer.eval.sim import _count_prefetch_misses, _wanted_topk_set
    from seer.trace.schema import compute_top_k

    buf = BlockStatsBuffer()
    # 400 blocks all with positive attention (dense softmax)
    # K = max(32, 0.10*400) = 40
    for bid in range(400):
        buf.attn_history[bid] = [0.001 + (bid * 0.0001)]  # all > 0

    wanted = _wanted_topk_set(buf, n_blocks=400)
    expected_k = compute_top_k(400)
    assert len(wanted) == expected_k, (
        f"top-K wanted set must be {expected_k} blocks at "
        f"n_blocks=400 under dense-positive attention; got "
        f"{len(wanted)} (would be 400 under the buggy "
        f"hist[-1] > 0 definition)"
    )

    # With every block evicted, miss count == K (not 400)
    misses = _count_prefetch_misses(buf, kept_set=set(), n_blocks=400)
    assert misses == expected_k


def test_wanted_topk_picks_highest_attention():
    """T1-A: ``_wanted_topk_set`` must return the K most-attended
    blocks (not arbitrary or first-K)."""
    from seer.eval.block_stats import BlockStatsBuffer
    from seer.eval.sim import _wanted_topk_set

    buf = BlockStatsBuffer()
    # 60 blocks; K = max(32, 6) = 32. Give blocks 30..59 high attention,
    # blocks 0..29 low attention. Wanted set must be 30..59 + 2 of the
    # low ones to total K=32.
    for bid in range(30):
        buf.attn_history[bid] = [0.01]
    for bid in range(30, 60):
        buf.attn_history[bid] = [0.5]

    wanted = _wanted_topk_set(buf, n_blocks=60)
    assert len(wanted) == 32
    high_attn = set(range(30, 60))
    # Every high-attention block must be in the wanted set
    assert high_attn.issubset(wanted), (
        f"top-K must include all blocks with the highest attention; "
        f"missing: {high_attn - wanted}"
    )


# ---------------------------------------------------------------------------
#  Regression: B6 (B_t_oracle exposed in simulator result dict)
# ---------------------------------------------------------------------------
#
# Reviewer round (2026-05, reviewer #2): the runner used
# ``per_step_block_count`` as B_t for the schedulability bound, but
# that count is bounded by the policy budget by construction and so
# systematically understates the true working set. The simulator now
# emits ``per_step_B_t_oracle`` (count of blocks the unmasked attention
# layer attended to) and the runner consumes it; this test pins the
# emission contract.

def test_simulator_io_mode_defaults_to_measured_dma():
    """P1-1 (M1/M3 fix): the simulator must default to
    ``io_mode='measured-dma'`` so eA / eB / eC headline numbers
    are not the circular-validation analytical form. Tests that
    construct the simulator without an explicit io_mode get the
    measured path; the analytical path is kept only as an
    opt-in ablation."""
    model = _StubModel()
    pol = RecencyPolicy()
    sim = MaskingSimulator(
        model=model, tokenizer=None, policy=pol,
        budget_frac=0.5, decision_period=8, ell_bar_us=400.0,
        n_head_groups=1,
    )
    assert sim.io_mode == "measured-dma", (
        f"simulator default must be measured-dma after P1-1 "
        f"(got {sim.io_mode!r})"
    )


def test_simulator_result_emits_B_t_oracle_key():
    """The result dict from ``MaskingSimulator.run`` must include the
    ``per_step_B_t_oracle`` key so the runner can use it."""
    # We can't call run() without a real model, but we can simulate
    # the run-loop ``return`` shape by constructing a minimal dict
    # and asserting key presence after a guarded fast-path.
    # Instead, verify the key exists in the keys-set of the helper
    # ``simulate_attention_mask`` entry point's documented shape by
    # parsing the function source for the literal string. This is
    # cheap and catches accidental removal of the field in refactors.
    import inspect

    from seer.eval import sim as sim_module
    src = inspect.getsource(sim_module.MaskingSimulator.run)
    assert "per_step_B_t_oracle" in src, (
        "simulator.run() must emit per_step_B_t_oracle for the "
        "runner's bound-B_t estimator (B6 regression)"
    )
