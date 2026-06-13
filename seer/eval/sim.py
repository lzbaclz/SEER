"""Offline simulator for policy evaluation with **real attention masking**.

This simulator answers, by actually running the model with a per-step,
per-layer attention mask: *if we had evicted these blocks according to
policy P, what would the task quality and timing have been?*

What this simulator does
------------------------
1. **Captures** real attention scores at each decode step via post-hooks
   (the same hook protocol used by :class:`seer.trace.hook.AttentionTracer`).
   These scores feed the per-layer :class:`seer.eval.block_stats.BlockStatsBuffer`
   so every policy — H2O, SnapKV, SEER, … — actually observes block
   dynamics rather than receiving zero-fill stubs.

2. **Decides** every ``decision_period`` decode steps which blocks to keep,
   per layer. The policy is invoked with the snapshot dict from the buffer.

3. **Enforces** the decision by applying a per-layer additive mask: tokens
   whose 32-block id is *not* in the kept set get a -inf bias added to
   their attention pre-softmax score, mathematically equivalent to
   evicting them from the cache (cheaper to implement than rewriting
   ``past_key_values``).

4. **Adds an IO-cost penalty** to the wall-clock per-step latency: every
   block that the *previous* decision evicted but that the *current* step
   would still want pays ``ell_bar_us / decision_period`` µs (a
   simplified Lemma-1-style cost model). This is what makes SEER's
   ``λ·IO_cost`` utility function meaningful in the offline sim — without
   the penalty all policies would record identical wall-times.

5. **FORWARD-ONCE** mode skips the masking machinery entirely; used for
   the ``full`` cache oracle.

Implementation notes
--------------------
* We support Llama / Qwen2 / Mistral attention naming (``self_attn``)
  and fall back to anything ending in ``attention``.
* ``output_attentions=True`` is required during decode so the post-hook
  receives the attention weight tensor. This is forced by setting
  ``model.config._attn_implementation = "eager"`` upstream — the runner
  loads with ``attn_implementation="eager"`` already.
* Block id of token position ``p`` is ``p // BLOCK_SIZE``.

Sketches that did not survive
-----------------------------
We considered patching the model's attention modules to manipulate
``past_key_values`` directly. That would be faithful to peak HBM
utilization but requires rewriting attention for every supported
architecture; the additive mask is a clean, architecture-independent
approximation that suffices for *quality* and *per-step decode latency
shape* — the two things RTSS reviewers care about.
"""
from __future__ import annotations

import time
from typing import Any

from seer.eval.block_stats import BlockStatsBuffer
from seer.policy.base import KVPolicy
from seer.trace.schema import BLOCK_SIZE, compute_top_k

# ---------------------------------------------------------------------------
#  Utility — find self-attention modules across model families
# ---------------------------------------------------------------------------

def _find_attention_layers(model) -> list:
    """Return the list of self-attention modules in layer order.

    Mirrors :meth:`seer.trace.hook.AttentionTracer._find_attention_layers`.
    """
    layers = []
    for name, mod in model.named_modules():
        if name.endswith("self_attn"):
            layers.append(mod)
    if not layers:
        for name, mod in model.named_modules():
            if name.endswith("attention"):
                layers.append(mod)
    return layers


# ---------------------------------------------------------------------------
#  Prefetch-miss accounting (pure, unit-testable)
# ---------------------------------------------------------------------------

def _wanted_topk_set(buf, n_blocks: int) -> set[int]:
    """Schema-consistent "wanted block" set at the current step.

    Returns the top-K most-attended blocks under the trace-schema
    definition ``k = max(MIN_TOP_K, TOP_K_FRACTION * n_blocks)`` from
    :mod:`seer.trace.schema`. This is the SAME definition used to
    label training rows for the LAP, so the simulator's oracle
    ``wanted`` set, the ε(φ) estimator's ``wanted`` set, and the
    LAP's training labels all agree (P0 review-round T1-A).

    Why not "any attention > 0":
    softmax attention is dense-positive — almost every block has
    a positive weight under unmasked forward — so a "> 0" definition
    inflates the wanted set to ≈ ``n_blocks`` and makes ε / B_t / miss
    inconsistent with the LAP training target. Top-K is the canonical
    definition used by every other component in the pipeline.
    """
    if not buf.attn_history:
        return set()
    k = compute_top_k(int(n_blocks))
    scored = [
        (bid, hist[-1] if hist else 0.0)
        for bid, hist in buf.attn_history.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {bid for bid, _ in scored[:k]}


def _count_prefetch_misses(buf, kept_set: set[int], n_blocks: int) -> int:
    """Count top-K wanted blocks the policy evicted.

    Pure function so that :func:`MaskingSimulator._io_penalty_us` can
    be regression-tested without spinning up a transformer.

    A "miss" requires both:
      * the block is in the schema-consistent top-K wanted set
        (see :func:`_wanted_topk_set`);
      * the block is NOT in ``kept_set`` (i.e.\\ the policy evicted it).

    B5 fix (May 2026 review round): under the legacy "post-hook reads
    masked attention" path, ``buf`` was always the masked buffer, so
    evicted blocks had attention 0 and ``wanted_now ∩ evicted`` was
    structurally empty. The caller now passes the oracle buffer
    populated by an unmasked forward pass, so this count correctly
    identifies attended-but-evicted blocks.

    T1-A fix (May 2026 reviewer round R1/R2 convergence): wanted-set
    is now top-K (matching the LAP training schema) instead of
    ``hist[-1] > 0``. The previous form labelled essentially every
    block as wanted because softmax outputs are dense-positive.

    Parameters
    ----------
    buf : BlockStatsBuffer
        Source of truth for "what did attention attend to last
        step". For correct miss accounting this must be the oracle
        (unmasked) buffer.
    kept_set : set[int]
        Block IDs the policy decided to keep in cache.
    n_blocks : int
        Total number of blocks in the working set (= ceil(KV tokens
        / block_size)).

    Returns
    -------
    int
        Number of top-K wanted blocks not in kept set; lower bound zero.
    """
    wanted = _wanted_topk_set(buf, n_blocks)
    if not wanted:
        return 0
    return len(wanted - set(kept_set))


# ---------------------------------------------------------------------------
#  Masking simulator
# ---------------------------------------------------------------------------

class MaskingSimulator:
    """Per-request attention-mask driven simulator.

    Lifecycle::

        sim = MaskingSimulator(model, tokenizer, policy, budget_frac=0.2)
        sim.attach()
        result = sim.run(prompt, max_new_tokens=64)
        sim.detach()
    """

    def __init__(
        self,
        model,
        tokenizer,
        policy: KVPolicy,
        budget_frac: float,
        decision_period: int = 8,
        block_size: int = BLOCK_SIZE,
        ell_bar_us: float = 200.0,
        n_head_groups: int | None = None,
        io_mode: str = "measured-dma",
        compute_oracle: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy
        self.budget_frac = float(budget_frac)
        self.decision_period = int(decision_period)
        self.block_size = int(block_size)
        self.ell_bar_us = float(ell_bar_us)
        # io_mode controls _io_penalty_us:
        #   "analytical" (default) — synthetic Lemma 1 form (misses * ell_bar / K_dec)
        #   "measured-dma" — real cudaMemcpyAsync timing (J.6 pinned-host path)
        if io_mode not in ("analytical", "measured-dma"):
            raise ValueError(f"io_mode must be 'analytical' or 'measured-dma', got {io_mode}")
        self.io_mode = io_mode
        # B5 fix (review round May 2026): the masked attention captured
        # by the regular post-hook has zero weight on evicted blocks
        # (the additive -inf mask zeroes them after softmax), so the
        # original ``_io_penalty_us`` saw ``wanted_now ∩ evicted = ∅``
        # by construction and miss-counts were always 0. We now run a
        # secondary "oracle" forward pass per step that captures the
        # UNMASKED attention — i.e.\ what the layer would attend to if
        # all blocks were resident. The oracle buffer feeds
        # ``_io_penalty_us`` and the runner's ``B_t`` estimate.
        # Set ``compute_oracle=False`` to recover the legacy behavior
        # for unit tests that do not require oracle truth.
        self.compute_oracle = bool(compute_oracle)
        # Oracle-mode flag toggled inside ``run()`` around the oracle
        # forward pass. The pre-hook honors it to skip masking and the
        # post-hook honors it to write into ``_oracle_stats_buffers``
        # instead of the regular buffer.
        self._oracle_mode = False
        self._dma_pinned = None
        self._dma_device = None

        layers = _find_attention_layers(model)
        self._n_layers = len(layers)
        cfg = getattr(model, "config", None)
        if n_head_groups is None and cfg is not None:
            n_head_groups = getattr(cfg, "num_key_value_heads",
                                    getattr(cfg, "num_attention_heads", 1))
        self._n_head_groups = int(n_head_groups or 1)

        # Per-layer rolling stats (mean over head groups for the policy snapshot).
        self._stats_buffers: list[BlockStatsBuffer] = [
            BlockStatsBuffer() for _ in range(self._n_layers)
        ]
        # Parallel buffers populated by the oracle forward pass.
        # These hold the UNMASKED top-k truth used by ``_io_penalty_us``.
        self._oracle_stats_buffers: list[BlockStatsBuffer] = [
            BlockStatsBuffer() for _ in range(self._n_layers)
        ]
        # Per-layer kept-token bool masks, populated each decision tick.
        self._kept_block_mask_per_layer: list = []
        self._kept_set_per_layer: list[set[int]] = [set() for _ in range(self._n_layers)]
        # Tracks how many tokens actually exist at each decision so we can
        # spot newly arrived blocks that the previous decision didn't see.
        self._known_blocks_per_layer: list[set[int]] = [set() for _ in range(self._n_layers)]
        self._pre_handles: list = []
        self._post_handles: list = []

    # ------------------------------------------------------------------
    #  Hook attach / detach
    # ------------------------------------------------------------------

    def attach(self) -> None:
        layers = _find_attention_layers(self.model)
        for layer_id, mod in enumerate(layers):
            self._pre_handles.append(
                mod.register_forward_pre_hook(
                    self._make_pre_hook(layer_id), with_kwargs=True
                )
            )
            self._post_handles.append(
                mod.register_forward_hook(self._make_post_hook(layer_id))
            )

    def detach(self) -> None:
        for h in self._pre_handles + self._post_handles:
            h.remove()
        self._pre_handles.clear()
        self._post_handles.clear()

    # ------------------------------------------------------------------
    #  Pre-hook: install the per-layer eviction mask
    # ------------------------------------------------------------------

    def _make_pre_hook(self, layer_id: int):
        import torch

        def hook(module, args, kwargs):
            # B5 fix: oracle pass must run unmasked so the post-hook
            # captures the "what would the layer attend to if all
            # blocks were resident" truth.
            if self._oracle_mode:
                return args, kwargs
            mask = kwargs.get("attention_mask")
            if mask is None or layer_id >= len(self._kept_block_mask_per_layer):
                return args, kwargs
            kept_token_mask = self._kept_block_mask_per_layer[layer_id]
            if kept_token_mask is None:
                return args, kwargs
            # mask: [B, 1, Q, K] additive bias
            K = mask.shape[-1]
            ktm = kept_token_mask
            if ktm.shape[0] < K:
                # New tokens have been generated since the last decision
                # tick (we recompute every decision_period steps, but
                # every step appends one). Treat the unseen tokens as
                # kept — they belong to the latest block, which the
                # sliding-window pin would protect anyway.
                pad = torch.ones(
                    K - ktm.shape[0],
                    dtype=torch.bool,
                    device=ktm.device,
                )
                ktm = torch.cat([ktm, pad])
            elif ktm.shape[0] > K:
                ktm = ktm[:K]
            evicted = ~ktm
            if not evicted.any():
                return args, kwargs
            new_mask = mask.clone()
            neg_inf = torch.finfo(new_mask.dtype).min
            new_mask[..., evicted] = neg_inf
            kwargs["attention_mask"] = new_mask
            return args, kwargs

        return hook

    # ------------------------------------------------------------------
    #  Post-hook: capture real attention weights into the stats buffer
    # ------------------------------------------------------------------

    def _make_post_hook(self, layer_id: int):
        import torch

        def hook(module, inputs, outputs):
            if layer_id >= len(self._stats_buffers):
                return
            attn = self._extract_attn(outputs)
            if attn is None:
                return
            with torch.no_grad():
                self._ingest_attn(layer_id, attn, oracle=self._oracle_mode)

        return hook

    @staticmethod
    def _extract_attn(outputs):
        import torch
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            w = outputs[1]
            if torch.is_tensor(w) and w.ndim == 4:
                return w
        return None

    def _ingest_attn(self, layer_id: int, attn_weights, oracle: bool = False) -> None:
        """Aggregate attn weights into per-block scalar stats and update buffer.

        attn_weights: [B, H, Q, K]
        We mean-pool over (B, query) and average over heads inside each
        head-group, then mean-pool 32 tokens into one block. The result is
        one scalar per block; we feed the head-group-averaged scalar to
        the policy's per-layer view (the policy sees one number per block,
        per layer).

        When ``oracle=True`` (P0-5 fix), the values are written to
        :attr:`_oracle_stats_buffers` so :meth:`_io_penalty_us` and
        the runner's ``B_t`` estimator have an unmasked source of
        truth for what the attention layer would attend to.
        """
        import numpy as np
        import torch.nn.functional as F

        aw = attn_weights.detach().float()
        B, H, Q, K = aw.shape
        aw = aw.mean(dim=2)  # [B, H, K]
        # GQA: fold heads into head groups
        group_size = max(1, H // self._n_head_groups)
        G = max(1, H // group_size)
        aw = aw[:, : G * group_size].reshape(B, G, group_size, K).mean(dim=2)  # [B, G, K]
        # block-pool along K
        n_blocks = (K + self.block_size - 1) // self.block_size
        pad_len = n_blocks * self.block_size - K
        if pad_len > 0:
            aw = F.pad(aw, (0, pad_len))
        aw = aw.reshape(B, G, n_blocks, self.block_size).mean(dim=3)  # [B, G, n_blocks]
        # Average over batch + head groups so the per-layer buffer sees a
        # single scalar per block (matches the policy's "head-aggregated"
        # view; per-head policies would key on (layer, head_group) instead).
        aw = aw.mean(dim=(0, 1)).cpu().numpy()  # [n_blocks]

        # Compute is_top_k labels for this step (per layer, head-aggregated).
        k = compute_top_k(int(n_blocks))
        if n_blocks <= k:
            top_set = set(range(n_blocks))
        else:
            top_set = set(np.argpartition(-aw, k - 1)[:k].tolist())

        per_block_attn = {bid: float(aw[bid]) for bid in range(int(n_blocks))}
        per_block_top = {bid: int(bid in top_set) for bid in range(int(n_blocks))}

        buf = (
            self._oracle_stats_buffers[layer_id]
            if oracle
            else self._stats_buffers[layer_id]
        )
        buf.update_step(buf.current_step, per_block_attn, per_block_top)
        for bid in range(int(n_blocks)):
            buf.set_position(bid, bid * self.block_size)

    # ------------------------------------------------------------------
    #  Decision: invoke policy and update per-layer masks
    # ------------------------------------------------------------------

    def _recompute_kept_masks(self, total_kv_tokens: int, device, dtype) -> None:
        import torch

        n_blocks_total = (total_kv_tokens + self.block_size - 1) // self.block_size
        budget = max(1, int(round(self.budget_frac * n_blocks_total)))
        new_masks: list = []
        for layer_id in range(self._n_layers):
            buf = self._stats_buffers[layer_id]
            stats = buf.snapshot(
                layer_scalar=layer_id / max(1, self._n_layers - 1),
                head_scalar=0.0,  # head-aggregated mask
            )
            # Stub-fill any blocks the buffer hasn't seen yet (e.g. brand-new
            # newly-decoded tokens at the very first decision after each step).
            for bid in range(n_blocks_total):
                if bid not in stats:
                    stats[bid] = {
                        "attn_score_now": 0.0,
                        "attn_history": [],
                        "position": bid * self.block_size,
                        "position_norm": (bid * self.block_size)
                        / max(1, (n_blocks_total - 1) * self.block_size),
                        "last_top_k_step": -1,
                        "steps_since_top_k": 1 << 20,
                        "persistence": 0.0,
                        "layer_scalar": layer_id / max(1, self._n_layers - 1),
                        "head_scalar": 0.0,
                        "io_cost": 1.0,  # cold block — fetching is "expensive"
                    }
                else:
                    # Tier-aware io_cost: blocks not in HBM at last decision
                    # would need to be re-fetched. We use the previous
                    # kept-set to approximate this.
                    stats[bid]["io_cost"] = (
                        0.0 if bid in self._kept_set_per_layer[layer_id] else 1.0
                    )

            kept = self.policy.select_to_keep(stats, budget=budget, step=buf.current_step)
            self._kept_set_per_layer[layer_id] = set(kept)

            kept_tensor = torch.zeros(n_blocks_total, dtype=torch.bool, device=device)
            for bid in kept:
                if 0 <= bid < n_blocks_total:
                    kept_tensor[bid] = True
            token_block_id = torch.arange(total_kv_tokens, device=device) // self.block_size
            token_mask = kept_tensor[token_block_id]
            new_masks.append(token_mask)
        self._kept_block_mask_per_layer = new_masks

    # ------------------------------------------------------------------
    #  IO-cost penalty: per-step Lemma-1-style synthetic latency add
    # ------------------------------------------------------------------

    def _io_penalty_us(self, n_kv_tokens: int) -> float:
        r"""IO penalty for blocks evicted-then-needed-again.

        Two modes:
          * **analytical (default)**: ``(# evicted blocks attended) *
            ell_bar_us / decision_period`` --- the Lemma-1 first-moment
            cost. Synthetic but matches the bound being validated.
          * **measured-dma** (``self.io_mode = 'measured-dma'``): time
            a real ``cudaMemcpyAsync`` on the pinned-host transport for
            each missed block, then divide by decision_period. This
            addresses the OSDI/RTSS-reviewer concern that the
            analytical mode is "circular validation" --- the measured
            mode plumbs through the same pinned-host DMA that vLLM-V1
            would use, so the per-step latency is faithful to the
            production substrate at the cost of $\sim 2$x simulator
            wall-clock.

        B5 fix (review round May 2026): the "wanted_now" set is now
        derived from the ORACLE buffer (the per-step unmasked
        attention pass) when :attr:`compute_oracle` is True; otherwise
        we fall back to the legacy masked buffer (kept for
        backwards-compat with tests that opt out of the oracle pass).

        See ``seer/integration/vllm_connector.py`` for the
        pinned-host transport reference implementation (J.6).
        """
        n_blocks = (n_kv_tokens + self.block_size - 1) // self.block_size
        if not self._stats_buffers:
            return 0.0
        # T2-J (May 2026 sixth-round reviewer fix): the original
        # implementation read only ``_oracle_stats_buffers[0]`` and
        # ``_kept_set_per_layer[0]`` even though the schedulability
        # model in §3 applies across all L attention layers. We now
        # average the per-layer miss count across the layers whose
        # oracle buffer is populated, which removes the layer-0
        # quirk without changing the per-step magnitude in the
        # homogeneous-layer regime that the ε/σ calibration was
        # validated against (B2 audit). The conservative reading is
        # mean(misses_l), not sum(misses_l); the layer-summed reading
        # would change the IO penalty by ~L=32x and is not consistent
        # with the legacy calibrated σ_clean=1203\,μs the bound
        # consumes. We report this caveat in §6.eC explicitly.
        use_oracle = self.compute_oracle and any(
            buf.attn_history for buf in self._oracle_stats_buffers
        )
        per_layer_misses: list[int] = []
        n_kept = len(self._kept_set_per_layer)
        for li in range(self._n_layers):
            if use_oracle:
                bufl = self._oracle_stats_buffers[li]
                if not bufl.attn_history:
                    continue
            else:
                bufl = self._stats_buffers[li]
            kept_l = self._kept_set_per_layer[li] if li < n_kept else set()
            per_layer_misses.append(_count_prefetch_misses(
                buf=bufl, kept_set=kept_l, n_blocks=n_blocks,
            ))
        if not per_layer_misses:
            return 0.0
        mean_misses = sum(per_layer_misses) / len(per_layer_misses)
        if mean_misses <= 0:
            return 0.0
        # Measured-DMA path. Requires self.io_mode = 'measured-dma' and
        # a registered pinned-host scratch buffer; falls back to
        # analytical if the scratch is not available (e.g. unit tests).
        if getattr(self, "io_mode", "analytical") == "measured-dma":
            ell_us = self._measure_dma_per_block_us(int(mean_misses + 0.5))
            if ell_us is not None:
                return mean_misses * ell_us / max(1, self.decision_period)
        # Analytical default — matches Lemma 1.
        return float(mean_misses) * self.ell_bar_us / max(1, self.decision_period)

    def _measure_dma_per_block_us(self, n_blocks: int) -> float | None:
        """Time a pinned-host -> HBM DMA of ``n_blocks`` KV blocks.

        Uses the same per-block KV shape as the J.6 probe
        (2 x 32 x 16 x 128, fp16, 256 KiB). Returns the per-block
        time in microseconds via averaging across n_blocks (so the
        return is comparable to a per-block ell_bar). If torch / CUDA
        are unavailable or the scratch buffers aren't allocated,
        returns None so the caller falls back to analytical.

        R28 advisor disclosure: this is the \"batch-amortised DMA\"
        path -- one large H2D copy of (n_blocks, 2, 32, 16, 128) fp16
        bytes, then divide elapsed by n_blocks. The substrate / A7
        microbenches under experiments/eC_bound_tightness use a
        \"per-transfer burst\" protocol instead -- one CUDA event-
        timed H2D+D2H pair around each KV block, n_reps=5000. The
        two protocols answer different questions:

        - batch-amortised here measures the operator-side end-to-end
          IO cost a single decode step pays for ``mean_misses`` blocks
          in aggregate (a worst-case full-batch refill);
        - per-transfer-burst in the microbench characterises the
          A7 / Lemma 2''' per-step burst distribution that the bound
          is sensitive to (each block is a distinct event).

        Numbers from the two paths therefore should NOT be directly
        compared; the per-transfer-burst numbers are what the §V/§VI
        bound consumes, and the batch-amortised numbers are what the
        sim's deadline-miss accounting consumes. This split is
        documented in paper/sections/A2_setup.tex and called out in
        §VI scope.
        """
        try:
            import time as _time

            import torch
        except Exception:
            return None
        if not torch.cuda.is_available() or n_blocks <= 0:
            return None
        # Allocate scratch lazily.
        if not hasattr(self, "_dma_pinned") or self._dma_pinned is None:
            shape = (n_blocks, 2, 32, 16, 128)
            try:
                self._dma_pinned = torch.empty(shape, dtype=torch.float16, pin_memory=True)
                self._dma_device = torch.empty(shape, dtype=torch.float16, device="cuda")
            except Exception:
                self._dma_pinned = None
                return None
        # If shape doesn't fit, re-allocate.
        if self._dma_pinned.shape[0] < n_blocks:
            shape = (n_blocks, 2, 32, 16, 128)
            try:
                self._dma_pinned = torch.empty(shape, dtype=torch.float16, pin_memory=True)
                self._dma_device = torch.empty(shape, dtype=torch.float16, device="cuda")
            except Exception:
                return None
        # Time HtoD on a dedicated stream.
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        self._dma_device[:n_blocks].copy_(self._dma_pinned[:n_blocks], non_blocking=True)
        torch.cuda.synchronize()
        elapsed_us = (_time.perf_counter() - t0) * 1e6
        # Per-block ell (the average over n_blocks).
        return elapsed_us / max(1, n_blocks)

    # ------------------------------------------------------------------
    #  Run one prompt
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> dict:
        import torch

        self.policy.reset()
        for buf in self._stats_buffers:
            buf.attn_history.clear()
            buf.last_top_k_step.clear()
            buf.persistence_window.clear()
            buf.position.clear()
            buf.current_step = 0
        for buf in self._oracle_stats_buffers:
            buf.attn_history.clear()
            buf.last_top_k_step.clear()
            buf.persistence_window.clear()
            buf.position.clear()
            buf.current_step = 0
        self._kept_set_per_layer = [set() for _ in range(self._n_layers)]
        self._kept_block_mask_per_layer = []
        self._oracle_mode = False

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.model.config, "max_position_embeddings", 32768),
        ).to(self.model.device)
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        device = input_ids.device

        # ---------- Prefill (no masking yet — full cache during prefill) ----
        t_start = time.perf_counter()
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
                output_attentions=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([input_ids, next_token], dim=1)
        prefill_us = (time.perf_counter() - t_start) * 1e6

        # First decision based on prefill-time attention.
        self._recompute_kept_masks(generated.shape[1], device, dtype=torch.float32)

        # ---------- Decode loop ----
        # We track total per-step latency *and* the (base, io) decomposition
        # so the bound can be calibrated against the IO-free residual rather
        # than the IO-contaminated total. base_us is the measured forward-pass
        # time under the current mask (= C_LAP + C_attn + C_ffn empirically);
        # io_us is the synthetic IO penalty derived from prefetch-miss count.
        per_step_us: list[float] = []
        per_step_base_us: list[float] = []
        per_step_io_us: list[float] = []
        per_step_block_count: list[int] = []
        # P0-6 fix: track the ORACLE working-set size per step so the
        # runner / schedulability bound see B_t = "blocks the layer
        # actually wanted" rather than B_t = "blocks the policy kept"
        # (which is bounded by the budget by construction).
        per_step_B_t_oracle: list[int] = []
        # T1-B fix (May 2026 reviewer round): per-step measured ε —
        # fraction of the top-K wanted set that the policy evicted at
        # this step. Replaces the 0.15 fallback that runner used to
        # hand to the schedulability bound; the runner aggregates this
        # list into a single eps_measured field in the output JSON.
        per_step_eps_measured: list[float] = []
        for step in range(1, max_new_tokens):
            for buf in self._stats_buffers:
                buf.current_step = step
            for buf in self._oracle_stats_buffers:
                buf.current_step = step

            # P0-5 oracle pass: run the model UNMASKED on a clone of
            # ``past`` so the per-step attention captured by the
            # post-hook is the counterfactual "what would the layer
            # attend to if all blocks were resident". K, V projections
            # are mask-independent so we can discard the oracle's
            # returned past_key_values; we only use its attentions.
            if self.compute_oracle:
                self._oracle_mode = True
                try:
                    with torch.no_grad():
                        _oracle_out = self.model(
                            input_ids=next_token,
                            attention_mask=torch.ones(
                                generated.shape, dtype=torch.long, device=device
                            ),
                            past_key_values=past,
                            use_cache=False,  # do not append; just probe
                            output_attentions=True,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                finally:
                    self._oracle_mode = False
                del _oracle_out
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    attention_mask=torch.ones(
                        generated.shape, dtype=torch.long, device=device
                    ),
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            base_us = (t1 - t0) * 1e6
            io_us = self._io_penalty_us(generated.shape[1])
            # Speculative-warmup wrappers expose an extra IO cost on
            # warmup steps via warmup_overhead_us(); honest accounting
            # adds it so the warmup step's latency reflects the real
            # production prefetch cost (~N_warmup × ℓ̄ / decision_period).
            warmup_us_fn = getattr(self.policy, "warmup_overhead_us", None)
            warmup_us = float(warmup_us_fn()) if callable(warmup_us_fn) else 0.0
            io_us = io_us + warmup_us
            step_us = base_us + io_us
            per_step_us.append(step_us)
            per_step_base_us.append(base_us)
            per_step_io_us.append(io_us)
            # T2-J: working-set + wanted-top-K + per-step ε are now
            # averaged across all populated layers, not just layer 0
            # (the paper's schedulability model is across-layer).
            if self._kept_set_per_layer:
                per_step_block_count.append(int(round(
                    sum(len(k) for k in self._kept_set_per_layer)
                    / max(1, len(self._kept_set_per_layer))
                )))
            else:
                per_step_block_count.append(0)
            kept = self._kept_set_per_layer[0] if self._kept_set_per_layer else set()
            # T1-A fix: use top-K wanted-set (matches LAP training
            # schema) instead of "any attention > 0".
            n_blocks_layer0 = max(
                int(generated.shape[1] // self.block_size + 1),
                len(self._known_blocks_per_layer[0]) if self._known_blocks_per_layer else 0,
            )
            if self.compute_oracle and self._oracle_stats_buffers:
                # T2-J: average B_t and ε across layers (was layer-0).
                B_t_per_layer: list[int] = []
                eps_per_layer: list[float] = []
                for li, oracle_buf in enumerate(self._oracle_stats_buffers):
                    if not oracle_buf.attn_history:
                        continue
                    wanted_l = _wanted_topk_set(oracle_buf, n_blocks_layer0)
                    if not wanted_l:
                        continue
                    B_t_per_layer.append(int(len(wanted_l)))
                    kept_l = (self._kept_set_per_layer[li]
                              if li < len(self._kept_set_per_layer) else set())
                    fn_l = len(wanted_l - kept_l)
                    eps_per_layer.append(float(fn_l) / float(len(wanted_l)))
                if B_t_per_layer:
                    per_step_B_t_oracle.append(int(round(
                        sum(B_t_per_layer) / len(B_t_per_layer))))
                # keep wanted_topk for the legacy code path below
                oracle_buf = self._oracle_stats_buffers[0]
                wanted_topk = _wanted_topk_set(oracle_buf, n_blocks_layer0)
                # Per-step measured ε: layer-averaged false-negative
                # rate (T1-B). The runner aggregates this into a single
                # eps_measured field for the JSON.
                if eps_per_layer:
                    per_step_eps_measured.append(
                        float(sum(eps_per_layer)) / float(len(eps_per_layer)))
                elif wanted_topk:
                    fn_topk = len(wanted_topk - kept)
                    per_step_eps_measured.append(float(fn_topk) / float(len(wanted_topk)))
                else:
                    per_step_eps_measured.append(0.0)
            else:
                per_step_B_t_oracle.append(int(len(kept)))
                per_step_eps_measured.append(0.0)

            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            self.policy.on_step_end(step, step_latency_us=step_us)
            # Notify policies that track miss-rate (e.g.\ SpeculativeWarmup).
            # io_us > 0 ⇔ at least one block the previous decision evicted
            # was attended now — a "miss" in the prefetch sense.
            rec_miss = getattr(self.policy, "record_miss", None)
            if callable(rec_miss):
                rec_miss(io_us > 0)

            if step % self.decision_period == 0:
                self._recompute_kept_masks(generated.shape[1], device, dtype=torch.float32)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        gen = generated[0][input_ids.shape[1]:]
        pred = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        ref = _extract_ruler_reference(prompt)

        return {
            "pred": pred,
            "ref": ref,
            "n_gen_tokens": int(gen.shape[0]),
            "policy": self.policy.name,
            "budget_frac": self.budget_frac,
            "decision_period": self.decision_period,
            "prefill_us": float(prefill_us),
            "per_step_us": [float(x) for x in per_step_us],
            # Decomposition into (base, IO penalty) for bound calibration:
            # base_us is the IO-free residual (LAP + attn + FFN under mask),
            # io_us is the synthetic prefetch-miss penalty. The bound's
            # base_cost should be calibrated from per_step_base_us only;
            # adding the analytic ε·B·ℓ̄ on top of per_step_us would
            # double-count the IO term.
            "per_step_base_us": [float(x) for x in per_step_base_us],
            "per_step_io_us": [float(x) for x in per_step_io_us],
            "per_step_block_count": [int(x) for x in per_step_block_count],
            # P0-6: oracle-derived working-set size per step. Use this
            # as B_t for schedulability-bound calibration rather than
            # ``per_step_block_count`` (which equals the policy budget
            # by construction and so under-counts the true working set).
            "per_step_B_t_oracle": [int(x) for x in per_step_B_t_oracle],
            # T1-B (May 2026 review round): per-step measured ε against
            # the top-K wanted set. The runner aggregates to a single
            # ``eps_measured`` scalar and writes it to the cell JSON.
            "per_step_eps_measured": [float(x) for x in per_step_eps_measured],
        }


# ---------------------------------------------------------------------------
#  Convenience entry — used by :mod:`seer.eval.runner`
# ---------------------------------------------------------------------------

def simulate_attention_mask(
    model: Any,
    tokenizer: Any,
    prompt: str,
    policy: KVPolicy,
    budget_frac: float,
    max_new_tokens: int = 128,
    decision_period: int = 8,
    skip_masking: bool = False,
    ell_bar_us: float = 200.0,
    io_mode: str = "measured-dma",
) -> dict:
    """Run ``prompt`` under ``policy`` and return per-step latency + quality.

    When ``skip_masking=True`` (the default for ``policy in {full}``) we
    bypass the hook installation entirely — same model wall-time as the
    masked path is meaningless when no mask is ever applied.

    ``io_mode`` controls the IO-penalty model:
      * ``"analytical"`` (default): synthetic Lemma 1 form
        (misses * ell_bar_us / decision_period).
      * ``"measured-dma"``: time a real cudaMemcpyAsync HtoD on a
        pinned-host scratch (J.6 transport) instead of the analytical
        approximation. Slower but addresses the "simulator IO is
        circular validation" reviewer concern.
    """
    if skip_masking or policy.name == "full":
        return _forward_once(model, tokenizer, prompt, policy, budget_frac,
                             max_new_tokens, decision_period)
    sim = MaskingSimulator(
        model=model,
        tokenizer=tokenizer,
        policy=policy,
        budget_frac=budget_frac,
        decision_period=decision_period,
        ell_bar_us=ell_bar_us,
        io_mode=io_mode,
    )
    sim.attach()
    try:
        return sim.run(prompt, max_new_tokens=max_new_tokens)
    finally:
        sim.detach()


def _forward_once(
    model, tokenizer, prompt, policy, budget_frac, max_new_tokens, decision_period
) -> dict:
    """Plain greedy generate; no masking, no hooks. Pre-pivot behavior.

    SYNTHETIC PER-STEP TIMING — important caveat for the bound
    ----------------------------------------------------------
    This path measures only the *total* generate-time and synthesises
    ``per_step_us = [wall_us / n_gen] * n_gen``. Every per-step entry
    is therefore identical (== average), so:

    * P50 / P99 / P999 of this ``per_step_us`` are all the same number
      and DO NOT reflect real per-step decode latency variance.
    * The ``full`` baseline cannot be used as a calibration anchor
      for ``base_cost_us`` or ``σ_residual`` of Lemma 2 / 2$'$.
    * For an honest no-IO calibration baseline, run ``--policy seer``
      (or any masked policy) at ``--hbm_budget 1.0`` so the masking
      simulator runs but the budget keeps every block — the simulator
      then emits real per-step decode times in ``per_step_base_us``.
    """
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(model.config, "max_position_embeddings", 32768),
    ).to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    wall_us = (time.perf_counter() - t0) * 1e6
    gen = out[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
    ref = _extract_ruler_reference(prompt)
    n_gen = int(gen.shape[0])
    per_step_us = [wall_us / max(1, n_gen)] * n_gen
    return {
        "pred": pred,
        "ref": ref,
        "n_gen_tokens": n_gen,
        "policy": policy.name,
        "budget_frac": budget_frac,
        "decision_period": decision_period,
        "prefill_us": 0.0,
        "per_step_us": per_step_us,
    }


def _extract_ruler_reference(prompt: str) -> str:
    """Extract the needle from a RULER-synthetic prompt."""
    import re
    m = re.search(r"secret password is (\d+)", prompt)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
#  eR/e4 single-cell simulator entry (model cached across sweep cells)
# ---------------------------------------------------------------------------

_SIM_CELL_CACHE: dict[str, tuple] = {}


def _parse_workload_spec(workload: str) -> tuple[str, int]:
    """Parse strings like ``mooncake-24`` → (``mooncake``, 24)."""
    if "-" in workload:
        base, tail = workload.rsplit("-", 1)
        if tail.isdigit():
            return base, int(tail)
    return workload, 24


def run_simulation_cell(
    policy: KVPolicy,
    workload: str = "mooncake-24",
    D_ms: float = 200.0,
    rho: float = 0.01,
    substrate: str = "DRAM",
    ell_bar_us: float = 200.0,
    ell_max_us: float = 1500.0,
    seed: int = 0,
    budget_frac: float = 0.20,
    max_new_tokens: int = 64,
    decision_period: int = 8,
) -> dict:
    """Run one in-process simulator cell for the eR/e4 head-to-head sweep."""
    import os
    import random

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from seer.eval.metrics import LatencyStats
    from seer.timing import derive_deadline_us
    from seer.timing.slo import SLOClass
    from seer.trace.datasets import load_prompts_with_refs

    del ell_max_us  # reserved for future substrate-specific IO caps

    wl_name, num_prompts = _parse_workload_spec(workload)
    model_name = os.environ.get("SEER_MODEL", "meta-llama/Llama-2-7b-hf")
    device = os.environ.get("SEER_DEVICE", "cuda")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_name not in _SIM_CELL_CACHE:
        dtype = torch.float16
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device).eval()
        _SIM_CELL_CACHE[model_name] = (tok, model)
    tok, model = _SIM_CELL_CACHE[model_name]

    slo = SLOClass(
        name=f"chat-{D_ms:g}ms",
        kind="TPOT",
        percentile=99.0,
        threshold_ms=float(D_ms),
        miss_target=float(rho),
    )
    deadline_us = derive_deadline_us(slo)

    prompts, _answers = load_prompts_with_refs(
        wl_name, [8192], num_prompts, tok,
    )

    all_latencies_us: list[float] = []
    miss_ratios: list[float] = []
    for prompt in prompts:
        policy.reset()
        r = simulate_attention_mask(
            model, tok, prompt, policy,
            budget_frac=budget_frac,
            max_new_tokens=max_new_tokens,
            decision_period=decision_period,
            ell_bar_us=ell_bar_us,
            io_mode="analytical",
        )
        per_step = r.get("per_step_us", [])
        stats = LatencyStats.from_vector(per_step, deadline_us=deadline_us)
        miss_ratios.append(float(stats.miss_ratio))
        all_latencies_us.extend(per_step)

    overall = LatencyStats.from_vector(all_latencies_us, deadline_us=deadline_us)
    return {
        "chat_miss_ratio": float(sum(miss_ratios) / max(1, len(miss_ratios))),
        "p50_tpot_ms": float(overall.p50_us) / 1000.0,
        "p99_tpot_ms": float(overall.p99_us) / 1000.0,
        "n_prompts": len(prompts),
        "substrate": substrate,
    }
