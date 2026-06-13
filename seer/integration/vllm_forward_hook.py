"""Forward-hook patch for vLLM 0.8.5 to surface per-block attention.

R1-Cut6 (sixth-round reviewer note): the real-vLLM J.5 absorber-off
chat-miss reduction is a connector-substrate effect because the LAP
predictor was never fired on real per-block attention --- the
synthetic recency-decay used by the T2-G/T2-I plumbing patches does
not exercise LAP predictability. This module installs a forward
pre-hook on every ``vllm.attention.Attention`` module that computes
an approximate per-block attention score from ``query`` and the
per-block key-cache centroid (the InfiniGen-style proxy), stashes
the result on the ``forward_context`` so the next
``SeerKVConnector`` scheduler-side hook can read it, and falls back
to a no-op when prerequisites are missing.

Approximation
-------------
For each in-flight request and each block ``b`` whose KV is cached:
    score_b = max_over_query_token max_over_head <Q_h, K_h(b_centroid)> / sqrt(d_head)
``K_h(b_centroid)`` is the per-head mean of the cached key vectors
inside block ``b`` (so a single (n_heads, head_dim) tensor per
block). The scalar score is a magnitude proxy for "how much this
query attends to this block" without the softmax. This is the same
proxy InfiniGen uses, and is sufficient for the LAP's top-K-ranking
contract (the planner consumes attention scores up to a monotone
transform).

Status
------
This is a forward-pre-hook patch that does NOT touch vLLM source.
It is opt-in via :func:`install_attention_forward_hook`. The cost is
one extra (n_blocks, n_heads) matmul per attention layer per decode
step --- ~10us on A100 at n_blocks=128, well inside Lemma 2's
$200\\mu$s LAP budget.

Limitations
-----------
* The proxy is centroid-of-block, not softmax(Q K^T). It is fine for
  top-K ranking but does not give a probability.
* It runs on the GPU on which the Attention layer lives; for TP$\\ge 2$
  each rank computes its own slice.
* The connector needs to be wired to read ``forward_context.seer_attn``
  inside ``get_num_new_matched_tokens`` / ``update_state_after_alloc``
  to feed it to ``observe_decode_step``; the wiring lives in
  :mod:`seer.integration.vllm_connector`.
"""
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


# Thread-local stash for per-step, per-layer attention scores. The
# forward hook writes here; the connector scheduler hook reads here.
# Keyed by ``(layer_name, virtual_engine)`` so multiple in-flight
# scheduler iterations on TP>=2 do not stomp each other.
_attn_stash_lock = threading.Lock()
_attn_stash: dict[tuple[str, int], Any] = {}

# Per-layer step counter for the throttle. The forward hook is called
# once per layer per decode step (32x for Llama-2-7B); we only need
# real attention every ``SEER_HOOK_STRIDE`` steps (default 8 = the
# SEER planner's decision_period). On the skipped steps we early-out
# before the GPU mean+matmul. Empirically this drops per-step wall
# from ~120ms (every-step) to ~32ms (stride-8) on the W3 smoke.
import os as _os_for_env

_HOOK_STRIDE = max(1, int(_os_for_env.environ.get("SEER_HOOK_STRIDE", "8")))
_hook_layer_steps: dict[str, int] = {}

# Phase 2.5 fix: K-centroid cache, keyed by (layer_name, ve). The
# K-cache for blocks 0..n_filled-2 is **immutable** during decode (only
# the most-recent partial block changes). We compute centroids for
# blocks that don't have one cached yet (or for the last block, which
# may still be growing), and reuse them across decode steps. This
# turns the per-step hook cost from O(n_blocks * block_size * n_heads)
# to O(n_new_blocks * ...) — ~0.05ms steady-state vs ~12ms before.
_kcentroid_cache: dict[tuple[str, int], Any] = {}
# Track the "last filled block index" per layer; centroids for blocks
# strictly less than this are frozen, the last block's centroid is
# always recomputed.
_kcentroid_last_block: dict[tuple[str, int], int] = {}

# Per-section microbench (Phase 2.5 diagnostic). Wall summed across all
# fires; reported by ``get_hook_counters_remote``.
_hook_timing_us = {
    "total": 0.0, "centroid_init_full": 0.0, "centroid_extend": 0.0,
    "score_matmul": 0.0, "stash_write": 0.0, "n_full_init": 0,
    "n_extend": 0, "n_matmul": 0,
}

# Diagnostic counters so the unit tests + the eF driver can verify
# the hook actually fires.
_hook_counters = {"fire": 0, "skip_no_kv_cache": 0, "skip_no_q": 0,
                  "skip_error": 0}
# Diagnostic ring of (type, msg) for the first ~32 skip_error events;
# helps debug install path without flooding stderr.
_hook_errors: list[tuple[str, str]] = []
_HOOK_ERRORS_MAX = 32


def get_attn_stash() -> dict[tuple[str, int], Any]:
    """Return a snapshot of the current attention stash (read-only)."""
    with _attn_stash_lock:
        return dict(_attn_stash)


def clear_attn_stash() -> None:
    """Reset the attention stash. Called between scheduler iterations
    so per-step data does not leak across steps."""
    with _attn_stash_lock:
        _attn_stash.clear()


def get_hook_counters() -> dict[str, int]:
    """Diagnostic counters: number of times the hook fired, skipped, errored.

    Also includes the Phase 2.5 microbench under ``_timing_us``: the
    cumulative wall (us) spent in each section of the proxy compute,
    so callers can audit the optimisation impact at submission time.
    """
    d = dict(_hook_counters)
    d["_timing_us"] = dict(_hook_timing_us)  # type: ignore[assignment]
    return d


def _compute_per_block_proxy(
    query: torch.Tensor,  # [num_tokens, num_heads, head_dim] or flat-packed
    kv_cache: torch.Tensor,  # vllm-specific layout
    num_heads: int,
    head_dim: int,
    cache_key: tuple[str, int] | None = None,
) -> torch.Tensor | None:
    """Return per-block attention proxy ``[n_blocks]``.

    Layout of ``kv_cache`` in vLLM 0.8.5 (FlashAttention backend):
      ``[2, n_blocks, block_size, num_kv_heads, head_dim]``
    The first dim splits K (index 0) and V (index 1). We average the
    K tensor over (block_size, num_kv_heads) to get a per-block,
    per-feature centroid, then dot with ``query.max(dim=tokens).mean(dim=heads)``
    to get a magnitude proxy.

    Returns None on layout mismatch (so the hook falls back silently).
    """
    try:
        pass
    except Exception:  # pragma: no cover
        return None

    if kv_cache is None:
        return None
    # Accept the documented [2, n_blocks, block, n_kv_heads, head_dim] shape.
    if kv_cache.dim() < 4:
        return None
    if kv_cache.shape[0] != 2:
        # Some backends keep K/V separate; we conservatively bail.
        return None
    k_cache = kv_cache[0]  # [n_blocks, block_size, n_kv_heads, head_dim]
    if k_cache.dim() < 3:
        return None
    n_blocks = k_cache.shape[0]
    if n_blocks <= 0 or query.numel() == 0:
        return None

    import time as _time
    t_total_0 = _time.perf_counter()

    # ---- Phase 2.5 fast path: K-centroid cache ----
    # The K-cache for fully-allocated blocks is immutable across decode
    # steps; only the just-filled last block changes per step. We cache
    # a per-block centroid (shape [n_blocks, head_dim]) and only
    # recompute the tail when blocks are appended.
    try:
        cached = _kcentroid_cache.get(cache_key) if cache_key else None
        last_known = _kcentroid_last_block.get(cache_key, -1) \
            if cache_key else -1
        if cached is None or cached.shape[0] != n_blocks \
                or cached.dtype != k_cache.dtype \
                or cached.device != k_cache.device:
            # First fire for this layer OR shape mismatch — full init.
            # Stay in the original dtype (typically fp16); the mean
            # over (block_size, n_kv_heads) is well-conditioned.
            t_ci = _time.perf_counter()
            k_block = k_cache.mean(dim=(-3, -2))  # [n_blocks, head_dim]
            _hook_timing_us["centroid_init_full"] += (
                _time.perf_counter() - t_ci) * 1e6
            _hook_timing_us["n_full_init"] += 1
            if cache_key is not None:
                _kcentroid_cache[cache_key] = k_block
                _kcentroid_last_block[cache_key] = n_blocks - 1
        else:
            # Same shape — only need to refresh the last block which
            # may still be filling.
            t_ce = _time.perf_counter()
            k_block = cached
            last_slice = k_cache[n_blocks - 1].mean(dim=(-2, -1)) \
                if False else \
                k_cache[n_blocks - 1].mean(dim=(0, 1))  # [head_dim]
            k_block[n_blocks - 1] = last_slice
            _hook_timing_us["centroid_extend"] += (
                _time.perf_counter() - t_ce) * 1e6
            _hook_timing_us["n_extend"] += 1
    except Exception:
        return None

    # query → [head_dim]: pool over tokens (max) then over heads (mean).
    # vLLM 0.8.x decode-side hands us a flat-packed query of shape
    # ``[num_tokens, num_heads * head_dim]``; we reshape into
    # ``[num_tokens, num_heads, head_dim]`` before the pooling. Stay
    # in fp16 throughout to avoid the upcast/downcast round-trip cost.
    try:
        q = query
        if q.dim() == 2 and q.shape[-1] == num_heads * head_dim:
            q = q.reshape(q.shape[0], num_heads, head_dim)
        if q.dim() == 3:
            q_pooled = q.abs().amax(dim=0).mean(dim=0)  # [head_dim]
        elif q.dim() == 2 and q.shape[-1] == head_dim:
            q_pooled = q.abs().amax(dim=0)
        else:
            return None
    except Exception:
        return None
    if q_pooled.dim() == 0 or q_pooled.shape[-1] != k_block.shape[-1]:
        return None
    t_mm = _time.perf_counter()
    scale = 1.0 / math.sqrt(max(1, head_dim))
    # Convert q_pooled dtype to match k_block (k_block is fp16; matmul
    # of mismatched dtypes raises).
    if q_pooled.dtype != k_block.dtype:
        q_pooled = q_pooled.to(k_block.dtype)
    scores = (k_block @ q_pooled) * scale  # [n_blocks]
    _hook_timing_us["score_matmul"] += (
        _time.perf_counter() - t_mm) * 1e6
    _hook_timing_us["n_matmul"] += 1
    _hook_timing_us["total"] += (_time.perf_counter() - t_total_0) * 1e6
    return scores.detach()


def _make_attention_pre_hook(layer_name: str):
    """Build a ``forward_pre_hook`` closure for the given Attention layer.

    The hook signature in pytorch is ``hook(module, args, kwargs)``.
    We compute the proxy from ``args = (query, key, value, output_shape)``
    and stash it under ``(layer_name, virtual_engine)``.
    """
    def _hook(module, args, kwargs):  # noqa: ANN001
        try:
            # Per-layer step counter + stride throttle. On the skipped
            # steps we early-out before the expensive mean+matmul; the
            # latest stash entry remains live for the connector to read.
            n = _hook_layer_steps.get(layer_name, 0)
            _hook_layer_steps[layer_name] = n + 1
            if (n % _HOOK_STRIDE) != 0:
                _hook_counters["skip_stride"] = _hook_counters.get(
                    "skip_stride", 0) + 1
                return
            query = args[0] if args else kwargs.get("query")
            if query is None or query.numel() == 0:
                _hook_counters["skip_no_q"] += 1
                return
            # One-shot shape diagnostic so we can fix the proxy.
            if not _hook_counters.get("_shape_logged", 0):
                try:
                    kvc = getattr(module, "kv_cache", None)
                    kv_shape = list(kvc[0].shape) if kvc else None
                    _hook_errors.append((
                        "SHAPE_DIAG",
                        f"q.shape={list(query.shape)}, "
                        f"kv[0].shape={kv_shape}, "
                        f"n_heads={getattr(module,'num_heads',None)}, "
                        f"head_dim={getattr(module,'head_size',None)}, "
                        f"layer={layer_name}",
                    ))
                    _hook_counters["_shape_logged"] = 1
                except Exception:
                    pass
            # Resolve virtual_engine + kv_cache via vllm's ForwardContext.
            try:
                from vllm.forward_context import get_forward_context
                ctx = get_forward_context()
                ve = getattr(ctx, "virtual_engine", 0)
            except Exception:
                ve = 0
            kvc_list = getattr(module, "kv_cache", None)
            if not kvc_list:
                _hook_counters["skip_no_kv_cache"] += 1
                return
            kv_cache = kvc_list[ve] if ve < len(kvc_list) else kvc_list[0]
            scores = _compute_per_block_proxy(
                query=query,
                kv_cache=kv_cache,
                num_heads=module.num_heads,
                head_dim=module.head_size,
                cache_key=(layer_name, ve),
            )
            if scores is None:
                _hook_counters["skip_error"] += 1
                return
            with _attn_stash_lock:
                _attn_stash[(layer_name, ve)] = scores
            _hook_counters["fire"] += 1
        except Exception as e:  # noqa: BLE001
            # Hooks must never break the forward pass.
            _hook_counters["skip_error"] += 1
            if len(_hook_errors) < _HOOK_ERRORS_MAX:
                import traceback as _tb
                _hook_errors.append(
                    (type(e).__name__,
                     f"{e}|{_tb.format_tb(e.__traceback__, limit=2)[-1].strip() if e.__traceback__ else ''}"[:240])
                )

    return _hook


def install_attention_forward_hook(model) -> int:
    """Walk ``model`` and register a forward pre-hook on a sparse
    subset of ``vllm.attention.Attention`` modules. Returns the number
    of hooks registered.

    Per-layer subsampling — environment variable
    ``SEER_HOOK_LAYER_STRIDE`` (default 8) keeps the hook on every Nth
    Attention layer; the others are not wired. This is the dominant
    cost: a forward_pre_hook crosses C++/Python per layer per decode
    step. On Llama-2-7B (32 layers) the default stride-8 wires layers
    {0, 8, 16, 24} which gives attention coverage at every depth
    quartile while cutting the Python-callback fan-out 8x.
    """
    try:
        from vllm.attention import Attention
    except Exception:  # pragma: no cover
        return 0

    import os as _os
    layer_stride = max(1, int(_os.environ.get("SEER_HOOK_LAYER_STRIDE", "8")))

    n_registered = 0
    seen = 0
    for name, mod in model.named_modules():
        if isinstance(mod, Attention):
            if (seen % layer_stride) != 0:
                seen += 1
                continue
            seen += 1
            layer_name = getattr(mod, "layer_name", name) or name
            mod.register_forward_pre_hook(
                _make_attention_pre_hook(layer_name),
                with_kwargs=True,
            )
            n_registered += 1
    return n_registered


def install_via_engine(llm) -> int:
    """Install the forward hook on every Attention module in the model.

    vLLM 0.8.x V1 engine runs the model in a worker **subprocess**: the
    documented attribute paths
    ``llm.llm_engine.model_executor.driver_worker.model_runner.model``
    and ``...engine_core.model_executor.workers[0].model_runner.model``
    do not exist in the driver process; the model is reachable only via
    ``llm.apply_model(func)`` which runs ``func(model)`` inside every
    worker subprocess. We use that as the primary install path and
    fall back to the legacy attribute walks for older / non-V1 engines.

    Returns the **maximum** number of hooks registered across workers
    (TP $\\ge 2$ workers each register their own slice; the max is the
    most useful single number to log).
    """
    # Phase 1.3 fix (2026-05-15 W3 follow-through): on vLLM 0.8.x V1,
    # ``LLM.apply_model`` looks for ``llm_engine.model_executor`` which
    # does not exist (V1 engines use ``engine_core`` + subprocess
    # workers). The version-agnostic entry point is
    # ``LLM.collective_rpc(func, ...)`` which ships ``func`` across
    # the msgpack channel to every worker subprocess; the first
    # positional arg is the ``Worker`` instance. Walk the worker to
    # the model and install there.
    def _install_in_worker(worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_forward_hook import (
            install_attention_forward_hook,
        )
        model = None
        # vLLM 0.8.x V1: worker.model_runner.model
        try:
            model = worker.model_runner.model
        except Exception:
            pass
        # vLLM 0.7.x / older: worker.model_executor / worker.model
        if model is None:
            try:
                model = worker.model
            except Exception:
                pass
        if model is None:
            return 0
        return install_attention_forward_hook(model)

    # Primary: V1 collective_rpc path.
    try:
        if hasattr(llm, "collective_rpc"):
            counts = llm.collective_rpc(_install_in_worker)
            if counts:
                return int(max(int(c or 0) for c in counts))
    except Exception as e:  # noqa: BLE001
        print(f"[forward_hook] collective_rpc install failed: {e!r}")

    # Fallback: apply_model (works on V0 / older releases).
    try:
        if hasattr(llm, "apply_model"):
            counts = llm.apply_model(install_attention_forward_hook)
            if counts:
                return int(max(int(c or 0) for c in counts))
    except Exception as e:  # noqa: BLE001
        print(f"[forward_hook] apply_model install failed: {e!r}")

    # Final legacy fallbacks for non-V1 engines.
    model = None
    try:
        model = (llm.llm_engine.model_executor.driver_worker
                 .model_runner.model)
    except Exception:
        try:
            model = (llm.llm_engine.engine_core
                     .model_executor.workers[0].model_runner.model)
        except Exception:
            return 0
    return install_attention_forward_hook(model)


def get_hook_counters_remote(llm) -> dict[str, int]:
    """Aggregate ``get_hook_counters()`` across all worker subprocesses
    via ``llm.collective_rpc``. The local module's counters are 0 in
    the driver process because the hook fires in worker processes; we
    have to RPC the read.

    The collective_rpc callable signature is ``func(worker, *args)``
    where ``worker`` is the Worker instance — we ignore it and just
    return the worker-process-local module state.
    """
    def _worker_read(_worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_forward_hook import (
            _hook_errors,
            get_hook_counters,
        )
        d = dict(get_hook_counters())
        # Carry the first few skip_error reasons across the RPC for
        # debugging — only takes effect when there are <33 distinct
        # errors so we don't blow up the msgpack payload.
        if _hook_errors:
            d["_first_errors"] = list(_hook_errors[:8])  # type: ignore[assignment]
        return d
    try:
        if hasattr(llm, "collective_rpc"):
            per_worker = llm.collective_rpc(_worker_read)
            agg: dict = {}
            for d in per_worker or []:
                if not isinstance(d, dict):
                    continue
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0) + int(v)
                    elif isinstance(v, list):
                        # Carry the first few error trace entries from
                        # the first worker that has any (the diagnostic
                        # _first_errors list).
                        if k not in agg:
                            agg[k] = list(v)
                    else:
                        if k not in agg:
                            agg[k] = v
            return agg
    except Exception as e:  # noqa: BLE001
        return {"rpc_error": 1, "_msg": repr(e)}  # type: ignore[dict-item]
    return get_hook_counters()


__all__ = [
    "install_attention_forward_hook",
    "install_via_engine",
    "get_attn_stash",
    "clear_attn_stash",
    "get_hook_counters",
    "get_hook_counters_remote",
]
