"""Step 0c — *batched-compute* attention proxy: one Python callback
+ one big-tensor compute that fuses all 32 layers' K-centroids and
matmuls into a small number of CUDA kernel launches.

Step 0 (monkey-patch) and Step 0b (single hook + 32-layer Python loop)
both hit a CUDA-kernel-launch ceiling: 32 layers × ~5 tensor ops each
× ~30 us/launch = ~5 ms cumulative overhead per engine.step. P999
spikes to 200 ms regardless of dispatch trigger.

This module amortises the launch overhead by:

1. **Caching all layers' KV-cache references** in a single list at
   install time.
2. On each fire, **stacking** the per-layer K-centroids into one
   ``[n_layers, n_blocks, head_dim]`` tensor (a single stack +
   single mean kernel, instead of 32 separate mean calls).
3. Doing **one** batched matmul ``[n_layers, n_blocks, d] @ [d]``
   → ``[n_layers, n_blocks]`` via einsum (single CUDA kernel).
4. Slicing the result per layer and writing each slice into the
   shared ``_attn_stash``.

The win over Step 0b is the **constant** number of CUDA launches
per fire (5--10 instead of 100+), independent of layer count.

This is the last pure-Python lever before C++ extension. If P999
still spikes here, the launch overhead really is dominated by the
PyTorch hook machinery itself plus the Q tensor's per-layer access
patterns, and the only remaining path is a C++/CUDA extension that
does the entire compute in a single fused kernel.
"""
from __future__ import annotations

import threading
from typing import Any

_batched_lock = threading.Lock()
_attn_module_list: list[Any] = []
_kcentroid_stacked: Any = None  # [n_layers, max_blocks_seen, head_dim]
_last_n_blocks: list[int] = []
_batched_counters = {
    "n_install": 0,
    "n_fires": 0,
    "n_layers_processed": 0,
    "n_errors": 0,
}
_first_errors: list[tuple[str, str]] = []


def _make_batched_compute_hook(attn_modules: list[Any]):
    """Closure: on each fire, batch-process all 32 layers' KV caches
    and queries into a small number of CUDA kernels."""
    from seer.integration.vllm_forward_hook import (
        _HOOK_STRIDE,
        _attn_stash,
        _attn_stash_lock,
        _hook_counters,
        _hook_layer_steps,
    )
    try:
        from vllm.forward_context import get_forward_context as _get_ctx
    except Exception:
        _get_ctx = None

    # Pre-fetch per-layer attributes once
    layer_names = [
        (getattr(m, "layer_name", None) or f"layer_{i}")
        for i, m in enumerate(attn_modules)
    ]
    num_heads = attn_modules[0].num_heads
    head_dim = attn_modules[0].head_size
    n_layers = len(attn_modules)

    def _hook(module, args, kwargs):  # noqa: ANN001
        global _kcentroid_stacked
        try:
            _batched_counters["n_fires"] += 1
            n = _hook_layer_steps.get("__batched_compute__", 0)
            _hook_layer_steps["__batched_compute__"] = n + 1
            if (n % _HOOK_STRIDE) != 0:
                _hook_counters["skip_stride"] = _hook_counters.get(
                    "skip_stride", 0) + 1
                return

            query = args[0] if args else kwargs.get("query")
            if query is None or query.numel() == 0:
                _hook_counters["skip_no_q"] += 1
                return

            ve = 0
            if _get_ctx is not None:
                try:
                    ctx = _get_ctx()
                    ve = getattr(ctx, "virtual_engine", 0)
                except Exception:
                    pass

            import torch

            # Gather KV caches for all layers. Cheap: just attribute reads.
            k_caches = []
            for attn_mod in attn_modules:
                kvc_list = getattr(attn_mod, "kv_cache", None)
                if not kvc_list:
                    k_caches.append(None)
                    continue
                kv_cache = (kvc_list[ve] if ve < len(kvc_list)
                            else kvc_list[0])
                if kv_cache is None or kv_cache.dim() < 4 \
                        or kv_cache.shape[0] != 2:
                    k_caches.append(None)
                    continue
                k_caches.append(kv_cache[0])  # [n_blocks, block, n_kv_h, d]

            # All layers should share the same n_blocks (they grow
            # in lockstep). Use the first valid one.
            n_blocks = None
            for kc in k_caches:
                if kc is not None:
                    n_blocks = int(kc.shape[0])
                    break
            if n_blocks is None or n_blocks == 0:
                return

            # Reduce per-layer FIRST (mean over block_size + n_kv_heads
            # while still indexed by layer), then stack the small
            # centroids — stacking full kv caches OOMs at ~28 GB on
            # Llama-2-7B / 32 layers.
            valid_idx = [i for i, kc in enumerate(k_caches) if kc is not None]
            centroids_list = []
            for i in valid_idx:
                # [n_blocks, block_size, n_kv_heads, head_dim] -> [n_blocks, head_dim]
                centroids_list.append(k_caches[i].mean(dim=(-3, -2)))
            # Stack -> [n_layers_valid, n_blocks, head_dim] (small: ~55 MB)
            k_centroids = torch.stack(centroids_list, dim=0)

            # Pool query: [num_tokens, n_heads*d] or [num_tokens, h, d]
            q = query
            if q.dim() == 2 and q.shape[-1] == num_heads * head_dim:
                q = q.reshape(q.shape[0], num_heads, head_dim)
            if q.dim() == 3:
                q_pooled = q.abs().amax(dim=0).mean(dim=0)  # [d]
            elif q.dim() == 2 and q.shape[-1] == head_dim:
                q_pooled = q.abs().amax(dim=0)
            else:
                return
            if q_pooled.dtype != k_centroids.dtype:
                q_pooled = q_pooled.to(k_centroids.dtype)

            # Batched matmul: [n_layers_valid, n_blocks, d] @ [d]
            # = [n_layers_valid, n_blocks]
            import math
            scale = 1.0 / math.sqrt(max(1, head_dim))
            # einsum is one kernel launch for the whole batch
            scores_batch = torch.einsum(
                "lbd,d->lb", k_centroids, q_pooled) * scale  # [L, n_blocks]

            # Stash per-layer slices. Move all to CPU in ONE copy.
            scores_cpu = scores_batch.detach().to("cpu", non_blocking=False)
            with _attn_stash_lock:
                for write_idx, mod_idx in enumerate(valid_idx):
                    _attn_stash[(layer_names[mod_idx], ve)] = \
                        scores_cpu[write_idx]
                    _batched_counters["n_layers_processed"] += 1
                    _hook_counters["fire"] += 1
        except Exception as e:  # noqa: BLE001
            _batched_counters["n_errors"] += 1
            if len(_first_errors) < 8:
                import traceback as _tb
                _first_errors.append(
                    (type(e).__name__,
                     f"{e}|{_tb.format_tb(e.__traceback__, limit=2)[-1].strip() if e.__traceback__ else ''}"[:280])
                )

    return _hook


def install_batched_compute_hook(model) -> int:
    try:
        from vllm.attention import Attention
    except Exception:
        return 0
    attn_modules = []
    for _name, mod in model.named_modules():
        if isinstance(mod, Attention):
            attn_modules.append(mod)
    if not attn_modules:
        return 0
    with _batched_lock:
        _attn_module_list[:] = attn_modules

    attn_modules[0].register_forward_pre_hook(
        _make_batched_compute_hook(attn_modules),
        with_kwargs=True,
    )
    _batched_counters["n_install"] += 1
    return len(attn_modules)


def install_via_engine(llm) -> int:
    def _do_install(worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_attention_batched_compute import (
            install_batched_compute_hook,
        )
        model = None
        try:
            model = worker.model_runner.model
        except Exception:
            pass
        if model is None:
            return 0
        return install_batched_compute_hook(model)

    try:
        if hasattr(llm, "collective_rpc"):
            counts = llm.collective_rpc(_do_install)
            return int(max(int(c or 0) for c in counts)) if counts else 0
    except Exception as e:  # noqa: BLE001
        print(f"[batched_compute] install failed: {e!r}")
    return 0


def get_counters_remote(llm) -> dict[str, int]:
    def _read(_worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_attention_batched_compute import (
            _batched_counters,
            _first_errors,
        )
        from seer.integration.vllm_forward_hook import get_hook_counters
        d = dict(_batched_counters)
        d.update(get_hook_counters())
        if _first_errors:
            d["_first_errors"] = list(_first_errors)  # type: ignore
        return d
    try:
        if hasattr(llm, "collective_rpc"):
            per_worker = llm.collective_rpc(_read)
            agg: dict = {}
            for d in per_worker or []:
                if not isinstance(d, dict):
                    continue
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0) + int(v)
                    elif isinstance(v, list) and k not in agg:
                        agg[k] = list(v)
            return agg
    except Exception as e:  # noqa: BLE001
        return {"rpc_error": 1, "_msg": repr(e)}  # type: ignore[dict-item]
    return dict(_batched_counters)


__all__ = [
    "install_batched_compute_hook",
    "install_via_engine",
    "get_counters_remote",
]
