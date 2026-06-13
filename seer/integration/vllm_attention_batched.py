"""Step 0b — *batched* attention proxy: single forward_pre_hook on layer
0 only, processes all 32 layers' KV caches in one Python callback.

Phase 2.5 short-form bisection (commit ``b8256db``) and Step 0
monkey-patch (commit pending) both showed that per-layer per-step
intervention is structurally slow on vLLM 0.8.5 — independent of
whether the trigger is ``forward_pre_hook`` or class-method monkey
patch, the **work** of 32 layer-level callbacks at ~60-80\\,us each
adds 2-5\\,ms per step and an outsized P999 spike.

This module amortises the dispatch + Python call cost by registering
a **single** forward_pre_hook on layer 0 only. Inside that one
callback we walk a pre-built list of all 32 ``Attention`` modules
and compute their per-block attention proxy in a tight Python loop
that does not pay the PyTorch hook-dispatch overhead per layer.

Trade-off:
* Layer 0's KV cache + query are current. Layers 1--31 see their
  KV cache from the just-completed previous step (one step stale).
  For the SEER planner's block-ranking purpose, one-step staleness
  is negligible — K-cache changes very slowly during decode.
* We use layer 0's pooled query as the query proxy for all layers
  (attention queries are correlated across layers; this is a
  documented assumption shared with InfiniGen-style top-K selection).

If this drops streaming P999 back near the no-hook 20.7\\,ms baseline,
Phase 2.5 ATC follow-up does not need a C++ extension — the same
LAP signal load-bearing claim holds with a vLLM-Python install.
"""
from __future__ import annotations

import threading
from typing import Any

_batched_lock = threading.Lock()
_attn_module_list: list[Any] = []
_batched_counters = {
    "n_install": 0,
    "n_fires": 0,
    "n_layers_processed": 0,
    "n_errors": 0,
}


def _make_batched_pre_hook(attn_modules: list[Any]):
    """Closure: on each fire of layer 0's pre-hook, walk
    ``attn_modules`` and compute proxies for all 32 layers using
    layer 0's pooled query.

    Imports are resolved once at closure creation; the body runs at
    native Python speed.
    """
    from seer.integration.vllm_forward_hook import (
        _HOOK_STRIDE,
        _attn_stash,
        _attn_stash_lock,
        _compute_per_block_proxy,
        _hook_counters,
        _hook_layer_steps,
    )
    try:
        from vllm.forward_context import get_forward_context as _get_ctx
    except Exception:  # pragma: no cover
        _get_ctx = None

    def _hook(module, args, kwargs):  # noqa: ANN001
        try:
            _batched_counters["n_fires"] += 1
            # Per-step stride throttle (counted on layer 0 only).
            n = _hook_layer_steps.get("__batched__", 0)
            _hook_layer_steps["__batched__"] = n + 1
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

            # Process all attention layers in a tight loop. Each
            # ``_compute_per_block_proxy`` call uses the per-layer
            # centroid cache so cumulative cost is small.
            for layer_idx, attn_mod in enumerate(attn_modules):
                try:
                    kvc_list = getattr(attn_mod, "kv_cache", None)
                    if not kvc_list:
                        continue
                    kv_cache = (kvc_list[ve]
                                if ve < len(kvc_list)
                                else kvc_list[0])
                    layer_name = (getattr(attn_mod, "layer_name", None)
                                  or f"layer_{layer_idx}")
                    scores = _compute_per_block_proxy(
                        query=query,  # shared layer-0 query proxy
                        kv_cache=kv_cache,
                        num_heads=attn_mod.num_heads,
                        head_dim=attn_mod.head_size,
                        cache_key=(layer_name, ve),
                    )
                    if scores is not None:
                        with _attn_stash_lock:
                            _attn_stash[(layer_name, ve)] = scores
                        _batched_counters["n_layers_processed"] += 1
                        _hook_counters["fire"] += 1
                except Exception:  # noqa: BLE001
                    _batched_counters["n_errors"] += 1
        except Exception:  # noqa: BLE001
            _batched_counters["n_errors"] += 1

    return _hook


def install_batched_hook(model) -> int:
    """Register one forward_pre_hook on the **first** Attention layer
    only. Inside that hook we process all 32 layers in a tight loop.

    Returns the number of layers that will be processed per fire
    (the size of the ``attn_modules`` list).
    """
    try:
        from vllm.attention import Attention
    except Exception:  # pragma: no cover
        return 0

    attn_modules = []
    for _name, mod in model.named_modules():
        if isinstance(mod, Attention):
            attn_modules.append(mod)
    if not attn_modules:
        return 0
    with _batched_lock:
        _attn_module_list[:] = attn_modules

    # Install the single hook on attn_modules[0] only.
    attn_modules[0].register_forward_pre_hook(
        _make_batched_pre_hook(attn_modules),
        with_kwargs=True,
    )
    _batched_counters["n_install"] += 1
    return len(attn_modules)


def install_via_engine(llm) -> int:
    """Driver-side: ship the batched installer into every worker."""
    def _do_install(worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_attention_batched import (
            install_batched_hook,
        )
        model = None
        try:
            model = worker.model_runner.model
        except Exception:
            pass
        if model is None:
            return 0
        return install_batched_hook(model)

    try:
        if hasattr(llm, "collective_rpc"):
            counts = llm.collective_rpc(_do_install)
            return int(max(int(c or 0) for c in counts)) if counts else 0
    except Exception as e:  # noqa: BLE001
        print(f"[batched] collective_rpc install failed: {e!r}")
    return 0


def get_counters_remote(llm) -> dict[str, int]:
    """Aggregate batched counters across workers."""
    def _read(_worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_attention_batched import (
            _batched_counters,
        )
        from seer.integration.vllm_forward_hook import get_hook_counters
        d = dict(_batched_counters)
        d.update(get_hook_counters())
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
            return agg
    except Exception as e:  # noqa: BLE001
        return {"rpc_error": 1, "_msg": repr(e)}  # type: ignore[dict-item]
    return dict(_batched_counters)


__all__ = [
    "install_batched_hook",
    "install_via_engine",
    "get_counters_remote",
]
