"""Step 0 sanity test — monkey-patch ``Attention.forward`` instead of
registering a ``forward_pre_hook``.

The Phase 2.5 short-form bisection (commit ``b8256db``) proved that
a registered ``forward_pre_hook`` on vLLM 0.8.5.post1's
:class:`vllm.attention.Attention` adds enough PyTorch C++/Python
dispatch overhead (hook-dict lookup + GIL + callable invoke) to lift
per-prompt P999 from $20.7$\\,ms (no hook) to $75.8$\\,ms even with
an empty-body hook. That dispatch path is fundamentally what slows
us down — not the hook body's compute.

This module tries a different interception mechanism that may bypass
the slow dispatch: **class-level monkey-patch of ``Attention.forward``**.
Python's normal method dispatch (attribute lookup on instance ->
class -> MRO) is the same path vLLM already uses to call
``forward()``; replacing the bound method on the class adds zero
new dispatch overhead. The wrapped method calls the original at the
end, so vLLM's attention compute is unchanged; we just get one cheap
Python pre-call slot per layer per step where we can stash the query
tensor + a KV-cache pointer.

If this monkey-patch reproduces the no-hook P999 (~21 ms), the C++
extension work is unnecessary for the dispatch-side problem (we keep
the centroid-cache Python compute and stash without paying the
forward_pre_hook penalty). If P999 still spikes to ~76 ms, the
dispatch ceiling really is structural and the next step is the C++
extension.
"""
from __future__ import annotations

import threading
from typing import Any

# Per-layer step counter + the per-(layer, ve) stash, shared with
# vllm_forward_hook so the connector's planner can read either.
_patch_lock = threading.Lock()
_patched_class: Any = None
_patched_original_forward: Any = None
_patch_counters = {
    "n_classes_patched": 0,
    "n_calls": 0,
    "n_stashed": 0,
    "n_errors": 0,
}


def _hooked_forward(original_forward):
    """Closure factory: wrap the bound method ``original_forward`` so
    that every call records the query tensor + KV-cache snapshot in
    the shared stash before delegating to the original implementation.

    All imports are resolved once at closure creation time (not per
    call) so the wrapped method runs at native Python method-call
    speed plus the proxy compute itself.
    """
    # Resolve imports + module-level refs ONCE; per-call body is hot path.
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

    def _forward(self, query, key, value, *args, **kwargs):  # noqa: ANN001
        try:
            _patch_counters["n_calls"] += 1
            layer_name = getattr(self, "layer_name", None) \
                or self.__class__.__name__
            n = _hook_layer_steps.get(layer_name, 0)
            _hook_layer_steps[layer_name] = n + 1
            if (n % _HOOK_STRIDE) == 0:
                ve = 0
                if _get_ctx is not None:
                    try:
                        ctx = _get_ctx()
                        ve = getattr(ctx, "virtual_engine", 0)
                    except Exception:
                        pass
                kvc_list = getattr(self, "kv_cache", None)
                if kvc_list:
                    kv_cache = kvc_list[ve] if ve < len(kvc_list) \
                        else kvc_list[0]
                    scores = _compute_per_block_proxy(
                        query=query,
                        kv_cache=kv_cache,
                        num_heads=self.num_heads,
                        head_dim=self.head_size,
                        cache_key=(layer_name, ve),
                    )
                    if scores is not None:
                        with _attn_stash_lock:
                            _attn_stash[(layer_name, ve)] = scores
                        _patch_counters["n_stashed"] += 1
                        _hook_counters["fire"] += 1
            else:
                _hook_counters["skip_stride"] = _hook_counters.get(
                    "skip_stride", 0) + 1
        except Exception:  # noqa: BLE001
            _patch_counters["n_errors"] += 1
        return original_forward(self, query, key, value, *args, **kwargs)

    return _forward


def install_attention_monkey_patch(model) -> int:
    """Replace ``Attention.forward`` on the *class* (not per-instance)
    with our wrapped method. Returns the number of classes patched
    (typically 1 since all Attention modules share a class).

    Run this from inside a worker subprocess via ``collective_rpc``,
    same pattern as the forward_pre_hook install.
    """
    global _patched_class, _patched_original_forward

    try:
        from vllm.attention import Attention
    except Exception:  # pragma: no cover
        return 0

    # Find all Attention classes actually used (could be subclasses).
    classes: set[type] = set()
    for _name, mod in model.named_modules():
        if isinstance(mod, Attention):
            classes.add(type(mod))
    if not classes:
        return 0

    n_patched = 0
    with _patch_lock:
        for cls in classes:
            if getattr(cls, "_seer_monkey_patched", False):
                continue
            orig = cls.forward
            cls.forward = _hooked_forward(orig)
            cls._seer_monkey_patched = True
            cls._seer_monkey_original = orig
            n_patched += 1
            _patched_class = cls
            _patched_original_forward = orig
    _patch_counters["n_classes_patched"] += n_patched
    return n_patched


def uninstall_attention_monkey_patch() -> int:
    """Restore the original ``forward`` on patched classes. Used for
    ablation comparisons (with-patch vs without-patch within a single
    process). Returns the count restored.
    """
    n = 0
    with _patch_lock:
        # We patched the class in-place, no list kept — fall back to
        # the singleton _patched_class.
        if _patched_class is not None and _patched_original_forward is not None:
            try:
                _patched_class.forward = _patched_original_forward
                if hasattr(_patched_class, "_seer_monkey_patched"):
                    delattr(_patched_class, "_seer_monkey_patched")
                n += 1
            except Exception:  # noqa: BLE001
                pass
    return n


def install_via_engine(llm) -> int:
    """Driver-side entry: ship the install closure into every worker
    subprocess via ``collective_rpc``. Mirrors the
    :func:`seer.integration.vllm_forward_hook.install_via_engine`
    pattern but installs the class-method monkey-patch instead of a
    forward_pre_hook.
    """
    def _do_install(worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_attention_monkey import (
            install_attention_monkey_patch,
        )
        model = None
        try:
            model = worker.model_runner.model
        except Exception:
            pass
        if model is None:
            return 0
        return install_attention_monkey_patch(model)

    try:
        if hasattr(llm, "collective_rpc"):
            counts = llm.collective_rpc(_do_install)
            return int(max(int(c or 0) for c in counts)) if counts else 0
    except Exception as e:  # noqa: BLE001
        print(f"[monkey] collective_rpc install failed: {e!r}")
    return 0


def get_patch_counters() -> dict[str, int]:
    return dict(_patch_counters)


def get_patch_counters_remote(llm) -> dict[str, int]:
    """Aggregate ``get_patch_counters()`` across all worker subprocesses
    via ``llm.collective_rpc``. The local module's counter is 0 in
    the driver because the patched method runs in workers.
    """
    def _worker_read(_worker, *_args, **_kwargs):  # noqa: ANN001
        from seer.integration.vllm_attention_monkey import (
            get_patch_counters,
        )
        return get_patch_counters()
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
            return agg
    except Exception as e:  # noqa: BLE001
        return {"rpc_error": 1, "_msg": repr(e)}  # type: ignore[dict-item]
    return get_patch_counters()


__all__ = [
    "install_attention_monkey_patch",
    "uninstall_attention_monkey_patch",
    "install_via_engine",
    "get_patch_counters",
    "get_patch_counters_remote",
]
