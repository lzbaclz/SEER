"""R36e A: vLLM hot-path decode-step hook (replaces R31 skeleton).

The R31 cycle-2 scaffold shipped ``vllm_patches/0002-hot-path-allocation-hook.patch.skeleton``
with placeholder identifiers (``_seer_connector``, ``_deadline_slack_us``,
``HOOK_INSERTION_SITE``) that did not exist in vLLM. This module
replaces the patch-based approach with a **runtime monkey-patch** that
wraps :meth:`vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks`
so SEER can decide which subset of matched prefix-cache blocks to
materialise to HBM under the per-step slack budget — **on the
decode-step hot path, not on request termination**.

Why monkey-patch instead of source patch:
  - No vLLM fork to maintain. Works against any
    ``vllm.__version__`` that exposes ``KVCacheManager.get_computed_blocks``
    with the existing ``(request) -> tuple[list[KVCacheBlock], int]``
    contract (verified for ``0.8.5.post1``).
  - Reversible (:func:`uninstall_hot_path_hook`).
  - The exact same decision logic (:mod:`seer.integration.hot_path_router`)
    is shared between the unit tests and the runtime hook.

Operator workflow (cycle-2 ATC):

.. code-block:: python

    from vllm import LLM
    from seer.integration.vllm_hot_path_hook import (
        install_hot_path_hook, register_router,
        set_request_slo_us, set_request_last_forward_us)
    from seer.integration.hot_path_router import HotPathRouter

    install_hot_path_hook()
    router = HotPathRouter(scorer=my_lap_scorer, ell_bar_us=200.0)
    register_router(router)

    llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_prefix_caching=True)
    # Before each request, push the SLO + last-forward stats so the
    # hook can compute slack:
    for req in batch:
        set_request_slo_us(req, slo_us=200_000)
        set_request_last_forward_us(req, last_forward_us=110_000)
    llm.generate(prompts)

The hook does NOT modify vLLM internals on disk. It rebinds
``KVCacheManager.get_computed_blocks`` to the wrapped form when
:func:`install_hot_path_hook` is called and restores the original via
:func:`uninstall_hot_path_hook`.

Paper-text update (R36e): the §VI.D / §VIII "diagnosed non-binding"
framing should be retracted only when the hook is active in the
sweep. The hook + router constitute the A2 cycle-2 deliverable;
this module is what closes ``todo_atc.md`` items A2.1--A2.5.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any

from seer.integration.hot_path_router import HotPathRouter

LOG = logging.getLogger(__name__)

# Module-level state. Single router per process; matches the
# single-engine vLLM deployment model.
_router: HotPathRouter | None = None
_original_get_computed_blocks: Any | None = None
_install_count: int = 0


def register_router(router: HotPathRouter) -> None:
    """Install (or replace) the global hot-path decision router.

    Must be called *after* :func:`install_hot_path_hook`. Setting
    ``router = None`` disables routing (the hook falls through to
    vLLM's original behaviour).
    """
    global _router
    _router = router


def unregister_router() -> None:
    """Convenience: alias for ``register_router(None)``."""
    register_router(None)  # type: ignore[arg-type]


def set_request_slo_us(request: Any, slo_us: float) -> None:
    """Attach a per-request SLO budget the hook reads when computing slack.

    Defaults to ``200_000`` µs (the relaxed-$D$ band §I uses) if not set.
    """
    object.__setattr__(request, "_seer_slo_us", float(slo_us))


def set_request_last_forward_us(request: Any, last_forward_us: float) -> None:
    """Attach the most-recent forward-pass elapsed time the hook
    uses to estimate per-step slack.

    Defaults to ``0`` µs (i.e. the full SLO is treated as slack) if
    not set; the user's SLO scheduler should push this on every
    iteration for a tight estimate.
    """
    object.__setattr__(request, "_seer_last_forward_us",
                       float(last_forward_us))


def _estimate_slack_us(request: Any) -> float:
    slo_us = float(getattr(request, "_seer_slo_us", 200_000.0))
    last_forward_us = float(getattr(request, "_seer_last_forward_us", 0.0))
    return max(0.0, slo_us - last_forward_us)


def install_hot_path_hook() -> None:
    """Monkey-patch :class:`KVCacheManager.get_computed_blocks`.

    Idempotent: calling twice does not double-wrap. To restore the
    original, call :func:`uninstall_hot_path_hook`.
    """
    global _original_get_computed_blocks, _install_count
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except ImportError as e:
        raise RuntimeError(
            "vLLM is not importable; hot-path hook unavailable. "
            "Install vLLM 0.8.5+ in this env."
        ) from e

    if _install_count > 0:
        LOG.info("hot-path hook already installed (count=%d)",
                 _install_count)
        _install_count += 1
        return

    _original_get_computed_blocks = KVCacheManager.get_computed_blocks

    # Optional debug counter for diagnosing why the hook doesn't fire
    # in V1 (set SEER_HOT_PATH_DEBUG=1 to print every call).
    _call_counter = {"n_calls": 0, "n_with_blocks": 0}

    @functools.wraps(_original_get_computed_blocks)
    def _wrapped(self, request):
        blocks, num_tokens = _original_get_computed_blocks(self, request)
        _call_counter["n_calls"] += 1
        if blocks:
            _call_counter["n_with_blocks"] += 1
        if os.environ.get("SEER_HOT_PATH_DEBUG"):
            print(f"[hot-path-hook] get_computed_blocks called: "
                  f"n_calls={_call_counter['n_calls']} "
                  f"n_with_blocks={_call_counter['n_with_blocks']} "
                  f"this_call: blocks={len(blocks)} num_tokens={num_tokens} "
                  f"router_is_None={_router is None} pid={os.getpid()}",
                  flush=True)
        if _router is None or not blocks:
            return blocks, num_tokens

        slack_us = _estimate_slack_us(request)
        if slack_us <= 0:
            # Deadline already exhausted; behave as full materialisation
            # (vLLM's allocator will surface the deadline miss).
            return blocks, num_tokens

        block_ids = [b.block_id for b in blocks]
        try:
            order = _router.partial_materialisation_order(
                request_id=request.request_id,
                matched_blocks=block_ids,
                slack_budget_us=slack_us,
            )
            max_blocks = _router.max_blocks_under_budget(slack_us)
        except Exception as e:  # router bug must not crash vLLM
            LOG.warning("hot-path router raised %s; falling through",
                        e, exc_info=True)
            return blocks, num_tokens

        if max_blocks >= len(blocks):
            # Slack budget covers everything; no truncation.
            return blocks, num_tokens

        # Truncate to the router's top-`max_blocks` choices, preserving
        # the router's order. The dropped suffix is rebuilt on the
        # decode path under SEER's eviction policy.
        kept_indices = order[:max_blocks]
        kept_blocks = [blocks[i] for i in kept_indices]
        # `num_tokens` must remain a multiple of block_size; vLLM
        # guarantees `len(computed_blocks) * block_size` ==
        # `num_computed_tokens`.
        kept_num_tokens = len(kept_blocks) * self.block_size
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "hot-path: req=%s slack=%.0fus matched=%d kept=%d",
                request.request_id, slack_us,
                len(blocks), len(kept_blocks),
            )
        return kept_blocks, kept_num_tokens

    KVCacheManager.get_computed_blocks = _wrapped
    _install_count = 1
    LOG.info("hot-path hook installed on KVCacheManager.get_computed_blocks")


def uninstall_hot_path_hook() -> None:
    """Restore the original method. Idempotent."""
    global _original_get_computed_blocks, _install_count
    if _install_count == 0 or _original_get_computed_blocks is None:
        return
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except ImportError:
        return
    KVCacheManager.get_computed_blocks = _original_get_computed_blocks
    _original_get_computed_blocks = None
    _install_count = 0
    LOG.info("hot-path hook uninstalled")


def is_hot_path_hook_installed() -> bool:
    return _install_count > 0


def current_router() -> HotPathRouter | None:
    return _router
