"""One-shot diagnostic: where does the model object live in vLLM 0.8.5.post1?

Loads Llama-2-7B via vllm.LLM, walks the engine attribute graph, and prints
every attribute path that leads to a torch.nn.Module containing an
Attention class. Output is consumed by the rewrite of
``install_via_engine`` so it stops returning 0 layers on this release.
"""
from __future__ import annotations

import inspect
import sys
import time


def _is_module(obj):
    try:
        import torch
        return isinstance(obj, torch.nn.Module)
    except Exception:
        return False


def _is_attention(obj):
    try:
        from vllm.attention import Attention
        return isinstance(obj, Attention)
    except Exception:
        return False


def _walk_attrs(root, prefix="root", depth=0, max_depth=6, seen=None):
    """Yield (path, obj) for every object reachable from root."""
    if seen is None:
        seen = set()
    if id(root) in seen or depth > max_depth:
        return
    seen.add(id(root))
    yield prefix, root
    # callable / built-in classes are dead ends
    if isinstance(root, (str, int, float, bool, bytes, type(None))):
        return
    # Try common containers
    for attr in dir(root):
        if attr.startswith("_") and not attr.startswith("__"):
            pass
        if attr.startswith("__") and attr.endswith("__"):
            continue
        try:
            child = getattr(root, attr)
        except Exception:
            continue
        if inspect.ismethod(child) or inspect.isfunction(child) or inspect.isbuiltin(child):
            continue
        if isinstance(child, type):
            continue
        path = f"{prefix}.{attr}"
        yield from _walk_attrs(child, path, depth + 1, max_depth, seen)


def main() -> int:
    print("[probe] importing vllm + loading Llama-2-7B (slow)...")
    t0 = time.time()
    import vllm
    print(f"[probe] vllm.__version__ = {vllm.__version__}")
    llm = vllm.LLM(
        model="meta-llama/Llama-2-7b-hf",
        dtype="float16",
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        enforce_eager=True,
        disable_log_stats=True,
    )
    print(f"[probe] LLM loaded in {time.time() - t0:.1f}s")

    # Top-level attrs on llm
    print("\n=== llm top-level attrs (Module / candidate containers) ===")
    for attr in sorted(dir(llm)):
        if attr.startswith("_"):
            continue
        try:
            child = getattr(llm, attr)
        except Exception:
            continue
        kind = type(child).__name__
        if _is_module(child):
            print(f"  llm.{attr}  -- nn.Module: {kind}")
        elif hasattr(child, "__dict__"):
            print(f"  llm.{attr}  -- {kind}")

    # llm_engine introspection
    print("\n=== walking llm.llm_engine looking for Module + Attention ===")
    paths_module = []
    paths_attention = []
    try:
        engine = llm.llm_engine
        for path, obj in _walk_attrs(engine, prefix="llm.llm_engine", max_depth=6):
            if _is_attention(obj):
                paths_attention.append((path, type(obj).__name__))
            elif _is_module(obj):
                cls = type(obj).__name__
                # Only print 'interesting' modules; skip stdlib torch
                if "Attention" in cls or "Model" in cls or "Llama" in cls:
                    paths_module.append((path, cls))
    except Exception as e:
        print(f"[probe] traversal failed: {e!r}")

    print(f"\n=== Module-like hits (Llama/Model/Attention in classname): {len(paths_module)} ===")
    for p, c in paths_module[:30]:
        print(f"  {c:>40s}  @  {p}")

    print(f"\n=== vllm.attention.Attention instances: {len(paths_attention)} ===")
    for p, c in paths_attention[:50]:
        print(f"  {c}  @  {p}")

    # If we found any nn.Module that looks like the LLM root, try named_modules()
    print("\n=== named_modules() on most-promising root ===")
    candidates = [(p, type(o).__name__, o) for p, o in
                  ((q, eval(q.replace("llm.", "llm."))) for q, c in paths_module)
                  if "Llama" in type(o).__name__ or "Model" in type(o).__name__]
    # Fallback: try the workers path on V1
    try:
        ec = llm.llm_engine.engine_core
        for attr in dir(ec):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(ec, attr)
            except Exception:
                continue
            kind = type(child).__name__
            print(f"  llm.llm_engine.engine_core.{attr} -- {kind}")
    except Exception as e:
        print(f"[probe] engine_core inspection: {e!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
