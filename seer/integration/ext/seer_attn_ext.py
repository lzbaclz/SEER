"""Load + verify the fused SEER attention-proxy CUDA kernel.

Phase 2.5 long-form: Step 0 / 0b / 0c (pure-Python paths) hit a CUDA
kernel-launch-overhead ceiling. This module JIT-compiles a single
fused CUDA kernel that processes all 32 layers in one launch, then
exposes ``seer_attn_proxy(k_caches, q_pooled) -> [L, max_blocks]``
to the Python connector hook.

Usage::

    from seer.integration.ext.seer_attn_ext import load_kernel
    ext = load_kernel()  # JIT-compiles on first call; cached after.
    scores = ext.seer_attn_proxy(k_caches_list, q_pooled_fp16)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KERNEL = _HERE / "seer_attn_kernel.cu"
_LOAD_LOCK = threading.Lock()
_CACHED_EXT = None


def load_kernel():
    """JIT-compile + cache the extension. Idempotent."""
    global _CACHED_EXT
    with _LOAD_LOCK:
        if _CACHED_EXT is not None:
            return _CACHED_EXT
        from torch.utils.cpp_extension import load
        # Force nvcc to match torch's CUDA version if both are installed.
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        # Build dir: a sibling so PyTorch puts the .so somewhere stable.
        build_dir = _HERE / "build"
        build_dir.mkdir(exist_ok=True)
        _CACHED_EXT = load(
            name="seer_attn_kernel",
            sources=[str(_KERNEL)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                f"-I{cuda_home}/include",
                # Target Ampere (A100 sm_80) + Hopper (H100 sm_90).
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_90,code=sm_90",
            ],
            extra_cflags=["-O3"],
            build_directory=str(build_dir),
            verbose=False,
        )
        return _CACHED_EXT


__all__ = ["load_kernel"]
