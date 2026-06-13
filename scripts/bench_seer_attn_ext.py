"""Microbench: fused CUDA kernel vs pure-Python centroid+matmul.

Builds synthetic K caches with the same shape as Llama-2-7B at
n_blocks=6744, block_size=16, n_kv_heads=32, head_dim=128, and
compares wall time across N iterations.
"""
import sys
import time

import torch


def bench():
    L = 32
    B = 6744
    S = 16
    H = 32
    D = 128
    N_ITERS = 100

    device = "cuda"
    dtype = torch.float16

    print(f"[bench] building {L} K caches shape=[{B},{S},{H},{D}] fp16 on {device}")
    k_caches = [torch.randn(B, S, H, D, dtype=dtype, device=device)
                for _ in range(L)]
    q_pooled = torch.randn(D, dtype=dtype, device=device)
    torch.cuda.synchronize()

    # Python centroid+stack+einsum (Step 0c body)
    def python_path():
        centroids = []
        for kc in k_caches:
            centroids.append(kc.mean(dim=(-3, -2)))  # [B, D]
        k_centroids = torch.stack(centroids, dim=0)  # [L, B, D]
        scores = torch.einsum("lbd,d->lb", k_centroids, q_pooled)
        return scores

    # Warmup
    for _ in range(10):
        s_py = python_path()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        s_py = python_path()
    torch.cuda.synchronize()
    t_py = (time.perf_counter() - t0) / N_ITERS * 1e3
    print(f"[bench] python centroid+einsum:       {t_py:8.3f} ms/iter  (output {tuple(s_py.shape)}, {s_py.dtype})")

    # Load extension
    from seer.integration.ext.seer_attn_ext import load_kernel
    print("[bench] JIT-compiling extension (first run only)...")
    ext = load_kernel()
    print(f"[bench] extension loaded: {ext}")

    # Warmup
    for _ in range(10):
        s_cu = ext.seer_attn_proxy(k_caches, q_pooled)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        s_cu = ext.seer_attn_proxy(k_caches, q_pooled)
    torch.cuda.synchronize()
    t_cu = (time.perf_counter() - t0) / N_ITERS * 1e3
    print(f"[bench] fused CUDA kernel:            {t_cu:8.3f} ms/iter  (output {tuple(s_cu.shape)}, {s_cu.dtype})")
    print(f"[bench] speedup vs python:            {t_py / t_cu:8.2f}x")

    # Numerical sanity: compare on first layer/block
    diff = (s_py.float() - s_cu.float()).abs()
    print(f"[bench] max abs diff: {diff.max().item():.4f}  mean abs diff: {diff.mean().item():.4f}")
    print(f"[bench] python[0,:5]: {s_py[0,:5].tolist()}")
    print(f"[bench]  fused[0,:5]: {s_cu[0,:5].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(bench())
