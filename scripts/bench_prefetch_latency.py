"""J.6 reference-backend prefetch latency benchmark.

Measures end-to-end H2D / D2H transport latency on the SEER
connector's pinned-host backend (now load-bearing, not a no-op).
The point of the experiment is to put a real number on the
$\\bar\\ell$ in Lemma 2's bound — published as a vendor proxy
($200\\,\\mu$s for DRAM, etc.) but never previously measured under
the actual connector flow.

Per layer / per block / per direction we run K=200 trials, drop the
first 20 as warm-up, and report P50 / P99 / P999. Output JSON goes
to ``experiments/eF_mixed_slo/results_vllm_xfer/bench.json``.

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_prefetch_latency.py \\
        --block_size 16 --num_blocks 64 --num_layers 32
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_blocks", type=int, default=64,
                    help="Number of KV blocks per layer (≈ residency).")
    ap.add_argument("--num_layers", type=int, default=32,
                    help="Llama-2-7B has 32 layers.")
    ap.add_argument("--block_size", type=int, default=16,
                    help="Tokens per block (vLLM default).")
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--num_kv_heads", type=int, default=32,
                    help="Llama-2-7B is 32 query heads, 32 kv heads (no GQA).")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--n_warmup", type=int, default=20)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="experiments/eF_mixed_slo/"
                    "results_vllm_xfer/bench.json")
    return ap


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("[bench-xfer] CUDA not available; aborting.")
        return 2

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    # Per-block KV shape: (2 [k,v], num_heads, block_size, head_dim).
    block_shape = (2, args.num_kv_heads, args.block_size, args.head_dim)
    bytes_per_block = (2 * args.num_kv_heads * args.block_size *
                       args.head_dim * (2 if dtype == torch.float16 else 4))
    print(f"[bench-xfer] block_shape={block_shape} dtype={dtype} "
          f"bytes/block={bytes_per_block:,}")

    # Allocate per-layer HBM (the "victim" KV pool) and host-pinned mirror.
    hbm_pool = [
        torch.randn((args.num_blocks, *block_shape), dtype=dtype, device="cuda")
        for _ in range(args.num_layers)
    ]
    host_pool = [
        torch.empty((args.num_blocks, *block_shape), dtype=dtype,
                    device="cpu", pin_memory=True)
        for _ in range(args.num_layers)
    ]
    print(f"[bench-xfer] HBM pool: {args.num_layers}×{args.num_blocks} blocks "
          f"= {args.num_layers * args.num_blocks * bytes_per_block / 1e9:.2f} GiB")

    transport = torch.cuda.Stream()

    # ---- Save (DtoH): per-block latency
    save_lat_us: list[float] = []
    for trial in range(args.n_trials):
        block_id = trial % args.num_blocks
        layer_id = trial % args.num_layers
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(transport):
            host_pool[layer_id][block_id].copy_(
                hbm_pool[layer_id][block_id], non_blocking=True
            )
        transport.synchronize()
        save_lat_us.append((time.perf_counter() - t0) * 1e6)

    # ---- Load (HtoD): per-block latency
    load_lat_us: list[float] = []
    for trial in range(args.n_trials):
        block_id = trial % args.num_blocks
        layer_id = trial % args.num_layers
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(transport):
            hbm_pool[layer_id][block_id].copy_(
                host_pool[layer_id][block_id], non_blocking=True
            )
        transport.synchronize()
        load_lat_us.append((time.perf_counter() - t0) * 1e6)

    # ---- Batch save/load: 8 blocks at once (typical eF prefetch budget)
    batch_n = 8
    batch_save_us: list[float] = []
    batch_load_us: list[float] = []
    for trial in range(args.n_trials):
        layer_id = trial % args.num_layers
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(transport):
            for b in range(batch_n):
                bid = (trial + b) % args.num_blocks
                host_pool[layer_id][bid].copy_(
                    hbm_pool[layer_id][bid], non_blocking=True
                )
        transport.synchronize()
        batch_save_us.append((time.perf_counter() - t0) * 1e6)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(transport):
            for b in range(batch_n):
                bid = (trial + b) % args.num_blocks
                hbm_pool[layer_id][bid].copy_(
                    host_pool[layer_id][bid], non_blocking=True
                )
        transport.synchronize()
        batch_load_us.append((time.perf_counter() - t0) * 1e6)

    def _summary(name, vec, n_warmup):
        v = sorted(vec[n_warmup:])
        n = len(v)
        if not n:
            return None
        pct = lambda p: v[min(n - 1, int((n - 1) * p))]
        return {
            "name": name, "n": n,
            "p50_us": pct(0.50), "p99_us": pct(0.99),
            "p999_us": pct(0.999), "max_us": v[-1],
            "mean_us": statistics.fmean(v),
        }

    summary = {
        "block_shape": list(block_shape),
        "dtype": str(dtype),
        "bytes_per_block": bytes_per_block,
        "num_blocks_per_layer": args.num_blocks,
        "num_layers": args.num_layers,
        "single_save_HtoD": _summary("single_save_DtoH", save_lat_us, args.n_warmup),
        "single_load_HtoD": _summary("single_load_HtoD", load_lat_us, args.n_warmup),
        "batch8_save_DtoH": _summary("batch8_save_DtoH", batch_save_us, args.n_warmup),
        "batch8_load_HtoD": _summary("batch8_load_HtoD", batch_load_us, args.n_warmup),
        "vendor_DRAM_proxy_us": 200.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n[bench-xfer] === results (n={args.n_trials}, drop {args.n_warmup}) ===")
    for k in ("single_save_HtoD", "single_load_HtoD",
              "batch8_save_DtoH", "batch8_load_HtoD"):
        s = summary[k]
        print(f"  {k:18}  P50={s['p50_us']:7.1f}μs  "
              f"P99={s['p99_us']:7.1f}μs  P999={s['p999_us']:7.1f}μs  "
              f"max={s['max_us']:7.1f}μs")
    print(f"\n[bench-xfer] vs vendor DRAM ℓ̄ proxy = "
          f"{summary['vendor_DRAM_proxy_us']:.0f}μs")
    print(f"[bench-xfer] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
