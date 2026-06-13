"""NVLink P2P bandwidth + latency probe for the SEER timing model.

Why this exists
---------------
The schedulability bound (Lemma 2) is parameterized by ``ell_bar`` — the
mean miss-tier IO latency that fires when a block isn't in HBM at the
moment the attention layer needs it. In the paper we list canonical
``ell_bar`` values (DRAM ≈ 200µs, NVMe ≈ 1ms, SSD ≈ 10ms). On a
multi-GPU node connected by NVLink, the *other GPU's HBM* is a viable
fourth tier with its own ``ell_bar`` — typically much smaller than DRAM,
because NVLink-attached HBM keeps the IO entirely on the GPU complex.

This probe measures the actual NVLink P2P latency and bandwidth on the
local host so we can put a real number for ``ell_bar_nvlink`` in the
appendix table.

What it measures
----------------
For every (src_gpu, dst_gpu) pair where ``src ≠ dst``:

* ``bw_GBps`` — sustained throughput at 256MB transfer size.
* ``p50_us``, ``p99_us`` — round-trip latency at 64KB transfer (one
  KV block, fp16, head_dim=128, 32 tokens ≈ 8KB; we use 64KB as a
  rounder unit since the cache layer often coalesces 4-8 blocks).
* ``per_block_us_64KB`` — same as ``p50_us`` but tagged for the paper.

Output is JSON to ``--out``. Example invocation::

    python experiments/eE_lap_wcet/nvlink_probe.py \\
        --out experiments/eE_lap_wcet/results/nvlink.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class NVLinkProbe:
    src: int
    dst: int
    bw_GBps: float
    p50_us: float
    p99_us: float
    p999_us: float
    per_block_us_64KB: float
    n_reps: int


def _measure_pair(src: int, dst: int, n_reps: int = 1000) -> NVLinkProbe:
    import torch

    torch.cuda.set_device(src)
    src_dev = torch.device(f"cuda:{src}")
    dst_dev = torch.device(f"cuda:{dst}")

    # Bandwidth: 256 MB of fp16 at sustained transfer.
    bw_size_bytes = 256 * 1024 * 1024
    n_floats = bw_size_bytes // 2  # fp16
    a = torch.randn(n_floats, dtype=torch.float16, device=src_dev)
    b = torch.empty_like(a, device=dst_dev)
    # Warmup
    for _ in range(8):
        b.copy_(a, non_blocking=True)
    torch.cuda.synchronize()

    # Bandwidth: time 32 sustained copies, divide.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n_bw_iter = 32
    start.record()
    for _ in range(n_bw_iter):
        b.copy_(a, non_blocking=True)
    end.record()
    end.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0
    bw_GBps = (bw_size_bytes * n_bw_iter / 1e9) / elapsed_s

    # Latency: 64KB transfers, P50/P99/P999 over n_reps.
    lat_size = 64 * 1024
    a_lat = torch.randn(lat_size // 2, dtype=torch.float16, device=src_dev)
    b_lat = torch.empty_like(a_lat, device=dst_dev)
    for _ in range(50):
        b_lat.copy_(a_lat, non_blocking=True)
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(n_reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        b_lat.copy_(a_lat, non_blocking=True)
        e.record()
        e.synchronize()
        samples_us.append(s.elapsed_time(e) * 1000.0)

    samples_us.sort()
    def pct(p): return samples_us[int((len(samples_us) - 1) * p / 100.0)]
    return NVLinkProbe(
        src=src, dst=dst,
        bw_GBps=float(bw_GBps),
        p50_us=pct(50.0),
        p99_us=pct(99.0),
        p999_us=pct(99.9),
        per_block_us_64KB=pct(50.0),
        n_reps=n_reps,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/eE_lap_wcet/results/nvlink.json")
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit(
            "Need at least 2 CUDA devices; have "
            f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}."
        )

    n = torch.cuda.device_count()
    # Enable peer access between every pair we'll touch.
    enabled = []
    for s in range(n):
        torch.cuda.set_device(s)
        for d in range(n):
            if s == d:
                continue
            try:
                torch.cuda.current_stream().synchronize()
                # PyTorch enables peer access lazily on first copy_; we just
                # confirm the topology can talk.
                _ = torch.empty(1, device=f"cuda:{s}").to(f"cuda:{d}")
                enabled.append((s, d))
            except Exception as e:  # noqa: BLE001
                print(f"[nvlink] peer {s}->{d} not enabled: {e}")
    print(f"[nvlink] peering OK for {len(enabled)} ordered pairs")

    probes: list[dict] = []
    for s, d in enabled:
        print(f"[nvlink] probing {s} -> {d}")
        p = _measure_pair(s, d, n_reps=args.reps)
        probes.append(asdict(p))
        print(f"  bw = {p.bw_GBps:.1f} GB/s  "
              f"p50/p99/p999 (64KB) = {p.p50_us:.1f} / {p.p99_us:.1f} / "
              f"{p.p999_us:.1f} µs")

    summary = {
        "device_count": n,
        "device_names": [torch.cuda.get_device_name(i) for i in range(n)],
        "probes": probes,
    }
    if probes:
        bws = [p["bw_GBps"] for p in probes]
        p50s = [p["p50_us"] for p in probes]
        summary["bw_GBps_mean"] = statistics.fmean(bws)
        summary["per_block_us_64KB_mean"] = statistics.fmean(p50s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[nvlink] wrote {args.out}")


if __name__ == "__main__":
    main()
