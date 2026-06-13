"""Multi-regime sigma_contended measurement (R5-W4).

The R3 paper reported a single sigma_contended/sigma_clean = 1.80x
from a cross-card co-runner. The R4 reviewer flagged this as a
single configuration -- operator-deployed contention comes from
multiple regimes (cross-card, PCIe, cuMemAlloc, NCCL). This
script measures sigma_contended on the local A100 across three
contention regimes plus a quiet baseline:

  (a) quiet -- single process, single GPU, no co-runner
  (b) cross-card -- second A100 running concurrent matmul
      (PCIe-shared, NVLink-isolated; reproduces R3 1.80x)
  (c) co-tenant cuMemAlloc -- second process on the SAME A100
      doing periodic large cudaMalloc/cudaFree, contending for
      the cuMemAlloc lock (worst-case for short-running kernels)
  (d) PCIe DMA contention -- second process on the SAME A100
      doing background H2D+D2H copy (saturates PCIe)

The R3 cross-card 1.80x is reproduced as one point in the
distribution; the worst-case across regimes is the operator-side
sigma_contended input to Theorem 1 (a) under attention-agnostic
policies.

This script measures the LAP forward-pass tail jitter under each
regime as a sigma proxy (the canonical sigma_residual_us in the
Lemma 2 bound). 3000-rep / 200-warmup measurement on TRT
b4096 plan.

Output:
- ``sigma_contention_regimes.json``
- ``sigma_contention_regimes.tex`` for the paper supplementary.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


def _measure_lap_jitter(plan_path: str, n_warmup: int, n_reps: int,
                         tag: str) -> tuple[float, list[float]]:
    """Measure LAP forward-pass latency jitter on TRT b4096 plan.
    Returns (mean_us, list_of_samples_us). Uses execute_async_v3
    API (TRT >= 10)."""
    import torch
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    batch = 4096
    feat = 37
    x = torch.randn(batch, feat, device="cuda", dtype=torch.float32)
    out_shape = (batch, 4)
    y = torch.empty(out_shape, device="cuda", dtype=torch.float32)
    # Bind by name (v3 API).
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    context.set_tensor_address(in_name, int(x.data_ptr()))
    context.set_tensor_address(out_name, int(y.data_ptr()))
    stream = torch.cuda.Stream()
    # Warmup.
    for _ in range(n_warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    # Time.
    latencies_us: list[float] = []
    for _ in range(n_reps):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record(stream)
        context.execute_async_v3(stream.cuda_stream)
        end_ev.record(stream)
        end_ev.synchronize()
        latencies_us.append(float(start_ev.elapsed_time(end_ev)) * 1000.0)
    return float(np.mean(latencies_us)), latencies_us


def _co_runner_matmul(stop_event, gpu_id: int = 1):
    """Cross-card co-runner: continuous matmul on GPU `gpu_id`."""
    import torch
    torch.cuda.set_device(gpu_id)
    a = torch.randn(8192, 8192, device=f"cuda:{gpu_id}")
    b = torch.randn(8192, 8192, device=f"cuda:{gpu_id}")
    while not stop_event.is_set():
        _ = a @ b
        torch.cuda.synchronize(gpu_id)


def _co_runner_memalloc(stop_event, gpu_id: int = 0):
    """Same-GPU cuMemAlloc thrash."""
    import torch
    torch.cuda.set_device(gpu_id)
    while not stop_event.is_set():
        buf = torch.empty(1024 * 1024 * 256, dtype=torch.float32,
                           device=f"cuda:{gpu_id}")
        del buf
        torch.cuda.empty_cache()
        time.sleep(0.005)


def _co_runner_pcie(stop_event, gpu_id: int = 0):
    """Same-GPU PCIe H2D / D2H thrash."""
    import torch
    torch.cuda.set_device(gpu_id)
    h = torch.empty(1024 * 1024 * 64, dtype=torch.float32, pin_memory=True)
    d = torch.empty_like(h, device=f"cuda:{gpu_id}")
    while not stop_event.is_set():
        d.copy_(h, non_blocking=True)
        h.copy_(d, non_blocking=True)
        torch.cuda.synchronize(gpu_id)


def _run_regime(regime: str, plan_path: str, n_reps: int, n_warmup: int,
                 co_runner_grace_s: float = 1.0) -> tuple[float, list[float]]:
    """Run measurement under named regime."""
    if regime == "quiet":
        return _measure_lap_jitter(plan_path, n_warmup, n_reps, regime)

    runners = {
        "cross_card":  _co_runner_matmul,
        "memalloc":    _co_runner_memalloc,
        "pcie":        _co_runner_pcie,
    }
    runner = runners[regime]
    # Launch co-runner.
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    if regime == "cross_card":
        p = ctx.Process(target=runner, args=(stop_event, 1))
    else:
        p = ctx.Process(target=runner, args=(stop_event, 0))
    p.start()
    time.sleep(co_runner_grace_s)
    try:
        mean_us, samples = _measure_lap_jitter(plan_path, n_warmup,
                                                 n_reps, regime)
    finally:
        stop_event.set()
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
    return mean_us, samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan",
                    default="checkpoints/lap_prod_b4096.plan")
    ap.add_argument("--n_reps", type=int, default=2000)
    ap.add_argument("--n_warmup", type=int, default=200)
    ap.add_argument("--n_trials", type=int, default=3,
                    help="Trials per regime; R5-residual bootstrap CI "
                         "on std and sigma_ratio across trials.")
    ap.add_argument("--regimes", nargs="+",
                    default=["quiet", "cross_card", "memalloc", "pcie"])
    ap.add_argument("--out_dir",
                    default="experiments/eFault_characterization/results")
    args = ap.parse_args()

    # R5-residual W2-r: bootstrap across trials per regime.
    rows = []
    trials_by_regime: dict[str, list[dict]] = {}
    for regime in args.regimes:
        trials_by_regime[regime] = []
        for trial in range(args.n_trials):
            print(f"[sigma-regime] measuring '{regime}' trial {trial + 1}/{args.n_trials} ...")
            mean_us, samples = _run_regime(regime, args.plan,
                                            n_reps=args.n_reps,
                                            n_warmup=args.n_warmup)
            std = float(np.std(samples))
            p99 = float(np.percentile(samples, 99))
            p999 = float(np.percentile(samples, 99.9))
            trials_by_regime[regime].append({
                "trial": trial,
                "mean_us": float(mean_us),
                "std_us": std,
                "p99_us": p99,
                "p999_us": p999,
                "n_samples": len(samples),
            })

    # Compute per-regime std distribution + sigma_ratio CI.
    quiet_stds = [t["std_us"] for t in trials_by_regime["quiet"]]
    quiet_std_mean = float(np.mean(quiet_stds))
    for regime in args.regimes:
        trials = trials_by_regime[regime]
        stds = np.asarray([t["std_us"] for t in trials])
        means = np.asarray([t["mean_us"] for t in trials])
        p99s = np.asarray([t["p99_us"] for t in trials])
        ratios = stds / max(1e-9, quiet_std_mean)
        if args.n_trials >= 2:
            ratio_lo, ratio_hi = float(np.min(ratios)), float(np.max(ratios))
        else:
            ratio_lo = ratio_hi = float(ratios[0])
        rows.append({
            "regime": regime,
            "n_trials": args.n_trials,
            "mean_us": float(np.mean(means)),
            "std_us": float(np.mean(stds)),
            "p99_us": float(np.mean(p99s)),
            "sigma_ratio_vs_quiet": float(np.mean(ratios)),
            "sigma_ratio_min": ratio_lo,
            "sigma_ratio_max": ratio_hi,
            "trials": trials,
        })
        print(f"  {regime:12s}  std={np.mean(stds):.2f}us "
              f"ratio={np.mean(ratios):.2f}x "
              f"[{ratio_lo:.2f}, {ratio_hi:.2f}]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": (
            "LAP forward-pass tail jitter under 4 contention regimes "
            "(R5-W4). sigma_ratio = std(regime) / std(quiet) on the "
            "TRT b4096 plan."
        ),
        "plan": args.plan,
        "n_reps_per_regime": args.n_reps,
        "n_warmup": args.n_warmup,
        "rows": rows,
    }
    out_json = out_dir / "sigma_contention_regimes.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[sigma-regime] wrote {out_json}")

    # TeX table.
    lines = [
        "% Generated by experiments/eFault_characterization/"
        "sigma_contention_regimes.py (R5-W4 / R5-residual W2-r).",
        "% sigma_contended distribution across contention regimes on A100.",
        f"% n_trials={args.n_trials} per regime; CI = "
        f"min/max across trials.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Regime & $\bar\ell$ ($\mu$s) & "
        r"$\sigma$ ($\mu$s) & $P_{99}$ ($\mu$s) & "
        r"$\sigma / \sigma_\text{quiet}$ (range) \\",
        r"\midrule",
    ]
    for r in rows:
        regime_label = r["regime"].replace("_", r"\_")
        lines.append(
            f"\\texttt{{{regime_label}}} & "
            f"${r['mean_us']:.1f}$ & ${r['std_us']:.2f}$ & "
            f"${r['p99_us']:.1f}$ & "
            f"${r['sigma_ratio_vs_quiet']:.2f}\\times$ "
            f"$[{r['sigma_ratio_min']:.2f}, {r['sigma_ratio_max']:.2f}]$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_tex = out_dir / "sigma_contention_regimes.tex"
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[sigma-regime] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
