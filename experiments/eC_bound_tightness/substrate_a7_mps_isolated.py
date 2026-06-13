"""R26 A7 under production isolation: NVIDIA MPS thread-percentage gating.

R25 W2 (审稿人2 W2; 审稿人3 W2): A7 fails sharply at 0.016 MiB
co-tenant. Reviewers asked whether production isolation
mechanisms (cgroup + MPS + MIG) can recover HOLDS.
This script tests the MPS path:

1. Spawn a background contender process pinned via
   ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=PCT_BG`` (default 50) on
   the chosen ``--device-index``. The contender continuously
   issues ``cudaMemcpyAsync`` on a fp16 buffer (default 1 MiB).
2. In the foreground (same process, same MPS daemon) we set
   ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=PCT_FG`` (default 50) and
   run the standard quiesced-threshold A7 microbenchmark at
   the {4, 16, 32, 64} KB grid.
3. Compare per-block CP 95% upper $q_\\mathrm{step}^\\text{A2}$
   under MPS isolation vs the R25 no-isolation result. If MPS
   isolation recovers $q_\\mathrm{UB} \\le \\rho$, the
   ``full-tail-PS`` bound is deployable on MPS-isolated
   production hardware.

Important caveats (R26 honest):

* MPS thread-percentage is a \\emph{soft} cap on SM share, not
  a hard isolation on PCIe bandwidth. We test PCIe-bound
  workloads where MPS cap on SM share matters less than the
  bandwidth split, so the result is a lower-bound on what
  production-grade MIG (which IS hard-partitioned) would
  achieve.
* The MPS daemon must already be running (\\texttt{ps aux | grep
  nvidia-cuda-mps}). We do not start/stop the daemon here.
* The contender and probe share the same CUDA context group
  under MPS, so context-switch cost is below the cross-process
  level.

Output: ``substrate_a7_mps_isolated.{json,tex}``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

from seer.timing.substrate_measure import (
    cp_upper_975 as _cp_upper_975,
    percentile as _percentile,
    safe_write_stub,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"
CONTENDER_SCRIPT = ROOT / "experiments/eC_bound_tightness/_mps_contender.py"


def _ensure_contender_script() -> None:
    """Write the background contender script if it does not exist."""
    if CONTENDER_SCRIPT.exists():
        return
    CONTENDER_SCRIPT.write_text(
        '"""Background contender used by substrate_a7_mps_isolated.py."""\n'
        "import os\n"
        "import time\n"
        "\n"
        "import torch\n"
        "dev = torch.device('cuda', int(os.environ.get('SEER_DEVICE_INDEX', '0')))\n"
        "buf_mb = float(os.environ.get('SEER_CONTENDER_MB', '1.0'))\n"
        "nbytes = int(buf_mb * 1024 * 1024)\n"
        "nelem = nbytes // 2\n"
        "host = torch.empty(nelem, dtype=torch.float16, pin_memory=True)\n"
        "devv = torch.empty(nelem, dtype=torch.float16, device=dev)\n"
        "stream = torch.cuda.Stream(device=dev)\n"
        "with torch.cuda.device(dev):\n"
        "    while True:\n"
        "        with torch.cuda.stream(stream):\n"
        "            devv.copy_(host, non_blocking=True)\n"
        "            host.copy_(devv, non_blocking=True)\n"
        "        if int(time.perf_counter() * 100) % 200 == 0:\n"
        "            stream.synchronize()\n"
    )


def measure_under_mps(
    block_size_kb: int, n_reps: int, warmup: int,
    device_index: int, fg_pct: int, fixed_threshold_us: float,
) -> dict:
    import statistics
    import torch
    dev = torch.device("cuda", device_index)
    nbytes = block_size_kb * 1024
    nelem = nbytes // 2
    host = torch.empty(nelem, dtype=torch.float16, pin_memory=True)
    devv = torch.empty(nelem, dtype=torch.float16, device=dev)
    stream = torch.cuda.Stream(device=dev)
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            devv.copy_(host, non_blocking=True)
            host.copy_(devv, non_blocking=True)
    torch.cuda.synchronize(device=dev)
    timings = []
    for _ in range(n_reps):
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            s_evt.record(stream)
            devv.copy_(host, non_blocking=True)
            host.copy_(devv, non_blocking=True)
            e_evt.record(stream)
        e_evt.synchronize()
        timings.append(s_evt.elapsed_time(e_evt) * 1000.0)
    mean = statistics.mean(timings)
    n_over_fixed = sum(1 for t in timings if t > fixed_threshold_us)
    cp_upper_fixed = _cp_upper_975(n_over_fixed, n_reps)
    return {
        "block_size_kb": block_size_kb,
        "n_reps": n_reps,
        "fg_pct": fg_pct,
        "mean_us": mean,
        "median_us": statistics.median(timings),
        "p99_us": _percentile(timings, 99.0),
        "p999_us": _percentile(timings, 99.9),
        "max_us": max(timings),
        "fixed_threshold_us": fixed_threshold_us,
        "n_over_fixed": n_over_fixed,
        "q_step_a2_fixed_emp": n_over_fixed / n_reps,
        "q_step_a2_fixed_upper95": cp_upper_fixed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-sizes-kb", nargs="+", type=int,
                    default=[4, 16, 32, 64])
    ap.add_argument("--n-reps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--contender-mb", type=float, default=1.0)
    ap.add_argument("--bg-pct", type=int, default=50,
                    help="MPS thread % for the contender (default 50)")
    ap.add_argument("--fg-pct", type=int, default=50,
                    help="MPS thread % for the foreground probe (default 50)")
    ap.add_argument("--device-index", type=int, default=1)
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--force-stub-overwrite", action="store_true",
                    help="R28: clobber existing measured artifact when "
                         "emitting the harness-only stub. Default refuses.")
    ap.add_argument("--out-stem", default="substrate_a7_mps_isolated",
                    help="R29: stem (without extension) for both output "
                         "JSON and TeX. Default 'substrate_a7_mps_isolated' "
                         "preserves the canonical R26 1 MiB heavy artifact "
                         "name. Use 'substrate_a7_mps_cross_0p016' for the "
                         "R27 cross-contention companion so the gen-claim-"
                         "evidence table finds it under the expected "
                         "filename.")
    args = ap.parse_args()

    def _has_cuda() -> bool:
        try:
            import torch  # noqa: F401
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    if not args.cuda or not _has_cuda():
        stub = {"status": "HARNESS_READY_AWAITING_GPU_RUN",
                "block_sizes_kb": args.block_sizes_kb,
                "out_stem": args.out_stem,
                "contender_mb": args.contender_mb,
                "rationale":
                    "R26 production-isolation A7 (MPS thread-%): "
                    "tests whether MPS thread-% gating recovers A7 "
                    "HOLDS under co-tenant pressure that R25 sharp-"
                    "transition sweep showed breaks chat-tier."}
        tex_lines = ["% Generated by substrate_a7_mps_isolated.py "
                     "(R26, harness-only).",
                     r"\begin{tabular}{ll}\toprule Field & Value \\ "
                     r"\midrule Status & \textsc{harness-only} \\ "
                     r"\bottomrule \end{tabular}"]
        wrote = safe_write_stub(
            RESULTS / f"{args.out_stem}.json",
            RESULTS / f"{args.out_stem}.tex",
            stub, "\n".join(tex_lines) + "\n",
            force=args.force_stub_overwrite,
        )
        if wrote:
            print(f"[a7-mps] STATUS: harness-only ({args.out_stem})")
        return 0

    # Check MPS daemon is alive.
    mps_alive = False
    try:
        ps = subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control"],
                            capture_output=True, timeout=2)
        mps_alive = ps.returncode == 0
    except Exception:
        pass
    if not mps_alive:
        print("[a7-mps] WARN: nvidia-cuda-mps-control daemon not "
              "detected; results will be NON-MPS interference, not "
              "MPS-isolated. To start MPS:")
        print("  sudo nvidia-cuda-mps-control -d")

    # Phase 0: quiesced calibration to derive fixed threshold per block.
    print(f"[a7-mps] R26 production-isolation A7 (MPS thread-%); "
          f"contender={args.contender_mb} MiB, bg={args.bg_pct}%, "
          f"fg={args.fg_pct}%, GPU={args.device_index}, "
          f"MPS-daemon-alive={mps_alive}")

    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100)
    quiet_rows = {}
    for bs in args.block_sizes_kb:
        r = measure_under_mps(
            bs, n_reps=args.n_reps, warmup=args.warmup,
            device_index=args.device_index, fg_pct=100,
            # Phase 0: use a sentinel large threshold so the
            # n_over_fixed column from this row is meaningless;
            # we extract just the mean and compute the canonical
            # 4*ell_bar threshold from it.
            fixed_threshold_us=1e9,
        )
        thr = 4.0 * r["mean_us"]
        quiet_rows[bs] = {
            "block_size_kb": bs,
            "mean_us": r["mean_us"],
            "fixed_threshold_us": thr,
        }
        print(f"[a7-mps] quiesced(MPS={args.fg_pct}%) {bs}KB: "
              f"mean={r['mean_us']:.2f}us  thr={thr:.2f}us")

    # Phase 1: spawn contender, then measure foreground under MPS gating.
    _ensure_contender_script()
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.bg_pct)
    env["SEER_DEVICE_INDEX"] = str(args.device_index)
    env["SEER_CONTENDER_MB"] = str(args.contender_mb)
    print(f"[a7-mps] spawning contender PID for "
          f"MPS-bg={args.bg_pct}%, contender={args.contender_mb}MiB")
    cont = subprocess.Popen(
        [sys.executable, str(CONTENDER_SCRIPT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2.0)  # let contender warm up + saturate
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.fg_pct)
        cont_rows = []
        for bs in args.block_sizes_kb:
            thr = quiet_rows[bs]["fixed_threshold_us"]
            r = measure_under_mps(
                bs, n_reps=args.n_reps, warmup=args.warmup,
                device_index=args.device_index, fg_pct=args.fg_pct,
                fixed_threshold_us=thr,
            )
            r["a7_holds_at_rho"] = bool(
                r["q_step_a2_fixed_upper95"] < args.rho)
            r["mean_inflation_vs_quiet"] = (
                r["mean_us"] / max(quiet_rows[bs]["mean_us"], 1e-6))
            cont_rows.append(r)
            print(f"[a7-mps] cont {bs}KB: mean={r['mean_us']:.2f}us "
                  f"infl={r['mean_inflation_vs_quiet']:.2f}x  "
                  f"k/n_fixed={r['n_over_fixed']}/{r['n_reps']}  "
                  f"qUB95={r['q_step_a2_fixed_upper95']:.3e}  "
                  f"A7@rho={r['a7_holds_at_rho']}")
    finally:
        cont.terminate()
        try:
            cont.wait(timeout=3)
        except Exception:
            cont.kill()

    overall_holds = all(r["a7_holds_at_rho"] for r in cont_rows)
    out = {
        "status": "MEASUREMENT_COMPLETE",
        "mps_daemon_alive": mps_alive,
        "bg_pct": args.bg_pct,
        "fg_pct": args.fg_pct,
        "contender_mb": args.contender_mb,
        "rho": args.rho,
        "device_index": args.device_index,
        "block_sizes_kb": args.block_sizes_kb,
        "quiesced_phase0": quiet_rows,
        "mps_contended_rows": cont_rows,
        "a7_holds_under_mps": overall_holds,
        "interpretation": (
            "R26 MPS thread-% isolation A7 test. If overall_holds "
            "is True, MPS thread-% gating recovers A7 chat-tier "
            "under contention that R25's no-isolation sweep shows "
            "fails sharply at 0.016 MiB. If False, MPS thread-% "
            "is insufficient and operators need MIG (hard "
            "partitioning) or cgroup-PCIe (queued, todo_atc.md~D)."
        ),
    }
    out_json = RESULTS / f"{args.out_stem}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[a7-mps] wrote {out_json}")
    print(f"[a7-mps] A7 holds under MPS thread-%: {overall_holds}")

    tex_lines = [
        "% Generated by substrate_a7_mps_isolated.py (R26).",
        r"\begin{tabular}{rrrrrl}",
        r"\toprule",
        r"Block & $\bar\ell_\text{quiet}$ & $\bar\ell_\text{MPS}$ & "
        r"inflation & $q_\mathrm{step}^\text{A2,fixed,UB}$ & "
        r"A7@$\rho{=}10^{-2}$ \\",
        r"(KB) & ($\mu$s) & ($\mu$s) & ($\times$) & (k/n) & \\",
        r"\midrule",
    ]
    for r in cont_rows:
        q = quiet_rows[r["block_size_kb"]]
        verdict = (r"\textsc{holds}" if r["a7_holds_at_rho"]
                   else r"\textsc{fails}")
        tex_lines.append(
            f"{r['block_size_kb']} & "
            f"{q['mean_us']:.2f} & "
            f"{r['mean_us']:.2f} & "
            f"{r['mean_inflation_vs_quiet']:.2f} & "
            f"{r['n_over_fixed']}/{r['n_reps']} "
            f"($\\le {r['q_step_a2_fixed_upper95']:.2e}$) & "
            f"{verdict} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / f"{args.out_stem}.tex"
    out_tex.write_text("\n".join(tex_lines) + "\n")
    print(f"[a7-mps] wrote {out_tex}")
    return 0 if overall_holds else 2


if __name__ == "__main__":
    sys.exit(main())
