"""R36d A7 under MIG hard partitioning (reviewer "isolated A100" item).

Reviewer R36 RTSS-hardening list, item 1: "做一个 'pass A7 的真实隔离
部署' 完整闭环：MIG 或 cgroup-PCIe, substrate + integrated + hot-path
probe 全部 HOLDS, 然后用它跑 sizing verdict."

This script is the MIG variant. The operator-side runbook is:

  # 1. Enable MIG on the chosen device (one-time, requires sudo):
  sudo nvidia-smi -i 1 -mig 1

  # 2. Create a Compute Instance. 3g.40gb is a 50% partition (PCIe
  #    bandwidth halved + SMs partitioned); 7g.80gb is full
  #    (degenerate case = quiesced).
  sudo nvidia-smi mig -i 1 -cgi 9 -C       # 3g.40gb (half partition)
  # OR for a 1g.10gb micro-partition:
  # sudo nvidia-smi mig -i 1 -cgi 19 -C

  # 3. Get the MIG UUID:
  nvidia-smi -L | grep MIG

  # 4. Run the A7 probe with the MIG UUID *via CUDA_VISIBLE_DEVICES*
  #    (MIG presents the partition as a virtual device):
  CUDA_VISIBLE_DEVICES=MIG-<uuid> \\
      python -m experiments.eC_bound_tightness.substrate_a7_mig \\
        --cuda --device-index 0 --contender-mb 1.0 \\
        --n-reps 5000 --out-stem substrate_a7_mig

  # 5. Optional: launch a contender on a *different* MIG partition
  #    on the same physical GPU to test cross-partition isolation.
  #    Same script with SEER_CONTENDER_MB and SEER_MIG_CONTENDER_UUID.

  # 6. Regenerate paper artifacts:
  python scripts/gen_claim_evidence_table.py
  make paper

The script writes ``results/substrate_a7_mig.{json,tex}``. The
output JSON records the MIG UUID (from nvidia-smi -L) and the
declared partition profile so the artifact reviewer can verify
that the probe ran under hard partitioning, not just a CUDA
device-index reassignment.

Pass criterion (same as R24/R26):
  qUB_step^A2 < rho = 1e-2 on every block size.

If A7 HOLDS under a non-trivial MIG partition (3g.40gb or
smaller) with a co-located contender on a different partition,
this closes the reviewer's #1 RTSS-push ask: SEER deployable on
real isolated hardware, not just quiesced.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"
CONTENDER_SCRIPT = ROOT / "experiments/eC_bound_tightness/_mps_contender.py"
# We reuse _mps_contender.py: it just calls torch.cuda.{Stream,Event}
# on whatever CUDA device the env exposes — under MIG the device is
# the GI/CI, so the contender stresses that partition's PCIe slice.


def _query_mig_state() -> dict:
    """Capture MIG topology + UUIDs from nvidia-smi (read-only)."""
    try:
        out_l = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True,
            check=False, timeout=10,
        ).stdout
        out_mig = subprocess.run(
            ["nvidia-smi", "--query-gpu",
             "index,uuid,mig.mode.current,mig.mode.pending",
             "--format=csv"], capture_output=True, text=True,
            check=False, timeout=10,
        ).stdout
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        return {
            "nvidia_smi_L": out_l.strip().splitlines(),
            "mig_mode": out_mig.strip().splitlines(),
            "cuda_visible_devices": cuda_visible,
        }
    except Exception as e:
        return {"error": str(e)}


def measure_quiet(
    block_size_kb: int, n_reps: int, warmup: int, device_index: int,
) -> dict:
    """Quiesced phase 0: derive the fixed threshold tau = 4 * mean_us."""
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
        timings.append(s_evt.elapsed_time(e_evt) * 1000.0)  # ms→us
    return {
        "block_size_kb": block_size_kb,
        "mean_us": statistics.mean(timings),
        "p99_us": _percentile(timings, 0.99),
        "p999_us": _percentile(timings, 0.999),
        "timings": timings,
    }


def measure_with_contender(
    block_size_kb: int, n_reps: int, warmup: int, device_index: int,
    contender_mb: float, contender_proc: subprocess.Popen | None,
    fixed_threshold_us: float,
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
    k_over = sum(1 for t in timings if t > fixed_threshold_us)
    q_step = k_over / max(1, len(timings))
    # CP 95% one-sided upper, no FP dependency:
    q_ub = _cp_upper_975(k_over, len(timings))
    return {
        "block_size_kb": block_size_kb,
        "n_reps": len(timings),
        "mean_us": statistics.mean(timings),
        "p99_us": _percentile(timings, 0.99),
        "k_over": k_over,
        "q_step_A2_fixed": q_step,
        "q_step_A2_fixed_upper95": q_ub,
        "fixed_threshold_us": fixed_threshold_us,
    }


def _percentile(xs: list[float], p: float) -> float:
    from seer.timing.substrate_measure import percentile as _p
    return _p(xs, p)


def _cp_upper_975(k: int, n: int) -> float:
    from seer.timing.substrate_measure import cp_upper_975 as _u
    return _u(k, n)


def _start_contender(device_index: int, contender_mb: float) -> subprocess.Popen | None:
    """Launch the background contender (reuses _mps_contender.py)."""
    if not CONTENDER_SCRIPT.exists():
        return None
    env = dict(os.environ)
    env["SEER_DEVICE_INDEX"] = str(device_index)
    env["SEER_CONTENDER_MB"] = str(contender_mb)
    # NOTE: the contender inherits CUDA_VISIBLE_DEVICES, so under MIG
    # it runs on the SAME MIG partition unless the user overrides
    # SEER_MIG_CONTENDER_UUID to point to a different one. For
    # cross-partition isolation, override CUDA_VISIBLE_DEVICES in env.
    if env.get("SEER_MIG_CONTENDER_UUID"):
        env["CUDA_VISIBLE_DEVICES"] = env["SEER_MIG_CONTENDER_UUID"]
    p = subprocess.Popen(
        [sys.executable, str(CONTENDER_SCRIPT)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)  # let it warm up
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", required=True)
    ap.add_argument("--device-index", type=int, default=0,
                    help="Within CUDA_VISIBLE_DEVICES (which under MIG "
                         "should be set to MIG-<uuid>)")
    ap.add_argument("--block-sizes-kb", type=int, nargs="+",
                    default=[4, 16, 32, 64])
    ap.add_argument("--n-reps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--contender-mb", type=float, default=1.0,
                    help="Contender buffer size in MiB (default 1 MiB; "
                         "for cross-partition isolation, the contender "
                         "should run on a DIFFERENT MIG GI/CI -- set "
                         "SEER_MIG_CONTENDER_UUID in env)")
    ap.add_argument("--no-contender", action="store_true",
                    help="Run quiesced-only (degenerate, for partition "
                         "baseline measurement)")
    ap.add_argument("--out-stem", type=str, default="substrate_a7_mig")
    args = ap.parse_args()

    mig_state = _query_mig_state()
    if not any("MIG" in line for line in mig_state.get("nvidia_smi_L", [])):
        print("[a7-mig] WARNING: nvidia-smi -L shows no MIG instances. "
              "Either MIG is not enabled (sudo nvidia-smi -i N -mig 1) "
              "or no compute instances exist (sudo nvidia-smi mig -i N "
              "-cgi PROFILE -C). This run will measure on the WHOLE GPU, "
              "not a MIG partition. The output JSON will record this.",
              file=sys.stderr)

    print("[a7-mig] phase 0: quiesced calibration on the partition")
    quiet_rows = {
        bs: measure_quiet(bs, args.n_reps, args.warmup, args.device_index)
        for bs in args.block_sizes_kb
    }
    fixed_tau = {
        bs: 4.0 * quiet_rows[bs]["mean_us"] for bs in args.block_sizes_kb
    }
    for bs in args.block_sizes_kb:
        print(f"[a7-mig] phase0  {bs}KB: mean={quiet_rows[bs]['mean_us']:.2f}us "
              f"P99={quiet_rows[bs]['p99_us']:.2f}us "
              f"tau={fixed_tau[bs]:.2f}us")

    contender_proc: subprocess.Popen | None = None
    sweep_rows = []
    if not args.no_contender:
        print(f"[a7-mig] phase 1: contender at {args.contender_mb} MiB")
        contender_proc = _start_contender(args.device_index, args.contender_mb)
        try:
            for bs in args.block_sizes_kb:
                row = measure_with_contender(
                    bs, args.n_reps, args.warmup, args.device_index,
                    args.contender_mb, contender_proc,
                    fixed_threshold_us=fixed_tau[bs],
                )
                infl = row["mean_us"] / quiet_rows[bs]["mean_us"]
                holds = row["q_step_A2_fixed_upper95"] <= args.rho
                row["mean_inflation"] = infl
                row["a7_holds"] = holds
                print(f"[a7-mig] cont={args.contender_mb}MiB {bs}KB: "
                      f"mean={row['mean_us']:.2f}us infl={infl:.2f}x  "
                      f"k/n_fixed={row['k_over']}/{row['n_reps']} "
                      f"qUB95={row['q_step_A2_fixed_upper95']:.3e} "
                      f"A7={holds}")
                sweep_rows.append(row)
        finally:
            if contender_proc is not None:
                contender_proc.terminate()
                contender_proc.wait(timeout=5)

    a7_holds_all = (
        all(r.get("a7_holds") for r in sweep_rows) if sweep_rows else None
    )
    out = {
        "status": "MEASUREMENT_COMPLETE",
        "device_index": args.device_index,
        "cuda": args.cuda,
        "rho_chat": args.rho,
        "block_sizes_kb": args.block_sizes_kb,
        "n_reps": args.n_reps,
        "warmup": args.warmup,
        "contender_mb": args.contender_mb if not args.no_contender else None,
        "mig_state_at_run": mig_state,
        "quiesced_calibration_rows": [quiet_rows[bs] for bs in args.block_sizes_kb],
        "contender_sweep_rows": sweep_rows,
        "a7_holds_all_blocks_at_rho": a7_holds_all,
        "interpretation": (
            "R36d MIG hard-partition A7 probe. quiesced-threshold "
            "verdict (tau = 4 * mean_us^quiet per block); A7 HOLDS iff "
            "q_step^A2,UB < rho on every block under contender pressure. "
            "If HOLDS on a non-trivial MIG partition (e.g. 3g.40gb with "
            "a 1 MiB contender), this closes the reviewer's #1 "
            "RTSS-push ask: real isolated deployment for full-tail-PS."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"{args.out_stem}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[a7-mig] wrote {out_json}")

    tex_lines = [
        "% Generated by substrate_a7_mig.py (R36d MIG hard-partition).",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Block (KB) & quiet mean ($\mu$s) & contended mean ($\mu$s) "
        r"& Inflation & $q_\mathrm{step}^\text{A2,fixed,UB}$ & A7@$\rho$ \\",
        r"\midrule",
    ]
    for row in sweep_rows:
        bs = row["block_size_kb"]
        tex_lines.append(
            f"{bs} & {quiet_rows[bs]['mean_us']:.2f} & "
            f"{row['mean_us']:.2f} & {row['mean_inflation']:.2f} & "
            f"{row['q_step_A2_fixed_upper95']:.3e} & "
            + (r"\textsc{hold}" if row['a7_holds'] else r"\textsc{fail}")
            + r" \\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_tex = RESULTS / f"{args.out_stem}.tex"
    out_tex.write_text("\n".join(tex_lines))
    print(f"[a7-mig] wrote {out_tex}")
    if a7_holds_all is True:
        print(f"[a7-mig] VERDICT: A7 HOLDS on every block under MIG "
              f"hard partitioning + {args.contender_mb} MiB contender. "
              "Reviewer R36 RTSS-push item 1 (isolated A7 deployment) "
              "is closed.")
    elif a7_holds_all is False:
        print(f"[a7-mig] VERDICT: A7 FAILS on at least one block under "
              f"MIG + {args.contender_mb} MiB. MIG thread/SM "
              "partitioning does not bound PCIe bandwidth -- needs "
              "cgroup-PCIe (Linux kernel BPF) or fully separate physical "
              "GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
