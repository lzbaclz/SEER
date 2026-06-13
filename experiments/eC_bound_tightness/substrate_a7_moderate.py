"""R21 moderate-contention A7 microbenchmark (fixed quiesced threshold).

R21 reviewer (W1) flagged that the heavy-contention sanity in
``substrate_a7_contended.py`` (8 MiB contender, $\\bar\\ell$
inflates $\\sim 58\\times$ to $\\sim 655\\,\\mu$s) makes the
$>4\\bar\\ell_\\text{cont}$ threshold trivially unreachable
because the PCIe queue caps per-block latency at $\\sim 710\\,\\mu$s
$\\ll 4\\bar\\ell_\\text{cont}\\approx 2.6$~ms; A7 holds for a
threshold-scaling reason rather than because of any non-trivial
substrate property. The reviewer's suggested fix:

  > Use a weaker contender (e.g. 1~MiB transfers instead of
  > 8~MiB) so $\\bar\\ell$ inflates only 2--5\\times{} rather
  > than 58\\times{}, and measure $q_\\mathrm{step}^\\text{A2}$
  > against a *fixed* quiesced threshold $4\\bar\\ell_\\text{quiet}$.

This script does exactly that. We sweep contender sizes
$\\in \\{0.25, 1, 4\\}$~MiB and, for each, report the per-block
exceedance count against a fixed quiesced threshold
$\\tau_\\text{quiesced} = 4\\bar\\ell_\\text{quiet}^\\text{4KB}
\\approx 44\\,\\mu$s$, computed from a same-script quiesced
calibration run at the start. This gives a non-trivial A7
validation: if $q_\\mathrm{step}^\\text{A2,fixed}$ remains
$\\le \\rho$ as $\\bar\\ell$ inflates moderately, A7 is robust
under PCIe contention against the bound's own threshold; if not,
A7 fails honestly on moderate contention.

Output: ``substrate_a7_moderate.{json,tex}``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import threading
import time

from seer.timing.substrate_measure import (
    cp_upper_975 as _cp_upper_975,
    percentile as _percentile,
    safe_write_stub,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"


class _Contender:
    def __init__(self, buffer_bytes: int, device_index: int = 0) -> None:
        import torch
        self.dev = torch.device("cuda", device_index)
        nb_elem = buffer_bytes // 2
        self.host = torch.empty(nb_elem, dtype=torch.float16, pin_memory=True)
        self.devv = torch.empty(nb_elem, dtype=torch.float16, device=self.dev)
        self.stream = torch.cuda.Stream(device=self.dev)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.iterations = 0

    def _loop(self) -> None:  # pragma: no cover - hardware path
        import torch
        with torch.cuda.device(self.dev):
            while not self.stop_event.is_set():
                with torch.cuda.stream(self.stream):
                    self.devv.copy_(self.host, non_blocking=True)
                    self.host.copy_(self.devv, non_blocking=True)
                self.iterations += 1
                if self.iterations % 200 == 0:
                    self.stream.synchronize()

    def start(self) -> None:  # pragma: no cover
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:  # pragma: no cover
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        import torch
        with torch.cuda.device(self.dev):
            self.stream.synchronize()


def _measure(block_size_kb: int, n_reps: int, warmup: int,
             contender_buffer_mb: float | None,
             device_index: int) -> dict:  # pragma: no cover - HW
    import torch
    dev = torch.device("cuda", device_index)
    n_bytes = block_size_kb * 1024
    nb_elem = n_bytes // 2
    host = torch.empty(nb_elem, dtype=torch.float16, pin_memory=True)
    devv = torch.empty(nb_elem, dtype=torch.float16, device=dev)
    stream = torch.cuda.Stream(device=dev)

    contender = None
    if contender_buffer_mb is not None and contender_buffer_mb > 0:
        contender = _Contender(
            buffer_bytes=int(contender_buffer_mb * 1024 * 1024),
            device_index=device_index)
        contender.start()
        time.sleep(0.5)

    try:
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
    finally:
        if contender is not None:
            contender.stop()

    mean = statistics.mean(timings)
    return {
        "block_size_kb": block_size_kb,
        "contender_buffer_mb": (contender_buffer_mb
                                if contender_buffer_mb else 0),
        "n_reps": n_reps,
        "mean_us": mean,
        "median_us": statistics.median(timings),
        "p99_us": _percentile(timings, 99.0),
        "p999_us": _percentile(timings, 99.9),
        "max_us": max(timings),
        "_timings_for_threshold": timings,  # kept; stripped on dump
        "contender_iterations": (
            contender.iterations if contender is not None else 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-sizes-kb", nargs="+", type=int,
                    default=[4, 16, 32, 64])
    ap.add_argument("--n-reps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--contender-mb-sweep", nargs="+", type=float,
                    default=[0.25, 1.0, 4.0],
                    help="Contender buffer sizes (MiB) for moderate "
                         "stress sweep (default 0.25, 1, 4)")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--out-stem", default="substrate_a7_moderate",
                    help="Basename for output JSON/TeX in "
                         "experiments/eC_bound_tightness/results/. "
                         "Default 'substrate_a7_moderate' (R25). R34 "
                         "advisor: pass 'substrate_a7_sub016' for the "
                         "sub-0.016 MiB step-shape probe so the R25 "
                         "load-bearing artifact is not overwritten.")
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--qbase-calibrated", type=float, default=0.0067)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--force-stub-overwrite", action="store_true",
                    help="R28: clobber existing measured artifact when "
                         "emitting the harness-only stub. Default refuses.")
    ap.add_argument(
        "--fixed-threshold-source", default="local-phase0",
        choices=("local-phase0", "r20-substrate-json"),
        help="Where the fixed quiesced threshold per block comes "
             "from. local-phase0 (default): re-measure quiesced "
             "in-script (4*mean). r20-substrate-json: load the "
             "persisted R20 clean quiesced measurement from "
             "substrate_pcie_nvme.json (use when GPU has external "
             "load that contaminates phase 0).")
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
                "contender_mb_sweep": args.contender_mb_sweep,
                "rationale": "R21 W1(b) moderate-contention A7 sanity "
                             "against fixed quiesced threshold."}
        tex_lines = [
            "% Generated by substrate_a7_moderate.py "
            "(R21, harness-only).",
            r"\begin{tabular}{ll}", r"\toprule",
            r"Field & Value \\", r"\midrule",
            r"Status & \textsc{harness-only} \\",
            r"\bottomrule", r"\end{tabular}",
        ]
        wrote = safe_write_stub(
            RESULTS / f"{args.out_stem}.json",
            RESULTS / f"{args.out_stem}.tex",
            stub, "\n".join(tex_lines) + "\n",
            force=args.force_stub_overwrite,
        )
        if wrote:
            print(f"[a7-moderate] STATUS: harness-only ({args.out_stem})")
        return 0

    print(f"[a7-moderate] R21 moderate-contention A7 sweep; "
          f"contenders MiB={args.contender_mb_sweep}; "
          f"fixed-threshold source: {args.fixed_threshold_source}")

    # Phase 0: derive a fixed threshold per block size, either by
    # re-measuring locally or by reading R20 persisted clean
    # quiesced calibration.
    quiet_rows = {}
    if args.fixed_threshold_source == "r20-substrate-json":
        r20_path = RESULTS / "substrate_pcie_nvme.json"
        if not r20_path.exists():
            print(f"[a7-moderate] FAIL: {r20_path} missing")
            return 1
        r20 = json.loads(r20_path.read_text())
        r20_by_bs = {r["block_size_kb"]: r for r in r20["rows"]}
        for bs in args.block_sizes_kb:
            if bs not in r20_by_bs:
                print(f"[a7-moderate] FAIL: R20 has no row for {bs}KB")
                return 1
            r = r20_by_bs[bs]
            quiet_rows[bs] = {
                "block_size_kb": bs,
                "source": "r20-substrate-json",
                "mean_us": r["mean_us"],
                "median_us": r["median_us"],
                "p99_us": r["p99_us"],
                "p999_us": r["p999_us"],
                "max_us": r["max_us"],
                "fixed_threshold_us": 4.0 * r["mean_us"],
                "n_reps": r["n_reps"],
                "n_over_fixed": r["n_over_4x_mean"],
                "q_step_fixed_emp": r["q_step_a2_aligned"],
                "q_step_fixed_upper95": r["q_step_a2_aligned_upper95"],
            }
            print(f"[a7-moderate] quiesced(R20) {bs}KB: "
                  f"mean={quiet_rows[bs]['mean_us']:.2f}us  "
                  f"fixed_thresh="
                  f"{quiet_rows[bs]['fixed_threshold_us']:.2f}us")
    else:
        for bs in args.block_sizes_kb:
            rq = _measure(bs, n_reps=args.n_reps, warmup=args.warmup,
                           contender_buffer_mb=None,
                           device_index=args.device_index)
            rq["fixed_threshold_us"] = 4.0 * rq["mean_us"]
            thresh = rq["fixed_threshold_us"]
            rq["n_over_fixed"] = sum(
                1 for t in rq["_timings_for_threshold"] if t > thresh)
            rq["q_step_fixed_emp"] = rq["n_over_fixed"] / rq["n_reps"]
            rq["q_step_fixed_upper95"] = _cp_upper_975(
                rq["n_over_fixed"], rq["n_reps"])
            del rq["_timings_for_threshold"]
            rq["source"] = "local-phase0"
            quiet_rows[bs] = rq
            print(f"[a7-moderate] quiesced {bs}KB: mean={rq['mean_us']:.2f}us "
                  f"fixed_thresh={rq['fixed_threshold_us']:.2f}us "
                  f"k/n={rq['n_over_fixed']}/{rq['n_reps']} "
                  f"qUB95={rq['q_step_fixed_upper95']:.3e}")

    # Phase 1: sweep contender sizes; for each, measure all block
    # sizes against the fixed quiesced threshold from phase 0.
    sweep_rows = []
    for cmb in args.contender_mb_sweep:
        for bs in args.block_sizes_kb:
            rc = _measure(bs, n_reps=args.n_reps, warmup=args.warmup,
                           contender_buffer_mb=cmb,
                           device_index=args.device_index)
            thresh = quiet_rows[bs]["fixed_threshold_us"]
            n_over_fixed = sum(1 for t in rc["_timings_for_threshold"]
                                if t > thresh)
            rc["fixed_threshold_us"] = thresh
            rc["n_over_fixed"] = n_over_fixed
            rc["q_step_fixed_emp"] = n_over_fixed / rc["n_reps"]
            rc["q_step_fixed_upper95"] = _cp_upper_975(
                n_over_fixed, rc["n_reps"])
            rc["mean_inflation"] = (
                rc["mean_us"] / max(quiet_rows[bs]["mean_us"], 1e-6))
            rc["a7_holds_fixed_at_rho"] = bool(
                rc["q_step_fixed_upper95"] < args.rho)
            rc["a7_holds_fixed_at_calibrated_qbase"] = bool(
                rc["q_step_fixed_upper95"] <= args.qbase_calibrated)
            del rc["_timings_for_threshold"]
            sweep_rows.append(rc)
            print(f"[a7-moderate] cont={cmb}MiB {bs}KB: "
                  f"mean={rc['mean_us']:.2f}us "
                  f"infl={rc['mean_inflation']:.2f}x  "
                  f"k/n_fixed={rc['n_over_fixed']}/{rc['n_reps']} "
                  f"qUB95={rc['q_step_fixed_upper95']:.3e} "
                  f"A7@rho={rc['a7_holds_fixed_at_rho']}")

    # Group results by contender size; overall verdict per regime.
    by_contender = {}
    for r in sweep_rows:
        by_contender.setdefault(r["contender_buffer_mb"], []).append(r)
    regime_verdicts = []
    for cmb, rows in by_contender.items():
        holds = all(r["a7_holds_fixed_at_rho"] for r in rows)
        mean_infls = [r["mean_inflation"] for r in rows]
        worst_qub = max(r["q_step_fixed_upper95"] for r in rows)
        regime_verdicts.append({
            "contender_mb": cmb,
            "mean_inflation_min": min(mean_infls),
            "mean_inflation_max": max(mean_infls),
            "worst_q_step_fixed_upper95": worst_qub,
            "a7_holds_all_blocks_at_rho": holds,
        })
        print(f"[a7-moderate] REGIME cont={cmb}MiB: "
              f"infl=[{min(mean_infls):.2f}, {max(mean_infls):.2f}]x  "
              f"worstqUB={worst_qub:.3e}  A7@rho={holds}")

    out = {
        "status": "MEASUREMENT_COMPLETE",
        "device_index": args.device_index if args.cuda else None,
        "cuda": args.cuda,
        "rho_chat": args.rho,
        "qbase_calibrated": args.qbase_calibrated,
        "block_sizes_kb": args.block_sizes_kb,
        "n_reps": args.n_reps,
        "warmup": args.warmup,
        "contender_mb_sweep": args.contender_mb_sweep,
        "quiesced_calibration_rows": [
            quiet_rows[bs] for bs in args.block_sizes_kb],
        "moderate_sweep_rows": sweep_rows,
        "regime_verdicts": regime_verdicts,
        "interpretation": (
            "Per-block exceedance is measured against a FIXED "
            "threshold derived from the quiesced calibration "
            "(4 * mean_us^quiet, per block size). Contender "
            "intensities are chosen so that the foreground mean "
            "inflates by 2-10x rather than ~58x. If "
            "q_step^A2,fixed remains <= rho across the moderate-"
            "stress sweep, A7 is robust under PCIe contention "
            "against the bound's own threshold (a non-trivial "
            "validation, in contrast to the threshold-scaling "
            "tautology in substrate_a7_contended)."),
    }
    out_json = RESULTS / f"{args.out_stem}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[a7-moderate] wrote {out_json}")

    tex_lines = [
        "% Generated by substrate_a7_moderate.py "
        "(R21 W1(b) moderate-contention A7 sweep, fixed quiesced "
        "threshold).",
        r"\begin{tabular}{rrrrrl}",
        r"\toprule",
        r"Cont. & Infl. range & worst $\bar\ell$ infl. & "
        r"worst $q_\mathrm{step}^\text{A2,fixed,UB}$ & "
        r"A7@$\rho{=}10^{-2}$ all blocks & note \\",
        r"(MiB) & ($\times$) & ($\times$) & (k/n max) & & \\",
        r"\midrule",
    ]
    for rv in regime_verdicts:
        # find max k/n at this contender
        worst_k = max(r["n_over_fixed"]
                      for r in by_contender[rv["contender_mb"]])
        n_reps = by_contender[rv["contender_mb"]][0]["n_reps"]
        verd = ("\\textsc{holds}" if rv["a7_holds_all_blocks_at_rho"]
                else "\\textsc{fails}")
        note = ("substrate-non-trivial" if rv["mean_inflation_max"] < 10
                else "heavy-contention (queue cap effects)")
        tex_lines.append(
            f"{rv['contender_mb']} & "
            f"[{rv['mean_inflation_min']:.2f}, {rv['mean_inflation_max']:.2f}] & "
            f"{rv['mean_inflation_max']:.2f} & "
            f"{worst_k}/{n_reps} "
            f"($\\le {rv['worst_q_step_fixed_upper95']:.2e}$) & "
            f"{verd} & {note} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / f"{args.out_stem}.tex"
    out_tex.write_text("\n".join(tex_lines) + "\n")
    print(f"[a7-moderate] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
