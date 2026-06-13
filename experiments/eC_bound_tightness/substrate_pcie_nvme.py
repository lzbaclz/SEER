"""R17/R18/R19 CUDA pinned-host transfer microbenchmark for the
substrate $\\bar\\ell$ + A7 calibration.

R19 honest rename: this script is a \\emph{CUDA pinned-host
H2D+D2H microbenchmark} (CPU DRAM $\\leftrightarrow$ GPU HBM
via $\\texttt{cudaMemcpyAsync}$ on a dedicated stream). It is
NOT a cgroup-throttled NVMe stress test, even though that path
is named in $\\texttt{--nvme-stress}$. The NVMe layering is the
load-bearing follow-up queued in $\\texttt{todo\\_atc.md}$~D
(real NVMe via SPDK or io\\_uring under cgroup IOPS throttle).

What this script DOES measure (and what we report in the paper):
the per-block PCIe transfer-latency distribution at 4--64\\,KB
KV-block sizes, the A2-aligned tail event count
($\\#\\{t > 4\\bar\\ell\\}$, the Lemma 2 truncation point), and
the 3-sigma proxy used by the legacy $\\texttt{\\_empirical\\_base\\_escape\\_mass}$
calibrator (kept as a sanity column).

R19 reviewer feedback closed:
1. The A7 verdict is now reported at the A2-aligned
   $>4\\bar\\ell$ threshold (not 3-sigma). The 3-sigma proxy is
   secondary.
2. $\\texttt{--nvme-stress}$ is explicitly documented as a stub.
3. Output table column headers reconciled with Lemma 2's
   $\\ell_\\mathrm{max}=4\\bar\\ell$.

Output: \\texttt{substrate\\_pcie\\_nvme.\\{json,tex\\}}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

from seer.timing.substrate_measure import (
    a2_aligned_stats,
    measure_block_transfer,
    safe_write_stub,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"


def _has_cuda() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _measure_cudamemcpyasync(block_size_kb: int, n_reps: int = 5000,
                              warmup: int = 200) -> dict:
    """Measure cudaMemcpyAsync H2D + D2H over PCIe for block_size_kb.

    The measurement uses pinned host memory + a dedicated CUDA
    stream so the latency reflects the deployment KV-transport
    path. Returns mean / P99 / P99.9 / max in microseconds.

    R28: timings + per-block stats delegated to
    :mod:`seer.timing.substrate_measure` so the four substrate
    harnesses share a single Clopper-Pearson / percentile path.
    """
    timings = measure_block_transfer(
        block_size_kb=block_size_kb, n_reps=n_reps, warmup=warmup,
        device_index=0)
    stats = a2_aligned_stats(timings)
    n_over_4x = stats["n_over_threshold"]
    return {
        "block_size_kb": block_size_kb,
        "n_reps": stats["n_reps"],
        "mean_us": stats["mean_us"],
        "median_us": stats["median_us"],
        "p99_us": stats["p99_us"],
        "p999_us": stats["p999_us"],
        "max_us": stats["max_us"],
        # A2/A7-aligned tail event: per-block latency > 4*ell_bar
        # (the Lemma 2 ell_max_factor=4 truncation point). This is
        # what Lemma 2''' A7 actually requires.
        "n_over_4x_mean": n_over_4x,
        "q_step_a2_aligned": stats["q_step_a2_emp"],
        "q_step_a2_aligned_upper95": stats["q_step_a2_upper95"],
        "q_step_a2_upper95_method": (
            "rule-of-three (Clopper-Pearson k=0)" if n_over_4x == 0
            else f"Clopper-Pearson (k={n_over_4x})"),
        # Sanity proxy: 3-sigma threshold (noise-confounded;
        # included for completeness but not the A7 calibration).
        "burst_rate_3sigma_proxy":
            _burst_rate_at_3sigma(timings),
    }


def _burst_rate_at_3sigma(xs: list[float]) -> float:
    """Per-rep fraction exceeding mu + 3*sigma; this is the
    A7 q_step input under the deployment substrate."""
    if len(xs) < 50:
        return 0.0
    mu = statistics.mean(xs)
    sigma = statistics.stdev(xs)
    thresh = mu + 3.0 * sigma
    return sum(1 for x in xs if x > thresh) / len(xs)


def _emit_harness_stub(out_path: pathlib.Path) -> dict:
    return {
        "status": "HARNESS_READY_AWAITING_GPU_RUN",
        "config": {
            "block_sizes_kb": [4, 16, 32, 64],
            "n_reps": 5000,
            "warmup": 200,
            "stream": "dedicated_transport_stream",
            "dtype": "float16",
            "pinned_host_memory": True,
            "direction": "H2D + D2H round-trip",
        },
        "rationale":
            "RTSS reviewer #2 demanded real PCIe cudaMemcpyAsync "
            "+ cgroup-throttled NVMe measurement to validate A7 "
            "(q_step coverage) on the deployment substrate. Without "
            "a CUDA-capable host, the harness is the artifact "
            "deliverable.",
        "produces": {
            "bar_ell_us": "per-block mean for the bound's input",
            "q_step_3sigma": "A7 per-step burst rate calibration",
            "p99_999_us": "tail envelope for full-tail-PS validation",
        },
        "runner_command":
            "python -m experiments.eC_bound_tightness.substrate_pcie_nvme "
            "--cuda --nvme-stress",
        "queued_in": "todo_atc.md (D, R17-D-PCIe)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_sizes_kb", nargs="+", type=int,
                    default=[4, 16, 32, 64])
    ap.add_argument("--n_reps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--cuda", action="store_true",
                    help="Require CUDA + run the cudaMemcpyAsync "
                         "measurement. Without --cuda the script "
                         "emits a HARNESS_READY stub.")
    ap.add_argument("--nvme-stress", action="store_true",
                    help="Stub: cgroup-throttled NVMe stress is not yet "
                         "implemented in this script (R19 honest "
                         "disclosure). The current measurement is a "
                         "CUDA pinned-host fp16 H2D+D2H microbenchmark; "
                         "the NVMe layering is the load-bearing "
                         "follow-up queued in todo_atc.md~D.")
    ap.add_argument("--force-stub-overwrite", action="store_true",
                    help="R28: by default the harness-stub path refuses "
                         "to overwrite an existing measured artifact "
                         "(status=MEASUREMENT_COMPLETE). Pass this flag "
                         "to clobber a prior measurement.")
    args = ap.parse_args()
    if args.nvme_stress:
        print("[substrate-pcie-nvme] WARNING: --nvme-stress is a stub; "
              "real cgroup-throttled NVMe path is queued (todo_atc.md~D)")

    if not args.cuda or not _has_cuda():
        stub = _emit_harness_stub(RESULTS / "substrate_pcie_nvme.json")
        tex_lines = [
            "% Generated by substrate_pcie_nvme.py (R17, harness-only).",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Field & Value \\",
            r"\midrule",
            r"Status & \textsc{harness-only} (queued for GPU+NVMe host) \\",
            r"Block sizes & 4/16/32/64\,KB \\",
            r"$n_\mathrm{reps}$ per cell & 5000 \\",
            r"Reports & $\bar\ell$, $P_{99}/P_{99.9}/\max$, $q_\mathrm{step}$ \\",
            r"Queued in & \texttt{todo\_atc.md}~D \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        wrote = safe_write_stub(
            RESULTS / "substrate_pcie_nvme.json",
            RESULTS / "substrate_pcie_nvme.tex",
            stub, "\n".join(tex_lines) + "\n",
            force=args.force_stub_overwrite,
        )
        if wrote:
            print("[substrate-pcie-nvme] STATUS: harness-only "
                  "(CUDA unavailable or --cuda not passed)")
        return 0

    rows = []
    for bs in args.block_sizes_kb:
        r = _measure_cudamemcpyasync(bs, n_reps=args.n_reps,
                                      warmup=args.warmup)
        rows.append(r)
        print(f"[substrate-pcie-nvme] {bs}KB: "
              f"mean={r['mean_us']:.2f}us "
              f"P99={r['p99_us']:.2f}us "
              f"P99.9={r['p999_us']:.2f}us "
              f"max={r['max_us']:.2f}us "
              f"q_step_A2={r['q_step_a2_aligned']:.5f} "
              f"({r['n_over_4x_mean']}/{r['n_reps']}) "
              f"q_3sigma_proxy={r['burst_rate_3sigma_proxy']:.4f}")
    # A7 verdict per block size, using the A2-aligned >4*mean
    # threshold (the Lemma 2 ell_max_factor=4 truncation point).
    # We compare against (i) the chat-tier rho=1e-2 and (ii) the
    # calibrated q_base from R12-R16. The 3-sigma proxy is kept
    # for sanity but is no longer the A7 verdict.
    rho_chat = 1e-2
    qbase_calib = 0.0067
    for r in rows:
        q_a7 = r["q_step_a2_aligned"]
        q_a7_upper = r.get("q_step_a2_aligned_upper95")
        r["a7_chat_nonvacuous"] = bool(q_a7 < rho_chat)
        r["a7_holds_at_calibrated_qbase"] = bool(q_a7 <= qbase_calib)
        # Honest upper-bound verdict: even the 95% CI upper bound is
        # below chat tier (when rule-of-three applies).
        r["a7_chat_nonvacuous_upper95"] = (
            (q_a7_upper is not None and q_a7_upper < rho_chat)
            if q_a7_upper is not None else None)
    out = {
        "status": "MEASUREMENT_COMPLETE",
        "block_sizes_kb": args.block_sizes_kb,
        "n_reps": args.n_reps,
        "warmup": args.warmup,
        "rho_chat": rho_chat,
        "qbase_calibrated": qbase_calib,
        "rows": rows,
    }
    (RESULTS / "substrate_pcie_nvme.json").write_text(
        json.dumps(out, indent=2))
    print(f"[substrate-pcie-nvme] wrote "
          f"{RESULTS / 'substrate_pcie_nvme.json'}")

    # TeX summary so the paper can read it. R19: report the
    # A2-aligned >4*mean threshold as the primary A7 calibrator
    # (matches Lemma 2's ell_max_factor=4 truncation point); the
    # 3-sigma proxy is kept as a secondary sanity column.
    tex_lines = [
        "% Generated by substrate_pcie_nvme.py "
        "(R18 measurement-complete; R19 threshold reconciled to A2).",
        r"\begin{tabular}{rrrrrrll}",
        r"\toprule",
        r"Block & $\bar\ell$ & $P_{99.9}$ & $\max$ & "
        r"$q_\mathrm{step}^\text{A2}$ & $q_\mathrm{step}^{3\sigma}$ & "
        r"A7@$\rho{=}10^{-2}$ \\",
        r"(KB) & ($\mu$s) & ($\mu$s) & ($\mu$s) & "
        r"($>4\bar\ell$, k/n) & (proxy) & (A2-aligned) \\",
        r"\midrule",
    ]
    for r in rows:
        q_a7_upper = r.get("q_step_a2_aligned_upper95", 1.0)
        nv = "non-vac." if r["a7_chat_nonvacuous"] else "VAC."
        # Always report k/n + Clopper-Pearson 95% upper (R20:
        # 64KB has k=1, which needs CP not rule-of-three).
        bold = r["a7_holds_at_calibrated_qbase"]
        kn_str = (rf"\textbf{{{r['n_over_4x_mean']}/{r['n_reps']}}}"
                  if bold else
                  rf"{r['n_over_4x_mean']}/{r['n_reps']}")
        a2_str = rf"{kn_str} ($\le {q_a7_upper:.2e}$)"
        tex_lines.append(
            f"{r['block_size_kb']} & "
            f"{r['mean_us']:.2f} & "
            f"{r['p999_us']:.2f} & "
            f"{r['max_us']:.2f} & "
            f"{a2_str} & "
            f"{r['burst_rate_3sigma_proxy']:.4f} & "
            f"{nv} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS / "substrate_pcie_nvme.tex").write_text(
        "\n".join(tex_lines) + "\n")
    print(f"[substrate-pcie-nvme] wrote "
          f"{RESULTS / 'substrate_pcie_nvme.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
