"""R21 contended-PCIe A7 substrate microbenchmark.

Reviewer R20 noted that the existing substrate_pcie_nvme.py
measurement is on the \\emph{quiesced} A100 PCIe transfer path
(single foreground cudaMemcpyAsync stream, no concurrent IO).
A7 (per-step burst correlation) is supposed to be substrate-
dependent: production substrates have PCIe queue contention from
other ranks, NUMA effects, and concurrent host-managed transfers.
A quiesced microbenchmark is the most benign substrate; ``A7
trivially holds'' on it.

This script adds a second adversarial substrate proxy without
relying on cgroup-throttled NVMe (which is still queued in
todo_atc.md~D). It spawns a \\emph{CUDA contender stream} that
continuously cudaMemcpyAsync's a large fp16 buffer (default
8\\,MiB blocks H2D+D2H) on a dedicated stream while the
measurement stream times the standard 4/16/32/64\\,KB blocks.
The contender saturates the PCIe queue and forces the measurement
stream to compete for bandwidth, approximating a multi-tenant
production deployment.

Output: ``substrate_a7_contended.{json,tex}``. The TeX is a
2-column table (quiesced vs contended) reporting per-block
Clopper-Pearson 95% upper bounds on $q_\\mathrm{step}^\\text{A2}$
($\\#\\{t > 4\\bar\\ell\\}/n$, the A2-aligned threshold from
Lemma 2''').

Scope (R21 honest): this is still NOT cgroup-throttled NVMe.
The contender is co-resident on the same PCIe root complex,
which is the dominant production-bottleneck class for
multi-tenant LLM serving. NVMe-stress remains queued in
todo_atc.md~D (real SPDK/io_uring throttle).
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


class _PCIeContender:
    """Background thread that floods cudaMemcpyAsync on a separate stream.

    Uses an independent CUDA stream + large fp16 buffer to compete
    for PCIe bandwidth with the foreground measurement stream.
    Stops on ``self.stop_event``.
    """

    def __init__(self, buffer_bytes: int = 8 * 1024 * 1024,
                 device_index: int = 0) -> None:
        import torch
        self.dev = torch.device("cuda", device_index)
        nb_elem = buffer_bytes // 2  # fp16
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
                # Don't synchronize per iteration; we want backpressure.
                self.iterations += 1
                if self.iterations % 200 == 0:
                    # Drain occasionally so the queue does not blow up.
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


def _measure_quiesced_or_contended(
    block_size_kb: int, n_reps: int, warmup: int,
    contended: bool, contender_buffer_mb: int = 8,
    device_index: int = 0,
) -> dict:  # pragma: no cover - hardware path
    import torch
    dev = torch.device("cuda", device_index)
    n_bytes = block_size_kb * 1024
    nb_elem = n_bytes // 2
    host = torch.empty(nb_elem, dtype=torch.float16, pin_memory=True)
    devv = torch.empty(nb_elem, dtype=torch.float16, device=dev)
    stream = torch.cuda.Stream(device=dev)

    contender: _PCIeContender | None = None
    if contended:
        contender = _PCIeContender(
            buffer_bytes=contender_buffer_mb * 1024 * 1024,
            device_index=device_index)
        contender.start()
        # Let contender warm up before we begin measurement.
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
    n_over = sum(1 for t in timings if t > 4.0 * mean)
    cp_upper = _cp_upper_975(n_over, n_reps)
    return {
        "block_size_kb": block_size_kb,
        "contended": contended,
        "contender_buffer_mb": contender_buffer_mb if contended else 0,
        "n_reps": n_reps,
        "mean_us": mean,
        "median_us": statistics.median(timings),
        "p99_us": _percentile(timings, 99.0),
        "p999_us": _percentile(timings, 99.9),
        "max_us": max(timings),
        "n_over_4x_mean": n_over,
        "q_step_a2_emp": n_over / n_reps,
        "q_step_a2_upper95": cp_upper,
        "contender_iterations": (
            contender.iterations if contender is not None else 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-sizes-kb", nargs="+", type=int,
                    default=[4, 16, 32, 64])
    ap.add_argument("--n-reps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--contender-buffer-mb", type=int, default=8,
                    help="Contender block size (default 8 MiB; saturates "
                         "PCIe queue without blowing host RAM)")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--qbase-calibrated", type=float, default=0.0067)
    ap.add_argument("--cuda", action="store_true",
                    help="Require CUDA; without --cuda emits a harness stub")
    ap.add_argument("--force-stub-overwrite", action="store_true",
                    help="R28: clobber existing measured artifact "
                         "(status=MEASUREMENT_COMPLETE) when emitting "
                         "the harness-only stub. Default refuses.")
    args = ap.parse_args()

    def _has_cuda() -> bool:
        try:
            import torch  # noqa: F401
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    if not args.cuda or not _has_cuda():
        stub = {
            "status": "HARNESS_READY_AWAITING_GPU_RUN",
            "rationale":
                "R21 reviewer asked for an A7 sanity run under "
                "PCIe contention, not just quiesced cudaMemcpyAsync. "
                "Without CUDA we emit the harness stub.",
            "config": {
                "block_sizes_kb": args.block_sizes_kb,
                "n_reps": args.n_reps,
                "warmup": args.warmup,
                "contender_buffer_mb": args.contender_buffer_mb,
            },
            "queued_in":
                "todo_atc.md~D for cgroup-throttled real-NVMe layering",
        }
        tex_lines = [
            "% Generated by substrate_a7_contended.py (R21, harness-only).",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Field & Value \\",
            r"\midrule",
            r"Status & \textsc{harness-only} (awaiting CUDA host) \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        wrote = safe_write_stub(
            RESULTS / "substrate_a7_contended.json",
            RESULTS / "substrate_a7_contended.tex",
            stub, "\n".join(tex_lines) + "\n",
            force=args.force_stub_overwrite,
        )
        if wrote:
            print(f"[a7-contended] STATUS: harness-only; wrote "
                  f"{RESULTS / 'substrate_a7_contended.json'}")
        return 0

    print(f"[a7-contended] R21 dual-mode microbench: quiesced + contended; "
          f"contender={args.contender_buffer_mb} MiB H2D+D2H")
    quiesced_rows = []
    contended_rows = []
    for bs in args.block_sizes_kb:
        rq = _measure_quiesced_or_contended(
            bs, n_reps=args.n_reps, warmup=args.warmup,
            contended=False,
            contender_buffer_mb=args.contender_buffer_mb,
            device_index=args.device_index)
        quiesced_rows.append(rq)
        print(f"[a7-contended] quiesced {bs}KB: "
              f"mean={rq['mean_us']:.2f}us "
              f"P99.9={rq['p999_us']:.2f}us max={rq['max_us']:.2f}us "
              f"k/n={rq['n_over_4x_mean']}/{rq['n_reps']} "
              f"qUB95={rq['q_step_a2_upper95']:.3e}")
    for bs in args.block_sizes_kb:
        rc = _measure_quiesced_or_contended(
            bs, n_reps=args.n_reps, warmup=args.warmup,
            contended=True,
            contender_buffer_mb=args.contender_buffer_mb,
            device_index=args.device_index)
        contended_rows.append(rc)
        print(f"[a7-contended] contended {bs}KB: "
              f"mean={rc['mean_us']:.2f}us "
              f"P99.9={rc['p999_us']:.2f}us max={rc['max_us']:.2f}us "
              f"k/n={rc['n_over_4x_mean']}/{rc['n_reps']} "
              f"qUB95={rc['q_step_a2_upper95']:.3e}")

    rows_all = []
    for rq, rc in zip(quiesced_rows, contended_rows):
        rows_all.append({
            "block_size_kb": rq["block_size_kb"],
            "quiesced": rq,
            "contended": rc,
            "a7_holds_quiesced": bool(rq["q_step_a2_upper95"] < args.rho),
            "a7_holds_contended": bool(rc["q_step_a2_upper95"] < args.rho),
            "a7_holds_contended_at_qbase":
                bool(rc["q_step_a2_upper95"] <= args.qbase_calibrated),
            "mean_inflation": rc["mean_us"] / max(rq["mean_us"], 1e-6),
            "p999_inflation": rc["p999_us"] / max(rq["p999_us"], 1e-6),
        })

    overall_holds_contended = all(r["a7_holds_contended"] for r in rows_all)
    out = {
        "status": "MEASUREMENT_COMPLETE",
        "rho_chat": args.rho,
        "qbase_calibrated": args.qbase_calibrated,
        "block_sizes_kb": args.block_sizes_kb,
        "n_reps": args.n_reps,
        "warmup": args.warmup,
        "contender_buffer_mb": args.contender_buffer_mb,
        "rows": rows_all,
        "verdict_contended_a7_holds": overall_holds_contended,
        "verdict_explainer":
            ("Contended A2-aligned q_step^A2 95% upper holds below "
             "chat-tier rho on every block size; A7 is non-vacuous "
             "even under in-process PCIe contention.")
            if overall_holds_contended else
            ("Contended A2-aligned q_step^A2 95% upper EXCEEDS "
             "chat-tier rho on at least one block size; A7 is "
             "substrate-conditional and full-tail-PS MUST NOT be "
             "deployed on this substrate without recalibration."),
        "scope":
            ("Contender is a co-resident PCIe consumer on the same "
             "root complex. cgroup-throttled NVMe stress remains "
             "queued (todo_atc.md~D)."),
    }
    out_json = RESULTS / "substrate_a7_contended.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[a7-contended] wrote {out_json}")

    # 2-column TeX table.
    tex_lines = [
        "% Generated by substrate_a7_contended.py "
        "(R21 quiesced-vs-contended A7 sanity).",
        r"\begin{tabular}{rrrcrrcrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Quiesced} & & \multicolumn{2}{c}{Contended} & "
        r"& Mean-inflation \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
        r"Block & $\bar\ell$ & $P_{99.9}$ & "
        r"$q_\mathrm{step}^\text{A2,UB}$ & & "
        r"$\bar\ell$ & $q_\mathrm{step}^\text{A2,cont,UB}$ & & "
        r"$\bar\ell^\text{cont}/\bar\ell^\text{q}$ \\",
        r"(KB) & ($\mu$s) & ($\mu$s) & (k/n) & & ($\mu$s) & (k/n) & & $\times$ \\",
        r"\midrule",
    ]
    for r in rows_all:
        q = r["quiesced"]
        c = r["contended"]
        q_kn = f"{q['n_over_4x_mean']}/{q['n_reps']} ($\\le {q['q_step_a2_upper95']:.2e}$)"
        c_kn = f"{c['n_over_4x_mean']}/{c['n_reps']} ($\\le {c['q_step_a2_upper95']:.2e}$)"
        tex_lines.append(
            f"{r['block_size_kb']} & "
            f"{q['mean_us']:.2f} & {q['p999_us']:.2f} & {q_kn} & & "
            f"{c['mean_us']:.2f} & {c_kn} & & "
            f"{r['mean_inflation']:.2f} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS / "substrate_a7_contended.tex").write_text(
        "\n".join(tex_lines) + "\n")
    print(f"[a7-contended] wrote "
          f"{RESULTS / 'substrate_a7_contended.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
