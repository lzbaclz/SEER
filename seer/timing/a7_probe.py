"""R21 operator-side A7 validation probe.

Five-minute pre-deployment substrate probe. Emits a YES/NO on
whether assumption A7 (per-step burst correlation, the precondition
of Lemma 2''' / ``full-tail-PS``) holds on the operator's substrate
at the A2-aligned $>4\\bar\\ell$ threshold.

\\textbf{Scope (R21 honest)}: this is a \\emph{substrate-grade}
probe. It measures the bare CUDA DMA path
($\\texttt{cudaMemcpyAsync}$ H2D+D2H on a dedicated stream, fp16
pinned host memory). It does NOT exercise the production prefetch
path through vLLM's ``KVConnector_V1.start_load_kv`` (which
includes scheduler-side block-id mapping, host-side per-layer
pinned-buffer allocation, inter-stream sync via CUDA events, and
potential ``BlockPool`` lock contention). The
\\emph{deployment-grade} vLLM-integrated probe (which triggers
the real ``start_load_kv`` path and measures A7 against it) is
queued in ``todo_atc.md~A2``. Operators should treat the
HOLDS verdict as a substrate-prerequisite gate (necessary but
not sufficient for a deployment-grade A7 claim).

Usage:
    python -m seer.timing.a7_probe --rho 1e-2 --block-sizes 4 16 32 64

Reports, per block size:
  * mean / P99 / P99.9 / max
  * empirical $q_\\mathrm{step}^\\text{A2} = \\#\\{t > 4\\bar\\ell\\}/n$
  * Clopper-Pearson 95% upper $q_\\mathrm{step}^\\text{A2,UB}$
  * verdict (HOLDS / FAILS) against operator-supplied $\\rho$ and
    optionally a tightened $q_\\mathrm{base}$ calibration value.

Exit code: 0 if A7 HOLDS for every block size; 2 if any block size
FAILS; 1 on configuration / CUDA error. Operators can wire this
into a CI gate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Iterable

from seer.timing.substrate_measure import (
    cp_upper_975,
)
from seer.timing.substrate_measure import (
    percentile as _percentile,
)


def _clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Clopper-Pearson upper at two-sided confidence ``conf``.

    Delegates to :func:`seer.timing.substrate_measure.cp_upper_975`
    for ``conf=0.95`` (the only confidence level used in the paper).
    Kept under the old name so external callers / tests don't break.
    """
    if abs(conf - 0.95) > 1e-9:
        raise ValueError(
            "a7_probe._clopper_pearson_upper currently supports "
            "conf=0.95 only (use substrate_measure.cp_upper_975 for "
            "the canonical helper)."
        )
    return cp_upper_975(k, n)


def measure_block(block_size_kb: int, n_reps: int, warmup: int,
                  mode: str = "substrate",
                  pool_size_blocks: int = 1024,
                  device_index: int = 0) -> dict:
    """Time fp16 H2D+D2H ``cudaMemcpyAsync`` at ``block_size_kb``.

    Two ``mode`` settings:

    ``substrate`` (default, R21): bare DMA path -- one pinned host
    tensor, one device tensor, one dedicated stream, single
    timing event around the H2D+D2H pair. This is the lower
    bound on what the vLLM-integrated connector path can ever do.

    ``integrated`` (R23 W2): mimics the integration overheads of
    ``SeerKVConnector.start_load_kv`` and the per-block
    ``_xfer_stream`` pattern in ``seer/integration/vllm_connector.py``:
    a per-block pinned-host tensor pool of size ``pool_size_blocks``
    (matches the connector's ``_host_pool`` dict), a Python-side
    dict lookup per block, a per-block CUDA event record-and-sync
    (rather than a single batch event), and a periodic
    ``stream.synchronize()`` every 32 blocks to drain backpressure
    (matches the connector's BlockPool-locking discipline). This
    is an in-process probe and remains a proxy for the full
    runtime path (no scheduler-side rebuild,
    BlockPool mutex contention, or KVCacheManager lock acquisition),
    but it captures the dominant per-block integration overhead
    the substrate probe ignores.

    Returns a dict with the per-block stats. Caller is responsible
    for asserting CUDA availability beforehand.
    """
    import torch
    dev = torch.device("cuda", device_index)
    n_bytes = block_size_kb * 1024
    nb_elem = n_bytes // 2  # fp16
    stream = torch.cuda.Stream(device=dev)

    if mode == "integrated":
        host_pool = {
            i: torch.empty(nb_elem, dtype=torch.float16, pin_memory=True)
            for i in range(pool_size_blocks)
        }
        devv = torch.empty(nb_elem, dtype=torch.float16, device=dev)
        import random as _rnd
        _rng = _rnd.Random(0)

        for _ in range(warmup):
            blk = _rng.randrange(pool_size_blocks)
            host = host_pool[blk]
            with torch.cuda.stream(stream):
                devv.copy_(host, non_blocking=True)
                host.copy_(devv, non_blocking=True)
        torch.cuda.synchronize()

        timings = []
        for it in range(n_reps):
            blk = _rng.randrange(pool_size_blocks)
            host = host_pool[blk]  # Python dict lookup, lock-free
            s_evt = torch.cuda.Event(enable_timing=True)
            e_evt = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                s_evt.record(stream)
                devv.copy_(host, non_blocking=True)
                host.copy_(devv, non_blocking=True)
                e_evt.record(stream)
            e_evt.synchronize()
            timings.append(s_evt.elapsed_time(e_evt) * 1000.0)
            if (it & 31) == 0:
                stream.synchronize()
    else:
        host = torch.empty(nb_elem, dtype=torch.float16, pin_memory=True)
        devv = torch.empty(nb_elem, dtype=torch.float16, device=dev)
        for _ in range(warmup):
            with torch.cuda.stream(stream):
                devv.copy_(host, non_blocking=True)
                host.copy_(devv, non_blocking=True)
        torch.cuda.synchronize()
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
    n_over = sum(1 for t in timings if t > 4.0 * mean)
    return {
        "block_size_kb": block_size_kb,
        "mode": mode,
        "pool_size_blocks": (pool_size_blocks
                             if mode == "integrated" else 1),
        "n_reps": n_reps,
        "mean_us": mean,
        "median_us": statistics.median(timings),
        "p99_us": _percentile(timings, 99.0),
        "p999_us": _percentile(timings, 99.9),
        "max_us": max(timings),
        "n_over_4x_mean": n_over,
        "q_step_a2_emp": n_over / n_reps,
        "q_step_a2_upper95": _clopper_pearson_upper(n_over, n_reps),
    }


def run_probe(
    block_sizes_kb: Iterable[int],
    n_reps: int = 5000,
    warmup: int = 200,
    rho: float = 1e-2,
    q_base_calibrated: float | None = 0.0067,
    mode: str = "substrate",
    pool_size_blocks: int = 1024,
    device_index: int = 0,
) -> dict:
    """Top-level probe. Returns the full result dict with a verdict.

    ``mode`` selects the substrate-grade DMA path (``substrate``,
    R21 default) or the R23 W2 ``integrated`` path that mimics the
    per-block pool-lookup + per-block CUDA event sync of the vLLM
    SeerKVConnector.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return {"status": "CUDA_UNAVAILABLE",
                "error": f"torch import failed: {exc!r}"}
    import torch
    if not torch.cuda.is_available():
        return {"status": "CUDA_UNAVAILABLE",
                "error": "torch.cuda.is_available() returned False"}

    t0 = time.perf_counter()
    rows = []
    fail_blocks = []
    for bs in block_sizes_kb:
        r = measure_block(bs, n_reps=n_reps, warmup=warmup,
                          mode=mode, pool_size_blocks=pool_size_blocks,
                          device_index=device_index)
        ub = r["q_step_a2_upper95"]
        r["a7_holds_at_rho"] = bool(ub < rho)
        if q_base_calibrated is not None:
            r["a7_holds_at_calibrated_qbase"] = bool(ub <= q_base_calibrated)
        if not r["a7_holds_at_rho"]:
            fail_blocks.append(bs)
        rows.append(r)
    wall_s = time.perf_counter() - t0
    overall_holds = len(fail_blocks) == 0
    verdict = "HOLDS" if overall_holds else "FAILS"
    return {
        "status": "MEASUREMENT_COMPLETE",
        "rho": rho,
        "q_base_calibrated": q_base_calibrated,
        "n_reps": n_reps,
        "warmup": warmup,
        "mode": mode,
        "pool_size_blocks": (pool_size_blocks
                             if mode == "integrated" else 1),
        "wall_seconds": wall_s,
        "rows": rows,
        "fail_blocks_kb": fail_blocks,
        "verdict": verdict,
        "verdict_overall_holds": overall_holds,
        "verdict_explainer":
            ("A7 holds at the operator-supplied rho on every block "
             "size; full-tail-PS may be flipped on for admission "
             "control on this substrate.") if overall_holds else
            ("At least one block size FAILS the A7 upper-bound test "
             "at the operator-supplied rho. full-tail-PS MUST NOT "
             "be used as an admission rule on this substrate; fall "
             "back to full-tail-guard or recalibrate."),
    }


def format_report(out: dict) -> str:
    if out.get("status") != "MEASUREMENT_COMPLETE":
        return (f"[a7-probe] STATUS: {out.get('status')}  "
                f"{out.get('error', '')}".rstrip())
    lines = []
    rho = out["rho"]
    qbase = out.get("q_base_calibrated")
    mode = out.get("mode", "substrate")
    lines.append(f"[a7-probe] R21/R23 A7 validation probe ({mode})")
    pool = out.get("pool_size_blocks", 1)
    lines.append(f"[a7-probe] rho={rho:.1e}  "
                 f"q_base_calibrated={qbase}  "
                 f"pool_blocks={pool}  "
                 f"wall={out['wall_seconds']:.1f}s")
    lines.append("[a7-probe] {:>5} {:>9} {:>9} {:>9} {:>9} "
                 "{:>8} {:>12} {:>9}".format(
                     "KB", "mean_us", "P99_us", "P99.9_us", "max_us",
                     "k/n", "qUB95", "verdict"))
    for r in out["rows"]:
        verdict = "HOLDS" if r["a7_holds_at_rho"] else "FAILS"
        lines.append("[a7-probe] {:>5} {:>9.2f} {:>9.2f} {:>9.2f} "
                     "{:>9.2f} {:>8} {:>12.3e} {:>9}".format(
                         r["block_size_kb"], r["mean_us"], r["p99_us"],
                         r["p999_us"], r["max_us"],
                         f"{r['n_over_4x_mean']}/{r['n_reps']}",
                         r["q_step_a2_upper95"], verdict))
    lines.append(f"[a7-probe] VERDICT: {out['verdict']}")
    if out["fail_blocks_kb"]:
        lines.append(f"[a7-probe] FAILED block sizes (KB): "
                     f"{out['fail_blocks_kb']}")
    lines.append(f"[a7-probe] {out['verdict_explainer']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block-sizes", nargs="+", type=int,
                    default=[4, 16, 32, 64],
                    help="KV-block sizes in KB (default: 4 16 32 64)")
    ap.add_argument("--n-reps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--rho", type=float, default=1e-2,
                    help="Operator-side miss target (default: 1e-2 chat)")
    ap.add_argument("--q-base", type=float, default=0.0067,
                    help="Calibrated q_base for cross-check "
                         "(default: SEER chat-tier 0.0067; "
                         "pass <=0 to skip)")
    ap.add_argument("--mode", choices=("substrate", "integrated"),
                    default="substrate",
                    help="substrate: bare DMA (R21); "
                         "integrated: per-block pool-lookup + "
                         "per-block CUDA event sync + periodic "
                         "stream drain (R23 W2 vLLM-connector "
                         "integration-path proxy)")
    ap.add_argument("--pool-size-blocks", type=int, default=1024,
                    help="Size of the simulated host-pinned block "
                         "pool for --mode=integrated (default 1024)")
    ap.add_argument("--device-index", type=int, default=0,
                    help="CUDA device index (default 0)")
    ap.add_argument("--out-json", default=None,
                    help="Optional path to write JSON result")
    args = ap.parse_args(argv)
    qb = args.q_base if args.q_base and args.q_base > 0 else None
    out = run_probe(
        block_sizes_kb=args.block_sizes,
        n_reps=args.n_reps,
        warmup=args.warmup,
        rho=args.rho,
        q_base_calibrated=qb,
        mode=args.mode,
        pool_size_blocks=args.pool_size_blocks,
        device_index=args.device_index,
    )
    print(format_report(out))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[a7-probe] wrote {args.out_json}")
    if out.get("status") != "MEASUREMENT_COMPLETE":
        return 1
    return 0 if out.get("verdict_overall_holds") else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
