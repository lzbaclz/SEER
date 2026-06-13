"""R15: representative-state MBPTA hygiene on per-chunk subsets.

The pooled MBPTA hygiene (\\texttt{mbpta\\_hygiene.py}) rejects
i.i.d.\\ on the persisted $n{=}3{,}000$ sample (runs $p{<}10^{-3}$,
Ljung-Box $p{<}10^{-3}$, AD on GPD rejected at 5\\%). This is the
expected outcome on a pooled long-running benchmark where the
GPU goes through warm-up and steady-state phases that violate
stationarity \\emph{between} phases, even when each phase is
internally well-behaved.

R15 reviewer asked for representative-state subset hygiene: split
the raw sample into $K$ equal-length chunks and re-run the
hygiene tests per chunk. A chunk \\emph{passes} when all three
tests pass (or, more conservatively, when runs + Ljung-Box both
fail to reject i.i.d.\\ at $5\\%$). The fraction of chunks that
pass is the representative-state coverage of the empirical
envelope; the pass set is the subset on which a strict MBPTA
WCET claim would be defensible.

We use a non-overlapping sliding-window split (\\texttt{K=10}
default, $n_\\mathrm{chunk}{=}300$ samples), drop the first chunk
(warm-up), and report the per-chunk pass/fail along with the
fraction. This is a hygiene diagnostic, not a WCET certification;
the empirical envelope is reported in §VI.B regardless.

Output: \\texttt{mbpta\\_representative\\_state.\\{json,tex\\}}.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

from experiments.eE_lap_wcet.mbpta_hygiene import (
    load_raw, runs_test, ljung_box,
    anderson_darling_gpd_threshold,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eE_lap_wcet/results"

ALPHA = 0.05


def chunk_split(x: np.ndarray, K: int, drop_first: int = 1
                ) -> list[np.ndarray]:
    """Non-overlapping equal-length chunks; drop_first chunks
    treated as warm-up."""
    n = len(x)
    chunk_n = n // K
    chunks = []
    for i in range(K):
        lo = i * chunk_n
        hi = (i + 1) * chunk_n
        chunks.append(x[lo:hi])
    return chunks[drop_first:]


def chunk_hygiene(chunk: np.ndarray) -> dict:
    rt = runs_test(chunk)
    lb = ljung_box(chunk, lags=10)
    # GPD AD needs at least 30 excesses; on n=300 chunks at q=0.95
    # excess count is ~15 which is below threshold. We report
    # only runs and Ljung-Box for the chunk pass decision; AD is
    # checked on the pooled pass set instead.
    pass_runs = rt["p_value"] > ALPHA if not math.isnan(rt["p_value"]) else False
    pass_lb = lb["p_value"] > ALPHA if not math.isnan(lb["p_value"]) else False
    return {
        "n": int(len(chunk)),
        "p50_us": float(np.median(chunk)),
        "p999_us": float(np.quantile(chunk, 0.999)),
        "runs_p": float(rt["p_value"]) if not math.isnan(rt["p_value"]) else None,
        "lb_p": float(lb["p_value"]) if not math.isnan(lb["p_value"]) else None,
        "pass_runs": bool(pass_runs),
        "pass_lb": bool(pass_lb),
        "pass_both": bool(pass_runs and pass_lb),
    }


def analyse_file(path: pathlib.Path, K: int, drop_first: int) -> dict | None:
    x = load_raw(path)
    if x is None:
        return None
    chunks = chunk_split(x, K, drop_first)
    per_chunk = [chunk_hygiene(c) for c in chunks]
    n_pass = sum(1 for c in per_chunk if c["pass_both"])
    n_chunks = len(per_chunk)
    # Pool the passing chunks and run the AD GoF check on that.
    # Chunk pools are smaller than the pooled-sample case, so lower
    # the min-excess threshold from 30 (asymptotic AD rule) to 15
    # (matches POT EVT small-sample practice; flag explicitly).
    if n_pass > 0:
        pooled = np.concatenate(
            [c for c, h in zip(chunks, per_chunk) if h["pass_both"]])
        ad = anderson_darling_gpd_threshold(pooled, 0.95, min_excess=15)
    else:
        ad = {"A2_adj": float("nan"), "fit_pass_5pct": False,
              "n_excess": 0,
              "note": "no chunks passed runs+LB"}
    return {
        "file": path.name,
        "n_total": int(len(x)),
        "K": K,
        "drop_first": drop_first,
        "n_chunks_analysed": n_chunks,
        "n_passing_both": n_pass,
        "pass_fraction": n_pass / n_chunks if n_chunks else 0.0,
        "per_chunk": per_chunk,
        "ad_on_passing_pool": ad,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10,
                    help="Number of non-overlapping chunks.")
    ap.add_argument("--drop_first", type=int, default=1,
                    help="Number of warm-up chunks to discard.")
    args = ap.parse_args()

    candidates = [
        RESULTS / "lap_wcet_A100_local_tiny_mlp_trt_b4096_raw_samples.json",
        RESULTS / "lap_wcet_A100_local_tiny_mlp_torch_b4096_raw_samples.json",
    ]
    results = []
    for p in candidates:
        r = analyse_file(p, args.K, args.drop_first)
        if r is not None:
            results.append(r)
            backend = "TRT" if "trt" in p.name else "torch"
            ad_pass = r["ad_on_passing_pool"].get("fit_pass_5pct", False)
            print(f"[mbpta-rep-state] {backend}: "
                  f"{r['n_passing_both']}/{r['n_chunks_analysed']} chunks "
                  f"pass runs+LB at alpha={ALPHA}; "
                  f"AD-on-pass-pool GPD-pass={ad_pass}")
        else:
            print(f"[mbpta-rep-state] SKIP {p.name}")

    out = {
        "K": args.K, "drop_first": args.drop_first,
        "alpha": ALPHA,
        "results": results,
        "interpretation": (
            "Per-chunk hygiene tests the local stationarity of the "
            "MBPET sample. A chunk passes when both Wald-Wolfowitz "
            "runs and Ljung-Box at lag 10 fail to reject i.i.d. at "
            "alpha=0.05. The pass fraction is the representative-state "
            "coverage; the AD-on-passing-pool row checks GPD GoF on "
            "the subset where stationarity is locally defensible. We "
            "do NOT claim a hard MBPTA WCET; the empirical envelope is "
            "reported in §VI.B regardless."
        ),
    }
    out_json = RESULTS / "mbpta_representative_state.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[mbpta-rep-state] wrote {out_json}")

    # TeX summary table. AD column carries the actual A2_adj + the
    # GPD-fit verdict so a reviewer cannot misread "no" as a successful
    # rejection when in fact the test was skipped for lack of excesses.
    lines = [
        "% Generated by mbpta_representative_state.py (R15+R16).",
        "% Per-chunk hygiene on the persisted MBPET raw sample.",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Backend & $K$ chunks & passing & $\Pr[\text{pass}]$ "
        r"& AD on pass-pool ($A^2_\mathrm{adj}$; $n_\mathrm{ex}$) \\",
        r"\midrule",
    ]
    for r in results:
        backend = "TRT" if "trt" in r["file"] else "torch"
        ad = r["ad_on_passing_pool"]
        a2 = ad.get("A2_adj", float("nan"))
        n_ex = ad.get("n_excess", 0)
        if not isinstance(a2, (int, float)) or math.isnan(a2):
            ad_str = f"n/a ($n_\\mathrm{{ex}}{{=}}{n_ex}$)"
        else:
            verdict = ("rejects GPD" if a2 > 2.492
                       else "fails to reject GPD")
            ad_str = (f"${a2:.2f}$, $n_\\mathrm{{ex}}{{=}}{n_ex}$ "
                      f"({verdict})")
        lines.append(
            f"\\texttt{{{backend}}} & {r['n_chunks_analysed']} & "
            f"{r['n_passing_both']} & "
            f"{r['pass_fraction']:.2f} & {ad_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "mbpta_representative_state.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[mbpta-rep-state] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
