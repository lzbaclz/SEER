"""Bootstrap confidence intervals for tail latency percentiles.

Reviewers (and ourselves) need to know how much of the SEER-vs-baseline
gap is real vs. sampling noise. The published eA percentiles
(P99 = 35.79 ms, P999 = 49.92 ms) are computed from a single sample of
20 prompts × ≤32 tokens ≈ 600 steps; a 1-in-1000 quantile estimated
from ~600 observations is statistically thin without a confidence
interval.

This module provides a non-parametric bootstrap CI estimator:

    >>> latencies_us = [...]                 # per-step latency vector
    >>> ci = bootstrap_percentile_ci(latencies_us, p=99.9, n_resamples=10_000)
    >>> ci.point, ci.lo, ci.hi
    (49920.0, 47800.5, 52310.2)

It also has a CLI that walks a directory of eA / eB / eD JSONs, pulls
``results[*].per_step_us``, and emits a sidecar
``<file>.boot_ci.json`` per input plus a roll-up
``<dir>/bootstrap_ci.csv`` for the paper.

CPU-only, no torch dependency. Median-of-10000 resamples runs in
~30 ms per percentile per file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path


@dataclass
class PercentileCI:
    p: float
    point: float
    lo: float
    hi: float
    confidence: float
    n_samples: int
    n_resamples: int


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs_sorted) - 1)
    frac = k - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def bootstrap_percentile_ci(
    samples,
    p: float = 99.0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PercentileCI:
    """Non-parametric bootstrap CI for the p-th percentile of ``samples``.

    Standard percentile method: resample with replacement n_resamples
    times, compute the percentile each time, then take the empirical
    [(1-c)/2, (1+c)/2] quantiles of the resampled percentiles.

    Uses numpy when available (≈50× faster on n ≥ 1000) and a pure
    Python fallback otherwise. NaN samples are dropped before resampling.
    """
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"p must be in [0, 100], got {p}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    try:
        import numpy as np
    except ImportError:
        np = None  # pure-python fallback below

    if np is not None:
        arr = np.asarray(list(samples), dtype=float)
        arr = arr[~np.isnan(arr)]
        n = arr.size
        if n == 0:
            return PercentileCI(p=p, point=float("nan"), lo=float("nan"),
                                hi=float("nan"), confidence=confidence,
                                n_samples=0, n_resamples=n_resamples)
        point = float(np.percentile(arr, p, method="linear"))
        if n < 2:
            return PercentileCI(p=p, point=point, lo=point, hi=point,
                                confidence=confidence, n_samples=n,
                                n_resamples=0)
        rng = np.random.default_rng(seed)
        idxs = rng.integers(0, n, size=(n_resamples, n))
        resampled = np.percentile(arr[idxs], p, axis=1, method="linear")
        lo_q = (1.0 - confidence) / 2.0 * 100.0
        hi_q = (1.0 + confidence) / 2.0 * 100.0
        lo = float(np.percentile(resampled, lo_q))
        hi = float(np.percentile(resampled, hi_q))
        return PercentileCI(p=p, point=point, lo=lo, hi=hi,
                            confidence=confidence, n_samples=int(n),
                            n_resamples=n_resamples)

    # Pure-Python fallback (slow; only for environments without numpy).
    xs = [x for x in samples if x == x]
    n = len(xs)
    if n == 0:
        return PercentileCI(p=p, point=float("nan"), lo=float("nan"),
                            hi=float("nan"), confidence=confidence,
                            n_samples=0, n_resamples=n_resamples)
    point = _percentile(xs, p)
    if n < 2:
        return PercentileCI(p=p, point=point, lo=point, hi=point,
                            confidence=confidence, n_samples=n, n_resamples=0)
    rng = random.Random(seed)
    resampled = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        resampled.append(_percentile([xs[i] for i in idxs], p))
    resampled.sort()
    lo_q = (1.0 - confidence) / 2.0 * 100.0
    hi_q = (1.0 + confidence) / 2.0 * 100.0
    lo = _percentile(resampled, lo_q)
    hi = _percentile(resampled, hi_q)
    return PercentileCI(p=p, point=point, lo=lo, hi=hi,
                        confidence=confidence, n_samples=n,
                        n_resamples=n_resamples)


def _flatten_per_step(result_dict) -> list[float]:
    """Pull per-step latencies out of a runner JSON.

    Schemas this knows about:
        {results: [{per_step_us: [...]}, ...]}    — eA / eB / eD style
        {samples_us: [...]}                        — wcet sidecar
        {per_step_us: [...]}                       — single-request flat
    """
    if "results" in result_dict and isinstance(result_dict["results"], list):
        out: list[float] = []
        for r in result_dict["results"]:
            out.extend(r.get("per_step_us", []) or [])
        return out
    if "samples_us" in result_dict:
        return list(result_dict["samples_us"])
    if "per_step_us" in result_dict:
        return list(result_dict["per_step_us"])
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m seer.eval.bootstrap")
    ap.add_argument("paths", nargs="+",
                    help="JSON files or globs (e.g. "
                         "experiments/eA_tail_latency/results_mooncake_full/*.json)")
    ap.add_argument("--percentiles", type=float, nargs="+",
                    default=[50.0, 99.0, 99.9])
    ap.add_argument("--n_resamples", type=int, default=10_000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default=None,
                    help="If set, also write a roll-up CSV at this path.")
    args = ap.parse_args(argv)

    all_files: list[str] = []
    for p in args.paths:
        if any(c in p for c in "*?["):
            all_files.extend(sorted(glob(p)))
        else:
            all_files.append(p)
    if not all_files:
        print("[bootstrap] no files matched", file=__import__("sys").stderr)
        return 2

    rows = []
    for f in all_files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"[bootstrap] {f}: failed to load ({e})")
            continue
        samples = _flatten_per_step(d)
        if not samples:
            print(f"[bootstrap] {f}: no per-step samples")
            continue
        cis = {}
        for p in args.percentiles:
            ci = bootstrap_percentile_ci(samples, p=p,
                                         n_resamples=args.n_resamples,
                                         confidence=args.confidence,
                                         seed=args.seed)
            cis[f"p{p}"] = asdict(ci)
            rows.append({
                "file": os.path.basename(f),
                "policy": d.get("policy", "?"),
                "hbm_budget": d.get("hbm_budget", float("nan")),
                "workload": d.get("workload", "?"),
                "p": p,
                "n_samples": ci.n_samples,
                "point_us": ci.point,
                "lo_us": ci.lo,
                "hi_us": ci.hi,
                "ci_pct": ci.confidence,
            })

        sidecar = Path(f).with_suffix(".boot_ci.json")
        sidecar.write_text(json.dumps({
            "source": os.path.basename(f),
            "policy": d.get("policy"),
            "hbm_budget": d.get("hbm_budget"),
            "workload": d.get("workload"),
            "n_samples": cis[f"p{args.percentiles[0]}"]["n_samples"],
            "n_resamples": args.n_resamples,
            "confidence": args.confidence,
            "percentiles": cis,
        }, indent=2))
        line = f"[bootstrap] {os.path.basename(f)}  n={cis[f'p{args.percentiles[0]}']['n_samples']}  "
        for p in args.percentiles:
            ci = cis[f"p{p}"]
            line += f"P{p:g}={ci['point']/1000:.2f}ms (CI {ci['lo']/1000:.2f}-{ci['hi']/1000:.2f})  "
        print(line.rstrip())

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[bootstrap] wrote roll-up to {args.out_csv} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
