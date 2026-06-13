"""Estimate the residual sub-Gaussian σ from a short pilot run.

Operators using Lemma 2 / Lemma 2′ as a sizing tool need σ_residual
*before* committing to a deployment. The eC `calibrate.py` script
recovers σ post-hoc from a steady-state P99.5 − P50 gap, but that
requires hundreds of decode steps already. This module provides:

1. ``sigma_from_pilot(latencies_us)`` — recover σ from a small pilot
   sample (≥ 20 steps), using the (P95 − P50) / sqrt(2 ln 20) gap so
   the estimator is well-defined at small n.
2. ``sigma_from_running_mean(mean_us, std_us)`` — closed-form when
   the operator can supply mean / std from log streams (no per-step
   array needed).
3. CLI: ``python -m seer.timing.sigma_estimate trace.json --warmup_steps 1``
   reads a runner JSON and emits {sigma_us, n_used, p50, p95, mean,
   std} as a single JSON record.

The estimator drops the first ``warmup_steps`` observations per
request to exclude the JIT / CUDA Graph cold-start spike that
dominates the warm-up-included percentile.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SigmaEstimate:
    sigma_us: float
    n_used: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    std_us: float
    method: str    # "p95_p50_gap" or "running_std"


def _percentile(xs, p):
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs_sorted) - 1)
    frac = k - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def sigma_from_pilot(latencies_us, p_high: float = 95.0,
                     p_low: float = 50.0) -> SigmaEstimate:
    """Estimate σ from the (p_high - p_low) gap of a pilot sample.

    Maps the empirical gap to a sub-Gaussian σ via the standard
    one-sided sub-Gaussian tail factor:

        gap_q = sigma * sqrt(2 ln(1/(1-q)))

    where q = p_high / 100. Robust when p_high ≤ 95: the 95-th
    percentile of 20 samples is still empirically defined (it is
    the 19th sorted sample, so finite-sample bias is small).

    Returns σ in µs.
    """
    xs = [x for x in latencies_us if x == x]
    n = len(xs)
    if n < 5:
        raise ValueError(f"need at least 5 pilot samples, got {n}")
    if not (0.0 < p_low < p_high < 100.0):
        raise ValueError("require 0 < p_low < p_high < 100")
    p_low_v = _percentile(xs, p_low)
    p_high_v = _percentile(xs, p_high)
    p99_v = _percentile(xs, 99.0)
    gap = max(0.0, p_high_v - p_low_v)
    factor = math.sqrt(2.0 * math.log(1.0 / (1.0 - p_high / 100.0)))
    sigma = gap / factor if factor > 0 else 0.0
    return SigmaEstimate(
        sigma_us=float(sigma),
        n_used=n,
        p50_us=float(p_low_v),
        p95_us=float(p_high_v),
        p99_us=float(p99_v),
        mean_us=float(statistics.fmean(xs)),
        std_us=float(statistics.pstdev(xs)) if n > 1 else 0.0,
        method="p95_p50_gap",
    )


def sigma_bootstrap_ci(
    latencies_us,
    n_resamples: int = 1000,
    p_high: float = 99.5,
    p_low: float = 50.0,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap a confidence interval on the one-sided sub-Gaussian
    $\\sigma$ proxy $(P_\\mathrm{high} - P_\\mathrm{low}) / Z_q$.

    P3-1 fix (review round May 2026, reviewer #1 + #3): the eC
    calibrated bound depends on $\\sigma$ as a point estimate from
    ${\\sim}600$ pilot steps. The bound itself is exponential in
    $1/\\sigma^2$, so a $\\pm 10\\%$ uncertainty in $\\sigma$ moves the
    bound by an order of magnitude. We expose a percentile-bootstrap
    CI so the operator can publish a $(\\sigma_\\mathrm{lo},
    \\sigma_\\mathrm{hi})$ pair and a $(\\mathrm{bound}_\\mathrm{lo},
    \\mathrm{bound}_\\mathrm{hi})$ pair.

    Parameters
    ----------
    latencies_us : iterable of float
        Per-step IO-free residual samples (recommended source:
        ``per_step_base_us`` from the simulator).
    n_resamples : int
        Number of percentile-bootstrap resamples; default 1000.
    p_high, p_low : float
        Percentiles used in the $\\sigma$ estimator; default 99.5 / 50.
    confidence : float
        Two-sided confidence level in $(0, 1)$; default 0.95.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys ``sigma_lo / sigma_med / sigma_hi``, the bootstrap
    median, and the $\\sigma$ at the input's full sample. The CI
    is the central
    $(1-\\text{confidence})/2$ -- $(1+\\text{confidence})/2$
    percentile of the resampled $\\sigma$ values.
    """
    import random
    xs = [float(x) for x in latencies_us if x == x]
    n = len(xs)
    if n < 20:
        raise ValueError(f"need at least 20 samples for bootstrap, got {n}")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    factor = math.sqrt(2.0 * math.log(1.0 / (1.0 - p_high / 100.0)))

    def _sigma(sample):
        ph = _percentile(sample, p_high)
        pl = _percentile(sample, p_low)
        gap = max(0.0, ph - pl)
        return gap / factor if factor > 0 else 0.0

    sigma_point = _sigma(xs)
    rng = random.Random(seed)
    sigmas = []
    for _ in range(n_resamples):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        sigmas.append(_sigma(sample))
    sigmas.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples))
    med_idx = n_resamples // 2
    return {
        "sigma_point_us": float(sigma_point),
        "sigma_median_us": float(sigmas[med_idx]),
        "sigma_lo_us": float(sigmas[lo_idx]),
        "sigma_hi_us": float(sigmas[hi_idx]),
        "confidence": float(confidence),
        "n_samples": int(n),
        "n_resamples": int(n_resamples),
    }


def sigma_cross_trace_table(
    pilots: dict[str, list[float]],
    p_high: float = 99.5,
    p_low: float = 50.0,
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Compute $\\sigma$ + 95\\% bootstrap CI for each named pilot trace.

    P3-1 fix: when the operator wants to know whether the
    Mooncake-calibrated $\\sigma$ transfers to a new workload, this
    helper runs :func:`sigma_bootstrap_ci` on each pilot and returns a
    row per trace ready for the cross-trace table. The non-overlapping
    CI test is the operator's signal that the bound needs re-calibration
    on the new workload.

    Parameters
    ----------
    pilots : dict[str, list[float]]
        Maps a label (workload name) to its IO-free residual sample.
    p_high, p_low, n_resamples, seed : forwarded to
        :func:`sigma_bootstrap_ci`.
    """
    rows = []
    for name, samples in pilots.items():
        try:
            ci = sigma_bootstrap_ci(
                samples,
                n_resamples=n_resamples,
                p_high=p_high,
                p_low=p_low,
                seed=seed,
            )
        except ValueError as e:
            ci = {"error": str(e), "n_samples": len(samples)}
        ci["trace"] = name
        rows.append(ci)
    return rows


def sigma_from_running_mean(mean_us: float, std_us: float,
                            n: int = 100) -> SigmaEstimate:
    """When the operator only logs running mean/std, just trust std.

    This is the cheapest possible estimator and the one we recommend
    when the operator does not have per-step latency arrays. Returns
    σ = std (the sample standard deviation as a sub-Gaussian proxy).
    """
    return SigmaEstimate(
        sigma_us=float(max(0.0, std_us)),
        n_used=int(n),
        p50_us=float("nan"),
        p95_us=float("nan"),
        p99_us=float("nan"),
        mean_us=float(mean_us),
        std_us=float(std_us),
        method="running_std",
    )


def _flatten_per_step(d, warmup_steps: int,
                      prefer_io_free: bool = True) -> tuple[list[float], str]:
    """Flatten per-step latencies from a runner JSON.

    The bound's σ_residual proxies the FFN+attn+LAP jitter only — the
    Bernoulli-IO variance is added analytically by Lemma 2 / 2$'$, so
    feeding it the *total* (per_step_us, which includes the IO penalty)
    double-counts. When the runner has emitted the (base, IO)
    decomposition, prefer ``per_step_base_us`` (IO-free residual);
    otherwise fall back to ``per_step_us`` (total) and flag it.

    Returns (samples, source_tag) where source_tag is one of
    ``"per_step_base_us"`` (preferred), ``"per_step_us"`` (legacy
    fallback, may double-count IO), or ``"samples_us"`` (raw pilot).
    """
    out: list[float] = []
    source = "per_step_us"
    if "results" in d and isinstance(d["results"], list):
        if prefer_io_free and any(
            r.get("per_step_base_us") for r in d["results"]
        ):
            source = "per_step_base_us"
            for r in d["results"]:
                steps = r.get("per_step_base_us", []) or []
                out.extend(steps[warmup_steps:])
        else:
            for r in d["results"]:
                steps = r.get("per_step_us", []) or []
                out.extend(steps[warmup_steps:])
    elif "samples_us" in d:
        out = list(d["samples_us"])[warmup_steps:]
        source = "samples_us"
    elif "per_step_us" in d:
        out = list(d["per_step_us"])[warmup_steps:]
        source = "per_step_us"
    return out, source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m seer.timing.sigma_estimate")
    ap.add_argument("source", help="Runner JSON (eA / eB style) or .csv pilot trace")
    ap.add_argument("--warmup_steps", type=int, default=1,
                    help="Drop this many leading per-request decode steps "
                         "(default 1: the JIT / CUDA Graph cold-start step).")
    ap.add_argument("--p_high", type=float, default=95.0,
                    help="Upper percentile for the (p_high - p_low) gap.")
    ap.add_argument("--p_low",  type=float, default=50.0)
    ap.add_argument("--out", default=None,
                    help="Write JSON to this path; otherwise stdout.")
    args = ap.parse_args(argv)

    src = Path(args.source)
    if src.suffix == ".csv":
        with open(src) as fh:
            xs = []
            for line in fh:
                line = line.strip()
                if not line or not line.replace(".", "", 1).replace("-", "", 1).isdigit():
                    continue
                xs.append(float(line))
    else:
        d = json.load(open(src))
        xs, source_tag = _flatten_per_step(d, args.warmup_steps)

    if len(xs) < 5:
        print(f"[sigma_est] need at least 5 samples post-warmup, got {len(xs)}",
              file=sys.stderr)
        return 2

    est = sigma_from_pilot(xs, p_high=args.p_high, p_low=args.p_low)
    payload = asdict(est)
    payload["source"] = str(src)
    payload["warmup_steps"] = args.warmup_steps
    if src.suffix != ".csv":
        payload["sample_source"] = source_tag
        if source_tag == "per_step_us":
            payload["caveat"] = (
                "σ derived from total per-step latency, which contains "
                "Bernoulli-IO variance. Lemma 2 also adds that variance "
                "analytically — there is a potential double-count. Re-run "
                "with the per_step_base_us decomposition for a clean σ."
            )
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    print(f"\n[sigma_est] σ ≈ {est.sigma_us/1000:.3f} ms  (n={est.n_used}, "
          f"P50={est.p50_us/1000:.2f} ms, P95={est.p95_us/1000:.2f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
