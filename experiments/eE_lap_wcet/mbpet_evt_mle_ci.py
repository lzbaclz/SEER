"""GPD MLE + parametric-bootstrap CI on raw LAP MBPET tail samples
(R4-W2 follow-through).

The quantile-LS fit in `mbpet_evt_fit.py` gives a point estimate of
the GPD shape parameter $\\xi$ on 5 tail order statistics. RTSS / MBPTA
reviewers asked for a CI on $\\xi$ from raw threshold-exceedance
samples (Cucu-Grosjean 2012, Cazorla 2013). This script:

1. Loads raw $n{\\approx}3{,}000$-rep tail samples from the
   `*_raw_samples.json` sidecar emitted by ``seer.lap.wcet``.
2. Filters to the threshold-exceedance set above the empirical
   P90 (Pickands--Balkema--de~Haan theorem).
3. Fits GPD by MLE using `scipy.stats.genpareto.fit`.
4. Bootstraps the threshold-exceedance set $1{,}000\\times$ and
   re-fits to give a $95\\%$ profile-style CI on $\\xi$ and on
   the extrapolated $P_{0.9999}$.

Output:
- `mbpet_evt_mle_ci.json`: $\\xi_{\\mathrm{MLE}}$ + 95% CI + extrapolated
  quantile CI per available raw-samples file.
- `mbpet_evt_mle_ci.tex`: paper supplementary table.

This script is non-blocking: if no raw-samples file is found in
`results/`, it prints an instructive message and exits 0 (so the
reproduce-smoke pipeline does not fail when the operator hasn't
re-run the WCET measurement with ``--samples_out``).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def _gpd_mle_fit(excesses: np.ndarray) -> tuple[float, float]:
    """Fit GPD to the threshold-exceedance set ``excesses`` (each
    sample is $x - u$ for $x > u$). Returns (xi, beta)."""
    from scipy.stats import genpareto
    # scipy parameterises GPD as (c, loc, scale) where c = xi. We fix
    # loc=0 (the threshold subtraction is done upstream).
    xi, _, beta = genpareto.fit(excesses, floc=0.0)
    return float(xi), float(beta)


def _gpd_quantile(xi: float, beta: float, threshold: float,
                   p_excess: float) -> float:
    """Inverse GPD CDF at conditional excess probability p_excess."""
    if abs(xi) < 1e-8:
        return threshold - beta * math.log(1.0 - p_excess)
    return threshold + beta * ((1.0 - p_excess) ** (-xi) - 1.0) / xi


def _bootstrap_ci(samples_us: np.ndarray, threshold_q: float,
                   n_boot: int = 1000, seed: int = 1) -> dict:
    """Parametric bootstrap: resample the raw samples with
    replacement, re-fit GPD on the threshold-exceedance set, and
    report 95% CI on (xi, beta, P_0.9999). Heavy-tailed fits where
    scipy fails to converge are filtered out and reported."""
    rng = np.random.default_rng(seed)
    threshold = float(np.quantile(samples_us, threshold_q))
    n = len(samples_us)
    xis: list[float] = []
    betas: list[float] = []
    p9999s: list[float] = []
    n_fails = 0
    # Conditional excess probability for absolute P0.9999 above the
    # empirical P(threshold_q):
    p_excess_target = (0.9999 - threshold_q) / (1.0 - threshold_q)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = samples_us[idx]
        excesses = boot[boot > threshold] - threshold
        if len(excesses) < 30:
            n_fails += 1
            continue
        try:
            xi_b, beta_b = _gpd_mle_fit(excesses)
        except Exception:
            n_fails += 1
            continue
        if not math.isfinite(xi_b) or not math.isfinite(beta_b):
            n_fails += 1
            continue
        xis.append(xi_b)
        betas.append(beta_b)
        try:
            p9999s.append(_gpd_quantile(xi_b, beta_b, threshold,
                                          p_excess_target))
        except Exception:
            pass
    return {
        "n_boot_succeeded": len(xis),
        "n_boot_failed": n_fails,
        "xi_mean": float(np.mean(xis)) if xis else float("nan"),
        "xi_ci95_lo": float(np.quantile(xis, 0.025)) if xis else float("nan"),
        "xi_ci95_hi": float(np.quantile(xis, 0.975)) if xis else float("nan"),
        "beta_mean": float(np.mean(betas)) if betas else float("nan"),
        "beta_ci95_lo": float(np.quantile(betas, 0.025)) if betas else float("nan"),
        "beta_ci95_hi": float(np.quantile(betas, 0.975)) if betas else float("nan"),
        "p9999_mean": float(np.mean(p9999s)) if p9999s else float("nan"),
        "p9999_ci95_lo": float(np.quantile(p9999s, 0.025)) if p9999s else float("nan"),
        "p9999_ci95_hi": float(np.quantile(p9999s, 0.975)) if p9999s else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="experiments/eE_lap_wcet/results")
    ap.add_argument("--out_dir",
                    default="experiments/eE_lap_wcet/results")
    ap.add_argument("--threshold_q", type=float, default=0.90,
                    help="Threshold quantile for GPD fit "
                         "(Pickands--Balkema--de~Haan).")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    rows = []
    candidates = sorted(glob.glob(f"{args.in_dir}/*raw_samples*.json"))
    if not candidates:
        print(f"[mbpet-evt-mle] no *_raw_samples*.json in {args.in_dir}; "
              f"re-run `python -m seer.lap.wcet ... --samples_out <path>` "
              f"on the target device to produce raw samples, then re-run "
              f"this script.")
        return 0

    for path in candidates:
        with open(path) as fh:
            d = json.load(fh)
        samples_us = np.asarray(d.get("samples_us", []), dtype=float)
        if len(samples_us) < 500:
            print(f"[mbpet-evt-mle] skip {path}: only {len(samples_us)} samples")
            continue
        threshold = float(np.quantile(samples_us, args.threshold_q))
        excesses = samples_us[samples_us > threshold] - threshold
        if len(excesses) < 30:
            print(f"[mbpet-evt-mle] skip {path}: only {len(excesses)} excess samples")
            continue
        xi_mle, beta_mle = _gpd_mle_fit(excesses)
        ci = _bootstrap_ci(samples_us, args.threshold_q,
                            n_boot=args.n_boot)
        device = d.get("device", "unknown")
        backend = d.get("backend", "unknown")
        # Empirical max as cross-check on the GPD extrapolation.
        empirical_max = float(samples_us.max())
        empirical_p999 = float(np.quantile(samples_us, 0.999))
        rows.append({
            "path": path,
            "device": device,
            "backend": backend,
            "n_samples": int(len(samples_us)),
            "n_excess": int(len(excesses)),
            "threshold_q": args.threshold_q,
            "threshold_us": threshold,
            "empirical_p999_us": empirical_p999,
            "empirical_max_us": empirical_max,
            "xi_mle": xi_mle,
            "beta_mle_us": beta_mle,
            "p9999_mle_us": _gpd_quantile(xi_mle, beta_mle, threshold,
                                            (0.9999 - args.threshold_q) /
                                            (1.0 - args.threshold_q)),
            "bootstrap_ci": ci,
        })
        ci_xi_str = (f"[{ci['xi_ci95_lo']:+.3f}, {ci['xi_ci95_hi']:+.3f}]"
                     if math.isfinite(ci["xi_ci95_lo"]) else "n/a")
        ci_p9999 = (f"[{ci['p9999_ci95_lo']:.1f}, {ci['p9999_ci95_hi']:.1f}]"
                    if math.isfinite(ci["p9999_ci95_lo"]) else "n/a")
        print(f"[mbpet-evt-mle] {device:14s} {backend:6s} "
              f"n={len(samples_us)} n_exc={len(excesses)}: "
              f"xi_MLE={xi_mle:+.3f} 95%CI={ci_xi_str}, "
              f"beta={beta_mle:.2f}us, "
              f"P9999_MLE={rows[-1]['p9999_mle_us']:.1f}us 95%CI={ci_p9999}, "
              f"empirical_max={empirical_max:.1f}us")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": (
            "GPD MLE + parametric bootstrap (n_boot=1000) on raw "
            "threshold-exceedance samples above the empirical P90 "
            "(Pickands--Balkema--de~Haan; Cucu-Grosjean 2012, "
            "Cazorla 2013). The xi_MLE 95% CI is the standard MBPTA "
            "tail-shape uncertainty quantification."
        ),
        "rows": rows,
    }
    out_json = out_dir / "mbpet_evt_mle_ci.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[mbpet-evt-mle] wrote {out_json}")

    # Supplementary TeX table.
    lines = [
        "% Generated by experiments/eE_lap_wcet/mbpet_evt_mle_ci.py (R4-W2).",
        "% GPD MLE + 95% bootstrap CI on raw LAP MBPET tail samples.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Device / backend & $n$ & $\xi_\mathrm{MLE}$ & $95\%$ CI on $\xi$ & "
        r"$P_{0.9999}^\mathrm{MLE}$ ($\mu$s) & empirical max ($\mu$s) \\",
        r"\midrule",
    ]
    for r in rows:
        ci = r["bootstrap_ci"]
        ci_str = (f"[{ci['xi_ci95_lo']:+.2f}, {ci['xi_ci95_hi']:+.2f}]"
                  if math.isfinite(ci.get("xi_ci95_lo", float("nan")))
                  else "n/a")
        device_tag = (r["device"] + " " + r["backend"]).replace("_", r"\_")
        lines.append(
            f"{device_tag} & ${r['n_samples']}$ & "
            f"${r['xi_mle']:+.2f}$ & ${ci_str}$ & "
            f"${r['p9999_mle_us']:.1f}$ & ${r['empirical_max_us']:.1f}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_tex = out_dir / "mbpet_evt_mle_ci.tex"
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[mbpet-evt-mle] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
