"""EVT / Generalized Pareto Distribution fit on LAP MBPET tail
(R3-W5 follow-through).

MBPTA standard (Cucu-Grosjean et al.\\ 2012, Cazorla et al.\\ 2013) fits
a Generalized Pareto Distribution (GPD) to the tail of the
forward-pass latency distribution and reports the shape parameter
$\\xi$:

  $\\xi \\to 0$  : exponential tail (well-behaved, MBPET converges fast)
  $\\xi > 0$    : heavy tail (Pareto-type; rare extreme events)
  $\\xi < 0$    : bounded support (truncated)

We fit a GPD to the LAP latency tail using the existing
$P_{50}, P_{90}, P_{99}, P_{99.9}, \\max$ quantiles in
`results/lap_wcet_*_tiny_mlp_*.json` (raw samples were not stored
post-hoc, so we treat the four upper quantiles + max as five tail
order statistics and solve the GPD CDF for $\\xi$ and $\\beta$ via
least-squares).

Output:
- `mbpet_evt_fit_summary.json`: per-SKU $\\xi, \\beta, P^{1{-}10^{-4}}$
  extrapolation.
- `mbpet_evt_fit.tex`: paper table.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _gpd_cdf(x: float, xi: float, beta: float, threshold: float) -> float:
    """GPD CDF at value $x$ above the threshold $u$. Returns
    $\\Pr(X \\le x \\mid X > u)$."""
    if x <= threshold:
        return 0.0
    z = (x - threshold) / beta
    if abs(xi) < 1e-8:
        return 1.0 - math.exp(-z)
    base = 1.0 + xi * z
    if base <= 0:
        return 1.0
    return 1.0 - base ** (-1.0 / xi)


def _quantile_at(xi: float, beta: float, threshold: float,
                  p: float) -> float:
    """Inverse: x such that GPD CDF = p (given excess above u)."""
    if p <= 0.0:
        return threshold
    if p >= 1.0:
        return float("inf")
    if abs(xi) < 1e-8:
        return threshold - beta * math.log(1.0 - p)
    return threshold + beta * ((1.0 - p) ** (-xi) - 1.0) / xi


def _fit_gpd_from_quantiles(quantiles: list[tuple[float, float]],
                              threshold: float) -> tuple[float, float, float]:
    """Fit GPD to the upper tail starting at ``threshold``. Each
    quantile is (p, x) where $\\Pr(X \\le x) = p$. We minimise the
    sum of squared residuals between predicted and measured
    quantiles. Returns (xi, beta, fit_resid)."""
    # Convert absolute probabilities to conditional-on-exceedance.
    # If the quantile p > Pr(X > threshold), the conditional excess
    # probability is (p - F(u)) / (1 - F(u)); when F(u) is unknown
    # we approximate F(u) by the lowest p (threshold percentile).
    if not quantiles:
        return 0.0, 0.0, float("inf")
    p_thr = min(p for p, _ in quantiles)
    F_u = p_thr
    excess = []
    for p, x in quantiles:
        if p > p_thr and x > threshold:
            p_excess = (p - F_u) / (1.0 - F_u)
            excess.append((p_excess, x - threshold))
    if len(excess) < 2:
        return 0.0, 0.0, float("inf")

    def residuals(params):
        xi, log_beta = params
        beta = math.exp(log_beta)
        out = []
        for p_ex, y in excess:
            try:
                pred_y = _quantile_at(xi, beta, threshold=0.0, p=p_ex)
            except Exception:
                pred_y = 0.0
            out.append((pred_y - y) / max(y, 1.0))
        return out

    # Initial guess: xi=0 (exponential), beta = mean excess.
    mean_excess = sum(y for _, y in excess) / len(excess)
    x0 = (0.0, math.log(max(1e-3, mean_excess)))
    try:
        sol = least_squares(residuals, x0, bounds=((-1.0, -10.0),
                                                    (5.0, 10.0)),
                             max_nfev=200)
        xi, log_beta = sol.x
        return float(xi), float(math.exp(log_beta)), float(np.sum(sol.fun ** 2))
    except Exception:
        return 0.0, mean_excess, float("inf")


def _extrapolate(xi: float, beta: float, threshold: float,
                  p_target: float, p_threshold: float) -> float:
    r"""Extrapolate to a tighter quantile. ``p_target`` is the
    \emph{absolute} target probability; we convert to conditional
    given exceedance of threshold."""
    if p_target <= p_threshold:
        return float("nan")
    p_excess = (p_target - p_threshold) / (1.0 - p_threshold)
    return threshold + _quantile_at(xi, beta, threshold=0.0,
                                     p=p_excess)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="experiments/eE_lap_wcet/results")
    ap.add_argument("--out_dir",
                    default="experiments/eE_lap_wcet/results")
    args = ap.parse_args()

    # Restrict to TinyMLP TRT b=4096 (the production setting).
    candidates = []
    for path in sorted(glob.glob(f"{args.in_dir}/lap_wcet_*tiny_mlp*trt_b4096.json")):
        with open(path) as fh:
            d = json.load(fh)
        if "card1" in path or "unlocked" in path:
            continue  # ablation runs, not headline
        candidates.append((path, d))

    # Group by (device, backend); prefer TRT > ONNX > torch.
    selected: dict[str, tuple[str, dict]] = {}
    rank = {"trt": 0, "trt_fp16": 0, "onnx": 1, "torch": 2}
    for path, d in candidates:
        dev = d.get("device", "cuda")
        # Try to extract a device key from filename.
        # lap_wcet_A100_tiny_mlp_*  or NVIDIA_H100_80GB_HBM3_tiny_mlp_*
        stem = Path(path).stem
        if "A100" in stem:
            dev_key = "A100"
        elif "H100" in stem:
            dev_key = "H100"
        elif "L40S" in stem:
            dev_key = "L40S"
        else:
            dev_key = dev
        backend = d.get("backend", "torch")
        if dev_key in selected:
            cur_rank = rank.get(selected[dev_key][1].get("backend", "torch"),
                                 99)
            new_rank = rank.get(backend, 99)
            if new_rank >= cur_rank:
                continue
        selected[dev_key] = (path, d)

    rows = []
    for dev_key in ("A100", "H100", "L40S"):
        if dev_key not in selected:
            print(f"[mbpet-evt] missing {dev_key}, skipping")
            continue
        path, d = selected[dev_key]
        p50 = float(d["p50_us"])
        p90 = float(d["p90_us"])
        p99 = float(d["p99_us"])
        p999 = float(d["p999_us"])
        pmax = float(d["max_us"])
        # Tail threshold at P90. We drop the single-sample `max`
        # because it is an empirical realisation, not a distribution
        # quantile; including it destabilises the LS fit (one
        # outlier dominates). The fitted GPD then *extrapolates* to
        # P99.99 / P99.999 and we report `max` separately for
        # cross-check.
        threshold = p90
        quantiles = [(0.50, p50), (0.90, p90), (0.99, p99),
                     (0.999, p999)]
        xi, beta, resid = _fit_gpd_from_quantiles(quantiles, threshold)
        # Extrapolated quantile at P99.99.
        p9999 = _extrapolate(xi, beta, threshold, 0.9999, 0.90)
        p99999 = _extrapolate(xi, beta, threshold, 0.99999, 0.90)
        rows.append({
            "device": dev_key,
            "backend": d.get("backend", "torch"),
            "batch_size": int(d.get("batch_size", 0)),
            "P50_us": p50, "P90_us": p90, "P99_us": p99,
            "P999_us": p999, "Pmax_us": pmax,
            "threshold_us": threshold,
            "gpd_xi": xi,
            "gpd_beta_us": beta,
            "fit_residual_norm_sq": resid,
            "P9999_extrap_us": p9999,
            "P99999_extrap_us": p99999,
            "headroom_at_P9999_x": 200.0 / max(p9999, 1e-3),
        })
        print(f"[mbpet-evt] {dev_key:5s} {d.get('backend',''):4s}: "
              f"xi={xi:+.3f}, beta={beta:.2f}us, "
              f"P9999={p9999:.1f}us, P99999={p99999:.1f}us, "
              f"200us-headroom@P9999={200/max(p9999,1e-3):.1f}x")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": (
            "GPD fit to LAP MBPET tail (threshold=P90) using "
            "{P50, P90, P99, P99.9, max} as tail order statistics. "
            "xi>0 = heavy tail; xi~0 = exponential; xi<0 = bounded."
        ),
        "rows": rows,
    }
    out_json = out_dir / "mbpet_evt_fit_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[mbpet-evt] wrote {out_json}")

    lines = [
        "% Generated by experiments/eE_lap_wcet/mbpet_evt_fit.py",
        "% GPD fit on the LAP MBPET tail (R3-W5 follow-through).",
        "% xi (shape) ~ 0 => exponential tail; xi > 0 => heavy tail.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"SKU & GPD $\xi$ & GPD $\beta$ ($\mu$s) & "
        r"$P_{0.9999}$ ($\mu$s) & headroom @ $P_{0.9999}$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['device']} & "
            f"${r['gpd_xi']:+.3f}$ & "
            f"${r['gpd_beta_us']:.1f}$ & "
            f"${r['P9999_extrap_us']:.1f}$ & "
            f"${r['headroom_at_P9999_x']:.1f}\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_tex = out_dir / "mbpet_evt_fit.tex"
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[mbpet-evt] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
