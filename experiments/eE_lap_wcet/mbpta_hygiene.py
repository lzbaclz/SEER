"""MBPTA hygiene tests on persisted MBPET raw samples (R7-UK-W2).

The UK-RT review correctly observes that calling the MBPET protocol
"MBPET-bounded" requires demonstrating (i) i.i.d.-or-weak-dependence
on the timing sample (Ljung-Box / runs test), (ii) representativity
across operational state (out of scope here -- only quiesced GPU
is persisted), and (iii) extreme-value model selection with a
goodness-of-fit check (Anderson-Darling). This script runs (i) and
(iii) on the existing TRT and torch raw samples; (ii) remains a
disclosure in §6 / §8 ("measurement-based tail-envelope, MBPTA
hygiene partial").

Outputs `mbpta_hygiene.json` + a one-line TeX summary used by §6.

Reference: Cucu-Grosjean et al. RTSS 2012; Cazorla et al. TECS 2013.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

try:
    from scipy import stats
    SCIPY = True
except ImportError:
    SCIPY = False


ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eE_lap_wcet/results"


def load_raw(path: pathlib.Path) -> np.ndarray | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    # different schemas across files; pick the most common keys.
    for k in ("raw_samples_us", "timings_us", "samples_us", "raw"):
        if k in d:
            return np.asarray(d[k], dtype=float)
    # fallback: look for any flat numeric array
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return np.asarray(v, dtype=float)
    return None


def runs_test(x: np.ndarray) -> dict:
    """Wald-Wolfowitz runs test for randomness around the median."""
    med = float(np.median(x))
    signs = (x > med).astype(int)
    # collapse to runs
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    n1 = int(signs.sum())
    n0 = int(len(signs) - n1)
    n = n0 + n1
    if min(n0, n1) < 1:
        return {"runs": runs, "p_value": float("nan"), "n0": n0, "n1": n1}
    mu = 2.0 * n0 * n1 / n + 1.0
    var = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / (n ** 2 * (n - 1.0))
    z = (runs - mu) / math.sqrt(max(var, 1e-12))
    # two-sided
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"runs": runs, "z": z, "p_value": p, "n0": n0, "n1": n1, "mu": mu}


def ljung_box(x: np.ndarray, lags: int = 10) -> dict:
    """Ljung-Box Q statistic at given lag count. Approximate chi-square p-value."""
    n = len(x)
    xc = x - x.mean()
    var0 = (xc ** 2).sum() / n
    if var0 <= 0:
        return {"Q": float("nan"), "p_value": float("nan"), "lags": lags}
    Q = 0.0
    for k in range(1, lags + 1):
        rk = (xc[k:] * xc[:-k]).sum() / n / var0
        Q += (rk ** 2) / max(n - k, 1)
    Q *= n * (n + 2)
    # asymptotic chi-square(lags) p-value
    if SCIPY:
        p = float(1.0 - stats.chi2.cdf(Q, df=lags))
    else:
        # crude wilson-hilferty approximation
        m = lags
        h = 2.0 / (9.0 * m)
        z = ((Q / m) ** (1.0 / 3.0) - (1.0 - h)) / math.sqrt(h)
        p = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return {"Q": Q, "p_value": p, "lags": lags}


def _gpd_cdf(x: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """GPD CDF with loc=0, scale=beta, shape=xi. Numpy-only."""
    if abs(xi) < 1e-10:
        return 1.0 - np.exp(-x / beta)
    z = 1.0 + xi * x / beta
    z = np.clip(z, 1e-300, None)
    return 1.0 - z ** (-1.0 / xi)


def _gpd_pwm_fit(excess: np.ndarray) -> tuple[float, float]:
    """Probability-weighted-moments (Hosking 1987) GPD fit, loc=0.

    Numerically stable on small samples and bias-bounded for $\\xi$
    in $[-0.5, 0.5]$, which covers our $\\xi \\approx 0.33$ regime.
    """
    n = len(excess)
    y = np.sort(excess)
    a0 = y.mean()
    # b1 = E[Y (1 - F(Y))] estimator using order statistics
    weights = (n - np.arange(1, n + 1)) / (n * (n - 1)) if n > 1 else np.zeros(n)
    a1 = float(np.sum(weights * y))
    if abs(a0 - 2.0 * a1) < 1e-12:
        # degenerate (exponential limit); fall back to MoM
        xi = 0.0
        beta = a0
    else:
        xi = 2.0 - a0 / (a0 - 2.0 * a1)
        beta = 2.0 * a0 * a1 / (a0 - 2.0 * a1)
    if beta <= 0:
        # constrain to feasible region with method of moments
        beta = max(a0, 1e-9)
        xi = 0.0
    return float(xi), float(beta)


def anderson_darling_gpd_threshold(
        x: np.ndarray, quantile_thr: float = 0.95,
        min_excess: int = 30) -> dict:
    """Fit GPD on the upper-tail threshold-exceedances and return AD GoF.

    SciPy provides AD only for a small family of distributions, none of
    which is GPD. We compute the AD statistic by hand against the fitted
    GPD CDF on the threshold-excess sample. When SciPy is available we
    use its MLE fit; otherwise we fall back to a numpy-only PWM fit so
    the GoF check is reportable in any environment.
    """
    thr = float(np.quantile(x, quantile_thr))
    excess = x[x > thr] - thr
    n = len(excess)
    if n < min_excess:
        return {"A2": float("nan"), "A2_adj": float("nan"),
                "n_excess": n, "fit": None,
                "fit_pass_5pct": False,
                "note": f"insufficient excesses (n={n}<{min_excess})"}
    fit_method = "MLE" if SCIPY else "PWM"
    if SCIPY:
        try:
            xi, _, beta = stats.genpareto.fit(excess, floc=0.0)
        except Exception as exc:
            return {"A2": float("nan"), "n_excess": n,
                    "fit": None, "note": f"fit failed: {exc}"}
    else:
        xi, beta = _gpd_pwm_fit(excess)
    F = _gpd_cdf(np.sort(excess), xi=xi, beta=beta)
    F = np.clip(F, 1e-12, 1.0 - 1e-12)
    i = np.arange(1, n + 1)
    A2 = -n - (1.0 / n) * np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    # n-correction (Stephens 1986 form) and EVT-literature threshold.
    A2_adj = float(A2 * (1.0 + 0.6 / n))
    return {"A2": float(A2), "A2_adj": A2_adj,
            "n_excess": n, "xi": float(xi), "beta": float(beta),
            "fit_method": fit_method,
            "fit_pass_5pct": bool(A2_adj <= 2.492),
            "threshold_us": thr}


def analyse(path: pathlib.Path) -> dict | None:
    x = load_raw(path)
    if x is None:
        return None
    out = {"file": str(path.name), "n": int(len(x)),
           "p50_us": float(np.median(x)),
           "p99_us": float(np.quantile(x, 0.99)),
           "p999_us": float(np.quantile(x, 0.999))}
    out["runs"] = runs_test(x)
    out["ljung_box_lag10"] = ljung_box(x, lags=10)
    out["anderson_darling_gpd"] = anderson_darling_gpd_threshold(x, 0.95)
    return out


def main() -> int:
    candidates = [
        RESULTS / "lap_wcet_A100_local_tiny_mlp_trt_b4096_raw_samples.json",
        RESULTS / "lap_wcet_A100_local_tiny_mlp_torch_b4096_raw_samples.json",
    ]
    results = []
    for p in candidates:
        r = analyse(p)
        if r is not None:
            results.append(r)
            print(f"[mbpta-hygiene] {p.name}")
            ad = r["anderson_darling_gpd"]
            a2 = ad.get("A2", float("nan"))
            print(f"  n={r['n']} runs p={r['runs']['p_value']:.3f}  "
                  f"LB(10) p={r['ljung_box_lag10']['p_value']:.3f}  "
                  f"AD(GPD@0.95) A2={a2:.2f}, "
                  f"pass={ad.get('fit_pass_5pct', '?')}")
        else:
            print(f"[mbpta-hygiene] SKIP {p.name} (missing or unreadable)")

    out = {"results": results,
           "interpretation": {
               "runs_test": "Wald-Wolfowitz runs around median; p>0.05 = "
                            "fail to reject randomness (cannot conclude i.i.d. "
                            "violation).",
               "ljung_box_lag10":
                   "Ljung-Box at lag 10. p>0.05 = no significant linear "
                   "autocorrelation. MBPTA prerequisite (i).",
               "anderson_darling_gpd":
                   "AD goodness-of-fit on POT excesses above the 0.95 "
                   "quantile. A2_adj <= 2.492 at 5% accepts GPD. MBPTA "
                   "prerequisite (iii)."}}
    out_path = RESULTS / "mbpta_hygiene.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[mbpta-hygiene] wrote {out_path}")

    # TeX summary line (one paragraph for §6).
    summary_lines = ["% Auto-generated by mbpta_hygiene.py (R7-UK-W2)."]
    for r in results:
        backend = "TRT" if "trt" in r["file"] else "torch"
        rp = r["runs"]["p_value"]
        lp = r["ljung_box_lag10"]["p_value"]
        ad = r["anderson_darling_gpd"]
        a2 = ad.get("A2_adj", float("nan"))
        pass_ad = ad.get("fit_pass_5pct", False)
        summary_lines.append(
            f"\\newcommand{{\\mbpta{backend}runs}}{{${rp:.3f}$}}"
        )
        summary_lines.append(
            f"\\newcommand{{\\mbpta{backend}lb}}{{${lp:.3f}$}}"
        )
        summary_lines.append(
            f"\\newcommand{{\\mbpta{backend}ad}}{{${a2:.2f}$}}"
        )
        summary_lines.append(
            f"\\newcommand{{\\mbpta{backend}adpass}}{{{'yes' if pass_ad else 'no'}}}"
        )
    (RESULTS / "mbpta_hygiene_macros.tex").write_text(
        "\n".join(summary_lines) + "\n")
    print(f"[mbpta-hygiene] wrote {RESULTS / 'mbpta_hygiene_macros.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
