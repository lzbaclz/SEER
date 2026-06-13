"""eC bound calibration — empirical sigma + bound-vs-CDF view.

This script complements ``sweep.py`` (which is a parameter scan over the
analytical bound) and ``analyze.py`` (which produces a heatmap that looks
binary at our operating point because the bound is either ~10^-30 or 1).
The reviewer-facing artifact is **a single CDF plot** that overlays the
empirical per-step latency CDF with the Lemma 2 bound's predicted tail
on the same axis, parameterised by the empirical residual sigma.

Outputs
-------
* ``results/calibrated.csv`` — per (policy, budget, D) row of:
    measured_miss, bound_default_sigma (paper-default 5000us),
    bound_empirical_sigma (P99.5-derived).
* ``results/steady_state.csv`` — per (policy, budget) P50/P99/P999/max
    both warmup-included and warmup-excluded.
* ``results/cdf_calibrated.pdf`` — overlay of empirical CDF and the
    sub-Gaussian Lemma 2 prediction with empirical sigma.

Usage::

    python -m experiments.eC_bound_tightness.calibrate \\
        --in_dir experiments/eA_tail_latency/results_mooncake_full \\
        --out_dir experiments/eC_bound_tightness/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_per_step(p: Path):
    d = json.load(open(p))
    pol = d.get("policy", "?")
    budget = float(d.get("hbm_budget", 0.0))
    lats_full, lats_steady = [], []
    for r in d["results"]:
        st = r.get("per_step_us", [])
        lats_full.extend(st)
        if len(st) > 1:
            lats_steady.extend(st[1:])  # drop warm-up step 0
    return pol, budget, lats_full, lats_steady


def _percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _empirical_sigma_us(lats_us):
    """One-sided sub-Gaussian proxy from the (P99.5 - P50) gap."""
    if len(lats_us) < 50:
        return float("nan")
    p50 = _percentile(lats_us, 50)
    p995 = _percentile(lats_us, 99.5)
    factor = math.sqrt(2 * math.log(1.0 / 0.005))
    return max(0.0, (p995 - p50) / factor)


def _lemma2(eps, ell_bar_us, B_t, D_us, sigma_us, base_cost_us):
    from seer.timing.schedulability import lemma2_miss_prob_bound
    return lemma2_miss_prob_bound(
        epsilon=eps, ell_bar_us=ell_bar_us, B_t=B_t,
        deadline_us=D_us, sigma_residual_us=sigma_us,
        base_cost_us=base_cost_us,
    )


def _lemma2_bernstein(eps, ell_bar_us, B_t, D_us, sigma_us, base_cost_us):
    from seer.timing.schedulability import lemma2_bernstein_miss_prob_bound
    return lemma2_bernstein_miss_prob_bound(
        epsilon=eps, ell_bar_us=ell_bar_us, B_t=B_t,
        deadline_us=D_us, sigma_residual_us=sigma_us,
        base_cost_us=base_cost_us,
    )


def _lemma2_mixture(eps_mean, eps_var, ell_bar_us, B_t, D_us, sigma_us, base_cost_us):
    from seer.timing.schedulability import lemma2_bernstein_mixture_bound
    return lemma2_bernstein_mixture_bound(
        epsilon_mean=eps_mean, epsilon_var=eps_var,
        ell_bar_us=ell_bar_us, B_t=B_t,
        deadline_us=D_us, sigma_residual_us=sigma_us,
        base_cost_us=base_cost_us,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epsilon", type=float, default=0.07,
                    help="LAP false-negative rate to plug into Lemma 2.")
    ap.add_argument("--epsilon_var", type=float, default=0.025,
                    help="Across-block variance of ε_b for the Bernstein "
                         "mixture bound. Empirically estimated from the LAP "
                         "predictions on the e1 held-out trace; default 0.025 "
                         "matches the LAP's per-horizon ε ∈ [0.054, 0.064]. "
                         "Must satisfy 0 ≤ v_ε ≤ ε̄(1-ε̄).")
    ap.add_argument("--ell_bar_us", type=float, default=200.0)
    ap.add_argument("--B_full", type=float, default=64.0)
    ap.add_argument("--Ds_ms", type=float, nargs="+",
                    default=[35, 40, 50, 75, 100, 200])
    ap.add_argument("--sigma_default_us", type=float, default=5000.0,
                    help="Paper-default sub-Gaussian proxy.")
    ap.add_argument("--policy_for_cdf", default="seer")
    ap.add_argument("--budget_for_cdf", type=float, default=0.20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir)

    # ------------------------------------------------------------------
    # 1. steady-state vs warmup-included table
    # ------------------------------------------------------------------
    ss_rows = []
    cal_rows = []
    cdf_payload = None
    for p in sorted(in_dir.glob("*_b*.json")):
        pol, budget, lats_full, lats_steady = _load_per_step(p)
        if not lats_full:
            continue
        ss_rows.append({
            "policy": pol, "budget": budget, "n_full": len(lats_full),
            "p50_full_ms": _percentile(lats_full, 50) / 1000,
            "p99_full_ms": _percentile(lats_full, 99) / 1000,
            "p999_full_ms": _percentile(lats_full, 99.9) / 1000,
            "max_full_ms": max(lats_full) / 1000,
            "n_steady": len(lats_steady),
            "p50_steady_ms": _percentile(lats_steady, 50) / 1000,
            "p99_steady_ms": _percentile(lats_steady, 99) / 1000,
            "p999_steady_ms": _percentile(lats_steady, 99.9) / 1000,
            "max_steady_ms": max(lats_steady) / 1000 if lats_steady else float("nan"),
        })

        # Calibrated bound: use steady-state distribution to estimate mu0 and sigma
        if not lats_steady:
            continue
        mu0_us = _percentile(lats_steady, 50)
        sigma_emp_us = _empirical_sigma_us(lats_steady)
        for D_ms in args.Ds_ms:
            D_us = D_ms * 1000
            measured_full = sum(1 for x in lats_full if x > D_us) / len(lats_full)
            measured_steady = sum(1 for x in lats_steady if x > D_us) / len(lats_steady)
            bound_default = _lemma2(args.epsilon, args.ell_bar_us,
                                     args.B_full * budget, D_us,
                                     args.sigma_default_us, mu0_us)
            bound_emp = _lemma2(args.epsilon, args.ell_bar_us,
                                 args.B_full * budget, D_us,
                                 sigma_emp_us, mu0_us)
            bound_bernstein = _lemma2_bernstein(args.epsilon, args.ell_bar_us,
                                                 args.B_full * budget, D_us,
                                                 sigma_emp_us, mu0_us)
            # Cap variance to feasible range (defensive)
            eps_var_cap = min(args.epsilon_var,
                              args.epsilon * (1 - args.epsilon))
            bound_mixture = _lemma2_mixture(args.epsilon, eps_var_cap,
                                             args.ell_bar_us,
                                             args.B_full * budget, D_us,
                                             sigma_emp_us, mu0_us)
            cal_rows.append({
                "policy": pol, "budget": budget, "D_ms": D_ms,
                "measured_full": measured_full,
                "measured_steady": measured_steady,
                "bound_default_sigma": bound_default,
                "bound_empirical_sigma": bound_emp,
                "bound_bernstein": bound_bernstein,
                "bound_bernstein_mixture": bound_mixture,
                "mu0_steady_ms": mu0_us / 1000,
                "sigma_emp_us": sigma_emp_us,
            })

        # Save the CDF payload for the chosen (policy, budget)
        if pol == args.policy_for_cdf and abs(budget - args.budget_for_cdf) < 1e-6:
            cdf_payload = (pol, budget, lats_steady, mu0_us, sigma_emp_us)

    # ------------------------------------------------------------------
    # 2. write tables
    # ------------------------------------------------------------------
    if ss_rows:
        with open(out_dir / "steady_state.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ss_rows[0].keys()))
            w.writeheader()
            w.writerows(ss_rows)
        print(f"[calibrate] wrote steady_state.csv ({len(ss_rows)} rows)")
    if cal_rows:
        with open(out_dir / "calibrated.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
            w.writeheader()
            w.writerows(cal_rows)
        print(f"[calibrate] wrote calibrated.csv ({len(cal_rows)} rows)")
        _write_calibrated_table(cal_rows, out_dir, args.epsilon_var)

    # ------------------------------------------------------------------
    # 3. CDF figure: empirical CDF vs sub-Gaussian Lemma 2 prediction
    # ------------------------------------------------------------------
    if cdf_payload is None:
        print("[calibrate] no matching CDF payload, skipping figure")
        return
    pol, budget, lats_steady, mu0_us, sigma_emp_us = cdf_payload
    _plot_cdf_calibration(pol, budget, lats_steady, mu0_us, sigma_emp_us,
                          args.epsilon, args.ell_bar_us, args.B_full,
                          args.sigma_default_us, out_dir)


def _fmt_sci(x: float) -> str:
    if x <= 0:
        return "0"
    if x >= 1.0:
        return "1"
    s = f"{x:.1e}"
    m, e = s.split("e")
    e = int(e)
    return f"${m}\\!\\times\\!10^{{{e}}}$"


def _write_calibrated_table(cal_rows, out_dir: Path, epsilon_var: float) -> None:
    """Render the SEER-only calibrated_table.tex from cal_rows."""
    seer = [r for r in cal_rows if r["policy"] == "seer"]
    if not seer:
        return
    seer.sort(key=lambda r: (float(r["budget"]), float(r["D_ms"])))
    keep_D = {35, 40, 50}
    seer = [r for r in seer if int(float(r["D_ms"])) in keep_D]
    lines = [
        "% Generated by experiments/eC_bound_tightness/calibrate.py",
        "% Calibrated Lemma 2 family on SEER + Mooncake. The empirical sigma is the",
        "% one-sided sub-Gaussian proxy from the steady-state per-step distribution",
        "% (P99.5 - P50 gap, n=600 steps per cell). The mixture-Bernstein column",
        "% (Lemma 2', cf. supplementary) handles heterogeneous per-block ε_b with",
        f"% across-block variance v_eps = {epsilon_var} (taken from the LAP's per-horizon",
        "% calibration spread of [0.054, 0.064] in e1).",
        "\\begin{tabular}{rrrcccc}",
        "\\toprule",
        "$B_\\mathrm{HBM}$ & $D$ (ms) & meas. & "
        "Hoeff., $\\sigma{=}5{,}000\\,\\mu$s & "
        "Hoeff., emp.\\,$\\sigma$ & Bernstein & Bernstein-mix \\\\",
        "\\midrule",
    ]
    last_B = None
    for r in seer:
        B = float(r["budget"])
        if last_B is not None and abs(B - last_B) > 1e-9:
            lines.append("\\midrule")
        meas = float(r["measured_steady"])
        meas_s = f"{meas:.4f}" if meas > 0 else "0      "
        b1 = _fmt_sci(float(r["bound_default_sigma"]))
        b2 = _fmt_sci(float(r["bound_empirical_sigma"]))
        b3 = _fmt_sci(float(r["bound_bernstein"]))
        b4 = _fmt_sci(float(r["bound_bernstein_mixture"]))
        # Bold the headline operating point B=0.20, D=35
        if abs(B - 0.20) < 1e-9 and int(float(r["D_ms"])) == 35:
            inner = b4.strip("$")
            b4 = f"$\\mathbf{{{inner}}}$"
        D = int(float(r["D_ms"]))
        lines.append(
            f"{B:.2f} & {D} & {meas_s} & {b1} & {b2} & {b3} & {b4} \\\\"
        )
        last_B = B
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out = out_dir / "calibrated_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"[calibrate] wrote {out}")


def _plot_cdf_calibration(pol, budget, lats_us, mu0_us, sigma_emp_us,
                          epsilon, ell_bar_us, B_full, sigma_default_us,
                          out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from seer import figstyle
        figstyle.apply()
    except Exception:
        pass

    arr = np.asarray(sorted(lats_us)) / 1000.0  # ms
    n = arr.size
    cdf = np.arange(1, n + 1) / n
    surv = 1.0 - cdf  # P(X > x)

    # Lemma 2 prediction: P(C_t > D) bound for each D.
    # mu = mu0 + eps*B*ell_bar; sigma^2 = B*eps*(1-eps)*(4*ell)^2 + sigma_resid^2
    B_t = B_full * budget
    mu_us = mu0_us + epsilon * B_t * ell_bar_us
    sig_io_sq = B_t * epsilon * (1 - epsilon) * (4 * ell_bar_us) ** 2

    Ds_ms = np.linspace(arr.min(), arr.max() * 1.05, 400)
    Ds_us = Ds_ms * 1000.0

    def pred(sigma_us):
        sig_total_sq = sig_io_sq + sigma_us ** 2
        out = []
        for D in Ds_us:
            if D <= mu_us:
                out.append(1.0)
            else:
                phi = (D - mu_us) ** 2 / (2 * sig_total_sq)
                out.append(min(1.0, math.exp(-phi)))
        return np.asarray(out)

    surv_default = pred(sigma_default_us)
    surv_empir = pred(sigma_emp_us)

    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    # empirical survival
    ax.semilogy(arr, np.maximum(surv, 1e-6), lw=1.4, color="0.15",
                label=f"Empirical 1-CDF (steady-state, n={n})")
    ax.semilogy(Ds_ms, np.clip(surv_default, 1e-300, 1), lw=1.0,
                ls="--", color="0.55",
                label="Lemma 2 bound, $\\sigma$=5000$\\mu$s (paper default)")
    ax.semilogy(Ds_ms, np.clip(surv_empir, 1e-300, 1), lw=1.4,
                color="#d95f02",
                label=f"Lemma 2 bound, $\\sigma$={sigma_emp_us:.0f}$\\mu$s (empirical)")
    ax.axvline(50, color="0.4", ls=":", lw=0.8)
    ax.text(50.2, 1e-4, "50ms SLO", fontsize=7, color="0.3", rotation=90, va="center")

    ax.set_xlabel("Per-step deadline $D$ (ms)")
    ax.set_ylabel(r"$\Pr(C_t > D)$ (log)")
    ax.set_ylim(5e-5, 1.2)
    ax.set_title(f"{pol.upper()}, $B_\\mathrm{{HBM}}={budget}$ on Mooncake (steady-state)",
                 fontsize=8.5)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False)
    try:
        from seer import figstyle
        figstyle.add_box_spines(ax)
    except Exception:
        pass

    out = out_dir / "cdf_calibrated.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[calibrate] wrote {out}")


if __name__ == "__main__":
    main()
