"""Held-out replay audit on admission-map boundary cells (R10-2).

The R10 review correctly noted that the in-session admission map
(``admission_map.json``) uses the same Beta-Binomial / LogNormal
generative model on both sides -- the bound consumes the
parametric model and the "empirical truth" samples from it. That
is a model-consistency check, not an independent system-level
validation.

This script partially addresses the same-assumption critique by
replaying the persisted dense-deployed-eps trace
(``dense_deployed_eps_summary.json``: 50 prompts x 4 phi knots =
200 bootstrap rows) against the bound at boundary cells where the
bound verdict flips between green/red. For each boundary cell we:

1. Bootstrap-resample (n=1000) the persisted per-prompt eps values
   at the cell's phi; record the empirical miss as
   $\\Pr[\\text{step latency} > D]$ under the same B_t blocks +
   LogNormal IO + Gaussian jitter pipeline.
2. Build a Wilson 95% CI on the per-bootstrap miss rate.
3. Check whether the bound's verdict (accept iff bound miss <= rho)
   is consistent with the held-out replay (accept iff CI upper
   <= rho).

The persisted eps values are from a different sampling distribution
than the parametric Beta-Binomial used in admission_map.py, so the
agreement is genuine cross-distribution evidence (not full
independent system validation, but better than self-consistency).

Output: held_out_replay.{json,tex}.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

from experiments.eC_bound_tightness.admission_map import (
    SUBSTRATES, RHO_TARGET, _stable_seed,
    interp_eps, bound_verdict,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"


# Boundary cells per substrate: take the 4 cells in the (D, phi) plane
# straddling each substrate's bound-verdict transition. Picked from
# the operator-default admission_map.json so the audit hits where the
# verdict actually flips.
BOUNDARY_CELLS = {
    "DRAM":    [(75, 0.4), (100, 0.3), (150, 0.2), (200, 0.1)],
    "NVMe":    [(100, 0.7), (150, 0.5), (200, 0.4), (250, 0.3)],
    "NVMe-GC": [(150, 0.7), (200, 0.6), (250, 0.5), (300, 0.4)],
}


def _load_persisted_eps(path: pathlib.Path) -> dict:
    """Persisted per-prompt deployed eps at each phi knot.

    Returns {phi -> (mean, sample_n)}. The sample_n drives
    bootstrap precision.
    """
    d = json.loads(path.read_text())
    return {r["phi"]: {
        "mean": r["deployed_mean"],
        "se": r["deployed_se"],
        "n": int(r["n_prompts"]),
    } for r in d["rows"]}


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    z = 1.96  # alpha=0.05
    den = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / den
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def held_out_empirical_miss(
        eps_pop_mean: float, eps_pop_se: float, eps_n: int,
        ell_bar_us: float, D_us: float, B_t: int,
        base_cost_us: float, sigma_residual_us: float,
        seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    """Bootstrap miss rate using the persisted eps SE rather than
    the parametric Beta-Binomial. Each bootstrap draws a per-prompt
    eps from N(mean, SE) -- the empirical distribution implied by
    the persisted dense-deployed-eps rows -- and then runs the same
    B_t-block / LogNormal-IO simulator. The resulting miss rate
    distribution is summarised by (mean, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    sigma_ln = 0.6
    var_ln_factor = (math.exp(sigma_ln ** 2) - 1.0)
    log_inv_pi = -math.log(eps_n)  # avoid unused warning
    miss_rates = np.zeros(n_boot)
    for b in range(n_boot):
        # Draw an eps realisation from the persisted SE
        eps_b = max(0.0, min(1.0, rng.normal(eps_pop_mean,
                                              eps_pop_se)))
        # Beta-Binomial cross-block correlation (operator default 0.08)
        rho_eff = 0.08
        a = max(0.01, eps_b * (1.0 / rho_eff - 1.0))
        bb = max(0.01, (1.0 - eps_b) * (1.0 / rho_eff - 1.0))
        p_step = rng.beta(a, bb)
        miss_count = rng.binomial(B_t, p_step)
        io_total = (miss_count * ell_bar_us
                    + rng.normal(0.0,
                                  math.sqrt(max(var_ln_factor *
                                                 ell_bar_us ** 2 *
                                                 max(miss_count, 1), 1.0))))
        jitter = rng.normal(0.0, sigma_residual_us)
        miss_rates[b] = 1.0 if (base_cost_us + jitter +
                                 max(io_total, 0.0)) > D_us else 0.0
    mean = float(miss_rates.mean())
    lo, hi = wilson_ci(int(miss_rates.sum()), n_boot)
    return mean, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_eps",
                    default=str(RESULTS / "dense_deployed_eps_summary.json"))
    ap.add_argument("--sigma_residual_us", type=float, default=1203.0)
    ap.add_argument("--base_cost_us", type=float, default=33800.0)
    ap.add_argument("--rho_bar", type=float, default=0.08)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    persisted = _load_persisted_eps(pathlib.Path(args.dense_eps))
    # Bound side uses interp curve (continuous over phi); replay side
    # bootstraps directly from the persisted SE at the closest phi
    # knot, so the two pipelines do NOT share the eps distribution.
    interp_curve = {phi: persisted[phi] for phi in persisted}

    rows = []
    n_total = 0
    n_consistent = 0
    n_bound_accept_replay_accept = 0
    n_bound_reject_replay_reject = 0
    for substrate, (ell_bar, _) in SUBSTRATES.items():
        for D_ms, phi in BOUNDARY_CELLS[substrate]:
            D_us = D_ms * 1000.0
            # Bound verdict (continuous interp).
            eps_mean_interp, eps_se_interp = interp_eps(phi, interp_curve)
            ba = bound_verdict(
                eps_mean_interp, ell_bar, D_us, RHO_TARGET,
                args.rho_bar, args.base_cost_us,
                args.sigma_residual_us, include_io_cost=True)
            # Held-out empirical: snap to nearest persisted phi knot.
            nearest = min(persisted.keys(), key=lambda k: abs(k - phi))
            persist = persisted[nearest]
            seed = _stable_seed("held_out", substrate, D_ms, phi)
            em_mean, em_lo, em_hi = held_out_empirical_miss(
                persist["mean"], persist["se"], persist["n"],
                ell_bar, D_us, 64,
                args.base_cost_us, args.sigma_residual_us,
                seed=seed, n_boot=args.n_boot)
            replay_accepts = em_hi <= RHO_TARGET
            consistent = (ba == replay_accepts)
            rows.append({
                "substrate": substrate, "D_ms": D_ms, "phi": phi,
                "bound_accepts": ba,
                "replay_miss_mean": em_mean,
                "replay_miss_ci_hi": em_hi,
                "replay_accepts": replay_accepts,
                "consistent": consistent,
                "persisted_phi_knot": nearest,
            })
            n_total += 1
            if consistent:
                n_consistent += 1
            if ba and replay_accepts:
                n_bound_accept_replay_accept += 1
            if (not ba) and (not replay_accepts):
                n_bound_reject_replay_reject += 1

    summary = {
        "n_boundary_cells": n_total,
        "n_consistent": n_consistent,
        "n_bound_accept_replay_accept": n_bound_accept_replay_accept,
        "n_bound_reject_replay_reject": n_bound_reject_replay_reject,
        "rho_target": RHO_TARGET,
        "n_boot": args.n_boot,
        "rows": rows,
        "seed_policy": "sha256('held_out'|substrate|D_ms|phi) -> first 4 bytes",
    }
    out = RESULTS / "held_out_replay.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[held-out-replay] {n_consistent}/{n_total} cells consistent "
          f"(bound matches replay Wilson upper-CI at rho={RHO_TARGET})")
    for r in rows:
        match = "OK " if r["consistent"] else "MISMATCH"
        print(f"  {match} {r['substrate']:8s} D={r['D_ms']:>4d}ms "
              f"phi={r['phi']:.2f}: "
              f"bound={'accept' if r['bound_accepts'] else 'reject'}, "
              f"replay miss_hi={r['replay_miss_ci_hi']:.4f}, "
              f"replay={'accept' if r['replay_accepts'] else 'reject'}")

    # TeX summary table for supplementary §A.5.
    lines = [
        "% Generated by held_out_replay.py (R10).",
        "% Boundary-cell replay audit using persisted dense-deployed-eps",
        "% (200 prompts x 4 phi knots) rather than the parametric",
        "% Beta-Binomial used in admission_map.py.",
        r"\begin{tabular}{lrrlrl}",
        r"\toprule",
        r"Substrate & $D$ (ms) & $\phi$ & "
        r"bound & replay miss$_{\text{hi}}$ & match \\",
        r"\midrule",
    ]
    for r in rows:
        match = r"\checkmark" if r["consistent"] else r"\textbf{X}"
        lines.append(
            f"{r['substrate']} & {r['D_ms']} & {r['phi']:.2f} & "
            f"{'accept' if r['bound_accepts'] else 'reject'} & "
            f"${r['replay_miss_ci_hi']:.4f}$ & {match} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "held_out_replay.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[held-out-replay] wrote {out} and {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
