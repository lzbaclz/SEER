"""R9 RTSS-strong-accept primary contribution: bound-vs-empirical
admission map across the (D, phi) plane on three substrates.

The R8 self-review made clear that the paper's most defensible
RTSS-track contribution is C1 admission control, not the SEER
policy. The strong-accept move is to demonstrate that the bound's
accept/reject verdict matches the empirical miss-rate outcome --
i.e. the bound is not a "synthetic sizing demo" but a
deployment-correct admission test.

For every cell (substrate, D, phi):

  bound_verdict   = epsilon(phi)*ell_bar / D  <= rho_target  (under
                    correlation-aware Cor 2'' at rho_bar=0.08)
  empirical_miss  = Monte Carlo over the same deployed eps(phi)
                    distribution with sample-level noise + IO
                    distribution.

Cell colour:
  green   = bound accepts AND empirical miss <= rho_target  (CORRECT ACCEPT)
  red     = bound rejects AND empirical miss > rho_target   (CORRECT REJECT)
  yellow  = bound rejects BUT empirical miss <= rho_target  (CONSERVATIVE)
  black   = bound accepts BUT empirical miss > rho_target   (UNSAFE FALSE ACCEPT)

We then run a no-IO-cost / i.i.d. *ablation* of the same map; the
ablations must produce more black cells (unsafe false accepts) to
demonstrate that each design piece prevents unsafe deployments.

Output: admission_map.{json, tex} for the headline; admission_map.pdf
for the figure (3-panel green/red/yellow/black scatter).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np


def _stable_seed(*parts) -> int:
    """Derive a stable integer seed from arbitrary parts via SHA-256.

    Python's built-in ``hash()`` is salted by ``PYTHONHASHSEED``, so
    repeated runs on the same data with the same Python interpreter
    can produce different seeds (and therefore different Monte Carlo
    outcomes -- a reproducibility hole flagged in R10). We anchor
    every seed to ``hashlib.sha256`` instead so the JSON outputs are
    reproducible across runs and environments.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

from seer.timing.schedulability import (
    lemma2_correlation_aware_bound,
    lemma2_per_step_conditional_bound,
    lemma2_truncated_heavy_tail_bound,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"


SUBSTRATES = {
    # name: (ell_bar_us, ell_max_factor)
    "DRAM":    (200.0,  4.0),
    "NVMe":    (1500.0, 4.0),
    "NVMe-GC": (2000.0, 6.0),
}

# Deadline grid (ms) and phi grid: dense enough to make a map, sparse
# enough to compute quickly on CPU.
D_GRID_MS = [50, 75, 100, 150, 200, 250, 300, 400, 500]
PHI_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RHO_TARGET = 1e-2


def load_dense_eps(path: pathlib.Path) -> dict:
    """Load deployed eps(phi) curve as a dict phi -> dict."""
    d = json.loads(path.read_text())
    out = {}
    for r in d["rows"]:
        out[r["phi"]] = {
            "mean": r["deployed_mean"],
            "se": r["deployed_se"],
            "ci_lo": r["deployed_ci95_lo"],
            "ci_hi": r["deployed_ci95_hi"],
        }
    return out


def interp_eps(phi: float, curve: dict) -> tuple[float, float]:
    """Piecewise-linear interpolation of (mean, se) over the curve."""
    phis = sorted(curve.keys())
    if phi <= phis[0]:
        r = curve[phis[0]]
        return r["mean"], r["se"]
    if phi >= phis[-1]:
        r = curve[phis[-1]]
        return r["mean"], r["se"]
    for i in range(len(phis) - 1):
        if phis[i] <= phi <= phis[i + 1]:
            lo, hi = phis[i], phis[i + 1]
            t = (phi - lo) / (hi - lo)
            mlo, mhi = curve[lo]["mean"], curve[hi]["mean"]
            slo, shi = curve[lo]["se"], curve[hi]["se"]
            return (1 - t) * mlo + t * mhi, (1 - t) * slo + t * shi
    return curve[phis[-1]]["mean"], curve[phis[-1]]["se"]


def bound_verdict(eps_mean, ell_bar_us, D_us, rho_target, rho_bar,
                  base_cost_us, sigma_residual_us,
                  include_io_cost=True, B_t=32,
                  q_base: float = 0.0067,
                  use_heavy_tail: bool = True,
                  use_per_step: bool = False) -> bool:
    """Operator-default sufficient admission test (Lemma 2 full form).

    Variants:
      * ``use_heavy_tail=False``: bernstein-cor2 (no q_base).
      * ``use_heavy_tail=True, use_per_step=False``: full-tail-guard
        (the original union-bound truncated heavy-tail variant).
      * ``use_heavy_tail=True, use_per_step=True``: full-tail-PS
        (R17 per-step conditional tail bound, Lemma 2''').
    """
    bc = base_cost_us if include_io_cost else 0.0
    eps_var = 0.0  # operator default: no a-priori variance reduction
    if use_heavy_tail and use_per_step:
        p_miss = lemma2_per_step_conditional_bound(
            epsilon_mean=eps_mean,
            epsilon_var=eps_var,
            ell_bar_us=ell_bar_us,
            B_t=B_t,
            deadline_us=D_us,
            sigma_residual_us=sigma_residual_us,
            q_step=q_base,
            base_cost_us=bc,
            rho_bar=rho_bar,
        )
    elif use_heavy_tail:
        p_miss = lemma2_truncated_heavy_tail_bound(
            epsilon_mean=eps_mean,
            epsilon_var=eps_var,
            ell_bar_us=ell_bar_us,
            B_t=B_t,
            deadline_us=D_us,
            sigma_residual_us=sigma_residual_us,
            escape_mass=q_base,
            base_cost_us=bc,
            rho_bar=rho_bar,
        )
    else:
        p_miss = lemma2_correlation_aware_bound(
            epsilon_mean=eps_mean,
            epsilon_var=eps_var,
            ell_bar_us=ell_bar_us,
            B_t=B_t,
            deadline_us=D_us,
            sigma_residual_us=sigma_residual_us,
            base_cost_us=bc,
            rho_bar=rho_bar,
        )
    return p_miss <= rho_target


def empirical_miss(eps_mean, eps_se, ell_bar_us, D_us, rho_bar,
                   base_cost_us, sigma_residual_us, n_samples=4000,
                   seed=0, B_t=32) -> float:
    """Monte-Carlo empirical miss rate.

    Per decode step: B_t blocks are queried; each has a (correlated)
    probability `eps_mean` of being a miss, drawn via a Beta-Binomial
    so cross-block correlation matches rho_bar=0.08. A miss costs a
    lognormal IO penalty centred on ell_bar. Per-step latency =
    base_cost + sigma_jitter + sum_of_miss_costs. The bound is checked
    against the empirical fraction of steps that exceed the deadline.
    """
    rng = np.random.default_rng(seed)
    # Beta-Binomial: draw a per-step "shared" eps, then Bernoulli B_t
    # blocks. rho_bar=0 -> very tight beta -> degenerates to Binomial.
    rho_eff = max(rho_bar, 1e-4)
    a = eps_mean * (1.0 / rho_eff - 1.0)
    b = (1.0 - eps_mean) * (1.0 / rho_eff - 1.0)
    p_step = rng.beta(max(a, 0.01), max(b, 0.01), size=n_samples)
    # Number of missing blocks per step.
    miss_counts = rng.binomial(B_t, p_step)
    # Each miss costs a lognormal IO penalty (median = ell_bar).
    # Vectorised: total miss cost = sum of `miss_counts[i]` lognormals.
    # Approximation: replace sum with mean*count for speed (high-n CLT).
    sigma_ln = 0.6
    mu_ln = math.log(max(ell_bar_us, 1.0)) - sigma_ln ** 2 / 2.0
    # E[X] = exp(mu + sigma^2/2) = ell_bar by construction.
    # Var[X] = (exp(sigma^2) - 1) * exp(2 mu + sigma^2)
    var_ln = (math.exp(sigma_ln ** 2) - 1.0) * ell_bar_us ** 2
    io_total = (miss_counts * ell_bar_us
                + rng.normal(0.0, np.sqrt(var_ln * np.maximum(miss_counts, 1)),
                             size=n_samples))
    io_total = np.maximum(io_total, 0.0)
    jitter = rng.normal(0.0, sigma_residual_us, size=n_samples)
    total = base_cost_us + jitter + io_total
    miss_rate = float(np.mean(total > D_us))
    return miss_rate


def classify(bound_accepts: bool, empirical: float, rho: float) -> str:
    """Four-colour classification."""
    safe = empirical <= rho
    if bound_accepts and safe:
        return "green"
    if not bound_accepts and not safe:
        return "red"
    if not bound_accepts and safe:
        return "yellow"
    return "black"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_eps",
                    default=str(RESULTS / "dense_deployed_eps_summary.json"))
    ap.add_argument("--sigma_residual_us", type=float, default=1203.0,
                    help="Operator-default sigma from contention sweep "
                         "(quiet regime; worst-case is 2.65x).")
    ap.add_argument("--base_cost_us", type=float, default=33800.0,
                    help="LAP MBPET base cost (A100 TRT P99.9).")
    ap.add_argument("--rho_bar", type=float, default=0.08,
                    help="Cor 2'' operator-default cross-block correlation.")
    ap.add_argument("--B_t", type=int, default=32,
                    help="Per-step working-set demand. R13: must match "
                         "raw_trace_replay and bound_variant_compare. "
                         "Default 32 = MIN_TOP_K floor "
                         "(seer.trace.schema.compute_top_k).")
    ap.add_argument("--q_base", type=float, default=0.0067,
                    help="Heavy-tail escape mass for full-tail-guard.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    curve = load_dense_eps(pathlib.Path(args.dense_eps))

    summary = {
        "rho_target": RHO_TARGET,
        "D_grid_ms": D_GRID_MS,
        "phi_grid": PHI_GRID,
        "substrates": list(SUBSTRATES.keys()),
        "B_t": args.B_t,
        "q_base": args.q_base,
        "regimes": {},
        # Seeds are derived via
        # hashlib.sha256(f"{args.seed}|{substrate}|{D_ms}|{phi}"), so
        # the same cli args yield identical Monte Carlo outcomes
        # across Python interpreters / PYTHONHASHSEED settings.
        "seed_policy": "sha256(args.seed|substrate|D_ms|phi) -> first 4 bytes",
        "cli_seed": args.seed,
    }

    # R13: 4 named bound variants and 4 named ablations, no
    # collisions. ablations are named after the variant they
    # disable a piece of.
    #
    #   bernstein-cor2      : Bernstein-mixture + Cor.~2'', no q_base
    #                          (the actionable sizing path)
    #   bernstein-no-IO     : bernstein-cor2 with base_cost_us=0
    #   bernstein-iid       : bernstein-cor2 with rho_bar=0
    #   full-tail-guard     : Bernstein-mixture + truncated heavy-tail
    #                          at q_base + Cor.~2'' (the safety variant)
    #   fulltail-no-IO      : full-tail-guard with base_cost_us=0
    #   fulltail-iid        : full-tail-guard with rho_bar=0
    #
    # On the parametric Beta-Binomial simulator the truth is
    # Bernstein-concentrated by construction, so on this map
    # bernstein-cor2 is already self-consistent. The full-tail-guard
    # row is added so the comparison table is complete; the
    # raw-trace replay (raw_trace_replay.py) is where the heavier
    # tails are actually exercised.
    regimes = {
        "bernstein-cor2":   dict(rho_bar=args.rho_bar,
                                  base_cost_us=args.base_cost_us,
                                  include_io_cost=True,
                                  use_heavy_tail=False),
        "bernstein-no-IO":  dict(rho_bar=args.rho_bar,
                                  base_cost_us=0.0,
                                  include_io_cost=False,
                                  use_heavy_tail=False),
        "bernstein-iid":    dict(rho_bar=0.0,
                                  base_cost_us=args.base_cost_us,
                                  include_io_cost=True,
                                  use_heavy_tail=False),
        "full-tail-guard":  dict(rho_bar=args.rho_bar,
                                  base_cost_us=args.base_cost_us,
                                  include_io_cost=True,
                                  use_heavy_tail=True),
        "fulltail-no-IO":   dict(rho_bar=args.rho_bar,
                                  base_cost_us=0.0,
                                  include_io_cost=False,
                                  use_heavy_tail=True),
        "fulltail-iid":     dict(rho_bar=0.0,
                                  base_cost_us=args.base_cost_us,
                                  include_io_cost=True,
                                  use_heavy_tail=True),
        "full-tail-PS":     dict(rho_bar=args.rho_bar,
                                  base_cost_us=args.base_cost_us,
                                  include_io_cost=True,
                                  use_heavy_tail=True,
                                  use_per_step=True),
    }

    overall_tally = {}
    for regime_name, kwargs in regimes.items():
        tally = {"green": 0, "red": 0, "yellow": 0, "black": 0}
        cells = []
        for substrate, (ell_bar, _) in SUBSTRATES.items():
            for D_ms in D_GRID_MS:
                D_us = D_ms * 1000.0
                for phi in PHI_GRID:
                    eps_mean, eps_se = interp_eps(phi, curve)
                    ba = bound_verdict(
                        eps_mean, ell_bar, D_us, RHO_TARGET,
                        kwargs["rho_bar"], kwargs["base_cost_us"],
                        args.sigma_residual_us,
                        include_io_cost=kwargs["include_io_cost"],
                        B_t=args.B_t,
                        q_base=args.q_base,
                        use_heavy_tail=kwargs["use_heavy_tail"],
                        use_per_step=kwargs.get("use_per_step", False))
                    em = empirical_miss(
                        eps_mean, eps_se, ell_bar, D_us,
                        # The simulator's "truth" uses the
                        # ground-truth correlation; only the *bound*
                        # uses the operator's chosen rho_bar.
                        rho_bar=0.08,
                        base_cost_us=args.base_cost_us,
                        sigma_residual_us=args.sigma_residual_us,
                        B_t=args.B_t,
                        seed=_stable_seed(args.seed, substrate, D_ms, phi))
                    cell_class = classify(ba, em, RHO_TARGET)
                    tally[cell_class] += 1
                    cells.append({
                        "substrate": substrate, "D_ms": D_ms, "phi": phi,
                        "bound_accepts": ba, "empirical_miss": em,
                        "class": cell_class,
                    })
        summary["regimes"][regime_name] = {
            "tally": tally, "cells": cells,
            "config": {k: v for k, v in kwargs.items()},
        }
        overall_tally[regime_name] = tally

    # R12+R13: record accepted / unsafe_accepts so a vacuous never-accept
    # variant cannot pass as operationally sound.
    for regime, t in overall_tally.items():
        t["accepted"] = t["green"] + t["black"]
        t["unsafe_accepts"] = t["black"]
        t["conservative_rejects"] = t["yellow"]
        summary["regimes"][regime]["tally"] = t

    out_json = RESULTS / "admission_map.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[admission-map] wrote {out_json} (B_t={args.B_t}, "
          f"q_base={args.q_base})")
    n_total = (len(SUBSTRATES) * len(D_GRID_MS) * len(PHI_GRID))
    for regime, t in overall_tally.items():
        unsafe = t["black"]
        if unsafe > 0:
            tag = "  [UNSAFE]"
        elif t["accepted"] == 0:
            tag = "  [SAFE-VACUOUS]"
        else:
            tag = "  [SAFE]"
        print(f"  {regime:20s}: green={t['green']:>3d}/{n_total} "
              f"red={t['red']:>3d}/{n_total} "
              f"yellow={t['yellow']:>3d}/{n_total} "
              f"black={t['black']:>3d}/{n_total} "
              f"accepted={t['accepted']:>3d}{tag}")

    # TeX summary table for paper §6.
    lines = [
        "% Generated by admission_map.py.",
        "% Bound-verdict vs empirical-truth across (D, phi) on 3 substrates.",
        f"% Working-set demand B_t={args.B_t}; q_base={args.q_base}.",
        "% green=correct-accept, red=correct-reject, yellow=conservative,",
        "% black=UNSAFE false-accept. Total cells per regime: "
        f"{n_total} ({len(SUBSTRATES)} substrates "
        f"x {len(D_GRID_MS)} deadlines x {len(PHI_GRID)} budgets).",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Regime & accepted & green & red & yellow & black & status \\",
        r"\midrule",
    ]
    for regime, t in overall_tally.items():
        if t["black"] > 0:
            status = r"\textbf{UNSAFE}"
        elif t["accepted"] == 0:
            status = "safe (vacuous)"
        else:
            status = "safe"
        lines.append(
            f"\\texttt{{{regime}}} & {t['accepted']} & "
            f"{t['green']} & {t['red']} & "
            f"{t['yellow']} & {t['black']} & {status} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "admission_map.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[admission-map] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
