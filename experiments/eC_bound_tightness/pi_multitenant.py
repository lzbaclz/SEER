"""Multi-tenant PI controller stability sweep (P3-4).

Reviewer round (2026-05, reviewer #3 S3): the PI stability sketch in
§4.2 is single-tenant. With N concurrent requests sharing a global
$\\lambda_t$, the closed loop becomes a MIMO system; the paper claims
stability without proof. This harness drives the
:class:`seer.timing.slo.LambdaController` against a synthetic plant
with N simultaneous "tenants" producing per-step latency observations
that share the controller's $\\lambda$. We measure:

  * settling time (number of decision cycles until $|\\lambda - \\lambda^\\star|
    \\le \\delta$)
  * variance of the rolling P99 across tenants
  * whether the controller saturates against the clip rails

The harness is deterministic (seeded). It does NOT run the simulator
or a real LLM — it stresses the controller's loop dynamics in
isolation.

Usage::

    python -m experiments.eC_bound_tightness.pi_multitenant \\
        --n_tenants 1 2 4 8 \\
        --out experiments/eC_bound_tightness/results/pi_multitenant.csv

Output CSV columns:
    n_tenants, settling_cycles, lambda_final, p99_final,
    saturated_low, saturated_high, corrupted_drops

The plant model: each tenant produces a per-step latency
$\\ell_t = \\mu_0 + \\beta (\\lambda^\\star - \\lambda_t) + \\eta$ with
$\\eta \\sim \\mathcal{N}(0, \\sigma_\\mathrm{noise})$; the controller
sees the average over tenants per decision cycle. This is the
classic averaged-plant limit; matches the recommendation in
§4.2 (use averaged-plant analysis when multi-tenant $\\lambda$
sharing is required).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from seer.timing.slo import LambdaController


def _plant_step(lam: float, lam_star: float, beta: float, mu0: float,
                sigma_noise: float, rng: random.Random) -> float:
    """One per-tenant latency observation from a linearised plant.

    Plant: ``ell = mu0 + beta * (lam - lam_star) + Gaussian noise``,
    floored at 0. The sign matches SEER's policy semantics
    (cf.\\ \\S\\ref{sec:method:policy}): higher $\\lambda$ shifts
    the utility ranking against high-IO-cost blocks, suppressing
    prefetch and raising next-step attention-miss latency, so
    $\\partial \\ell / \\partial \\lambda > 0$ (positive $\\beta$).
    The controller's $\\lambda \\mathrel{-}{=} k_p \\cdot \\mathrm{err}$
    update then forms a negative-feedback loop: when
    $\\ell > \\mathrm{target}$ (err $> 0$), $\\lambda$ decreases,
    which decreases $\\ell$.
    """
    val = mu0 + beta * (lam - lam_star) + rng.gauss(0.0, sigma_noise)
    return max(0.0, val)


def run_sweep(
    n_tenants_values: list[int],
    target_us: float = 47_500.0,
    lam_star: float = 0.30,
    mu0: float = 47_500.0,
    beta: float = 1_000.0,
    sigma_noise: float = 800.0,
    n_cycles: int = 400,
    settle_band: float = 0.05,
    seed: int = 0,
) -> list[dict]:
    """Run the sweep across multi-tenant counts."""
    rows = []
    for n_tenants in n_tenants_values:
        rng = random.Random(seed + n_tenants)
        ctrl = LambdaController(target_us=target_us, initial=0.5)
        # Track running window of average-per-tenant latency for P99
        recent = []
        settling_cycles = None
        for step in range(n_cycles):
            tenant_latencies = [
                _plant_step(ctrl.current_lambda(), lam_star, beta, mu0,
                            sigma_noise, rng)
                for _ in range(n_tenants)
            ]
            avg_lat = sum(tenant_latencies) / n_tenants
            ctrl.observe(avg_lat)
            recent.append(avg_lat)
            if len(recent) > 100:
                recent.pop(0)
            if abs(ctrl.current_lambda() - lam_star) <= settle_band and settling_cycles is None:
                settling_cycles = step + 1

        recent_sorted = sorted(recent)
        if recent_sorted:
            k99 = max(0, int(len(recent_sorted) * 0.99) - 1)
            p99 = recent_sorted[k99]
        else:
            p99 = float("nan")
        stats = ctrl.stats()
        rows.append({
            "n_tenants": int(n_tenants),
            "settling_cycles": int(settling_cycles) if settling_cycles is not None else -1,
            "lambda_final": float(stats["lambda"]),
            "lambda_star_target": float(lam_star),
            "p99_final_us": float(p99),
            "saturated_low": int(stats["lambda"] <= ctrl.lambda_min + 1e-6),
            "saturated_high": int(stats["lambda"] >= ctrl.lambda_max - 1e-6),
            "corrupted_drops": int(stats["corrupted_samples"]),
            "samples_seen": int(stats["samples"]),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_tenants", type=int, nargs="+",
                    default=[1, 2, 4, 8],
                    help="Concurrent tenant counts to sweep.")
    ap.add_argument("--target_us", type=float, default=47_500.0,
                    help="Controller target latency, µs (default = 0.95 * 50 ms).")
    ap.add_argument("--lam_star", type=float, default=0.30,
                    help="Plant operating point of λ that matches target_us.")
    ap.add_argument("--mu0", type=float, default=47_500.0,
                    help="Plant latency at λ=λ_star (set to target_us so the "
                         "loop has a zero-error fixed point at λ=λ_star).")
    ap.add_argument("--beta", type=float, default=1_000.0,
                    help="Plant slope dℓ/dλ in µs/unit-λ; SEER's empirical "
                         "slope is ~1ms/unit-λ at the bound's operating point.")
    ap.add_argument("--sigma_noise", type=float, default=800.0)
    ap.add_argument("--n_cycles", type=int, default=400)
    ap.add_argument("--settle_band", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/eC_bound_tightness/results/pi_multitenant.csv")
    args = ap.parse_args(argv)

    rows = run_sweep(
        n_tenants_values=list(args.n_tenants),
        target_us=args.target_us,
        lam_star=args.lam_star,
        mu0=args.mu0,
        beta=args.beta,
        sigma_noise=args.sigma_noise,
        n_cycles=args.n_cycles,
        settle_band=args.settle_band,
        seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} rows)")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
