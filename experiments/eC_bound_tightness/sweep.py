"""eC sweep: compute Lemma 2 bound across an (eps, deadline, budget) grid.

This is the *analytical* leg of eC — pure-python, no GPU, no model
required. We can ship the figure even if the empirical miss-ratio leg
isn't ready, by leaving the ``measured`` column blank.

Usage::

    python sweep.py \\
        --epsilons 0.05 0.10 0.15 0.20 0.30 \\
        --deadlines_ms 25 35 50 75 100 \\
        --budgets 0.1 0.2 0.4 0.8 \\
        --out results/grid.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--deadlines_ms", type=float, nargs="+",
                    default=[25.0, 35.0, 50.0, 75.0, 100.0])
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--ell_bar_us", type=float, default=200.0)
    ap.add_argument("--sigma_residual_us", type=float, default=100.0)
    ap.add_argument("--B_full", type=float, default=64.0)
    ap.add_argument("--base_cost_us", type=float, default=1500.0)
    ap.add_argument("--out", default="results/grid.csv")
    args = ap.parse_args()

    from seer.timing.schedulability import lemma2_miss_prob_bound

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for eps in args.epsilons:
        for D_ms in args.deadlines_ms:
            for phi in args.budgets:
                D_us = D_ms * 1000.0
                bound = lemma2_miss_prob_bound(
                    epsilon=eps,
                    ell_bar_us=args.ell_bar_us,
                    B_t=args.B_full * phi,
                    deadline_us=D_us,
                    sigma_residual_us=args.sigma_residual_us,
                    base_cost_us=args.base_cost_us,
                )
                rows.append({
                    "epsilon": eps,
                    "deadline_ms": D_ms,
                    "budget": phi,
                    "bound": bound,
                    "ell_bar_us": args.ell_bar_us,
                    "sigma_residual_us": args.sigma_residual_us,
                    "base_cost_us": args.base_cost_us,
                })
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[eC] wrote {len(rows)} grid points → {args.out}")


if __name__ == "__main__":
    main()
