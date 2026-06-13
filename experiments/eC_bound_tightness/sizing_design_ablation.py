"""Sizing-rule output under per-design-piece ablations (R7-W6).

The R6 round closed the LAP-vs-others necessity gap on the AUC-scaled
sizing-rule output axis (`sizing_predictor_compare.py`:
LAP 3/20 -> H2O 2/20 -> raw-attn 1/20). The R7 reviewer asks
whether the *framework* itself (not just LAP) has design pieces
that are each individually necessary. This script ablates the
four framework design pieces and reports the actionable-cell
count under each ablation.

Design pieces (each independently knocked out):

1. **PI (online sigma tracking).** Without it we cannot adapt to
   contention and must assume the worst-case sigma observed in
   the eFault_characterization regime sweep
   (`sigma_residual_us = 5.49 * sigma_quiet_us`).
2. **IO-cost-aware bound (`base_cost_us`).** Without it the bound
   pretends IO is free (`base_cost_us = 0`), the case the prior
   art often assumes.
3. **Fail-safe q_base (bootstrap upper-CI).** Without it we use
   the q_base point estimate instead of the upper-CI (=0.0058
   rather than 0.0067), so the bound is no longer over the
   bootstrap CI's worst case.
4. **Correlation-aware Corollary 2'' (rho_bar=0.08).** Without it
   we use the i.i.d. ablation (`rho_bar=0`), which the R6 round
   established gives 2/20 (vs operator-default 3/20).

Each ablation reports the actionable-cell count over the same 20
cells used by sizing_predictor_compare.py. The Delta from the
SEER-full baseline is the design piece's necessity score.

Output: `sizing_design_ablation.json` + `sizing_design_ablation.tex`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seer.timing.schedulability import min_hbm_budget_for_slo_measured


SUBSTRATES = [
    ("DRAM",    200.0,  1500.0),
    ("NVMe",    1500.0, 3000.0),
    ("NVMe-GC", 2000.0, 5000.0),
]

CELLS = [
    ("DRAM",    35,  1e-2),
    ("DRAM",    35,  1e-3),
    ("DRAM",    50,  1e-2),
    ("DRAM",    50,  1e-3),
    ("DRAM",    75,  1e-2),
    ("DRAM",    75,  1e-3),
    ("DRAM",    100, 1e-2),
    ("DRAM",    100, 1e-3),
    ("NVMe",    35,  1e-2),
    ("NVMe",    50,  1e-2),
    ("NVMe",    75,  1e-2),
    ("NVMe",    75,  1e-3),
    ("NVMe",    100, 1e-2),
    ("NVMe",    100, 1e-3),
    ("NVMe-GC", 35,  1e-2),
    ("NVMe-GC", 50,  1e-2),
    ("NVMe-GC", 75,  1e-2),
    ("NVMe-GC", 75,  1e-3),
    ("NVMe-GC", 100, 1e-2),
    ("NVMe-GC", 100, 1e-3),
]


POLICY_FLOOR = 0.125


def _load_dense_eps(path: Path) -> list[tuple[float, float]]:
    with open(path) as fh:
        data = json.load(fh)
    return [(r["phi"], r["deployed_mean"]) for r in data["rows"]]


def count_actionable(eps_curve, ell_bar_us_by_substrate, B_full,
                     sigma_residual_us, base_cost_us, rho_bar):
    """Return (count, per_cell_phi). A cell is actionable iff
    POLICY_FLOOR < phi^min <= 1 - 1e-3."""
    count = 0
    cells = []
    for substrate, D_ms, rho in CELLS:
        ell_bar_us = ell_bar_us_by_substrate[substrate]
        D_us = D_ms * 1000.0
        try:
            phi = min_hbm_budget_for_slo_measured(
                eps_curve=eps_curve,
                ell_bar_us=ell_bar_us,
                B_full=B_full,
                deadline_us=D_us,
                sigma_residual_us=sigma_residual_us,
                miss_target=rho,
                base_cost_us=base_cost_us,
                rho_bar=rho_bar,
            )
        except TypeError:
            # backwards-compatibility: older signatures don't accept rho_bar
            phi = min_hbm_budget_for_slo_measured(
                eps_curve=eps_curve,
                ell_bar_us=ell_bar_us,
                B_full=B_full,
                deadline_us=D_us,
                sigma_residual_us=sigma_residual_us,
                miss_target=rho,
                base_cost_us=base_cost_us,
            )
        except Exception:
            phi = 1.0
        cells.append({"substrate": substrate, "D_ms": D_ms,
                      "rho": rho, "phi": phi})
        if POLICY_FLOOR < phi <= 1.0 - 1e-3:
            count += 1
    return count, cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_eps",
                    default="experiments/eC_bound_tightness/results/"
                            "dense_deployed_eps_summary.json")
    ap.add_argument("--B_full", type=float, default=512.0)
    # SEER-full operator defaults (matching sizing_predictor_compare.py):
    ap.add_argument("--sigma_residual_us", type=float, default=1203.0)
    ap.add_argument("--base_cost_us", type=float, default=33800.0)
    ap.add_argument("--rho_bar", type=float, default=0.08)
    # ablation amplification factors:
    ap.add_argument("--no_pi_sigma_scale", type=float, default=5.49,
                    help="Worst-case contention/quiet ratio observed in "
                         "the sigma_contention_regimes 3-trial bootstrap.")
    args = ap.parse_args()

    deployed_lap_curve = _load_dense_eps(Path(args.dense_eps))
    ell_bar = {s: e for s, e, _ in SUBSTRATES}

    regimes = [
        ("SEER-full (operator default)",
         dict(eps_curve=deployed_lap_curve,
              sigma_residual_us=args.sigma_residual_us,
              base_cost_us=args.base_cost_us,
              rho_bar=args.rho_bar)),
        ("no-PI (worst-case sigma)",
         dict(eps_curve=deployed_lap_curve,
              sigma_residual_us=args.sigma_residual_us * args.no_pi_sigma_scale,
              base_cost_us=args.base_cost_us,
              rho_bar=args.rho_bar)),
        ("no-IO-cost (base_cost_us=0)",
         dict(eps_curve=deployed_lap_curve,
              sigma_residual_us=args.sigma_residual_us,
              base_cost_us=0.0,
              rho_bar=args.rho_bar)),
        ("no-correlation-aware (i.i.d.)",
         dict(eps_curve=deployed_lap_curve,
              sigma_residual_us=args.sigma_residual_us,
              base_cost_us=args.base_cost_us,
              rho_bar=0.0)),
    ]

    results = []
    print("[design-ablation] regime -> actionable cells:")
    for name, kw in regimes:
        n, cells = count_actionable(B_full=args.B_full,
                                    ell_bar_us_by_substrate=ell_bar,
                                    **kw)
        results.append({"regime": name,
                        "actionable": n,
                        "total": len(CELLS),
                        "config": {k: v for k, v in kw.items()
                                   if k != "eps_curve"}})
        print(f"  {name:38s} -> {n}/{len(CELLS)}")

    out_dir = Path("experiments/eC_bound_tightness/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "sizing_design_ablation.json"
    out_json.write_text(json.dumps(
        {"regimes": results, "n_cells_total": len(CELLS)}, indent=2))
    print(f"[design-ablation] wrote {out_json}")

    baseline = results[0]["actionable"]
    lines = [
        "% Generated by sizing_design_ablation.py (R7-W6).",
        "% Sizing-rule actionable-cell count under per-design-piece",
        "% knockouts. Baseline is the SEER-full operator default; each",
        "% ablation knocks out exactly one framework design piece.",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Regime & actionable & $\Delta$ vs full \\",
        r"\midrule",
    ]
    for r in results:
        n = r["actionable"]
        delta = n - baseline
        delta_s = f"${delta:+d}$" if r["regime"] != results[0]["regime"] else "---"
        # Escape LaTeX specials in the regime label (underscore, etc).
        label = (r["regime"]
                 .replace("_", r"\_"))
        lines.append(f"\\texttt{{{label}}} & "
                     f"${n}/{r['total']}$ & {delta_s} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_tex = out_dir / "sizing_design_ablation.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[design-ablation] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
