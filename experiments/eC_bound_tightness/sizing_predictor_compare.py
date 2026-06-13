"""Sizing-rule output under different predictor regimes (R5-W2).

The previous round (R3-W4) recomputed Lemma~2's $\\phi^{\\min}$
under the *deployed* $\\epsilon(\\phi)$ curve and reported 2/20
operator-actionable cells (vs 6/20 under the calibrated curve).
This script propagates the AUC$\\to\\epsilon$ gap from the e1
predictability experiment ($\\text{AUC}_\\text{LAP}{=}0.94$,
$\\text{AUC}_\\text{H2O}{=}0.89$,
$\\text{AUC}_\\text{raw-attn}{=}0.77$) into the deployed
$\\epsilon(\\phi)$ curve via the linear ratio
$\\epsilon_\\text{pred}(\\phi) = \\epsilon_\\text{LAP}(\\phi)
\\cdot (1{-}\\text{AUC}_\\text{pred}) / (1{-}\\text{AUC}_\\text{LAP})$
and recomputes the sizing rule's operator-actionable cell count
under each predictor.

This quantifies the framework's *LAP-necessity* on the sizing-rule
output axis (the deployment-grade output the framework consumes),
not just on the AUC axis: if raw-attn at AUC$=0.77$ collapses the
sizing rule's actionable count to ${\\le}1/20$, the LAP's
${+}0.17$ AUC gap is necessary in the deployment-grade sense.

Output:
- ``sizing_predictor_compare.json``
- ``sizing_predictor_compare.tex`` for the paper (R5-W2 table).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seer.timing.schedulability import min_hbm_budget_for_slo_measured


# Substrate definitions (match io_bound_regime.py).
SUBSTRATES = [
    ("DRAM",    200.0,  1500.0),
    ("NVMe",    1500.0, 3000.0),
    ("NVMe-GC", 2000.0, 5000.0),
]
# Cells from the headline `sizing_rule_io_bound.tex`.
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

# Predictor AUC values from e1 (paper sec:6.2).
PREDICTORS = {
    "LAP":          0.94,   # TinyMLP headline
    "H2O persist":  0.89,   # best non-LAP baseline
    "raw-attn":     0.77,   # current-step top-K, no learning
}
LAP_AUC = PREDICTORS["LAP"]


def _scaled_eps_curve(deployed_lap_curve: list[tuple[float, float]],
                       pred_auc: float) -> list[tuple[float, float]]:
    """Linear AUC->eps scaling. eps_pred(phi) = eps_LAP(phi) *
    (1 - AUC_pred) / (1 - AUC_LAP). Clamps to [0, 1]."""
    ratio = (1.0 - pred_auc) / max(1e-9, (1.0 - LAP_AUC))
    out = []
    for phi, eps in deployed_lap_curve:
        scaled = min(1.0, max(0.0, eps * ratio))
        out.append((phi, scaled))
    return out


def _load_dense_eps(path: Path) -> list[tuple[float, float]]:
    with open(path) as fh:
        data = json.load(fh)
    return [(r["phi"], r["deployed_mean"]) for r in data["rows"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_eps",
                    default="experiments/eC_bound_tightness/results/"
                            "dense_deployed_eps_summary.json")
    ap.add_argument("--B_full", type=float, default=512.0)
    ap.add_argument("--sigma_residual_us", type=float, default=1203.0)
    ap.add_argument("--base_cost_us", type=float, default=33800.0)
    ap.add_argument("--out_dir",
                    default="experiments/eC_bound_tightness/results")
    args = ap.parse_args()

    deployed_lap_curve = _load_dense_eps(Path(args.dense_eps))

    POLICY_FLOOR = 0.125

    # Build per-predictor eps curves.
    eps_curves = {
        name: _scaled_eps_curve(deployed_lap_curve, auc)
        for name, auc in PREDICTORS.items()
    }

    # Sizing inversion under each predictor.
    rows = []
    cell_counts: dict[str, int] = {n: 0 for n in PREDICTORS}
    cell_phi: dict[str, list[float]] = {n: [] for n in PREDICTORS}
    for substrate, D_ms, rho in CELLS:
        ell_bar_us = next(e for s, e, _ in SUBSTRATES if s == substrate)
        D_us = D_ms * 1000.0
        cell_phi_at_predictor: dict[str, float] = {}
        for name, curve in eps_curves.items():
            try:
                phi = min_hbm_budget_for_slo_measured(
                    eps_curve=curve,
                    ell_bar_us=ell_bar_us,
                    B_full=args.B_full,
                    deadline_us=D_us,
                    sigma_residual_us=args.sigma_residual_us,
                    miss_target=rho,
                    base_cost_us=args.base_cost_us,
                )
            except Exception:
                phi = 1.0
            cell_phi_at_predictor[name] = phi
            cell_phi[name].append(phi)
            if POLICY_FLOOR < phi <= 1.0 - 1e-3:
                cell_counts[name] += 1
        rows.append({
            "substrate": substrate,
            "D_ms": D_ms,
            "rho": rho,
            "phi_per_predictor": cell_phi_at_predictor,
        })

    print("[predictor-compare] AUC -> deployed eps -> actionable cells:")
    for name, auc in PREDICTORS.items():
        ratio = (1.0 - auc) / (1.0 - LAP_AUC)
        print(f"  {name:12s}  AUC={auc:.2f}  "
              f"eps_ratio={ratio:.2f}x  actionable={cell_counts[name]}/{len(CELLS)}")

    summary = {
        "policy_floor": POLICY_FLOOR,
        "predictors_auc": PREDICTORS,
        "deployed_lap_curve": deployed_lap_curve,
        "n_cells_total": len(CELLS),
        "n_actionable_per_predictor": cell_counts,
        "rows": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "sizing_predictor_compare.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[predictor-compare] wrote {out_json}")

    # TeX table.
    lines = [
        "% Generated by sizing_predictor_compare.py (R5-W2).",
        "% Sizing-rule output under three predictor regimes.",
        "% eps(phi) scaled by (1-AUC_pred)/(1-AUC_LAP) on the deployed",
        "% LAP eps(phi) curve. Operator-actionable count is the headline.",
        r"\begin{tabular}{lcrcr}",
        r"\toprule",
        r"Predictor & AUC$_\text{e1}$ & "
        r"$\varepsilon$-scale ($\times$ LAP) & "
        r"actionable cells & cell rate \\",
        r"\midrule",
    ]
    for name, auc in PREDICTORS.items():
        ratio = (1.0 - auc) / (1.0 - LAP_AUC)
        cells = cell_counts[name]
        pct = 100.0 * cells / len(CELLS)
        bold = r"\mathbf" if name == "LAP" else ""
        lines.append(
            f"\\texttt{{{name}}} & ${auc:.2f}$ & "
            f"${'' if name == 'LAP' else bold}{{{ratio:.2f}\\times}}$ & "
            f"${{{bold}{{{cells}}}}}/{len(CELLS)}$ & "
            f"${pct:.0f}\\%$ \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_tex = out_dir / "sizing_predictor_compare.tex"
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[predictor-compare] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
