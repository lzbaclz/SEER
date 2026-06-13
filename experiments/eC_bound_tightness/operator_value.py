"""Operator value of the SEER framework: cost-of-pessimism vs.
cost-of-naive-over-allocate.

R4 W4 fix: an operator considering deployment can ask:
"If the bound is systematically 3x pessimistic, why not just
allocate 3x more HBM and skip the framework entirely?" This script
quantifies the comparison cell by cell.

Three baselines compared per substrate x deadline x rho cell:
  * SEER framework: phi_min^SEER from Lemma 2 inversion, then
    pessimism-adjusted phi := phi_min^SEER * pessimism_factor
    (the operator pays the calibration cost as a provisioning
    cushion).
  * Naive: phi = 1.0 (the trivially-safe baseline that allocates
    full cache for every stream).
  * H2O@phi=0.20: the published H2O recommended operating point
    (a heavy-hitter policy at 20% active-mask fraction). This is
    the reviewer-preferred comparison: a deployed-policy baseline
    rather than a naive over-allocation strawman (R2-W2 fix).
  * HBM saving (vs naive) := 1 - phi_seer_adj.
  * HBM saving (vs H2O)  := (0.20 - phi_seer_adj) / 0.20 when
    phi_seer_adj < 0.20; otherwise SEER costs more cache and the
    delta is reported negatively.

The framework value proposition holds in cells where
phi_seer_adj < 1.0 with appreciable margin. Cells where
phi_seer_adj > 1.0 indicate the framework cannot deploy at the
operator's chosen (substrate, deadline, rho) target -- a real
deployment-regime answer, not a marketing failure.
"""
from __future__ import annotations

import json
from pathlib import Path

from seer.timing.schedulability import (
    _eps_eff_heuristic,
    lemma2_truncated_heavy_tail_bound,
)


def phi_min_for_cell(
    *, epsilon_clean: float, ell_bar_us: float, B_full: float,
    deadline_us: float, sigma_residual_us: float,
    miss_target: float, q_base: float = 0.0067,
    epsilon_var_scale: float = 0.1,
    ell_max_factor: float = 4.0,
    search_steps: int = 400,
) -> float:
    """Closed-form Lemma 2 inversion: smallest phi s.t. the
    truncated-heavy-tail bound stays under miss_target."""
    best = 1.0
    for i in range(search_steps + 1):
        phi = 1.0 - i / search_steps
        phi = max(1e-3, phi)
        eps_eff = _eps_eff_heuristic(epsilon_clean, phi)
        bound = lemma2_truncated_heavy_tail_bound(
            epsilon_mean=eps_eff,
            epsilon_var=epsilon_var_scale * eps_eff,
            ell_bar_us=ell_bar_us,
            B_t=B_full * phi,
            deadline_us=deadline_us,
            sigma_residual_us=sigma_residual_us,
            escape_mass=q_base,
            ell_max_factor=ell_max_factor,
        )
        if bound <= miss_target:
            best = phi
        else:
            break
    return float(best)


def main() -> int:
    # Use the paper's already-computed phi_min cells from
    # io_bound_regime_summary.json (sizing_rule_io_bound.tex). The
    # six cells with phi_min strictly above the policy floor (0.125)
    # are the ones where the framework adds real sizing value.
    # Median pessimism from io_bound_regime_summary.json:
    #   DRAM=3.21, NVMe=5.78, NVMe-GC=29.27.  (Worst-case in
    #   parens for the table caption.)
    PESSIMISM = {
        "DRAM": 3.21, "NVMe": 5.78, "NVMe-GC": 29.27,
    }
    PESSIMISM_MAX = {
        "DRAM": 5.63, "NVMe": 34.28, "NVMe-GC": 277.50,
    }
    cells = [
        # (substrate, D_ms, rho, phi_min_seer) — from sizing_rule_io_bound.tex
        ("DRAM",    50,  1e-3, 0.125),
        ("DRAM",    100, 1e-3, 0.125),
        ("NVMe",    75,  1e-2, 0.250),
        ("NVMe",    75,  1e-3, 0.295),
        ("NVMe",    100, 1e-3, 0.160),
        ("NVMe-GC", 75,  1e-2, 0.335),
        ("NVMe-GC", 100, 1e-2, 0.195),
        ("NVMe-GC", 100, 1e-3, 0.245),
    ]

    rows: list[dict] = []
    for substrate, D_ms, rho, phi_seer in cells:
        pess_med = PESSIMISM[substrate]
        pess_max = PESSIMISM_MAX[substrate]
        phi_adj_med = phi_seer * pess_med
        phi_adj_max = phi_seer * pess_max
        if phi_adj_med < 1.0:
            saving_med = 100.0 * (1.0 - phi_adj_med)
            verdict = "deployable at median pessimism"
        else:
            saving_med = 0.0
            verdict = (
                "median-pessimism provisioning exceeds full cache "
                f"(phi_adj_med = {phi_adj_med:.2f}); operator should "
                "either use a tighter bound (measured eps curve) or "
                "select a different substrate"
            )
        # H2O@phi=0.20 reference baseline (R2-W2 fix).
        H2O_PHI = 0.20
        if phi_adj_med < H2O_PHI:
            saving_vs_h2o = 100.0 * (H2O_PHI - phi_adj_med) / H2O_PHI
        else:
            saving_vs_h2o = -100.0 * (phi_adj_med - H2O_PHI) / H2O_PHI
        rows.append({
            "substrate": substrate,
            "deadline_ms": D_ms,
            "rho": rho,
            "phi_min_seer": phi_seer,
            "pessimism_median": pess_med,
            "pessimism_max": pess_max,
            "phi_adj_median": phi_adj_med,
            "phi_adj_max": phi_adj_max,
            "naive_phi": 1.0,
            "h2o_phi": H2O_PHI,
            "saving_pct_median_pessimism": saving_med,
            "saving_pct_vs_h2o": saving_vs_h2o,
            "verdict": verdict,
        })
        print(f"[op-value] {substrate:8s} D={D_ms:3d}ms rho={rho:.0e}: "
              f"phi_seer={phi_seer:.3f} x{pess_med:.2f}={phi_adj_med:.3f}  "
              f"saving_vs_naive={saving_med:.1f}%  "
              f"saving_vs_H2O@0.20={saving_vs_h2o:+.1f}%")

    out_dir = Path("experiments/eC_bound_tightness/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "explanation": (
            "phi_min_seer (taken from sizing_rule_io_bound.tex) is the "
            "smallest active-mask fraction at which Lemma 2's "
            "truncated-heavy-tail bound stays at or below rho. "
            "phi_adj := phi_min_seer * pessimism is the operator's "
            "actual provisioning if they respect the calibrated safety "
            "margin (median or worst-case pessimism from "
            "io_bound_regime_summary.json). saving_pct = 100 * (1 - "
            "phi_adj) vs the naive phi=1 baseline."
        ),
        "q_base_calibrated": 0.0067,
        "rows": rows,
    }
    with open(out_dir / "operator_value_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[op-value] wrote {out_dir / 'operator_value_summary.json'}")

    lines = [
        "% Generated by experiments/eC_bound_tightness/operator_value.py",
        "% Operator-value: SEER framework's pessimism-adjusted provisioning",
        "% vs naive phi=1 over-allocate AND vs H2O@phi=0.20 reference",
        "% (R2-W2 fix). Median pessimism from io_bound_regime_summary.json.",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Substrate & $D$ & $\rho$ & $\hbm^\mathrm{min}$ & "
        r"Pess.\ (med/max) & $\hbm^{\mathrm{adj},\mathrm{med}}$ & "
        r"Saving / na\"ive & Saving / H2O \\",
        r"\midrule",
    ]
    for r in rows:
        phi_adj_med = r["phi_adj_median"]
        if phi_adj_med < 1.0:
            saving_naive = f"${r['saving_pct_median_pessimism']:.1f}\\%$"
            phi_cell = f"${phi_adj_med:.3f}$"
        else:
            saving_naive = r"\textit{n/a}"
            phi_cell = f"${phi_adj_med:.2f}{{>}}1$"
        s_h2o = r["saving_pct_vs_h2o"]
        saving_h2o = (f"${s_h2o:+.1f}\\%$" if abs(s_h2o) < 9999
                       else r"\textit{n/a}")
        lines.append(
            f"{r['substrate']} & "
            f"${r['deadline_ms']}$\\,ms & "
            f"${r['rho']:.0e}$ & "
            f"${r['phi_min_seer']:.3f}$ & "
            f"${r['pessimism_median']:.1f}{{\\times}}$ / "
            f"${r['pessimism_max']:.1f}{{\\times}}$ & "
            f"{phi_cell} & "
            f"{saving_naive} & "
            f"{saving_h2o} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(out_dir / "operator_value.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[op-value] wrote {out_dir / 'operator_value.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
