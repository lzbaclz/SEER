"""R17 per-step conditional tail bound envelope (Lemma 2''').

Replaces the $B_t \\cdot q_\\mathrm{base}$ union with a per-step
burst cover $q_\\mathrm{step}$ under A7 (per-step burst
correlation). At the calibrated $q_\\mathrm{base}{=}0.0067$ the
heavy-tail contribution drops from $B_t \\cdot 0.0067 = 0.214$
(chat-tier vacuous) to $0.0067$ alone (chat-tier non-vacuous
condition $q_\\mathrm{step}{<}\\rho$ holds at $\\rho{=}10^{-2}$).

This script evaluates the per-step bound \\texttt{full-tail-PS}
on the same 20-cell deployed-$\\varepsilon$ sizing grid + same
$(B_t, \\rho)$ sweep used by
\\texttt{full\\_tail\\_guard\\_envelope.py}. The output table is
the headline evidence that the per-step refinement closes the
chat-tier safe-and-actionable gap without any operator-side
substrate contract tightening.

Output: \\texttt{per\\_step\\_envelope.\\{json,tex\\}}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from seer.timing.schedulability import lemma2_per_step_conditional_bound

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"

# Reuse the 20-cell deployed-eps sizing grid (matches the
# bound_variant_compare / contractual_qstar conventions).
SIZING_CELLS = [
    ("DRAM",    35,  1e-2), ("DRAM",    35,  1e-3),
    ("DRAM",    50,  1e-2), ("DRAM",    50,  1e-3),
    ("DRAM",    75,  1e-2), ("DRAM",    75,  1e-3),
    ("DRAM",    100, 1e-2), ("DRAM",    100, 1e-3),
    ("NVMe",    35,  1e-2), ("NVMe",    50,  1e-2),
    ("NVMe",    75,  1e-2), ("NVMe",    75,  1e-3),
    ("NVMe",    100, 1e-2), ("NVMe",    100, 1e-3),
    ("NVMe-GC", 35,  1e-2), ("NVMe-GC", 50,  1e-2),
    ("NVMe-GC", 75,  1e-2), ("NVMe-GC", 75,  1e-3),
    ("NVMe-GC", 100, 1e-2), ("NVMe-GC", 100, 1e-3),
]
SUBSTRATE_ELL = {"DRAM": 200.0, "NVMe": 1500.0, "NVMe-GC": 2000.0}

POLICY_FLOOR = 0.125
SIGMA_RES_US = 1203.0
BASE_COST_US = 33800.0
RHO_BAR = 0.08
Q_BASE_CALIB = 0.0067


def _load_dense_eps(path: pathlib.Path) -> list[tuple[float, float]]:
    d = json.loads(path.read_text())
    return [(r["phi"], r["deployed_mean"]) for r in d["rows"]]


def _eps_at(curve: list[tuple[float, float]], phi: float) -> float:
    s = sorted(curve)
    if phi <= s[0][0]:
        return s[0][1]
    if phi >= s[-1][0]:
        return s[-1][1]
    for (a, ea), (b, eb) in zip(s, s[1:]):
        if a <= phi <= b:
            t = (phi - a) / max(1e-12, b - a)
            return ea + t * (eb - ea)
    return s[-1][1]


def cell_actionable_per_step(substrate, D_ms, rho_target,
                              eps_curve, B_t, q_step) -> bool:
    ell_bar = SUBSTRATE_ELL[substrate]
    D_us = D_ms * 1000.0
    last_good = 1.0
    for i in range(201):
        phi = max(1e-3, 1.0 - i / 200)
        eps_phi = _eps_at(eps_curve, phi)
        eps_var = min(0.1 * eps_phi,
                      max(0.0, eps_phi * (1.0 - eps_phi) - 1e-9))
        bound = lemma2_per_step_conditional_bound(
            epsilon_mean=eps_phi, epsilon_var=eps_var,
            ell_bar_us=ell_bar, B_t=float(B_t), deadline_us=D_us,
            sigma_residual_us=SIGMA_RES_US, q_step=q_step,
            base_cost_us=BASE_COST_US, rho_bar=RHO_BAR)
        if bound <= rho_target:
            last_good = phi
        else:
            break
    return POLICY_FLOOR < last_good <= 1.0 - 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dense_eps",
        default=str(RESULTS / "dense_deployed_eps_summary.json"))
    ap.add_argument("--q_step", type=float, default=Q_BASE_CALIB,
                    help="Per-step burst probability; defaults to the "
                         "calibrated q_base=0.0067.")
    args = ap.parse_args()

    eps_curve = _load_dense_eps(pathlib.Path(args.dense_eps))

    # Same (B_t, rho) sweep as full_tail_guard_envelope.py.
    B_t_grid = [1, 2, 4, 8, 16, 32, 51, 64]
    rho_grid = [1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5]

    cells = []
    for B_t in B_t_grid:
        for rho in rho_grid:
            n_actionable = sum(
                cell_actionable_per_step(sub, D, rho, eps_curve, B_t,
                                          args.q_step)
                for (sub, D, _) in SIZING_CELLS)
            cells.append({
                "B_t": B_t, "rho": rho, "q_step": args.q_step,
                "non_vacuous_condition": args.q_step < rho,
                "actionable": n_actionable,
            })

    out = {
        "q_step": args.q_step,
        "B_t_grid": B_t_grid,
        "rho_grid": rho_grid,
        "rho_bar": RHO_BAR,
        "lemma": "Lemma 2''' per-step conditional tail bound (A7)",
        "cells": cells,
    }
    out_json = RESULTS / "per_step_envelope.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[per-step-env] wrote {out_json}")

    print(f"[per-step-env] q_step={args.q_step}; actionable/20 vs (B_t, rho):")
    print(f"{'B_t':>4s}  " + "  ".join(
        [f"rho={r:g}" for r in rho_grid]))
    for B_t in B_t_grid:
        row = [c for c in cells if c["B_t"] == B_t]
        print(f"{B_t:>4d}  " + "  ".join(
            [f"{c['actionable']:>3d}/20" for c in row]))

    # TeX table (compact, B_t rows x rho cols).
    lines = [
        f"% Generated by per_step_envelope.py (R17). q_step={args.q_step}.",
        "% Lemma 2''' per-step conditional tail bound: at calibrated",
        "% q_step=q_base=0.0067 the chat-tier non-vacuous condition",
        r"% $q_\mathrm{step}<\rho$ holds at $\rho{=}10^{-2}$.",
        r"\begin{tabular}{r" + "r" * len(rho_grid) + "}",
        r"\toprule",
        r"$B_t$ & " + " & ".join(
            [rf"$\rho{{=}}{r:g}$" for r in rho_grid]) + r" \\",
        r"\midrule",
    ]
    for B_t in B_t_grid:
        row = sorted([c for c in cells if c["B_t"] == B_t],
                     key=lambda r: r["rho"])
        cells_str = []
        for c in row:
            v = c["actionable"]
            if v > 0:
                cells_str.append(rf"\textbf{{{v}}}")
            else:
                cells_str.append(str(v))
        lines.append(f"${B_t}$ & " + " & ".join(cells_str) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "per_step_envelope.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[per-step-env] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
