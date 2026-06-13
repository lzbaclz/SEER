"""R15 contractual escape-mass q* sweep.

The R14 envelope showed that \\texttt{full-tail-guard} at the
calibrated $q_\\mathrm{base}{=}0.0067$ is chat-tier vacuous: the
heavy-tail term $B_t \\cdot q_\\mathrm{base}{=}0.214$ exceeds
$\\rho{=}10^{-2}$. The natural follow-up is to characterise the
\\emph{contractual} escape-mass regime: at what $q_\\mathrm{base}^*$
does \\texttt{full-tail-guard} become non-vacuous on the chat
tier, and how does that vary with $B_t$?

This script sweeps $q_\\mathrm{base}^* \\in \\{1{\\cdot}10^{-4}, ..., 0.0067\\}$
for each $B_t$ in the production grid, evaluating
\\texttt{full-tail-guard} on the 20-cell deployed-$\\varepsilon$
sizing grid at the chat operating point $\\rho{=}10^{-2}$. The
output is a small table showing:

  * the smallest $q_\\mathrm{base}^*$ at which the bound starts
    accepting non-trivial cells on the chat tier (the
    \\emph{operator contract floor});
  * for that $q^*$, the actionable count;
  * the implied substrate IOPS reservation level for an NVMe
    operator (back-of-envelope: $q^* < 10^{-4}$ corresponds to
    sub-$0.01\\%$ block-IO escape).

The intent is diagnostic: tell the operator what level of
substrate contract is needed to deploy SEER on chat. We
\\emph{do not claim} that this contract is achievable; the
real-substrate sprint is queued in \\texttt{todo\\_atc.md}.

Output: \\texttt{contractual\\_qstar.\\{json,tex\\}}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from seer.timing.schedulability import lemma2_truncated_heavy_tail_bound

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"

# Reuse the same 20-cell sizing grid as bound_variant_compare.
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


def cell_actionable_fulltail(substrate, D_ms, rho_target,
                             eps_curve, B_t, q_star) -> bool:
    ell_bar = SUBSTRATE_ELL[substrate]
    D_us = D_ms * 1000.0
    last_good = 1.0
    for i in range(201):
        phi = max(1e-3, 1.0 - i / 200)
        eps_phi = _eps_at(eps_curve, phi)
        eps_var = min(0.1 * eps_phi,
                      max(0.0, eps_phi * (1.0 - eps_phi) - 1e-9))
        bound = lemma2_truncated_heavy_tail_bound(
            epsilon_mean=eps_phi, epsilon_var=eps_var,
            ell_bar_us=ell_bar, B_t=float(B_t), deadline_us=D_us,
            sigma_residual_us=SIGMA_RES_US, escape_mass=q_star,
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
    args = ap.parse_args()

    eps_curve = _load_dense_eps(pathlib.Path(args.dense_eps))

    # Sweep q* from 1e-5 to 0.0067 (calibrated). Production B_t grid.
    qstar_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 6.7e-3]
    B_t_grid = [16, 32, 51, 64]
    rho_chat = 1e-2

    cells = []
    for B_t in B_t_grid:
        for q_star in qstar_grid:
            actionable = sum(
                cell_actionable_fulltail(sub, D, rho_chat,
                                          eps_curve, B_t, q_star)
                for (sub, D, _) in SIZING_CELLS)
            cells.append({
                "B_t": B_t, "q_star": q_star,
                "btq_star": B_t * q_star,
                "non_vacuous_condition": q_star < rho_chat / B_t,
                "actionable": actionable,
            })

    # For each B_t, find the largest q_star with at least 1
    # actionable cell on chat tier (= the operator's contract
    # ceiling), and the actionable count at that exact q*.
    ceilings = {}
    for B_t in B_t_grid:
        rows = [c for c in cells if c["B_t"] == B_t and c["actionable"] > 0]
        if rows:
            q_ceil = max(r["q_star"] for r in rows)
            # actionable AT the ceiling q*, not max over all rows.
            actionable_at = next(
                r["actionable"] for r in rows if r["q_star"] == q_ceil)
            # also report the peak actionable count and the
            # *loosest* q* that still hits the peak (the operator's
            # smallest required contract tightening).
            peak = max(r["actionable"] for r in rows)
            peak_q = max(r["q_star"] for r in rows if r["actionable"] == peak)
            ceilings[B_t] = {
                "q_star_ceiling": q_ceil,
                "actionable_at_ceiling": actionable_at,
                "peak_actionable": peak,
                "q_star_at_peak": peak_q,
            }
        else:
            ceilings[B_t] = {"q_star_ceiling": None,
                              "actionable_at_ceiling": 0,
                              "peak_actionable": 0,
                              "q_star_at_peak": None}

    out = {
        "rho_chat": rho_chat,
        "qstar_grid": qstar_grid,
        "B_t_grid": B_t_grid,
        "rho_bar": RHO_BAR,
        "ceilings": {str(k): v for k, v in ceilings.items()},
        "cells": cells,
    }
    out_json = RESULTS / "contractual_qstar.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[contractual-qstar] wrote {out_json}")

    print("[contractual-qstar] chat-tier (rho=1e-2) "
          "actionable/20 vs q* and B_t:")
    print(f"{'B_t':>4s}  " + "  ".join(
        [f"q*={q:.0e}" for q in qstar_grid]))
    for B_t in B_t_grid:
        row = [c for c in cells if c["B_t"] == B_t]
        print(f"{B_t:>4d}  " + "  ".join(
            [f"{c['actionable']:>6d}/20" for c in row]))

    # TeX rendering.
    lines = [
        "% Generated by contractual_qstar.py (R15).",
        "% full-tail-guard actionable/20 on the deployed-eps sizing",
        f"% grid at rho={rho_chat}, swept across (B_t, q_star). The",
        "% calibrated empirical value is q_base=0.0067 (rightmost column).",
        r"\begin{tabular}{r" + "r" * len(qstar_grid) + "}",
        r"\toprule",
        r"$B_t$ & " + " & ".join([
            rf"$q^*{{=}}{q:g}$".replace("e-0", "{e}{-}").replace("e-", "{e}{-}")
            for q in qstar_grid
        ]) + r" \\",
        r"\midrule",
    ]
    for B_t in B_t_grid:
        row = sorted([c for c in cells if c["B_t"] == B_t],
                     key=lambda r: r["q_star"])
        cells_str = []
        for c in row:
            v = c["actionable"]
            if v > 0:
                cells_str.append(rf"\textbf{{{v}}}")
            else:
                cells_str.append(str(v))
        lines.append(f"${B_t}$ & " + " & ".join(cells_str) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "contractual_qstar.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[contractual-qstar] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
