"""R13 full-tail-guard operating envelope.

R12 surfaced that the full-tail-guard variant (Bernstein-mixture +
truncated heavy-tail at $q_\\mathrm{base}=0.0067$ + Cor.~2'') is
safe but operationally vacuous at the headline chat op-point
($\\rho{=}10^{-2}$, $B_t{=}32$). R13's reviewer noted this is the
core RTSS sales weakness and demanded at least one non-vacuous safe
operating region.

We sweep ($B_t$, $\\rho$) on the deployed-$\\varepsilon$ sizing
grid and report the (a) smallest $\\rho$ at which full-tail-guard
becomes non-vacuous for each $B_t$, and (b) the smallest
$B_t \\cdot q_\\mathrm{base}$ inflation that is consistent with a
non-vacuous accept set. The intent is *diagnostic*: the operator
reads the envelope to choose between (i) relaxed-$\\rho$ workloads,
(ii) smaller-$B_t$ deployments (e.g. layer-sparse decoding), or
(iii) a contractual escape-mass guarantee ($q_\\mathrm{base}^*$)
that retires the $B_t \\cdot q_\\mathrm{base}$ term.

Output: ``full_tail_guard_envelope.{json,tex}``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from seer.timing.schedulability import (
    lemma2_correlation_aware_bound,
    lemma2_truncated_heavy_tail_bound,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"

# Sizing grid (matches sizing_deployed_eps.py).
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


def cell_actionable(substrate: str, D_ms: int, rho_target: float,
                    eps_curve: list[tuple[float, float]],
                    B_t: int, q_base: float,
                    use_heavy_tail: bool) -> bool:
    ell_bar = SUBSTRATE_ELL[substrate]
    D_us = D_ms * 1000.0
    last_good = 1.0
    for i in range(201):
        phi = max(1e-3, 1.0 - i / 200)
        eps_phi = _eps_at(eps_curve, phi)
        eps_var = min(0.1 * eps_phi,
                      max(0.0, eps_phi * (1.0 - eps_phi) - 1e-9))
        if use_heavy_tail:
            bound = lemma2_truncated_heavy_tail_bound(
                epsilon_mean=eps_phi, epsilon_var=eps_var,
                ell_bar_us=ell_bar, B_t=float(B_t),
                deadline_us=D_us,
                sigma_residual_us=SIGMA_RES_US,
                escape_mass=q_base, base_cost_us=BASE_COST_US,
                rho_bar=RHO_BAR)
        else:
            bound = lemma2_correlation_aware_bound(
                epsilon_mean=eps_phi, epsilon_var=eps_var,
                ell_bar_us=ell_bar, B_t=float(B_t),
                deadline_us=D_us,
                sigma_residual_us=SIGMA_RES_US,
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
    ap.add_argument("--q_base", type=float, default=0.0067)
    args = ap.parse_args()

    eps_curve = _load_dense_eps(pathlib.Path(args.dense_eps))

    # Sweep grid. B_t=51 is the production working-set demand
    # (compute_top_k(B_full=512)) so the envelope also covers what
    # the deployed sizing rule actually evaluates.
    B_t_list = [1, 2, 4, 8, 16, 32, 51, 64]
    rho_list = [1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]

    cells = []
    for B_t in B_t_list:
        # Bt*q_base = the heavy-tail floor (cell can only accept if
        # rho > B_t*q_base; this is the bound's vacuum boundary).
        floor = B_t * args.q_base
        for rho in rho_list:
            actionable = sum(
                cell_actionable(sub, D, rho, eps_curve, B_t,
                                args.q_base, use_heavy_tail=True)
                for (sub, D, _) in SIZING_CELLS)
            cells.append({
                "B_t": B_t, "rho_target": rho,
                "btq_floor": floor,
                # R14: the count is `actionable` not `accepted`:
                # cell_actionable returns POLICY_FLOOR < phi_min <= 1,
                # so cells that collapse to the floor at relaxed rho
                # stop counting and the count can be non-monotone
                # in rho (intentional, see paper §VI.F).
                "fulltail_actionable": actionable,
                "above_floor": rho > floor,
            })

    # Also report bernstein-cor2 (no q_base) on the same grid for
    # contrast.
    bern_cells = []
    for B_t in B_t_list:
        for rho in rho_list:
            actionable = sum(
                cell_actionable(sub, D, rho, eps_curve, B_t,
                                args.q_base, use_heavy_tail=False)
                for (sub, D, _) in SIZING_CELLS)
            bern_cells.append({
                "B_t": B_t, "rho_target": rho,
                "bernstein_actionable": actionable,
            })

    # Chat-tier non-vacuous q* threshold: rho > B_t * q* gives a
    # necessary condition for full-tail-guard to be able to accept
    # at all. We tabulate q*(B_t) at rho=1e-2 (chat) so the operator
    # can read off the required escape-mass contract.
    qstar_chat = {}
    for B_t in B_t_list:
        qstar_chat[B_t] = 1e-2 / B_t

    out = {
        "q_base": args.q_base,
        "B_t_grid": B_t_list,
        "rho_grid": rho_list,
        "fulltail_envelope": cells,
        "bernstein_envelope": bern_cells,
        "sizing_grid_n": len(SIZING_CELLS),
        "chat_tier_qstar_threshold": qstar_chat,
        "definition_actionable": (
            "POLICY_FLOOR < phi_min <= 1 - 1e-3, i.e. the bound "
            "returns a sizing recommendation strictly above the "
            "policy correctness floor (0.125) and strictly inside "
            "the feasible region; non-monotone in rho when relaxed "
            "rho collapses cells to the floor"),
    }
    out_json = RESULTS / "full_tail_guard_envelope.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[envelope] wrote {out_json}")

    # Print a small grid (B_t row, rho column).
    print(f"[envelope] full-tail-guard actionable cells on 20-cell "
          f"sizing grid (q_base={args.q_base:.4f}):")
    print(f"{'B_t':>4s} {'B_t*q_base':>11s}  "
          + "  ".join([f"rho={r:.0e}" for r in rho_list]))
    cells_by_bt = {b: {r: 0 for r in rho_list} for b in B_t_list}
    for c in cells:
        cells_by_bt[c["B_t"]][c["rho_target"]] = c["fulltail_actionable"]
    for B_t in B_t_list:
        row = cells_by_bt[B_t]
        print(f"{B_t:>4d} {B_t*args.q_base:>11.4f}  "
              + "  ".join([f"{row[r]:>7d}/{len(SIZING_CELLS)}"
                           for r in rho_list]))
    print("[envelope] chat-tier (rho=1e-2) q* thresholds:")
    for B_t in B_t_list:
        ratio = args.q_base / qstar_chat[B_t]
        print(f"  B_t={B_t:>3d}: q* < {qstar_chat[B_t]:.2e} "
              f"({ratio:.1f}x tighter than calibrated 0.0067)")

    # TeX rendering: heatmap-style table. The full JSON keeps all
    # (B_t, rho) cells; the rendered table is the diagnostic subset
    # spanning the boundary, plus the production B_t=51 row so the
    # reviewer can see the production working-set demand side by
    # side with the diagnostic B_t=32 floor.
    tex_Bt = [8, 16, 32, 51, 64]
    tex_rho = [1e-2, 1e-1, 2.5e-1, 5e-1]
    lines = [
        "% Generated by full_tail_guard_envelope.py.",
        "% Each cell = full-tail-guard *actionable*/20 on the "
        "deployed-eps sizing grid",
        f"% (q_base={args.q_base}, rho_bar={RHO_BAR}). actionable",
        "% = POLICY_FLOOR < phi_min <= 1; relaxed rho can collapse",
        "% cells to the floor and lower the count (non-monotone in rho).",
        "% Bold = non-vacuous safe regime.",
        r"\begin{tabular}{rr" + "r" * len(tex_rho) + "}",
        r"\toprule",
        r"$B_t$ & $B_t q_\mathrm{base}$ & "
        + " & ".join([
            rf"$\rho{{=}}{r:g}$".replace("e-0", "{e}{-}").replace("e-", "{e}{-}")
            for r in tex_rho]) + r" \\",
        r"\midrule",
    ]
    for B_t in tex_Bt:
        row = cells_by_bt[B_t]
        cells_str = []
        for r in tex_rho:
            acc = row[r]
            if acc > 0:
                cells_str.append(rf"\textbf{{{acc}}}")
            else:
                cells_str.append(str(acc))
        lines.append(f"${B_t}$ & ${B_t*args.q_base:.3f}$ & "
                     + " & ".join(cells_str) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "full_tail_guard_envelope.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[envelope] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
