"""R12 bound-variant comparison table.

The R12 reviewer correctly pointed out that the paper has been
calling two different bound variants `operator-default`:

  - On the model-consistency map and on the §VI deployed-epsilon
    sizing rule, the published code uses the Bernstein-mixture +
    correlation-aware Cor.~2'' bound (no q_base truncation). We
    name this variant `bernstein-cor2`.

  - On the raw trace replay, the published code (R11) uses the
    Bernstein-mixture + truncated heavy-tail at q_base + Cor.~2''.
    We name this variant `full-tail-guard`.

R12 wants a single comparison table that, for each of the three
maps and each variant, reports

    accepted | unsafe accepts | conservative rejects

so the safety/usability tradeoff is explicit.

This script aggregates that table from the existing JSONs and
writes ``bound_variant_compare.{json,tex}`` to results/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from seer.timing.schedulability import (
    lemma2_correlation_aware_bound,
    lemma2_per_step_conditional_bound,
    lemma2_truncated_heavy_tail_bound,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"

# 20-cell deployed-eps sizing grid (matches sizing_deployed_eps.py).
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


POLICY_FLOOR = 0.125


def sizing_cell_actionable(
        substrate: str, D_ms: int, rho: float,
        eps_curve: list[tuple[float, float]],
        use_heavy_tail: bool,
        B_t: int = 32,
        q_base: float = 0.0067,
        rho_bar: float = 0.08,
        sigma_residual_us: float = 1203.0,
        base_cost_us: float = 33800.0,
        use_per_step: bool = False) -> tuple[bool, float]:
    """Return (actionable, phi_min) matching sizing_deployed_eps.py.

    actionable = POLICY_FLOOR < phi_min <= 1 - 1e-3, i.e. the bound
    returns a recommendation strictly above the policy correctness
    floor (0.125) and strictly inside the feasible region. Cells
    that collapse to the floor or are infeasible do not count.

    R13: B_t is explicit (default 32 = MIN_TOP_K floor) so the
    sizing-grid comparison matches the model-consistency map and
    raw-trace replay exactly.
    """
    ell_bar = SUBSTRATE_ELL[substrate]
    D_us = D_ms * 1000.0
    B_t = float(B_t)
    last_good = 1.0
    for i in range(201):
        phi = max(1e-3, 1.0 - i / 200)
        eps_phi = _eps_at(eps_curve, phi)
        eps_var = min(0.1 * eps_phi,
                      max(0.0, eps_phi * (1.0 - eps_phi) - 1e-9))
        if use_heavy_tail and use_per_step:
            bound = lemma2_per_step_conditional_bound(
                epsilon_mean=eps_phi, epsilon_var=eps_var,
                ell_bar_us=ell_bar, B_t=B_t, deadline_us=D_us,
                sigma_residual_us=sigma_residual_us,
                q_step=q_base, base_cost_us=base_cost_us,
                rho_bar=rho_bar)
        elif use_heavy_tail:
            bound = lemma2_truncated_heavy_tail_bound(
                epsilon_mean=eps_phi, epsilon_var=eps_var,
                ell_bar_us=ell_bar, B_t=B_t, deadline_us=D_us,
                sigma_residual_us=sigma_residual_us,
                escape_mass=q_base, base_cost_us=base_cost_us,
                rho_bar=rho_bar)
        else:
            bound = lemma2_correlation_aware_bound(
                epsilon_mean=eps_phi, epsilon_var=eps_var,
                ell_bar_us=ell_bar, B_t=B_t, deadline_us=D_us,
                sigma_residual_us=sigma_residual_us,
                base_cost_us=base_cost_us, rho_bar=rho_bar)
        if bound <= rho:
            last_good = phi
        else:
            break
    actionable = (POLICY_FLOOR < last_good <= 1.0 - 1e-3)
    return actionable, last_good


def summarise_map(name: str, B_t: int,
                  variant_to_tally: dict[str, dict[str, int]],
                  ) -> list[dict]:
    """Convert raw tally to (accepted, unsafe, conservative) rows."""
    rows = []
    for variant, t in variant_to_tally.items():
        accepted = t["green"] + t["black"]
        rows.append({
            "map": name,
            "B_t": B_t,
            "variant": variant,
            "n_cells": t["green"] + t["red"] + t["yellow"] + t["black"],
            "accepted": accepted,
            "unsafe_accepts": t["black"],
            "conservative_rejects": t["yellow"],
            "green": t["green"], "red": t["red"],
            "yellow": t["yellow"], "black": t["black"],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dense_eps",
        default=str(RESULTS / "dense_deployed_eps_summary.json"))
    ap.add_argument("--B_t", type=int, default=32,
                    help="Working-set demand; must match admission_map "
                         "and raw_trace_replay (R13).")
    args = ap.parse_args()

    rows: list[dict] = []

    # ----- Map 1: 216-cell model-consistency admission map -----
    am_path = RESULTS / "admission_map.json"
    am = json.loads(am_path.read_text())
    am_B_t = am.get("B_t")
    if am_B_t is not None and am_B_t != args.B_t:
        raise RuntimeError(
            f"admission_map.json was generated with B_t={am_B_t} but "
            f"bound_variant_compare expects B_t={args.B_t}; "
            f"re-run `python -m experiments.eC_bound_tightness.admission_map "
            f"--B_t {args.B_t}` first.")
    am_tallies = {r: am["regimes"][r]["tally"]
                  for r in am["regimes"]}
    rows.extend(summarise_map(
        "model-consistency (216 cells, parametric Beta-Binomial truth)",
        args.B_t, am_tallies))

    # ----- Map 2: 36-cell raw trace replay -----
    rt_path = RESULTS / "raw_trace_replay.json"
    rt = json.loads(rt_path.read_text())
    rt_B_t = rt.get("B_t")
    if rt_B_t is not None and rt_B_t != args.B_t:
        raise RuntimeError(
            f"raw_trace_replay.json was generated with B_t={rt_B_t} but "
            f"bound_variant_compare expects B_t={args.B_t}; "
            f"re-run `python -m experiments.eC_bound_tightness.raw_trace_replay "
            f"--B_t {args.B_t}` first.")
    rt_tallies = {r: {k: rt["regimes"][r]["tally"][k]
                       for k in ("green", "red", "yellow", "black")}
                  for r in rt["regimes"]}
    rows.extend(summarise_map(
        "raw-trace replay (36 cells, raw per-step FN-rate trace)",
        args.B_t, rt_tallies))

    # ----- Map 3: 20-cell deployed-eps sizing grid -----
    # No ground truth on this grid, so unsafe/conservative are N/A;
    # we only report the accepted count under each variant. R13:
    # the deployed sizing publishes Table II at the production
    # working-set demand B_t=51 (compute_top_k of B_full=512); we
    # report the same grid at the unified diagnostic B_t (args.B_t)
    # so the comparison column is consistent with the maps above.
    eps_curve = _load_dense_eps(pathlib.Path(args.dense_eps))
    for variant, use_ht, use_ps in (
            ("bernstein-cor2", False, False),
            ("full-tail-guard", True, False),
            ("full-tail-PS", True, True)):
        accepted = 0
        for substrate, D_ms, rho in SIZING_CELLS:
            ok, _ = sizing_cell_actionable(substrate, D_ms, rho,
                                            eps_curve,
                                            use_heavy_tail=use_ht,
                                            use_per_step=use_ps,
                                            B_t=args.B_t)
            if ok:
                accepted += 1
        rows.append({
            "map": "deployed-eps sizing (20 cells, no MC ground truth)",
            "B_t": args.B_t,
            "variant": variant,
            "n_cells": len(SIZING_CELLS),
            "accepted": accepted,
            "unsafe_accepts": None,
            "conservative_rejects": None,
        })

    out_json = RESULTS / "bound_variant_compare.json"
    out_json.write_text(json.dumps(
        {"B_t": args.B_t, "rows": rows}, indent=2))
    print(f"[bound-variant-compare] wrote {out_json}")
    for r in rows:
        print(f"  {r['map']:60s} | {r['variant']:18s}"
              f"  acc={r['accepted']:>3d}/{r['n_cells']:<3d}  "
              f"unsafe={r['unsafe_accepts']}  "
              f"cons={r['conservative_rejects']}")

    # TeX summary table.
    lines = [
        "% Generated by bound_variant_compare.py (R13).",
        "% accepted = bound returns a non-trivial sizing recommendation;",
        "% unsafe = green-overshoot vs simulator/trace ground truth (`black' class);",
        "% conservative = bound rejects but truth would accept (`yellow' class);",
        "% B_t = per-step working-set demand (paper-correct top-K formula,",
        "% seer.trace.schema.compute_top_k; MIN_TOP_K=32 floor).",
        "% N/A in the 20-cell sizing grid because no ground truth exists there.",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Map & Variant & $B_t$ & cells & accepted & unsafe & conservative \\",
        r"\midrule",
    ]
    cur_map = None
    # Drop fulltail-no-IO / fulltail-iid from the rendered TeX:
    # they are identical to full-tail-guard on both maps (full-tail-guard
    # is already 0-accept everywhere, so its ablations cannot
    # produce a different outcome). They remain in the JSON for
    # auditability.
    # Headline-only rendering: keep bernstein-cor2 (main actionable),
    # bernstein-no-IO (most-unsafe ablation), and full-tail-guard
    # (safety variant). bernstein-iid is in the JSON but omitted
    # from the rendered table to save space.
    visible_variants = {
        "bernstein-cor2", "bernstein-no-IO", "full-tail-guard",
        "full-tail-PS",
    }
    for r in rows:
        if r["variant"] not in visible_variants:
            continue
        # Short-form the map column.
        m = r["map"]
        if "model-consistency" in m:
            mshort = "216-cell model"
        elif "raw-trace" in m:
            mshort = "36-cell raw"
        else:
            mshort = "20-cell sizing"
        ua = (r"\textbf{" + str(r["unsafe_accepts"]) + "}"
              if (r["unsafe_accepts"] or 0) > 0
              else (str(r["unsafe_accepts"]) if r["unsafe_accepts"] is not None
                    else "n/a"))
        cr = (str(r["conservative_rejects"])
              if r["conservative_rejects"] is not None else "n/a")
        if mshort != cur_map and cur_map is not None:
            lines.append(r"\midrule")
        cur_map = mshort
        lines.append(
            f"{mshort} & \\texttt{{{r['variant']}}} & "
            f"{r['B_t']} & {r['n_cells']} & {r['accepted']} & {ua} & {cr} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "bound_variant_compare.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[bound-variant-compare] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
