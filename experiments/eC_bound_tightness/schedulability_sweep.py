"""Schedulability test sweep (Theorem 1 in §4 of the SEER paper).

Three example task sets covering the typical operator-decision regimes:

  Phi_1  single-tenant chat at moderate budget — should be schedulable
  Phi_2  mixed chat+doc with split phi allocation — schedulable
  Phi_3  stress: high arrival rate that pushes utilization past 1
         (forces a constraint binding so the test has rejection power)

Outputs ``results/schedulability_summary.json`` and
``results/schedulability.tex`` for the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

from seer.timing.schedulability_test import Stream, is_schedulable


def main() -> int:
    # All streams use the paper's measured parameters at A100 SXM4-80GB
    # with Llama-2-7B: eps_clean = 0.060 (from eC bound table),
    # ell_bar 200 us (DRAM tier datasheet),
    # sigma_residual 1500 us (paper's iofree pool sigma_clean).
    eps_clean = 0.060
    ell_bar = 200.0
    sigma_res = 1500.0
    B_full = 16.0

    chat_5e2 = Stream(
        name="chat",
        lambda_rps=10.0,
        deadline_us=50_000.0,
        miss_target=5e-2,
        phi=0.20,
        epsilon_clean=eps_clean,
        ell_bar_us=ell_bar,
        sigma_residual_us=sigma_res,
        B_full=B_full,
        c_io_us_at_phi=ell_bar * 0.20,
    )

    chat_mixed = Stream(
        name="chat (mixed)",
        lambda_rps=8.0,
        deadline_us=50_000.0,
        miss_target=5e-2,
        phi=0.20,
        epsilon_clean=eps_clean,
        ell_bar_us=ell_bar,
        sigma_residual_us=sigma_res,
        B_full=B_full,
        c_io_us_at_phi=ell_bar * 0.20,
    )
    doc_mixed = Stream(
        name="doc (mixed)",
        lambda_rps=2.0,
        deadline_us=1_000_000.0,
        miss_target=1.5e-2,
        phi=0.10,
        epsilon_clean=eps_clean,
        ell_bar_us=ell_bar,
        sigma_residual_us=sigma_res,
        B_full=B_full,
        c_io_us_at_phi=ell_bar * 0.50,
    )

    # Stress: tight miss target (rho=1e-3) at the same phi forces the
    # Lemma 2 (a) constraint to bind. Operator can recover schedulability
    # by raising phi (the inversion is in seer.timing.schedulability.
    # min_hbm_budget_for_slo).
    chat_stress = Stream(
        name="chat (tight rho)",
        lambda_rps=10.0,
        deadline_us=50_000.0,
        miss_target=1e-3,
        phi=0.20,
        epsilon_clean=eps_clean,
        ell_bar_us=ell_bar,
        sigma_residual_us=sigma_res,
        B_full=B_full,
        c_io_us_at_phi=ell_bar * 0.20,
    )

    task_sets = [
        ("\\Phi_1", "single-tenant chat", [chat_5e2]),
        ("\\Phi_2", "mixed chat+doc", [chat_mixed, doc_mixed]),
        ("\\Phi_3", "chat at tight $\\rho$", [chat_stress]),
    ]

    from seer.timing.schedulability_test import per_stream_bound

    out = []
    print("[schedulability] running 3 task sets...")
    for tag, label, streams in task_sets:
        verdict = is_schedulable(streams)
        # Always compute per-stream bounds + utilization for display,
        # even when verdict short-circuits on (a).
        bounds_display = [
            {"name": s.name,
             "bound": per_stream_bound(s),
             "rho": s.miss_target}
            for s in streams
        ]
        u_lap = sum(s.lambda_rps * (33.8e-6) for s in streams)
        u_io = sum(s.lambda_rps * (s.c_io_us_at_phi * 1e-6)
                   for s in streams)
        row = {
            "tag": tag,
            "label": label,
            "phi_summary": ", ".join(
                f"{s.name}={s.phi}" for s in streams
            ),
            "schedulable": bool(verdict.schedulable),
            "binding": verdict.binding,
            "u_lap": u_lap,
            "u_io": u_io,
            "per_stream_bounds": bounds_display,
            "streams": [s.__dict__ for s in streams],
        }
        out.append(row)
        verdict_str = "SCHEDULABLE" if verdict.schedulable else "infeasible"
        print(f"[schedulability] {tag} {label}: {verdict_str}"
              + (f"  ({verdict.binding})" if verdict.binding else ""))

    out_dir = Path("experiments/eC_bound_tightness/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "theorem_1": (
            "Schedulable(Phi) := for every stream i, Lemma 2 "
            "bound(phi_i, D_i, lambda_i) <= rho_i (per-task) AND "
            "sum_i lambda_i * C_LAP <= 1 (shared-LAP) AND "
            "sum_i lambda_i * E[C_IO_i] <= 1 - q_base (substrate)."
        ),
        "c_lap_us": 33.8,
        "q_base_calibrated": 0.0067,
        "results": out,
    }
    with open(out_dir / "schedulability_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[schedulability] wrote {out_dir / 'schedulability_summary.json'}")

    # Tex table.
    def _check(ok: bool) -> str:
        return r"$\checkmark$" if ok else r"$\times$"

    def _fmt_bound(rs):
        if not rs:
            return "---"
        worst = max(d["bound"] for d in rs)
        return f"${worst:.4f}$"

    lines = [
        "% Generated by experiments/eC_bound_tightness/schedulability_sweep.py",
        "% Theorem 1: schedulability(Phi) := per-task (a) AND shared-LAP (b)",
        "% AND substrate-bandwidth (c). All three columns must check.",
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Task set & Allocation & (a) Lemma 2 & (b) $U^\text{LAP}$ & "
        r"(c) $U^\text{IO}$ & Verdict & Binding \\",
        r"\midrule",
    ]
    for row in out:
        check_a = all(d["bound"] <= d["rho"]
                      for d in row["per_stream_bounds"])
        u_lap = row["u_lap"] or 0.0
        u_io = row["u_io"] or 0.0
        check_b = (u_lap <= 1.0)
        check_c = (u_io <= 1.0 - 0.0067)
        worst_bound = _fmt_bound(row["per_stream_bounds"])
        verdict = ("\\textbf{schedulable}" if row["schedulable"]
                   else "\\textbf{infeasible}")
        binding = row.get("binding") or ""
        binding_tex = (binding.replace("_", r"\_")
                              .replace("%", r"\%")
                              .replace(">", r"$>$")
                              .replace("<", r"$<$"))
        if not binding_tex:
            binding_tex = "---"
        phi_tex = (row["phi_summary"].replace("_", " ")
                                     .replace("phi", r"$\phi$"))
        lines.append(
            f"${row['tag']}$ & "
            f"{phi_tex} & "
            f"{_check(check_a)} ({worst_bound}) & "
            f"{_check(check_b)} (${u_lap:.4f}$) & "
            f"{_check(check_c)} (${u_io:.4f}$) & "
            f"{verdict} & {binding_tex} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(out_dir / "schedulability.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[schedulability] wrote {out_dir / 'schedulability.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
