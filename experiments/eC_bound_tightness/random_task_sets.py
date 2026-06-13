"""Random multi-stream schedulability acceptance-ratio simulation.

R1-W3 / R2-W3 follow-through (third review round): the previous
Theorem 1 evaluation only ran on three hand-picked task sets. RTSS
standard practice is acceptance-ratio plots on 100--200 randomly
generated task sets across a (lambda, D, rho, phi) grid.

Protocol:
- For each total utilisation U in {0.1, 0.2, ..., 1.5} (15 op-points),
  generate `n_per_util` random task sets, each containing
  `n_streams` streams whose arrival rates lambda_i sum to U / C_LAP.
- Each stream draws (D, rho, phi) uniformly from a grid mixing chat
  and doc class operating points.
- Run the sufficient (a)+(b)+(c) schedulability test and record:
  (i) acceptance fraction, (ii) which constraint binds when rejected.

Output:
- `random_task_sets_summary.json` with acceptance fraction per U and
  binding-constraint distribution.
- `random_task_sets.tex` summarising the acceptance ratio curve and
  the binding-constraint breakdown.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from seer.timing.schedulability_test import (
    Stream,
    is_schedulable,
    per_stream_bound,
)

CHAT_PROFILE = dict(
    deadline_us_range=(50_000.0, 200_000.0),
    miss_target_range=(1e-3, 5e-2),
    phi_range=(0.10, 0.30),
)
DOC_PROFILE = dict(
    deadline_us_range=(500_000.0, 2_000_000.0),
    miss_target_range=(5e-3, 5e-2),
    phi_range=(0.05, 0.20),
)


def _sample_stream(rng: random.Random, name: str, lam_rps: float,
                    profile: dict) -> Stream:
    D = rng.uniform(*profile["deadline_us_range"])
    rho = rng.uniform(*profile["miss_target_range"])
    phi = rng.uniform(*profile["phi_range"])
    return Stream(
        name=name,
        lambda_rps=lam_rps,
        deadline_us=D,
        miss_target=rho,
        phi=phi,
        epsilon_clean=0.060,
        ell_bar_us=200.0,
        sigma_residual_us=1500.0,
        B_full=16.0,
        c_io_us_at_phi=200.0 * phi,
    )


def _sample_task_set(rng: random.Random, target_u_lap: float,
                     n_streams: int, c_lap_us: float = 33.8) -> list[Stream]:
    # Distribute total arrival rate evenly across streams; mix chat/doc
    # 50/50.
    total_lambda = target_u_lap / (c_lap_us * 1e-6)
    base_lam = total_lambda / max(1, n_streams)
    # Add ±25% jitter so streams are heterogeneous.
    streams = []
    for i in range(n_streams):
        jitter = rng.uniform(0.75, 1.25)
        lam = base_lam * jitter
        profile = CHAT_PROFILE if rng.random() < 0.5 else DOC_PROFILE
        name = f"s{i}_{'chat' if profile is CHAT_PROFILE else 'doc'}"
        streams.append(_sample_stream(rng, name, lam, profile))
    return streams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_util", type=int, default=200,
                    help="task sets per U op-point")
    ap.add_argument("--utils", nargs="+", type=float,
                    default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                             0.70, 0.80, 0.90, 1.00, 1.10, 1.20,
                             1.30, 1.40, 1.50])
    ap.add_argument("--n_streams_grid", nargs="+", type=int,
                    default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_dec_batched", type=int, default=2,
                    help="batched-decode amortisation factor for "
                         "Corollary 1' (R4-W4); default 2 = a 2-way "
                         "fused decode batch")
    ap.add_argument("--out_dir",
                    default="experiments/eC_bound_tightness/results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows: list[dict] = []
    binding_total = {"(a)": 0, "(b)": 0, "(c)": 0}
    binding_per_u: dict[float, dict[str, int]] = {}
    # R4-W4 (Corollary 1'): batched-decode amortisation factor.
    k_dec_batched: int = int(getattr(args, "k_dec_batched", 2))

    print(f"[random-task-sets] {args.n_per_util} task sets per U over "
          f"{len(args.utils)} U values, n_streams in "
          f"{args.n_streams_grid}; batched-aware k_dec={k_dec_batched}")

    def _is_a_feasible(streams: list[Stream]) -> bool:
        """Constraint (a): every stream individually meets Lemma 2's
        bound at the calibrated q_base. This is the per-stream feasibility
        of the task set independent of system utilisation."""
        return all(
            per_stream_bound(s, use_truncated=True, q_base=0.0067)
            <= s.miss_target
            for s in streams
        )

    for u in args.utils:
        binding_per_u[u] = {"(a)": 0, "(b)": 0, "(c)": 0}
        n_accept = 0
        n_total = 0
        n_a_feasible = 0
        n_accept_given_a = 0
        n_accept_given_a_batched = 0  # R4-W4: Corollary 1'
        for n_streams in args.n_streams_grid:
            for _ in range(args.n_per_util):
                streams = _sample_task_set(rng, target_u_lap=u,
                                           n_streams=n_streams)
                a_feas = _is_a_feasible(streams)
                v = is_schedulable(streams)
                v_batched = is_schedulable(streams, k_dec=k_dec_batched)
                n_total += 1
                if a_feas:
                    n_a_feasible += 1
                if v.schedulable:
                    n_accept += 1
                    if a_feas:
                        n_accept_given_a += 1
                else:
                    if v.binding.startswith("(a)"):
                        bkey = "(a)"
                    elif v.binding.startswith("(b)"):
                        bkey = "(b)"
                    elif v.binding.startswith("(c)"):
                        bkey = "(c)"
                    else:
                        bkey = "(a)"
                    binding_per_u[u][bkey] += 1
                    binding_total[bkey] += 1
                if v_batched.schedulable and a_feas:
                    n_accept_given_a_batched += 1
        accept_frac = n_accept / max(1, n_total)
        accept_given_a = (n_accept_given_a / max(1, n_a_feasible))
        accept_given_a_batched = (n_accept_given_a_batched /
                                  max(1, n_a_feasible))
        rows.append({
            "u_lap_target": u,
            "n_total": n_total,
            "n_accept": n_accept,
            "accept_frac": accept_frac,
            "n_a_feasible": n_a_feasible,
            "n_accept_given_a_feasible": n_accept_given_a,
            "accept_frac_given_a_feasible": accept_given_a,
            "n_accept_given_a_feasible_batched": n_accept_given_a_batched,
            "accept_frac_given_a_feasible_batched": accept_given_a_batched,
            "k_dec_batched": k_dec_batched,
            "binding_breakdown": dict(binding_per_u[u]),
        })
        print(f"[random-task-sets] U={u:.2f}: raw "
              f"{100 * accept_frac:5.1f}% | (a)-cond "
              f"{100 * accept_given_a:5.1f}% | (a)-cond+batched(k={k_dec_batched}) "
              f"{100 * accept_given_a_batched:5.1f}% "
              f"(n_a_feas={n_a_feasible}); reject: {binding_per_u[u]}")

    summary = {
        "n_total_task_sets": sum(r["n_total"] for r in rows),
        "binding_total": binding_total,
        "rows": rows,
    }
    out_json = out_dir / "random_task_sets_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[random-task-sets] wrote {out_json}")

    # TeX table.
    lines = [
        "% Generated by experiments/eC_bound_tightness/random_task_sets.py",
        "% Acceptance-ratio of the sufficient schedulability test on",
        "% random multi-stream task sets. R2-W3 follow-through.",
        "% R3-W1 follow-through: report both 'raw' acceptance and",
        "% 'conditional on (a)-feasibility' acceptance, the standard RT",
        "% acceptance-ratio form.",
        "% R4-W4 follow-through: also report Corollary 1' batched-aware",
        "% acceptance (LHS of (b) divided by k_dec). k_dec=1 is the",
        "% conservative sequential-server upper bound; k_dec>=2 is the",
        "% operator-side refinement under co-scheduled batched decode.",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$U_\text{LAP}$ & $n$ & raw (\%) & "
        r"(a)-cond (\%) & "
        rf"(a)-cond, $k_\text{{dec}}{{=}}{k_dec_batched}$ (\%) & "
        r"rej (a) & rej (b) & rej (c) \\",
        r"\midrule",
    ]
    for r in rows:
        b = r["binding_breakdown"]
        lines.append(
            f"${r['u_lap_target']:.2f}$ & ${r['n_total']}$ & "
            f"${100 * r['accept_frac']:.1f}\\%$ & "
            f"${100 * r['accept_frac_given_a_feasible']:.1f}\\%$ & "
            f"${100 * r['accept_frac_given_a_feasible_batched']:.1f}\\%$ & "
            f"{b['(a)']} & {b['(b)']} & {b['(c)']} \\\\"
        )
    lines.extend([
        r"\midrule",
        f"\\multicolumn{{8}}{{l}}{{\\textit{{Total task sets: "
        f"${sum(r['n_total'] for r in rows)}$; rejection-binding mix: "
        f"(a)~${binding_total['(a)']}$, (b)~${binding_total['(b)']}$, "
        f"(c)~${binding_total['(c)']}$. (a)-conditional is the standard "
        f"RT form; $k_\\text{{dec}}{{=}}{k_dec_batched}$ is "
        f"Corollary~$1'$ (batched-aware).}}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_tex = out_dir / "random_task_sets.tex"
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[random-task-sets] wrote {out_tex}")

    # R3-W1: also emit a matplotlib PDF figure (acceptance-ratio plot).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Us = [r["u_lap_target"] for r in rows]
        accept_raw = [100 * r["accept_frac"] for r in rows]
        accept_cond = [100 * r["accept_frac_given_a_feasible"] for r in rows]
        accept_cond_batched = [
            100 * r["accept_frac_given_a_feasible_batched"] for r in rows
        ]
        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        ax.plot(Us, accept_cond_batched, "^-", color="#009E73",
                label=rf"(a)-cond, $k_{{\mathrm{{dec}}}}{{=}}{k_dec_batched}$ "
                      r"(Corollary~$1'$)")
        ax.plot(Us, accept_cond, "o-", color="#0072B2",
                label=r"(a)-cond, $k_{\mathrm{dec}}{=}1$ (conservative)")
        ax.plot(Us, accept_raw, "s--", color="#888888",
                label="raw (incl.\\ (a)-infeasible)")
        ax.set_xlabel(r"$U_{\mathrm{LAP}}$")
        ax.set_ylabel(r"Acceptance fraction (\%)")
        ax.set_ylim(-2, 102)
        ax.set_xlim(0, 1.55)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower left",
                  frameon=False, handlelength=1.6,
                  handletextpad=0.4, labelspacing=0.25)
        fig.tight_layout()
        fig_path = out_dir / "random_task_sets.pdf"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"[random-task-sets] wrote {fig_path}")
    except Exception as e:
        print(f"[random-task-sets] skipped matplotlib figure: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
