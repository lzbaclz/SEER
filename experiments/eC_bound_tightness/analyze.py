"""eC analyze — pessimism heatmap of Theorem 1 bound vs measured miss ratio.

Inputs
------
1. ``--grid`` : analytical-only grid produced by
   ``experiments/eC_bound_tightness/sweep.py`` (CSV with columns
   ``epsilon, deadline_ms, budget, bound, ...``).
2. ``--measured`` : (optional) one or more JSON files produced by
   ``seer.eval.runner``; we read ``slo.threshold_ms``, ``hbm_budget``,
   and ``miss_ratio`` and join with the grid on
   ``(deadline_ms, budget)`` plus a chosen ``epsilon`` band.

Outputs
-------
* ``<dir>/pessimism.csv`` — bound, measured, ratio per joined row.
* ``<dir>/pessimism.pdf`` — paper-ready heatmap of the analytical
  bound (log10) on a (ε, D) grid for the budget slice given by
  ``--budget_filter`` (default 0.20). Uses the Patrick / ADSLab
  grayscale + deep-orange palette: cells are ``Greys`` (B&W-friendly)
  with measured points overlaid as deep-orange ringed scatter dots.
* ``<dir>/pessimism_summary.json`` — min / median / max / IQR of the
  ratio plus the parameter band over which they were computed.

If no measured points are provided we still emit a *bound-only*
heatmap so the reader can see the analytical structure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seer import figstyle  # noqa: E402


def _load_grid(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "epsilon": float(r["epsilon"]),
                "deadline_ms": float(r["deadline_ms"]),
                "budget": float(r["budget"]),
                "bound": float(r["bound"]),
            })
    return rows


def _load_measured(paths: Iterable[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        slo = d.get("slo") or {}
        out.append({
            "policy": d.get("policy"),
            "deadline_ms": float(slo.get("threshold_ms", 0.0)),
            "budget": float(d.get("hbm_budget", 0.0)),
            "miss_ratio": float(d.get("miss_ratio", 0.0)),
            "source": Path(p).name,
        })
    return out


def _join_pessimism(
    grid: list[dict], measured: list[dict], eps_band: tuple[float, float],
) -> list[dict]:
    out = []
    eps_lo, eps_hi = eps_band
    for m in measured:
        cands = [
            g for g in grid
            if eps_lo <= g["epsilon"] <= eps_hi
            and abs(g["deadline_ms"] - m["deadline_ms"]) < 1e-6
            and abs(g["budget"] - m["budget"]) < 1e-6
        ]
        for g in cands:
            ratio = (g["bound"] / m["miss_ratio"]) if m["miss_ratio"] > 0 else math.inf
            out.append({
                **m,
                "epsilon": g["epsilon"],
                "bound": g["bound"],
                "ratio": ratio,
            })
    return out


def _summarize_ratios(rows: list[dict]) -> dict:
    rs = [r["ratio"] for r in rows if math.isfinite(r["ratio"])]
    if not rs:
        return {"n": 0}
    rs_sorted = sorted(rs)

    def pct(p):
        k = (len(rs_sorted) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(rs_sorted) - 1)
        frac = k - lo
        return rs_sorted[lo] * (1 - frac) + rs_sorted[hi] * frac

    return {
        "n": len(rs),
        "min": rs_sorted[0],
        "median": pct(50),
        "max": rs_sorted[-1],
        "p25": pct(25),
        "p75": pct(75),
    }


def _plot_heatmap_paper(grid: list[dict], joined: list[dict], out_dir: Path,
                        budget_filter: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figstyle.apply()

    grid_f = [g for g in grid if abs(g["budget"] - budget_filter) < 1e-6]
    if not grid_f:
        # Fall back to closest budget present in the grid.
        budgets = sorted({g["budget"] for g in grid})
        if not budgets:
            print("[eC] heatmap skipped: empty grid")
            return
        budget_filter = min(budgets, key=lambda b: abs(b - budget_filter))
        grid_f = [g for g in grid if abs(g["budget"] - budget_filter) < 1e-6]
        print(f"[eC] no exact budget match, using closest = {budget_filter}")

    epsilons = sorted({g["epsilon"] for g in grid_f})
    deadlines = sorted({g["deadline_ms"] for g in grid_f})
    Z = np.full((len(deadlines), len(epsilons)), np.nan)
    for g in grid_f:
        i = deadlines.index(g["deadline_ms"])
        j = epsilons.index(g["epsilon"])
        Z[i, j] = math.log10(max(g["bound"], 1e-300))

    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    # Discrete grayscale cells (no smoothing) so reviewers can read each
    # (ε, D) value exactly. We use ``pcolormesh`` with explicit cell edges
    # rather than ``imshow(extent=…)`` so the axis ticks land mid-cell.
    eps_edges = _cell_edges(epsilons)
    d_edges = _cell_edges(deadlines)
    pcm = ax.pcolormesh(
        eps_edges, d_edges, Z,
        cmap="Greys", vmin=Z[np.isfinite(Z)].min() if np.any(np.isfinite(Z)) else -300, vmax=0,
        shading="flat", linewidth=0.0, edgecolor="none",
    )

    # Annotate cell text (log10) only where finite + > -100; otherwise the
    # cell is "vacuous" and we leave it solid black.
    for i, d in enumerate(deadlines):
        for j, e in enumerate(epsilons):
            v = Z[i, j]
            if not np.isfinite(v):
                continue
            if v > -10:
                txt = f"{v:.1f}"
            elif v > -50:
                txt = f"{v:.0f}"
            else:
                continue
            color = "white" if v < -3 else "black"
            ax.text(e, d, txt, ha="center", va="center",
                    fontsize=6.5, color=color)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(r"$\log_{10}\,\Pr(C_t > D)$ (Theorem 1 bound)", fontsize=8)

    # Overlay measured points (if any joined): scatter at (eps, D) with
    # deep-orange ring + grayscale fill encoding the pessimism ratio.
    if joined:
        xs = [r["epsilon"] for r in joined]
        ys = [r["deadline_ms"] for r in joined]
        ax.scatter(
            xs, ys,
            s=42, c="white",
            edgecolor=figstyle.OURS, linewidths=1.1, zorder=6,
            label="Measured SEER",
        )
        # legend with single proxy entry
        ax.legend(loc="upper right", fontsize=7.0, frameon=True,
                  edgecolor="black", framealpha=0.95).get_frame().set_linewidth(0.6)

    ax.set_xlabel(r"LAP false-negative rate $\epsilon$")
    ax.set_ylabel("Deadline $D$ (ms)")
    figstyle.add_box_spines(ax)
    ax.grid(False)

    out_path = figstyle.save_paper(fig, "pessimism", out_dir)
    print(f"[eC] heatmap → {out_path}")


def _cell_edges(centers: list[float]) -> list[float]:
    """Convert a list of cell centers to edges for ``pcolormesh``."""
    if len(centers) == 1:
        c = centers[0]
        return [c - 0.5, c + 0.5]
    edges = [centers[0] - 0.5 * (centers[1] - centers[0])]
    for a, b in zip(centers[:-1], centers[1:], strict=False):
        edges.append(0.5 * (a + b))
    edges.append(centers[-1] + 0.5 * (centers[-1] - centers[-2]))
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid.csv",
                    help="CSV produced by sweep.py")
    ap.add_argument("--measured", nargs="*", default=[],
                    help="JSON files from seer.eval.runner; optional")
    ap.add_argument("--eps_band", nargs=2, type=float, default=[0.05, 0.10],
                    metavar=("LO", "HI"),
                    help="ε band over which to match measured points to grid")
    ap.add_argument("--budget_filter", type=float, default=0.20,
                    help="Restrict heatmap to a single budget slice "
                         "(default 0.20 to match the eA headline budget).")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = _load_grid(args.grid)
    print(f"[eC] loaded {len(grid)} grid points from {args.grid}")
    measured = _load_measured(args.measured)
    print(f"[eC] loaded {len(measured)} measured points")

    joined = _join_pessimism(grid, measured, tuple(args.eps_band))  # type: ignore[arg-type]
    if joined:
        with open(out_dir / "pessimism.csv", "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["policy", "source", "epsilon", "deadline_ms",
                            "budget", "bound", "miss_ratio", "ratio"],
            )
            w.writeheader()
            w.writerows(joined)
        print(f"[eC] wrote pessimism.csv ({len(joined)} rows)")

    summary = {
        "n_grid": len(grid),
        "n_measured": len(measured),
        "n_joined": len(joined),
        "eps_band": list(args.eps_band),
        "budget_filter": args.budget_filter,
        "ratio_stats": _summarize_ratios(joined),
    }
    with open(out_dir / "pessimism_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eC] wrote pessimism_summary.json: {summary['ratio_stats']}")

    _plot_heatmap_paper(grid, joined, out_dir, args.budget_filter)


if __name__ == "__main__":
    main()
