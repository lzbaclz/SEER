"""Render the 4-color admission map figure (R9 RTSS strong-accept).

Three panels (one per substrate) on the (D, phi) plane, with cells
coloured green/red/yellow/black to show bound-vs-empirical agreement.

The figure is the load-bearing visual evidence that the framework's
admission control is self-consistent under the `bernstein-cor2`
variant and that the ablations (no-IO-cost, i.i.d.) produce unsafe
false-accept cells.

Reads admission_map.json; writes admission_map.pdf.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"
FIG_DIR = ROOT / "paper/figures"

COLOURS = {
    "green":  "#2ca02c",
    "red":    "#d62728",
    "yellow": "#f0c000",
    "black":  "#000000",
}
LEGEND = {
    "green":  "correct accept",
    "red":    "correct reject",
    "yellow": "conservative reject",
    "black":  "UNSAFE false accept",
}


def render(regime: str = "bernstein-cor2",
           summary_path: pathlib.Path | None = None,
           out_pdf: pathlib.Path | None = None) -> pathlib.Path:
    if summary_path is None:
        summary_path = RESULTS / "admission_map.json"
    if out_pdf is None:
        suffix = "" if regime == "bernstein-cor2" else "_" + regime.replace("-", "_")
        out_pdf = FIG_DIR / f"admission_map{suffix}.pdf"

    d = json.loads(summary_path.read_text())
    substrates = d["substrates"]
    D_grid = d["D_grid_ms"]
    phi_grid = d["phi_grid"]
    cells = d["regimes"][regime]["cells"]
    tally = d["regimes"][regime]["tally"]

    cells_by_sub: dict[str, dict[tuple[int, float], str]] = {
        s: {} for s in substrates}
    for c in cells:
        cells_by_sub[c["substrate"]][(c["D_ms"], c["phi"])] = c["class"]

    fig, axes = plt.subplots(
        1, len(substrates), figsize=(11.5, 3.6), sharey=True)
    if len(substrates) == 1:
        axes = [axes]

    for ax, substrate in zip(axes, substrates):
        for D_ms in D_grid:
            for phi in phi_grid:
                cls = cells_by_sub[substrate].get((D_ms, phi), "yellow")
                ax.scatter([D_ms], [phi], s=130,
                           c=COLOURS[cls],
                           edgecolors="black", linewidths=0.4,
                           marker="s")
        ax.set_xlabel(r"deadline $D$ (ms)")
        ax.set_title(substrate)
        ax.set_xticks([50, 100, 200, 300, 500])
        ax.set_xlim(D_grid[0] - 25, D_grid[-1] + 25)
        ax.set_yticks(phi_grid)
        ax.set_ylim(0.05, 0.85)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"logical attention budget $\phi$")

    # Legend with tallies as a banner row.
    legend_handles = []
    for k in ("green", "red", "yellow", "black"):
        legend_handles.append(plt.Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=COLOURS[k], markeredgecolor="black",
            markersize=8,
            label=f"{LEGEND[k]} ({tally[k]})"))
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.05), frameon=False,
               fontsize=9)
    fig.suptitle(f"Admission map [{regime}]: bound verdict vs "
                  f"empirical miss at $\\rho{{=}}10^{{-2}}$",
                  y=1.13, fontsize=10)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def main() -> int:
    for regime in ("bernstein-cor2", "bernstein-no-IO", "bernstein-iid",
                    "full-tail-guard", "fulltail-no-IO", "fulltail-iid"):
        p = render(regime=regime)
        print(f"[admission-map-fig] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
