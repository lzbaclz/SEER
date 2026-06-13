"""Render eE_cross_card.pdf: Card-0 vs Card-1 contended vs Card-1 quiescent.

Patrick/ADSLab style grouped bar over the four reported tail percentiles
(P50, P99, P99.9, Max). The story: Card-1 measured ~80% slower under
background load (`card1_contended`), but with the GPU quiesced the gap
disappears (`card1_unlocked`). All three series are TinyMLP+TRT @ B=4096.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seer import figstyle  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = _REPO / "paper" / "figures" / "eE_cross_card.pdf"

FILES = {
    "Card 0 (reference)":         "lap_wcet_A100_tiny_mlp_trt_b4096.json",
    "Card 1 (contended)":         "lap_wcet_A100_card1_contended_2026_05_08.json",
    "Card 1 (quiesced)":          "lap_wcet_A100_card1_unlocked_20260509_tiny_mlp_trt_b4096.json",
}
PCTS = [("P50", "p50_us"), ("P99", "p99_us"), ("P99.9", "p999_us"), ("Max", "max_us")]


def main() -> None:
    figstyle.apply()
    data = {label: json.load(open(RESULTS / fn)) for label, fn in FILES.items()}
    series = list(data.keys())
    values = np.array([[data[s][k] for _, k in PCTS] for s in series])

    n_series, n_pcts = values.shape
    width = 0.82 / n_series
    x = np.arange(n_pcts)

    fig, ax = plt.subplots(figsize=(5.6, 2.5))

    GRAYS = figstyle.GRAYS
    OURS = figstyle.OURS
    HATCHES = figstyle.HATCHES
    series_styles = [
        (GRAYS[3], HATCHES[1]),  # Card 0 reference: medium gray
        (GRAYS[4], HATCHES[3]),  # Card 1 contended: dark gray with xx
        (OURS,     HATCHES[0]),  # Card 1 quiesced: accent (matches Card 0)
    ]

    for i, name in enumerate(series):
        color, hatch = series_styles[i]
        bars = ax.bar(
            x + (i - (n_series - 1) / 2) * width,
            values[i],
            width=width,
            label=name,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
        )
        for j, b in enumerate(bars):
            ax.annotate(
                f"{values[i, j]:.1f}",
                xy=(b.get_x() + b.get_width() / 2, values[i, j]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([p for p, _ in PCTS])
    ax.set_xlabel("LAP inference latency percentile (TinyMLP + TRT, B = 4096)")
    ax.set_ylabel(r"Latency ($\mu$s)")
    ax.set_ylim(0, values.max() * 1.18)
    ax.legend(
        loc="upper left",
        ncol=1,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[eE_cross_card] wrote {OUT}")


if __name__ == "__main__":
    main()
