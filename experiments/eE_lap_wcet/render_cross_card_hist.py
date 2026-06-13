"""Render eE_cross_card_hist.pdf: true latency histogram (Card 0 vs Card 1
contended vs Card 1 quiesced), once raw samples are available.

Looks for `*_samples.json` siblings of the standard percentile JSONs;
falls back to a SystemExit message if any sample file is missing.

Usage:
    python experiments/eE_lap_wcet/render_cross_card_hist.py

The percentile bar chart in `render_cross_card.py` is the supported
figure when raw samples are missing; this histogram is the upgrade.
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
OUT = _REPO / "paper" / "figures" / "eE_cross_card_hist.pdf"

SERIES = [
    ("Card 0 (reference)",  "lap_wcet_A100_tiny_mlp_trt_b4096_samples.json"),
    ("Card 1 (contended)",  "lap_wcet_A100_card1_contended_2026_05_08_samples.json"),
    ("Card 1 (quiesced)",   "lap_wcet_A100_card1_unlocked_20260509_tiny_mlp_trt_b4096_samples.json"),
]


def main() -> None:
    figstyle.apply()

    series = []
    for label, fn in SERIES:
        p = RESULTS / fn
        if not p.exists():
            print(f"[eE_cross_card_hist] missing {p} — re-run smoke_tensorrt.sh "
                  f"with SAVE_SAMPLES=1 for the corresponding card. "
                  f"Falling back to render_cross_card.py (percentile bar).")
            return
        d = json.load(open(p))
        series.append((label, np.asarray(d["samples_us"], dtype=float)))

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    bins = np.linspace(20, 80, 61)

    GRAYS = figstyle.GRAYS
    OURS = figstyle.OURS
    styles = [
        dict(color=GRAYS[3], linestyle="-",  linewidth=1.4, alpha=0.85),
        dict(color=GRAYS[4], linestyle="--", linewidth=1.4, alpha=0.85),
        dict(color=OURS,     linestyle="-",  linewidth=2.0, alpha=0.95),
    ]

    for (label, samples), style in zip(series, styles):
        ax.hist(samples, bins=bins, histtype="step", label=label, **style)

    ax.set_xlabel(r"LAP forward-pass latency ($\mu$s, TinyMLP + TRT, $B = 4096$)")
    ax.set_ylabel("Count")
    ax.set_xlim(20, 80)
    ax.legend(
        loc="upper right",
        ncol=1,
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[eE_cross_card_hist] wrote {OUT}")


if __name__ == "__main__":
    main()
