"""Paper-ready figure style (Patrick Lee / ADSLab convention).

This module exposes a single ``apply()`` entry point that mutates
``matplotlib.rcParams`` to the conventions used in the FAST/ATC/PVLDB
papers from Patrick P.\\ C. Lee's ADSLab group (CUHK + HUST):

* Times serif font, embeddable Type-42 fonts.
* 4-side spines, outward ticks, horizontal-only dashed grid in light gray.
* Single-color palette (5 grays) plus one accent color reserved for the
  "ours" curve (deep orange ``#D55E00`` — colorblind-safe and prints to
  a deep gray on B&W paper).
* Hatch list to keep grouped bars readable on B&W printouts.

Helper constants
----------------
``GRAYS``   five-tone gray ramp (white through near-black).
``OURS``    accent color, reserved for our system on every plot.
``HATCHES`` ordered hatch ramp; index ``0`` is empty so the
            most-emphasised bar can stay solid.

Helper functions
----------------
``policy_style(policy)`` returns the canonical
``(color, linestyle, marker, hatch, linewidth)`` tuple for a SEER policy
name so that all six §6 figures use the same visual mapping.

``annotate_topbar(ax, x, y, text, ...)``
   small wrapper to write a number above a bar tip in the
   ADSLab convention (small serif, deep orange when "ours").

``save_paper(fig, name, out_dir)``
   save both PDF (vector, paper) and PNG (preview) with the conventional
   tight padding.

Style reference: ``Paper_style_extract/patrick/09_figure_style.md``.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

GRAYS = ["#FFFFFF", "#D9D9D9", "#A6A6A6", "#737373", "#404040"]
OURS = "#D55E00"
HATCHES = ["", "//", "\\\\", "xx", "..", "++", "**", "oo"]

POLICY_ORDER = [
    "full",
    "recency",
    "streaming",
    "snapkv",
    "quest",
    "h2o",
    "seer",
]

POLICY_PRETTY = {
    "full": "Full (no eviction)",
    "recency": "Recency",
    "streaming": "StreamingLLM",
    "snapkv": "SnapKV",
    "quest": "Quest",
    "h2o": "H2O",
    "seer": "SEER",
}

_POLICY_STYLE_TABLE = {
    "full":      dict(color=GRAYS[4], linestyle=(0, (1, 1)),         marker="x", hatch=HATCHES[1], lw=1.1, ms=4.6),
    "recency":   dict(color=GRAYS[3], linestyle=":",                  marker="s", hatch=HATCHES[2], lw=1.1, ms=4.6),
    "streaming": dict(color=GRAYS[3], linestyle="--",                 marker="^", hatch=HATCHES[3], lw=1.2, ms=4.6),
    "snapkv":    dict(color=GRAYS[2], linestyle="-.",                 marker="D", hatch=HATCHES[4], lw=1.2, ms=4.6),
    "quest":     dict(color=GRAYS[2], linestyle=(0, (3, 1, 1, 1)),    marker="P", hatch=HATCHES[5], lw=1.2, ms=4.6),
    "h2o":       dict(color=GRAYS[4], linestyle=(0, (5, 1)),          marker="v", hatch=HATCHES[6], lw=1.2, ms=4.6),
    "seer":      dict(color=OURS,     linestyle="-",                  marker="o", hatch=HATCHES[0], lw=1.9, ms=5.8),
}


def apply() -> None:
    """Install Patrick / ADSLab rcParams.

    Call once at the top of an analyze script before creating any
    figures. Idempotent — calling twice is harmless.
    """
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            # Match math symbols to the Times serif text — STIX is the
            # Times-compatible math font; without this matplotlib falls
            # back to Computer Modern for $C_t$, $D$, $\phi$, etc., and
            # the body text + math get visibly different shapes.
            "mathtext.fontset": "stix",
            "mathtext.rm":      "STIXGeneral",
            "mathtext.it":      "STIXGeneral:italic",
            "mathtext.bf":      "STIXGeneral:bold",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "grid.color": "#999999",
            # Frameless legend, sharp corners (GogetaFS / Patrick
            # reference). fancybox=False keeps text bboxes (Fig 5
            # stats panel) corners consistent with legends.
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "legend.framealpha": 0.95,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 150,
        }
    )


def policy_style(policy: str) -> dict:
    """Return the (color, linestyle, marker, hatch, lw, ms) dict for a policy.

    Unknown policies fall back to a neutral dark-gray solid line so that
    figures don't blow up if the policy set drifts; do add an explicit
    entry for any new system before final submission.
    """
    p = (policy or "").lower()
    if p in _POLICY_STYLE_TABLE:
        return dict(_POLICY_STYLE_TABLE[p])
    return dict(color=GRAYS[4], linestyle="-", marker="x",
                hatch=HATCHES[3], lw=1.1, ms=4.6)


def policy_label(policy: str) -> str:
    p = (policy or "").lower()
    return POLICY_PRETTY.get(p, policy or "?")


def policy_sort_key(policy: str) -> int:
    """Order policies the way Patrick's caption reads: worst -> best -> ours.

    SEER is always the *last* legend entry so the visual flow is
    "baselines... and finally Seer (ours)". Unknown policies sort to the
    front so they don't accidentally bury the system being defended.
    """
    p = (policy or "").lower()
    if p in POLICY_ORDER:
        return POLICY_ORDER.index(p)
    return -1


def annotate_topbar(ax, x: float, y: float, text: str, *,
                    is_ours: bool = False, dy_pt: float = 3.0) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, dy_pt),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=OURS if is_ours else "#404040",
        fontweight="bold" if is_ours else "normal",
    )


def save_paper(fig: plt.Figure, name: str, out_dir: Path | str,
               also_png: bool = True) -> Path:
    """Save *fig* as PDF (paper) and optionally a 300dpi PNG (preview)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / f"{name}.pdf"
    fig.savefig(pdf_path)
    if also_png:
        fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    return pdf_path


def add_box_spines(ax) -> None:
    """Re-enable all four spines (matplotlib 3.x defaults already do this,
    but this keeps the call site explicit and matches the §5 style spec)."""
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)


def sorted_by_policy(items: Iterable, key=lambda x: x):
    """Sort an iterable using ``policy_sort_key`` on a key extractor.

    Used by the analyze scripts so that legends always render
    SEER last (= "ours" reading order).
    """
    return sorted(items, key=lambda x: policy_sort_key(key(x)))


__all__ = [
    "apply",
    "policy_style",
    "policy_label",
    "policy_sort_key",
    "annotate_topbar",
    "save_paper",
    "add_box_spines",
    "sorted_by_policy",
    "GRAYS",
    "OURS",
    "HATCHES",
    "POLICY_ORDER",
    "POLICY_PRETTY",
]
