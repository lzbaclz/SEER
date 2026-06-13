"""Aggregate eA results: tail-latency CDF + summary table + schedulability.

Reads every ``*.json`` file produced by :mod:`seer.eval.runner` under
the given results directory, then:

* Prints a markdown summary table to stdout (P50 / P99 / P999 / miss%
  / F1 / Lemma 2 bound / pessimism).
* Writes ``cdf.pdf`` next to the results — one curve per policy at the
  --plot_budget HBM slice (default 0.20). The CDF is rendered in the
  Patrick / ADSLab paper style (Times-serif, 4-side spines, log-x,
  baselines in grayscale dashed lines, SEER in deep-orange solid),
  with a vertical SLO marker and inline P99 / P999 annotations on the
  SEER curve.
* Writes ``table.csv`` and ``table.tex`` (paper-ready LaTeX).
* Writes ``schedulability.json`` — single-point sanity check of
  Lemma 2 pessimism alongside the full eC sweep.
"""
from __future__ import annotations

import argparse
import csv
import json

# Allow ``python experiments/eA_tail_latency/analyze.py`` from any cwd.
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seer import figstyle  # noqa: E402  (sys.path tweak above)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs_sorted) - 1)
    frac = k - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--plot_budget", type=float, default=0.20,
                    help="HBM budget slice to draw the CDF for (default 0.20).")
    ap.add_argument("--slo_ms", type=float, default=50.0,
                    help="SLO line drawn on the CDF (default 50 ms).")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    skip_names = {"schedulability.json", "pessimism.csv",
                  "pessimism_summary.json"}
    files = sorted(
        f for f in rdir.glob("*.json") if f.name not in skip_names
    )
    if not files:
        raise SystemExit(f"no runner JSON files under {rdir}")

    rows = []
    cdfs: dict[str, list[float]] = {}     # tag -> latencies
    sched: list[dict] = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if not isinstance(d, dict):  # belt-and-braces
            continue
        latencies = []
        for r in d.get("results", []):
            latencies.extend(r.get("per_step_us", []))
        latencies_ms = [x / 1000.0 for x in latencies]
        rows.append({
            "tag": f.stem,
            "policy": d.get("policy"),
            "budget": d.get("hbm_budget"),
            "P50_ms": _percentile(latencies_ms, 50.0),
            "P99_ms": _percentile(latencies_ms, 99.0),
            "P999_ms": _percentile(latencies_ms, 99.9),
            "max_ms": max(latencies_ms) if latencies_ms else 0.0,
            "miss_ratio": d.get("miss_ratio", 0.0),
            "F1": d.get("f1_mean", 0.0),
            "tokens_per_s": d.get("tokens_per_second", 0.0),
            "bound": d.get("bound_lemma2_miss"),
            "pessimism": d.get("pessimism"),
        })
        cdfs[f.stem] = latencies_ms
        sched.append({
            "tag": f.stem,
            "policy": d.get("policy"),
            "deadline_us": d.get("deadline_us"),
            "miss_ratio": d.get("miss_ratio"),
            "bound_lemma2_miss": d.get("bound_lemma2_miss"),
            "pessimism": d.get("pessimism"),
            "slo": d.get("slo"),
        })

    # ---------------- Markdown table
    headers = ["tag", "policy", "budget", "P50_ms", "P99_ms", "P999_ms",
               "max_ms", "miss_ratio", "F1", "tokens_per_s", "bound", "pessimism"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row[h]) for h in headers) + " |")

    # ---------------- CSV
    csv_path = rdir / "table.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[analyze] wrote table.csv → {csv_path}")

    # ---------------- LaTeX (paper-ready)
    tex_path = rdir / "table.tex"
    _write_latex_table(tex_path, rows)
    print(f"[analyze] wrote table.tex → {tex_path}")

    # ---------------- schedulability.json (single-point eC sanity)
    sched_path = rdir / "schedulability.json"
    with open(sched_path, "w") as fh:
        json.dump(sched, fh, indent=2)
    print(f"[analyze] wrote schedulability.json → {sched_path}")

    # ---------------- CDF figure (paper-ready)
    if not args.no_plot:
        try:
            _plot_cdf_paper(rows, cdfs, rdir,
                            plot_budget=args.plot_budget,
                            slo_ms=args.slo_ms)
        except Exception as e:  # noqa: BLE001
            print(f"[analyze] CDF plot skipped: {e}")


# ---------------------------------------------------------------------------
# Patrick-style CDF
# ---------------------------------------------------------------------------
def _plot_cdf_paper(rows: list[dict], cdfs: dict[str, list[float]],
                    out_dir: Path, plot_budget: float, slo_ms: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figstyle.apply()

    # Pick the budget slice that matches plot_budget (within ε); fallback to
    # the budget closest to plot_budget if no exact match.
    eps = 1e-6
    sliced = [
        r for r in rows
        if r.get("budget") is not None and abs(float(r["budget"]) - plot_budget) < eps
    ]
    if not sliced:
        budgets = sorted({float(r["budget"]) for r in rows if r.get("budget") is not None})
        if not budgets:
            raise SystemExit("[eA] no rows have a budget; cannot pick a CDF slice")
        plot_budget = min(budgets, key=lambda b: abs(b - plot_budget))
        sliced = [r for r in rows if abs(float(r["budget"]) - plot_budget) < eps]
        print(f"[eA] no exact match for --plot_budget, using closest = {plot_budget}")

    sliced.sort(key=lambda r: figstyle.policy_sort_key(r.get("policy") or ""))

    fig, ax = plt.subplots(figsize=(3.45, 2.4))

    for r in sliced:
        tag = r["tag"]
        lat = cdfs.get(tag) or []
        if not lat:
            continue
        st = figstyle.policy_style(r["policy"])
        arr = np.sort(np.asarray(lat))
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        is_ours = (r["policy"] or "").lower() == "seer"
        ax.plot(
            arr, cdf,
            label=figstyle.policy_label(r["policy"]),
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["lw"] + (0.4 if is_ours else 0.0),
            zorder=4 if is_ours else 2,
        )

    # SLO marker (P99 = slo_ms) — drawn in the lower half so it doesn't
    # collide with the SEER P99 / P999 callouts at y ≈ 1.0.
    ax.axvline(slo_ms, color="black", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(slo_ms * 1.04, 0.30, f"SLO {slo_ms:g} ms",
            ha="left", va="bottom", fontsize=7.0, color="black",
            rotation=90)

    # Annotate SEER P99 / P999 inline (the headline number we defend in §6).
    # Both points have CDF very near 1.0, so we anchor the labels to the
    # right of the point and slightly below to avoid running off the panel.
    seer_row = next((r for r in sliced if (r["policy"] or "").lower() == "seer"), None)
    if seer_row is not None:
        p99 = float(seer_row["P99_ms"])
        p999 = float(seer_row["P999_ms"])
        ax.scatter([p99, p999], [0.99, 0.999], s=22,
                   color=figstyle.OURS, edgecolor="black", linewidths=0.6,
                   zorder=6)
        ax.annotate(
            f"P99 = {p99:.1f} ms",
            xy=(p99, 0.99), xytext=(-4, -28), textcoords="offset points",
            ha="right", va="bottom",
            fontsize=7.0, color=figstyle.OURS, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=figstyle.OURS, lw=0.6),
        )
        ax.annotate(
            f"P999 = {p999:.1f} ms",
            xy=(p999, 0.999), xytext=(4, -38), textcoords="offset points",
            ha="left", va="bottom",
            fontsize=7.0, color=figstyle.OURS, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=figstyle.OURS, lw=0.6),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Per-step TPOT (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.05)
    figstyle.add_box_spines(ax)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#999999")

    # Two-column legend so it fits in the bottom-right corner of the panel.
    leg = ax.legend(loc="lower right", ncol=2, fontsize=6.8)
    leg.get_frame().set_linewidth(0.6)

    out_path = figstyle.save_paper(fig, "cdf", out_dir)
    print(f"[analyze] wrote cdf.pdf → {out_path}")


def _write_latex_table(path: Path, rows: list[dict]) -> None:
    cols = ["policy", "budget", "P50_ms", "P99_ms", "P999_ms",
            "miss_ratio", "F1", "bound", "pessimism"]
    pretty = {
        "policy": "Policy",
        "budget": "$B_\\text{HBM}$",
        "P50_ms": "P50 (ms)",
        "P99_ms": "P99 (ms)",
        "P999_ms": "P999 (ms)",
        "miss_ratio": "Miss \\%",
        "F1": "F1",
        "bound": "L2 bound",
        "pessimism": "Pessimism",
    }
    lines = [
        "% Generated by experiments/eA_tail_latency/analyze.py",
        "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
        "\\toprule",
        " & ".join(pretty[c] for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append("--")
            elif c in ("P50_ms", "P99_ms", "P999_ms"):
                cells.append(f"{v:.2f}")
            elif c == "miss_ratio":
                cells.append(f"{v*100:.2f}")
            elif c == "F1":
                cells.append(f"{v:.3f}")
            elif c == "bound":
                cells.append(f"{v:.2e}" if isinstance(v, (int, float)) else "--")
            elif c == "pessimism":
                if not isinstance(v, (int, float)) or v != v:  # NaN check
                    cells.append("--")
                elif v == float("inf"):
                    # Infinite pessimism = bound > 0 with measured = 0;
                    # the SLO is met. Render as a strict-upper-envelope
                    # marker rather than a misleading "inf".
                    cells.append(r"--$^{\dagger}$")
                else:
                    cells.append(f"{v:.2f}$\\times$")
            elif c == "budget":
                cells.append(f"{v}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000 or v == 0:
            return f"{v:.1f}"
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


if __name__ == "__main__":
    main()
