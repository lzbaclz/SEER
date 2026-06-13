"""eB analyze — miss-ratio vs SLO threshold, one curve per policy.

Reads every ``*.json`` produced by :mod:`seer.eval.runner` under the
results directory, groups by policy, and plots miss_ratio vs the SLO's
``threshold_ms``. Writes both ``table.csv`` and the paper-ready
``miss_vs_slo.pdf`` in the Patrick / ADSLab style.

Optional ``--with_bound`` overlays the Lemma-2 bound for the SEER row
using a configurable ε.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seer import figstyle  # noqa: E402


def _load_runner_jsons(results_dir: str) -> list[dict]:
    rdir = Path(results_dir)
    files = sorted(rdir.glob("*.json"))
    out = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        slo = d.get("slo") or {}
        out.append({
            "policy": d.get("policy"),
            "tag": f.stem,
            "deadline_ms": float(slo.get("threshold_ms", 0.0)),
            "miss_ratio": float(d.get("miss_ratio", 0.0)),
            "p99_ms": float(d.get("tpot_p99_us", 0.0)) / 1000.0,
            "f1": float(d.get("f1_mean", 0.0)),
            "budget": float(d.get("hbm_budget", 0.0)),
            "bound": d.get("bound_lemma2_miss"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--with_bound", action="store_true",
                    help="Overlay analytical bound on the SEER curve")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--floor", type=float, default=1e-3,
                    help="Replacement value for zero miss-ratios on the log-y "
                         "axis. Defaults to 1e-3.")
    args = ap.parse_args()

    rows = _load_runner_jsons(args.results_dir)
    if not rows:
        raise SystemExit(f"no JSON under {args.results_dir}")

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)
    for v in by_policy.values():
        v.sort(key=lambda x: x["deadline_ms"])

    # ---- CSV
    out_dir = Path(args.results_dir)
    csv_path = out_dir / "table.csv"
    headers = ["policy", "tag", "deadline_ms", "budget",
               "miss_ratio", "p99_ms", "f1", "bound"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in headers})
    print(f"[eB] table.csv → {csv_path}")

    # ---- markdown
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        cells = [
            r["policy"], r["tag"],
            f"{r['deadline_ms']:.0f}", f"{r['budget']:.2f}",
            f"{r['miss_ratio']:.4f}", f"{r['p99_ms']:.2f}",
            f"{r['f1']:.3f}",
            (f"{r['bound']:.2e}" if r["bound"] is not None else "—"),
        ]
        print("| " + " | ".join(cells) + " |")

    # ---- plot
    if args.no_plot:
        return
    try:
        _plot_miss_vs_slo(by_policy, out_dir,
                          floor=args.floor,
                          with_bound=args.with_bound)
    except Exception as e:  # noqa: BLE001
        print(f"[eB] plot skipped: {e}")


def _plot_miss_vs_slo(by_policy: dict[str, list[dict]], out_dir: Path,
                      floor: float, with_bound: bool) -> None:
    import matplotlib.pyplot as plt

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(3.45, 2.4))

    # Sort policies so SEER renders last (= top of legend)
    items = sorted(by_policy.items(),
                   key=lambda kv: figstyle.policy_sort_key(kv[0]))

    for policy, recs in items:
        st = figstyle.policy_style(policy)
        is_ours = (policy or "").lower() == "seer"
        xs = [r["deadline_ms"] for r in recs]
        ys = [max(r["miss_ratio"], floor) for r in recs]
        ax.plot(
            xs, ys,
            label=figstyle.policy_label(policy),
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["lw"] + (0.4 if is_ours else 0.0),
            marker=st["marker"],
            markersize=st["ms"] + (0.6 if is_ours else 0.0),
            markerfacecolor=st["color"],
            markeredgecolor="black" if is_ours else st["color"],
            markeredgewidth=0.6 if is_ours else 0.0,
            zorder=4 if is_ours else 2,
        )

    if with_bound:
        seer = by_policy.get("seer", [])
        bxs = [r["deadline_ms"] for r in seer if r["bound"] is not None]
        bys = [max(float(r["bound"]), floor) for r in seer if r["bound"] is not None]
        if bxs:
            ax.plot(bxs, bys, linestyle="--", color="black",
                    linewidth=0.8, label="Theorem 1 bound (SEER)", zorder=3)

    ax.set_xlabel("Deadline $D$ (ms)")
    ax.set_ylabel("Deadline-miss ratio")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, 2.0)
    ax.axhline(0.01, color="black", linestyle=":", linewidth=0.7, zorder=1)
    ax.text(ax.get_xlim()[1], 0.012, "1% target",
            ha="right", va="bottom", fontsize=7.0, color="black")
    figstyle.add_box_spines(ax)
    leg = ax.legend(loc="lower left", ncol=2, fontsize=6.8)
    leg.get_frame().set_linewidth(0.6)

    out_path = figstyle.save_paper(fig, "miss_vs_slo", out_dir)
    print(f"[eB] miss_vs_slo.pdf → {out_path}")


if __name__ == "__main__":
    main()
