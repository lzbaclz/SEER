"""E2 analysis: collect per-(policy, budget) JSON results and plot Pareto."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="dir of *.json from runner.py")
    ap.add_argument("--metric", default="f1_mean",
                    choices=["f1_mean", "em_mean", "substring_mean"])
    ap.add_argument("--out", default="figures/e2_pareto.pdf")
    args = ap.parse_args()

    rows = []
    for p in Path(args.results_dir).glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "policy": d["policy"],
            "budget": d["budget"],
            "metric": d[args.metric],
        })
    if not rows:
        print(f"[E2] no results in {args.results_dir}")
        return

    # ---------- Print table
    policies = sorted({r["policy"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    print(f"\n{args.metric} table")
    header = "budget \\ policy  | " + "  ".join(f"{p:>10s}" for p in policies)
    print(header)
    print("-" * len(header))
    for b in budgets:
        line = f"{b:>15.2f}  | "
        for p in policies:
            m = next((r["metric"] for r in rows if r["policy"] == p and r["budget"] == b), None)
            line += f"  {m:>10.3f}" if m is not None else f"  {'-':>10s}"
        print(line)

    # ---------- Plot
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[E2] matplotlib not available, skipping plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in policies:
        pts = sorted([(r["budget"], r["metric"]) for r in rows if r["policy"] == p])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", label=p, linewidth=2 if p == "seer" else 1)
    ax.set_xlabel("HBM budget (fraction of full cache)")
    ax.set_ylabel(args.metric)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("E2: Quality–Budget Pareto")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"[E2] wrote {out}")


if __name__ == "__main__":
    main()
