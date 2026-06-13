#!/usr/bin/env python3
"""P5-close: Quality vs P999 Pareto plot on SQuAD150 (review #5).

Plots each policy as a point in (P999 ms, F1) space at the same
hbm_budget=0.20. Annotates the Pareto front and Streaming's
dominated location. Output: paper/figures/eA_squad150_pareto.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path


def main():
    src = Path("experiments/eA_tail_latency/results_squad150")
    files = {p.stem.split("_")[0]: p for p in src.glob("*.json")}
    if not files:
        print("[pareto] no SQuAD150 results")
        return
    pts = {}
    for name, path in files.items():
        d = json.load(open(path))
        pts[name] = {
            "p99": d["tpot_p99_us"] / 1000.0,
            "p999": d["tpot_p999_us"] / 1000.0,
            "f1": d.get("f1_mean", 0.0),
            "miss": d["miss_ratio"],
        }
    # Sort by p99 ascending (lower = better)
    for name, p in sorted(pts.items(), key=lambda x: x[1]["p99"]):
        print(f"  {name:10}  P99={p['p99']:5.1f}ms  P999={p['p999']:6.1f}ms  "
              f"F1={p['f1']:.3f}  miss={p['miss']:.4f}")

    # Use P99 as the latency axis — that's the bound's framework
    # (Pr(C_t > D) at deadline D). Excludes 'full' from comparison since
    # full uses 100% HBM and isn't an eviction policy.
    LATENCY_KEY = "p99"
    items_all = list(pts.items())
    items = [(n, p) for n, p in items_all if n != "full"]
    pareto = []
    for i, (n1, p1) in enumerate(items):
        dominated = False
        for j, (n2, p2) in enumerate(items):
            if i == j:
                continue
            # p2 dominates p1 if lower-latency AND higher-F1
            if (p2[LATENCY_KEY] <= p1[LATENCY_KEY] and p2["f1"] >= p1["f1"] and
                (p2[LATENCY_KEY] < p1[LATENCY_KEY] or p2["f1"] > p1["f1"])):
                dominated = True
                break
        if not dominated:
            pareto.append(n1)
    print(f"[pareto] Pareto front (eviction policies, P99): {pareto}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[pareto] matplotlib unavailable: {e}")
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    # Pareto front as connecting line
    pareto_pts = sorted([(pts[n]["p99"], pts[n]["f1"], n) for n in pareto])
    if pareto_pts:
        ax.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts],
                "k--", lw=1.0, alpha=0.5, zorder=1, label="Pareto front")
    # All points
    label_offset = {
        "snapkv": (4, -10), "full": (-10, 8), "h2o": (4, 6),
        "seer": (-30, 8), "streaming": (4, -10), "quest": (4, 6),
    }
    color_map = {
        "full": "tab:gray", "snapkv": "tab:green", "h2o": "tab:orange",
        "seer": "tab:blue", "streaming": "tab:red", "quest": "tab:purple",
    }
    for name, p in pts.items():
        is_pareto = name in pareto
        is_full = name == "full"
        ax.scatter(p["p99"], p["f1"],
                   s=90 if is_pareto else 50,
                   color=color_map.get(name, "k"),
                   marker="*" if is_full else "o",
                   edgecolors="black" if is_pareto or is_full else "none",
                   linewidths=1.5 if is_pareto else (1.0 if is_full else 0),
                   zorder=3)
        ann = ("\\textsc{Full} (oracle)" if is_full else
               ("\\textbf{SEER}" if name == "seer" else name.capitalize()))
        dx, dy = label_offset.get(name, (5, 5))
        ax.annotate(ann, (p["p99"], p["f1"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, fontweight="bold" if is_pareto else "normal")
    # SLO line at 50ms
    ax.axvline(50.0, ls=":", color="red", lw=0.8, alpha=0.6)
    ax.text(50.3, 0.005, "P99 SLO 50\\,ms", color="red", fontsize=7, alpha=0.8)
    ax.set_xlabel("Per-step P99 TPOT (ms)")
    ax.set_ylabel("Quality F1")
    ax.set_xlim(33, 41)
    ax.set_ylim(0, 0.25)
    ax.grid(True, alpha=0.3)
    ax.set_title("SQuAD150 quality--latency Pareto ($\\phi=0.20$, P99 axis)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    out = Path("paper/figures/eA_squad150_pareto.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"[pareto] PDF → {out}")


if __name__ == "__main__":
    main()
