#!/usr/bin/env python3
"""Q9: CDF of per-step decode latency, SEER vs SEER+warmup on SQuAD150.

Renders paper/figures/eA_q9_warmup_cdf.pdf. The salient comparison is
at the tail: SEER's P999 was 179 ms (above SLO); SEER+warmup should
fall in a tighter band centered near the warmup-step bound
(C_base + N_warmup·ℓ̄).
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("experiments/eA_tail_latency/results_q9_warmup")


def collect_per_step(files: list[Path]) -> list[float]:
    out = []
    for f in files:
        d = json.load(open(f))
        for r in d.get("results", []):
            out.extend(r.get("per_step_us", []))
    return out


def main():
    seer_files = sorted(RESULTS.glob("seer_s*.json"))
    warmup_files = sorted(RESULTS.glob("seer-warmup_s*.json"))
    if not seer_files or not warmup_files:
        print(f"[Q9-cdf] missing data: seer={len(seer_files)} warmup={len(warmup_files)}")
        return
    seer_step = sorted(collect_per_step(seer_files))
    warmup_step = sorted(collect_per_step(warmup_files))
    print(f"[Q9-cdf] SEER:    n={len(seer_step)}  P99={seer_step[int(0.99*len(seer_step))]/1000:.1f}ms"
          f"  P999={seer_step[int(0.999*len(seer_step))]/1000:.1f}ms  max={seer_step[-1]/1000:.1f}ms")
    print(f"[Q9-cdf] Warmup:  n={len(warmup_step)}  P99={warmup_step[int(0.99*len(warmup_step))]/1000:.1f}ms"
          f"  P999={warmup_step[int(0.999*len(warmup_step))]/1000:.1f}ms  max={warmup_step[-1]/1000:.1f}ms")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[Q9-cdf] matplotlib unavailable: {e}")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    def cdf(xs):
        xs = np.asarray(xs, dtype=float) / 1000.0  # → ms
        y = np.arange(1, len(xs) + 1) / len(xs)
        return xs, y

    seer_x, seer_y = cdf(seer_step)
    warm_x, warm_y = cdf(warmup_step)

    # Focus on the tail — clip below 30ms and above 200ms
    ax.step(seer_x, seer_y, where="post", color="tab:red", lw=1.2,
            label=f"SEER (P999={seer_x[int(0.999*len(seer_x))-1]:.0f}ms)")
    ax.step(warm_x, warm_y, where="post", color="tab:blue", lw=1.4,
            label=f"SEER+warmup (P999={warm_x[int(0.999*len(warm_x))-1]:.0f}ms)")
    ax.axvline(50.0, ls=":", color="red", lw=0.8, alpha=0.6)
    ax.text(50.5, 0.05, "50\\,ms SLO", color="red", fontsize=7, alpha=0.8,
            rotation=90, verticalalignment="bottom")
    ax.set_xscale("log")
    ax.set_xlim(30, 250)
    ax.set_ylim(0.5, 1.001)
    ax.set_xlabel("Per-step decode latency (ms)")
    ax.set_ylabel("CDF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("SQuAD150: speculative-warmup absorbs P999 outliers",
                 fontsize=9)
    plt.tight_layout()
    out = Path("paper/figures/eA_q9_warmup_cdf.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"[Q9-cdf] PDF → {out}")


if __name__ == "__main__":
    main()
