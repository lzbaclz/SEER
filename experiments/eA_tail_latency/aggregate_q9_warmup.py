#!/usr/bin/env python3
"""Aggregate the Q9 SEER vs SEER+warmup comparison on SQuAD150."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path("experiments/eA_tail_latency/results_q9_warmup")


def main():
    rows = defaultdict(list)
    for f in sorted(RESULTS.glob("*.json")):
        d = json.load(open(f))
        policy = d["policy"]
        rows[policy].append({
            "seed": int(f.stem.split("_s")[-1]),
            "p99": d["tpot_p99_us"] / 1000,
            "p999": d["tpot_p999_us"] / 1000,
            "miss": d["miss_ratio"],
            "f1": d.get("f1_mean", 0.0),
            "policy_stats": d.get("policy_stats", {}),
        })
    if not rows:
        print("[Q9] no results")
        return
    print(f"[Q9] loaded {sum(len(v) for v in rows.values())} rows")
    print()
    for policy in sorted(rows):
        rs = rows[policy]
        p99s = [r["p99"] for r in rs]
        p999s = [r["p999"] for r in rs]
        misses = [r["miss"] for r in rs]
        f1s = [r["f1"] for r in rs]
        mean_std = lambda xs: (statistics.mean(xs),
                                statistics.stdev(xs) if len(xs) > 1 else 0.0)
        p99_m, p99_s = mean_std(p99s)
        p999_m, p999_s = mean_std(p999s)
        miss_m, _ = mean_std(misses)
        f1_m, _ = mean_std(f1s)
        # Warmup counters (sum across seeds)
        ws = [r["policy_stats"] for r in rs if isinstance(r["policy_stats"], dict)]
        triggers = sum(s.get("trigger_count", 0) for s in ws)
        warmup_steps = sum(s.get("warmup_steps", 0) for s in ws)
        print(f"  {policy:15} P99={p99_m:6.1f}±{p99_s:4.1f}ms  "
              f"P999={p999_m:6.1f}±{p999_s:4.1f}ms  "
              f"miss={miss_m:.4f}  F1={f1_m:.3f}  "
              f"triggers={triggers}  warmup_steps={warmup_steps}")
    # LaTeX
    tex = RESULTS / "q9_warmup_table.tex"
    with open(tex, "w") as f:
        f.write("% Q9: SEER vs SEER+speculative-warmup on SQuAD150-stitched.\n")
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\caption{Speculative pre-warm absorbs SQuAD150's P999 outliers. "
                "Mean $\\pm$ std across 3 seeds, $\\hbm = 0.20$, $n=150$ prompts. "
                "\\textsc{triggers}: how often the detect predicate fired; "
                "\\textsc{warmup steps}: actual forced pre-fetches.}\n")
        f.write("\\label{tab:q9-warmup}\n")
        f.write("\\begin{tabular}{lrrrrrr}\\toprule\n")
        f.write("policy & P99 (ms) & P999 (ms) & miss & F1 & triggers & warmup\\\\\n\\midrule\n")
        for policy in sorted(rows):
            rs = rows[policy]
            p99 = statistics.mean([r["p99"] for r in rs])
            p99_s = statistics.stdev([r["p99"] for r in rs]) if len(rs) > 1 else 0
            p999 = statistics.mean([r["p999"] for r in rs])
            p999_s = statistics.stdev([r["p999"] for r in rs]) if len(rs) > 1 else 0
            miss = statistics.mean([r["miss"] for r in rs])
            f1 = statistics.mean([r["f1"] for r in rs])
            ws = [r["policy_stats"] for r in rs if isinstance(r["policy_stats"], dict)]
            triggers = sum(s.get("trigger_count", 0) for s in ws)
            warmup = sum(s.get("warmup_steps", 0) for s in ws)
            f.write(f"{policy} & "
                    f"{p99:.1f}$\\pm${p99_s:.1f} & "
                    f"{p999:.1f}$\\pm${p999_s:.1f} & "
                    f"{miss:.4f} & {f1:.3f} & {triggers} & {warmup} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"[Q9] LaTeX → {tex}")


if __name__ == "__main__":
    main()
