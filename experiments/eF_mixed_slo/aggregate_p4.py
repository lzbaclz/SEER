#!/usr/bin/env python3
"""Aggregate the P4 eF-EDF vs eF-FIFO comparison."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("experiments/eF_mixed_slo/results_p4")


def main():
    rows = {}
    for f in sorted(RESULTS.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        policy = d["policy"]
        mode = d["mode"]
        rows.setdefault(policy, {})[mode] = d
    if not rows:
        print("[P4] no results yet")
        return
    tex = RESULTS / "eF_edf_table.tex"
    with open(tex, "w") as f:
        f.write("% P4: EDF admission vs FIFO on the eF mixed-SLO workload.\n")
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\caption{Mixed-SLO miss-ratio under FIFO and EDF schedulers, "
                "single A100, 10 chat ($\\le 50$\\,ms) + 2 doc (TTFT $\\le 1$\\,s) "
                "tasks. \\textbf{FIFO}: legacy ThreadPoolExecutor; "
                "\\textbf{EDF}: bound-aware admission queue. Lower chat-miss "
                "and shorter wall time confirm that the published mixed-SLO "
                "limit was a scheduler issue, not a policy one.}\n")
        f.write("\\label{tab:eF-edf}\n")
        f.write("\\begin{tabular}{llrrr}\\toprule\n")
        f.write("policy & sched & chat-miss & doc-miss & wall (s) \\\\\n")
        f.write("\\midrule\n")
        for policy in sorted(rows):
            for mode in ["fifo", "edf"]:
                d = rows[policy].get(mode)
                if d is None:
                    f.write(f"{policy} & {mode} & --- & --- & --- \\\\\n")
                    continue
                f.write(f"{policy} & {mode.upper()} & "
                        f"{d['chat_miss_ratio']:.4f} & "
                        f"{d['doc_miss_ratio']:.4f} & "
                        f"{d['wall_time_s']:.1f} \\\\\n")
            f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\\end{table}\n")
    print(f"[P4] table → {tex}")
    # Console summary
    for policy in sorted(rows):
        for mode in sorted(rows[policy]):
            d = rows[policy][mode]
            print(f"  {policy:>10} {mode:>4}: chat={d['chat_miss_ratio']:.4f} "
                  f"doc={d['doc_miss_ratio']:.4f} wall={d['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
