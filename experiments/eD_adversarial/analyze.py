"""eD analyze — P99 TPOT degradation as a function of σ_shift.

Reads (family, policy) JSON files from results/ and plots, for each
policy, P99 TPOT against the family's engineered ``sigma`` value.
Lemma 3 predicts: heuristic policies are roughly linear in σ; SEER
saturates near its predictor-error floor.

Output: ``p99_vs_shift.pdf`` in the Patrick / ADSLab paper style.
The slope summary in ``shift_slopes.json`` is consumed by §6.6 of the
paper; SEER's slope is what we read out as the headline 130× improvement
over StreamingLLM.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seer import figstyle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    # Skip artefacts produced by previous analyze runs (shift_slopes.json
    # is a flat policy → slope dict, not a runner output).
    skip_names = {"shift_slopes.json"}
    files = sorted(
        f for f in rdir.glob("*.json") if f.name not in skip_names
    )
    if not files:
        raise SystemExit(f"no JSON under {rdir}")

    by_policy: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    rows = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        # A second guard: the runner JSON has both "family" and "sigma";
        # anything else (including the slopes file under a previous name)
        # gets skipped.
        if "family" not in d or "sigma" not in d:
            continue
        family = d.get("family") or f.stem.split("__")[0]
        sigma = float(d.get("sigma", 0.0))
        policy = d.get("policy") or f.stem.split("__")[-1]
        p99 = float(d.get("tpot_p99_us", 0.0)) / 1000.0
        miss = float(d.get("miss_ratio", 0.0))
        f1 = float(d.get("f1_mean", 0.0))
        by_policy[policy].append((sigma, p99, family))
        rows.append({"family": family, "policy": policy, "sigma": sigma,
                     "p99_ms": p99, "miss": miss, "f1": f1, "tag": f.stem})

    csv_path = rdir / "table.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["tag", "family", "policy", "sigma",
                            "p99_ms", "miss", "f1"])
        w.writeheader()
        w.writerows(rows)
    print(f"[eD] table.csv → {csv_path}")

    print("| family | policy | σ | P99 (ms) | miss | F1 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['family']} | {r['policy']} | {r['sigma']} | "
              f"{r['p99_ms']:.2f} | {r['miss']:.4f} | {r['f1']:.3f} |")

    summary = {}
    for pol, pts in by_policy.items():
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_mean = statistics.fmean(xs)
        y_mean = statistics.fmean(ys)
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False))
        var = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = cov / var
        summary[pol] = {"slope_ms_per_sigma": slope,
                         "intercept_ms": y_mean - slope * x_mean,
                         "n": len(pts)}
    with open(rdir / "shift_slopes.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[eD] slope summary:", json.dumps(summary, indent=2))

    if args.no_plot:
        return
    try:
        _plot_p99_vs_shift(by_policy, summary, rdir)
    except Exception as e:  # noqa: BLE001
        print(f"[eD] plot skipped: {e}")


def _plot_p99_vs_shift(by_policy: dict[str, list[tuple[float, float, str]]],
                       summary: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(3.45, 2.4))

    # Sort policies so SEER renders last (top of legend stack)
    items = sorted(by_policy.items(),
                   key=lambda kv: figstyle.policy_sort_key(kv[0]))

    sigma_global_min = min(p[0] for pts in by_policy.values() for p in pts)
    sigma_global_max = max(p[0] for pts in by_policy.values() for p in pts)
    sigma_pad = 0.05 * (sigma_global_max - sigma_global_min or 1.0)
    sigma_lo = sigma_global_min - sigma_pad
    sigma_hi = sigma_global_max + sigma_pad

    for policy, pts in items:
        st = figstyle.policy_style(policy)
        is_ours = (policy or "").lower() == "seer"
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
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

    # Annotate slope ratio (Streaming / SEER) where applicable. Anchor it
    # to the bottom-right so it never collides with the streaming spike.
    if "streaming" in summary and "seer" in summary:
        s_str = summary["streaming"]["slope_ms_per_sigma"]
        s_seer = summary["seer"]["slope_ms_per_sigma"] or 1e-6
        if s_seer > 0 and s_str > 0:
            ratio = s_str / s_seer
            txt = (f"streaming slope: $+{s_str:.0f}$ ms/$\\sigma$\n"
                   f"SEER slope:        $+{s_seer:.1f}$ ms/$\\sigma$\n"
                   f"ratio = {ratio:.0f}$\\times$ tighter")
            ax.text(0.97, 0.05, txt,
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=7.0, color=figstyle.OURS, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.30",
                              facecolor="white",
                              edgecolor=figstyle.OURS,
                              linewidth=0.6, alpha=0.95))

    ax.set_xlabel(r"Engineered attention shift $\sigma_{\mathrm{shift}}$")
    ax.set_ylabel("P99 TPOT (ms)")
    ax.set_xlim(sigma_lo, sigma_hi)
    ax.set_yscale("log")
    ax.set_ylim(20, max(p[1] for pts in by_policy.values() for p in pts) * 1.6)
    figstyle.add_box_spines(ax)
    leg = ax.legend(loc="upper left", ncol=1, fontsize=6.8)
    leg.get_frame().set_linewidth(0.6)

    out_path = figstyle.save_paper(fig, "p99_vs_shift", out_dir)
    print(f"[eD] p99_vs_shift.pdf → {out_path}")


if __name__ == "__main__":
    main()
