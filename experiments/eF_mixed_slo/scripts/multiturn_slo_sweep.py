"""Re-aggregate chat-miss at multiple SLO thresholds from quantile data.

Addresses reviewer W4: the β3 sweep used P99=200ms SLO; the paper's
motivating headline is 50ms. This script computes per-policy chat-miss
at SLO ∈ {50, 100, 150, 200, 250, 300, 400, 500}ms using piecewise-linear
CDF interpolation over the stored per-prompt quantiles
(p50_us, p90_us, p99_us, p999_us, max_us). The interpolation is conservative
(linear between quantile pairs); for fine-grained comparison the underlying
points are also reported.

For each cell, we treat the prompt's quantile summary as a 4-point CDF and
estimate miss_count = n * (1 - F(SLO)). Aggregated step counts and Wilson
95% CIs are computed from the pooled (sum-of-miss, sum-of-n) per policy.

The output is a (policy x SLO) chat-miss table plus per-policy Wilson CIs,
emitted to LaTeX (multiturn_slo_sweep.tex) and JSON (multiturn_slo_sweep.json).
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def two_sample_z(k_a: int, n_a: int, k_b: int, n_b: int) -> tuple[float, float]:
    """Two-sample test for difference in proportions p_a - p_b.

    Returns (z_score, two-sided p-value). Negative z means p_a < p_b.
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_a - p_b) / se
    # Two-sided p-value via the standard normal CDF.
    p_val = math.erfc(abs(z) / math.sqrt(2))
    return z, p_val


def cdf_from_quantiles(deadline_us: float, ts: dict, n: int) -> int:
    """Estimate miss_count at ``deadline_us`` from a stored ``tpot_stats``
    summary. The summary gives p50, p90, p99, p999, max; we linearly
    interpolate F(deadline) on this piecewise-CDF and return
    ``round(n * (1 - F(deadline)))``.
    """
    if n <= 0:
        return 0
    pts = [
        (float(ts.get("p50_us", 0.0)), 0.50),
        (float(ts.get("p90_us", 0.0)), 0.90),
        (float(ts.get("p99_us", 0.0)), 0.99),
        (float(ts.get("p999_us", 0.0)), 0.999),
        (float(ts.get("max_us", 0.0)), 1.0),
    ]
    # The mean might be informative but is not a quantile; skip.
    # Also include a sentinel for 0us → F=0 to anchor the left tail.
    pts = [(0.0, 0.0)] + sorted(pts, key=lambda t: t[0])
    # If deadline is past max, no misses.
    if deadline_us >= pts[-1][0]:
        return 0
    # If deadline is before p50, miss is heavy.
    if deadline_us <= pts[0][0]:
        return int(round(n * 1.0))
    # Find the bracket.
    for i in range(1, len(pts)):
        x_lo, F_lo = pts[i - 1]
        x_hi, F_hi = pts[i]
        if x_lo <= deadline_us <= x_hi:
            if x_hi == x_lo:
                F = F_hi
            else:
                F = F_lo + (deadline_us - x_lo) / (x_hi - x_lo) * (F_hi - F_lo)
            return int(round(n * (1.0 - F)))
    return 0


def aggregate(in_dir: Path, slos_ms: Sequence[int]) -> dict:
    """Aggregate per-policy step misses at each SLO threshold."""
    by_policy: dict[str, list[dict]] = {}
    for f in sorted(in_dir.glob("*_s*.json")):
        name = f.name
        if any(tag in name for tag in
               ("boot_ci", "aggregate", "summary", "slo_sweep",
                "per_seed", "mechanism")):
            continue
        if ".contention." in name:
            continue  # the H2O contention-outlier seed is preserved
            # for transparency but not used in the headline aggregate.
        policy = name.split("_")[0]
        try:
            with open(f) as fh:
                by_policy.setdefault(policy, []).append(json.load(fh))
        except Exception:
            continue
    out: dict = {"per_policy": {}, "slos_ms": list(slos_ms)}
    for policy, cells in by_policy.items():
        per_slo: dict[int, dict] = {}
        for slo_ms in slos_ms:
            total_n = 0
            total_miss = 0
            for c in cells:
                for r in c.get("results", []):
                    ts = r.get("tpot_stats") or r.get("stats") or {}
                    n = int(ts.get("n", 0))
                    total_n += n
                    total_miss += cdf_from_quantiles(slo_ms * 1000.0, ts, n)
            lo, hi = wilson_ci(total_miss, total_n)
            per_slo[slo_ms] = {
                "k": total_miss,
                "n": total_n,
                "miss_rate": (total_miss / total_n) if total_n else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        out["per_policy"][policy] = {
            "n_seeds": len(cells),
            "per_slo": per_slo,
        }
    # Add SEER vs {H2O, Streaming, SnapKV, Quest} two-sample tests
    # at each SLO (full 6-policy comparison panel).
    out["tests"] = {}
    seer = out["per_policy"].get("seer", {}).get("per_slo", {})
    h2o = out["per_policy"].get("h2o", {}).get("per_slo", {})
    streaming = out["per_policy"].get("streaming", {}).get("per_slo", {})
    snapkv = out["per_policy"].get("snapkv", {}).get("per_slo", {})
    quest = out["per_policy"].get("quest", {}).get("per_slo", {})
    for slo_ms in slos_ms:
        s = seer.get(slo_ms, {})
        h = h2o.get(slo_ms, {})
        st = streaming.get(slo_ms, {})
        sk = snapkv.get(slo_ms, {})
        qu = quest.get(slo_ms, {})
        def _z(other):
            return two_sample_z(
                int(s.get("k", 0)), int(s.get("n", 0)),
                int(other.get("k", 0)), int(other.get("n", 0)))
        z_sh, p_sh = _z(h)
        z_ss, p_ss = _z(st)
        z_sk, p_sk = _z(sk)
        z_sq, p_sq = _z(qu)
        out["tests"][slo_ms] = {
            "seer_vs_h2o_z": z_sh,
            "seer_vs_h2o_p": p_sh,
            "seer_vs_h2o_significant_at_alpha_005": p_sh < 0.05,
            "seer_vs_streaming_z": z_ss,
            "seer_vs_streaming_p": p_ss,
            "seer_vs_streaming_significant_at_alpha_005": p_ss < 0.05,
            "seer_vs_snapkv_z": z_sk,
            "seer_vs_snapkv_p": p_sk,
            "seer_vs_snapkv_significant_at_alpha_005": p_sk < 0.05,
            "seer_vs_quest_z": z_sq,
            "seer_vs_quest_p": p_sq,
            "seer_vs_quest_significant_at_alpha_005": p_sq < 0.05,
        }
    return out


def emit_tex(out: dict, path: Path) -> None:
    slos = out["slos_ms"]
    policies = ["full", "streaming", "h2o", "snapkv", "quest", "seer"]
    lines = [
        "% Generated by multiturn_slo_sweep.py.",
        "% Per-policy step-miss rate (with Wilson 95% CI) at each P99-TPOT SLO",
        "% threshold, re-aggregated from stored tpot_stats quantiles via",
        "% piecewise-linear CDF interpolation.",
        r"\begin{tabular}{l" + "r" * len(slos) + "}",
        r"\toprule",
        "Policy & " + " & ".join(f"${ms}$\\,ms" for ms in slos) + r" \\",
        r"\midrule",
    ]
    for p in policies:
        if p not in out["per_policy"]:
            continue
        row = [rf"\texttt{{{p}}}"]
        for ms in slos:
            d = out["per_policy"][p]["per_slo"].get(ms, {})
            if not d:
                row.append("--")
            else:
                row.append(rf"${d['miss_rate']:.4f}$")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(slos) + 1)
                 + r"}{l}{\textit{SEER vs.\ H2O 2-sample $z$-test:}} \\")
    row = [r"\hspace{1em}$p$-value"]
    for ms in slos:
        d = out["tests"].get(ms, {})
        p = d.get("seer_vs_h2o_p", float("nan"))
        if p < 0.001:
            row.append(r"$<$\,$10^{-3}$")
        else:
            row.append(rf"${p:.3f}$")
    lines.append(" & ".join(row) + r" \\")
    row = [r"\hspace{1em}sig.\ at $\alpha{=}0.05$?"]
    for ms in slos:
        d = out["tests"].get(ms, {})
        sig = d.get("seer_vs_h2o_significant_at_alpha_005", False)
        row.append("yes" if sig else "no")
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\multicolumn{" + str(len(slos) + 1)
                 + r"}{l}{\textit{SEER vs.\ Streaming 2-sample $z$-test:}} \\")
    row = [r"\hspace{1em}$p$-value"]
    for ms in slos:
        d = out["tests"].get(ms, {})
        p = d.get("seer_vs_streaming_p", float("nan"))
        if p < 0.001:
            row.append(r"$<$\,$10^{-3}$")
        else:
            row.append(rf"${p:.3f}$")
    lines.append(" & ".join(row) + r" \\")
    row = [r"\hspace{1em}sig.\ at $\alpha{=}0.05$?"]
    for ms in slos:
        d = out["tests"].get(ms, {})
        sig = d.get("seer_vs_streaming_significant_at_alpha_005", False)
        row.append("yes" if sig else "no")
    lines.append(" & ".join(row) + r" \\")
    # SEER vs SnapKV.
    if any("seer_vs_snapkv_p" in out["tests"].get(ms, {}) for ms in slos):
        lines.append(r"\multicolumn{" + str(len(slos) + 1)
                     + r"}{l}{\textit{SEER vs.\ SnapKV 2-sample $z$-test:}} \\")
        row = [r"\hspace{1em}$p$-value"]
        for ms in slos:
            d = out["tests"].get(ms, {})
            p = d.get("seer_vs_snapkv_p", float("nan"))
            if p < 0.001:
                row.append(r"$<$\,$10^{-3}$")
            else:
                row.append(rf"${p:.3f}$")
        lines.append(" & ".join(row) + r" \\")
        row = [r"\hspace{1em}sig.\ at $\alpha{=}0.05$?"]
        for ms in slos:
            d = out["tests"].get(ms, {})
            sig = d.get("seer_vs_snapkv_significant_at_alpha_005", False)
            row.append("yes" if sig else "no")
        lines.append(" & ".join(row) + r" \\")
    # SEER vs Quest.
    if any("seer_vs_quest_p" in out["tests"].get(ms, {}) for ms in slos):
        lines.append(r"\multicolumn{" + str(len(slos) + 1)
                     + r"}{l}{\textit{SEER vs.\ Quest 2-sample $z$-test:}} \\")
        row = [r"\hspace{1em}$p$-value"]
        for ms in slos:
            d = out["tests"].get(ms, {})
            p = d.get("seer_vs_quest_p", float("nan"))
            if p < 0.001:
                row.append(r"$<$\,$10^{-3}$")
            else:
                row.append(rf"${p:.3f}$")
        lines.append(" & ".join(row) + r" \\")
        row = [r"\hspace{1em}sig.\ at $\alpha{=}0.05$?"]
        for ms in slos:
            d = out["tests"].get(ms, {})
            sig = d.get("seer_vs_quest_significant_at_alpha_005", False)
            row.append("yes" if sig else "no")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def emit_per_seed_tex(in_dir: Path, slo_ms: int, path: Path) -> None:
    """Per-seed table at a single SLO to expose outliers (W3)."""
    by_policy: dict[str, list[tuple[int, float]]] = {}
    for f in sorted(in_dir.glob("*_s*.json")):
        if ".contention." in f.name or "summary" in f.name or "aggregate" in f.name:
            continue
        policy = f.name.split("_")[0]
        try:
            seed = int(f.stem.split("_s")[1])
        except Exception:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        total_n = 0
        total_miss = 0
        for r in d.get("results", []):
            ts = r.get("tpot_stats") or r.get("stats") or {}
            n = int(ts.get("n", 0))
            total_n += n
            total_miss += cdf_from_quantiles(slo_ms * 1000.0, ts, n)
        rate = (total_miss / total_n) if total_n else 0.0
        by_policy.setdefault(policy, []).append((seed, rate))
    seeds_seen = sorted({s for v in by_policy.values() for s, _ in v})
    lines = [
        "% Generated by multiturn_slo_sweep.py.",
        f"% Per-seed step-miss at P99={slo_ms}ms.",
        r"\begin{tabular}{l" + "r" * len(seeds_seen) + "r}",
        r"\toprule",
        "Policy & " + " & ".join(f"s{s}" for s in seeds_seen) + r" & mean \\",
        r"\midrule",
    ]
    for policy in ("full", "streaming", "h2o", "seer"):
        if policy not in by_policy:
            continue
        seed_map = {s: r for s, r in by_policy[policy]}
        rates = [seed_map.get(s) for s in seeds_seen]
        cells = []
        for r in rates:
            cells.append(rf"${r:.4f}$" if r is not None else "--")
        m = sum(r for r in rates if r is not None) / max(
            1, sum(1 for r in rates if r is not None))
        lines.append(
            rf"\texttt{{{policy}}} & " + " & ".join(cells)
            + rf" & ${m:.4f}$ \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="experiments/eF_mixed_slo/results_vllm_path_beta3")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    slos = [50, 100, 150, 200, 250, 300, 400, 500]
    out = aggregate(in_dir, slos)
    (out_dir / "multiturn_slo_sweep.json").write_text(
        json.dumps(out, indent=2))
    emit_tex(out, out_dir / "multiturn_slo_sweep.tex")
    emit_per_seed_tex(in_dir, 200, out_dir / "multiturn_per_seed_200ms.tex")
    emit_per_seed_tex(in_dir, 150, out_dir / "multiturn_per_seed_150ms.tex")
    print(f"Wrote {out_dir}/multiturn_slo_sweep.{{tex,json}} and per-seed tables.")
    print("\nSEER vs H2O two-sample z-test (p-values):")
    for ms in slos:
        d = out["tests"][ms]
        marker = "***" if d["seer_vs_h2o_p"] < 0.001 else (
            "**" if d["seer_vs_h2o_p"] < 0.01 else (
                "*" if d["seer_vs_h2o_p"] < 0.05 else " "))
        print(f"  {ms:>3}ms: z={d['seer_vs_h2o_z']:+.2f} p={d['seer_vs_h2o_p']:.4f} {marker}")
    print("\nSEER vs Streaming two-sample z-test (p-values):")
    for ms in slos:
        d = out["tests"][ms]
        marker = "***" if d["seer_vs_streaming_p"] < 0.001 else (
            "**" if d["seer_vs_streaming_p"] < 0.01 else (
                "*" if d["seer_vs_streaming_p"] < 0.05 else " "))
        print(f"  {ms:>3}ms: z={d['seer_vs_streaming_z']:+.2f} p={d['seer_vs_streaming_p']:.4f} {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
