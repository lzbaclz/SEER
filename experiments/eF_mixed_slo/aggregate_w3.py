"""W3 follow-through: aggregate vLLM-end-to-end three-seed runs.

R1/R2 W3 follow-through (2026-05-15): previously the only real vLLM
artifact (results_vllm_separation/*_nopfx_chat20.json) had
``planner_decisions=0`` and the post-submission t2i_verification
rerun got ``chat_miss=1.0`` from host CPU contention. This script
aggregates a clean three-seed sweep produced by
``experiments.eF_mixed_slo.driver --seed {0,1,2}`` per policy on
the same absorber-off / prefix-cache-off regime.

Per (policy) we report:
* Mean / Wilson 95\\% CI on chat-miss across the three seeds.
* P99 and P999 (mean across seeds) on per-step decode latency for
  seeds that used ``--use_engine_step``.
* SEER mechanistic counters (``planner_decisions``, ``hit_rate``,
  ``n_save / n_load``) from ``seer_connector_stats``.

Emits:
* ``results/vllm_w3_aggregate.csv``
* ``results/vllm_w3_aggregate.tex`` (drop-in replacement for the
  ``tab:vllm-separation`` source).
* ``results/vllm_w3_aggregate.json`` (machine-readable + paper
  hooks).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return lo, hi


def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _aggregate_one(jsons: list[dict]) -> dict:
    """Aggregate over 1+ seed runs of the same policy."""
    if not jsons:
        return {}
    # T2-M (R1-Cut1 follow-up): two miss-ratio definitions tracked.
    # ``chat_miss_step_level_mean`` is the step-level miss-ratio
    # (total decode misses / total decode steps), pooled across
    # seeds — this is the headline metric reviewers asked for, free
    # of the per-prompt-mean bimodal artefact. ``chat_miss_mean``
    # remains the legacy mean-of-per-prompt-miss-ratio for the
    # archival comparison vs. earlier sweeps.
    chat_misses = [float(d.get("chat_miss", 0.0)) for d in jsons]
    chat_miss_mean = statistics.mean(chat_misses) if chat_misses else 0.0
    # chat_miss_step_level is computed directly from
    # ``miss_total / n_total`` over the pooled step samples below;
    # we initialise here and finalise after the aggregation loop.
    chat_miss_step_level_mean = 0.0
    # Wilson over the *combined* per-prompt counts to get a proper CI.
    n_total = 0
    miss_total = 0
    p99_us, p999_us, max_us = [], [], []
    planner_decisions = []
    n_save = []
    n_load = []
    n_load_miss = []
    hit_rate = []
    save_us_mean = []
    n_attn_driven = []
    n_recency_fallback = []
    fwd_fire = []
    for d in jsons:
        rs = d.get("results", []) or []
        for r in rs:
            if r.get("kind") != "chat":
                continue
            # T2-M (R1-Cut1 follow-up): chat-miss / P99 / P999 now read
            # from ``tpot_stats`` (decode-only per-token latency, the
            # TPOT SLO metric). The legacy ``stats`` field (full trace
            # including the batched-prefill step) is retained in JSON
            # for back-compat but no longer aggregated as chat-miss.
            stats = r.get("tpot_stats") or r.get("stats", {}) or {}
            n = int(stats.get("n", 0))
            misses = int(stats.get("miss_count", stats.get("misses", 0)))
            n_total += n
            miss_total += misses
            if stats.get("p99_us") is not None:
                p99_us.append(float(stats["p99_us"]))
            if stats.get("p999_us") is not None:
                p999_us.append(float(stats["p999_us"]))
            if stats.get("max_us") is not None:
                max_us.append(float(stats["max_us"]))
        # connector stats (may be nested under aggregate)
        cs = d.get("seer_connector_stats") or {}
        agg = cs.get("aggregate") if isinstance(cs, dict) else None
        if agg:
            if agg.get("planner_decisions") is not None:
                planner_decisions.append(int(agg["planner_decisions"]))
            if agg.get("hit_rate") is not None:
                hit_rate.append(float(agg["hit_rate"]))
            if agg.get("n_attn_driven_decisions") is not None:
                n_attn_driven.append(int(agg["n_attn_driven_decisions"]))
            if agg.get("n_recency_fallback_decisions") is not None:
                n_recency_fallback.append(
                    int(agg["n_recency_fallback_decisions"]))
            xf = agg.get("xfer") or {}
            for k, dst in (("n_save", n_save), ("n_load", n_load),
                           ("n_load_miss", n_load_miss),
                           ("save_us_mean", save_us_mean)):
                if xf.get(k) is not None:
                    dst.append(float(xf[k]))
        fh = d.get("forward_hook_counters") or {}
        if isinstance(fh, dict) and "fire" in fh:
            fwd_fire.append(int(fh["fire"]))
    lo, hi = _wilson_ci(miss_total, max(1, n_total))
    chat_miss_step_level_mean = (miss_total / n_total) if n_total > 0 else 0.0
    return {
        "n_seeds": len(jsons),
        "n_chat_prompts_total": n_total,
        "chat_miss_mean": chat_miss_mean,
        "chat_miss_step_level_mean": chat_miss_step_level_mean,
        "chat_miss_wilson_lo": lo,
        "chat_miss_wilson_hi": hi,
        "p99_us_mean_across_prompts": statistics.mean(p99_us) if p99_us else None,
        "p999_us_mean_across_prompts": statistics.mean(p999_us) if p999_us else None,
        "max_us_mean_across_prompts": statistics.mean(max_us) if max_us else None,
        "planner_decisions_mean": statistics.mean(planner_decisions) if planner_decisions else None,
        "planner_decisions_total": sum(planner_decisions) if planner_decisions else None,
        "hit_rate_mean": statistics.mean(hit_rate) if hit_rate else None,
        "n_save_total": sum(n_save) if n_save else None,
        "n_load_total": sum(n_load) if n_load else None,
        "n_load_miss_total": sum(n_load_miss) if n_load_miss else None,
        "n_attn_driven_total": sum(n_attn_driven) if n_attn_driven else None,
        "n_recency_fallback_total": (
            sum(n_recency_fallback) if n_recency_fallback else None),
        "forward_hook_fire_total": sum(fwd_fire) if fwd_fire else None,
    }


def _fmt_us_ms(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{x / 1000:.1f}"


def _fmt_chat_miss(mean: float, lo: float, hi: float) -> str:
    return f"${mean:.4f}\\,[{lo:.4f},{hi:.4f}]$"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="Directory with *_s{0,1,2}.json per policy.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--policies", nargs="+",
                    default=["full", "streaming", "h2o", "seer"])
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_policy: dict[str, dict] = {}
    rows_csv: list[dict] = []
    for policy in args.policies:
        files = sorted(in_dir.glob(f"{policy}_s*.json"))
        jsons = []
        for f in files:
            try:
                jsons.append(json.load(open(f)))
            except Exception as e:  # noqa: BLE001
                print(f"[w3] skip {f}: {e}")
        if not jsons:
            print(f"[w3] no files for policy={policy}")
            continue
        agg = _aggregate_one(jsons)
        by_policy[policy] = agg
        cm_show = agg.get("chat_miss_step_level_mean", agg["chat_miss_mean"])
        print(
            f"[w3] {policy:>10}: n_seeds={agg['n_seeds']} "
            f"chat_miss={cm_show:.4f}"
            f" [{agg['chat_miss_wilson_lo']:.4f},"
            f"{agg['chat_miss_wilson_hi']:.4f}]"
            f"  P99={_fmt_us_ms(agg['p99_us_mean_across_prompts'])}ms"
            f"  P999={_fmt_us_ms(agg['p999_us_mean_across_prompts'])}ms"
            f"  planner_decisions(total)={agg['planner_decisions_total']}"
        )
        rows_csv.append({"policy": policy, **agg})

    # CSV
    if rows_csv:
        import csv
        with open(out_dir / "vllm_w3_aggregate.csv", "w", newline="") as fh:
            keys = list(rows_csv[0].keys())
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_csv)
        print(f"[w3] wrote {out_dir/'vllm_w3_aggregate.csv'}")

    # JSON
    with open(out_dir / "vllm_w3_aggregate.json", "w") as fh:
        json.dump({"by_policy": by_policy}, fh, indent=2, default=str)
    print(f"[w3] wrote {out_dir/'vllm_w3_aggregate.json'}")

    # TeX — drop-in replacement for tab:vllm-separation source.
    # When forward-hook fired in the underlying runs, widen the table
    # with the Phase 2 signal-source columns.
    any_attn = any(
        (by_policy.get(p) or {}).get("n_attn_driven_total")
        for p in args.policies
    )
    lines = [
        "% Generated by experiments/eF_mixed_slo/aggregate_w3.py",
        "% Three-seed vLLM 0.8.5.post1 absorber-off + prefix-cache-off",
        "% run. All four policies share the same SeerKVConnector",
        "% plumbing; only the extra_config[\"policy\"] selector differs.",
        "% planner_decisions > 0 confirms the worker-side plumbing fires",
        "% (R1/R2 W3 follow-through). Wilson 95% CI is computed over the",
        "% union of per-prompt step samples across seeds. The optional",
        "% n_attn / n_rec columns appear when the forward-hook patch",
        "% installed: n_attn = planner.plan() calls fed by the real",
        "% per-block attention proxy from the forward-hook stash; n_rec",
        "% = recency-tail synthetic fallback for un-hooked layers /",
        "% throttled steps (Phase 2 attention signal load-bearing).",
    ]
    if any_attn:
        lines.append("\\begin{tabular}{lrrrrrr}")
        lines.append("\\toprule")
        lines.append(
            "Policy & chat-miss (Wilson 95\\% CI) & P99 (ms) & P999 (ms) "
            "& planner\\_dec.\\ & n\\_attn & n\\_rec \\\\"
        )
    else:
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append(
            "Policy & chat-miss (Wilson 95\\% CI) & P99 (ms) & P999 (ms) "
            "& planner\\_dec.\\ & hit-rate \\\\"
        )
    lines.append("\\midrule")
    for policy in args.policies:
        if policy not in by_policy:
            continue
        a = by_policy[policy]
        # Prefer step-level miss-ratio when available; falls back to
        # the legacy per-prompt mean for older JSON dumps.
        cm_val = a.get("chat_miss_step_level_mean", a["chat_miss_mean"])
        cm = _fmt_chat_miss(cm_val, a["chat_miss_wilson_lo"],
                            a["chat_miss_wilson_hi"])
        p99 = _fmt_us_ms(a["p99_us_mean_across_prompts"])
        p999 = _fmt_us_ms(a["p999_us_mean_across_prompts"])
        pd = a.get("planner_decisions_total")
        pd_str = f"{int(pd)}" if pd is not None else "--"
        if any_attn:
            na = a.get("n_attn_driven_total") or 0
            nr = a.get("n_recency_fallback_total") or 0
            lines.append(
                f"\\texttt{{{policy}}} & {cm} & {p99} & {p999} & "
                f"{pd_str} & {int(na)} & {int(nr)} \\\\"
            )
        else:
            hr = a.get("hit_rate_mean")
            hr_str = f"{hr:.3f}" if hr is not None else "--"
            lines.append(
                f"\\texttt{{{policy}}} & {cm} & {p99} & {p999} & "
                f"{pd_str} & {hr_str} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    (out_dir / "vllm_w3_aggregate.tex").write_text("\n".join(lines) + "\n")
    print(f"[w3] wrote {out_dir/'vllm_w3_aggregate.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
