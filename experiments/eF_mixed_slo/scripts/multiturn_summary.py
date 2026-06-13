"""Post-sweep summary for the multi-turn cross-prompt-KV-reuse cell.

Reads results_vllm_multiturn/{full,streaming,h2o,seer}_s{0,1,2}.json,
emits ``multiturn_summary.tex`` for the paper, and reports the
three Path-beta acceptance gates:

  G1: n_load > 50 per seed -> namespace gap closed
  G2: SEER chat-miss < H2O chat-miss with non-overlap Wilson 95% CI
  G3: SEER chat-miss reduction relative to H2O >= 20%

Honesty rule: the table prints whatever the data says. If a gate
fails, the failure is recorded; the paper text references the
gate-pass / gate-fail status without rewriting the headline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z*z / n
    centre = (p + z*z / (2*n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z*z / (4 * n*n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def load_cells(in_dir: Path) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    for f in sorted(in_dir.glob("*_s*.json")):
        if "boot_ci" in f.name or "aggregate" in f.name:
            continue
        policy = f.name.split("_")[0]
        with open(f) as fh:
            cells.setdefault(policy, []).append(json.load(fh))
    return cells


def summarise(cells: dict[str, list[dict]]) -> dict:
    out = {"per_policy": {}}
    for policy in ("full", "streaming", "h2o", "snapkv", "quest", "seer"):
        cs = cells.get(policy, [])
        if not cs:
            continue
        chat_miss = [c.get("chat_miss", 0.0) for c in cs]
        chat_miss_legacy = [c.get("chat_miss_legacy_prefill_inclusive",
                                  c.get("chat_miss", 0.0)) for c in cs]
        # Sum step-level n + misses across seeds for Wilson CI
        total_n = 0
        total_misses = 0
        n_save_total = 0
        n_load_total = 0
        n_load_miss_total = 0
        fwd_fires = 0
        n_attn_driven = 0
        wall_total = 0.0
        # v-beta-3 real-eviction counters (patched vLLM fork's
        # KVCacheManager.free path; see seer/integration/vllm_patches/
        # 0001-add-eviction-order-hook.patch). The scheduler-side
        # stats file is the only place these appear (the eviction
        # hook fires in the scheduler process), so we read them from
        # the per-cell connector stats sources.
        n_evict_fires = 0
        n_evict_kept = 0
        n_evict_front = 0
        for c in cs:
            for r in c.get("results", []):
                ts = r.get("tpot_stats") or r.get("stats") or {}
                total_n += int(ts.get("n", 0))
                total_misses += int(ts.get("miss_count", ts.get("misses", 0)))
            agg = (c.get("seer_connector_stats") or {}).get("aggregate")
            if not agg:
                agg = c.get("seer_connector_stats") or {}
            xfer = agg.get("xfer") or {}
            n_save_total += int(xfer.get("n_save", 0))
            n_load_total += int(xfer.get("n_load", 0))
            n_load_miss_total += int(xfer.get("n_load_miss", 0))
            n_attn_driven += int(agg.get("n_attn_driven_decisions", 0))
            fh = c.get("forward_hook_counters") or {}
            fwd_fires += int(fh.get("fire", 0))
            wall_total += float(c.get("wall_time_s", 0.0))
            # Eviction-hook counters: try top-level aggregate, then
            # fall back to per-source raw stats file (scheduler-only).
            ev_f = int(agg.get("n_eviction_hook_fires", 0))
            ev_k = int(agg.get("n_eviction_blocks_kept", 0))
            ev_fr = int(agg.get("n_eviction_blocks_front", 0))
            if ev_f == 0:
                for src in (c.get("seer_connector_stats")
                            or {}).get("sources", []):
                    try:
                        raw = json.load(open(src["path"]))
                        a2 = raw.get("aggregate") or raw
                        ev_f = max(ev_f,
                                   int(a2.get("n_eviction_hook_fires", 0)))
                        ev_k = max(ev_k,
                                   int(a2.get("n_eviction_blocks_kept", 0)))
                        ev_fr = max(ev_fr,
                                    int(a2.get("n_eviction_blocks_front", 0)))
                    except Exception:  # noqa: BLE001
                        pass
            n_evict_fires += ev_f
            n_evict_kept += ev_k
            n_evict_front += ev_fr
        lo, hi = wilson_ci(total_misses, total_n)
        out["per_policy"][policy] = {
            "n_seeds": len(cs),
            "chat_miss_mean": mean(chat_miss) if chat_miss else None,
            "chat_miss_step_total_n": total_n,
            "chat_miss_step_total_misses": total_misses,
            "chat_miss_step_level": (total_misses / total_n)
                                    if total_n > 0 else None,
            "chat_miss_wilson_lo": lo, "chat_miss_wilson_hi": hi,
            "n_save_total": n_save_total,
            "n_load_total": n_load_total,
            "n_load_miss_total": n_load_miss_total,
            "n_load_per_seed_mean": (n_load_total / len(cs)) if cs else 0,
            "forward_hook_fires": fwd_fires,
            "n_attn_driven_decisions": n_attn_driven,
            "wall_s_mean": wall_total / max(1, len(cs)),
            "n_eviction_hook_fires": n_evict_fires,
            "n_eviction_blocks_kept": n_evict_kept,
            "n_eviction_blocks_front": n_evict_front,
            "n_eviction_hook_fires_per_seed_mean":
                (n_evict_fires / len(cs)) if cs else 0,
        }
    return out


def gates(summary: dict) -> dict:
    pp = summary["per_policy"]
    seer = pp.get("seer", {})
    h2o = pp.get("h2o", {})
    g1 = seer.get("n_load_per_seed_mean", 0) > 50
    seer_mr = seer.get("chat_miss_step_level") or 1.0
    h2o_mr = h2o.get("chat_miss_step_level") or 1.0
    g2 = (seer_mr < h2o_mr and
          seer.get("chat_miss_wilson_hi", 1.0)
          < h2o.get("chat_miss_wilson_lo", 0.0))
    g3 = h2o_mr > 0 and (h2o_mr - seer_mr) / h2o_mr >= 0.20
    return {
        "G1_n_load_gt_50": bool(g1),
        "G1_value": seer.get("n_load_per_seed_mean", 0),
        "G2_non_overlap_wilson": bool(g2),
        "G2_seer_ci": (seer.get("chat_miss_wilson_lo"),
                       seer.get("chat_miss_wilson_hi")),
        "G2_h2o_ci": (h2o.get("chat_miss_wilson_lo"),
                      h2o.get("chat_miss_wilson_hi")),
        "G3_reduction_ge_20pct": bool(g3),
        "G3_relative_reduction": ((h2o_mr - seer_mr) / h2o_mr
                                  if h2o_mr > 0 else None),
    }


def emit_tex(summary: dict, gates_res: dict, out_path: Path) -> None:
    pp = summary["per_policy"]
    lines = [
        "% Generated by experiments/eF_mixed_slo/scripts/multiturn_summary.py",
        "% Path-beta multi-turn cross-prompt-KV-reuse end-to-end sweep on",
        "% Llama-2-7B + 1xA100 SXM4-80GB. Canonical beta4 cell:",
        "% 6 policies x 3 seeds, 10 threads x 10 turns per cell,",
        "% max_tokens=64, --disable_prefix_cache,",
        "% gpu_memory_utilization=0.20 (KV cache = 2,544 tokens),",
        "% chat_slo P99=200ms. SEER_HOOK_LAYER_STRIDE=1 (all 32 Attn layers),",
        "% SEER_BLOCK_TABLE_LOAD=1, SEER_BLOCK_TABLE_LOAD_COPY=1,",
        "% SEER_POLICY_ROUTE_WORKER=1, SEER_LAP_DECISION_PERIOD=64.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & step miss & 95\% CI & $n_\text{save}$ & "
        r"$n_\text{load}$ & $n_\text{miss}$ & wall (s) \\",
        r"\midrule",
    ]
    for policy in ("full", "streaming", "h2o", "snapkv", "quest", "seer"):
        p = pp.get(policy)
        if not p:
            continue
        cm = p["chat_miss_step_level"]
        lo, hi = p["chat_miss_wilson_lo"], p["chat_miss_wilson_hi"]
        lines.append(
            rf"\texttt{{{policy}}} & "
            rf"${cm:.4f}$ & "
            rf"$[{lo:.4f}, {hi:.4f}]$ & "
            rf"${p['n_save_total']:,}$ & "
            rf"${p['n_load_total']:,}$ & "
            rf"${p['n_load_miss_total']:,}$ & "
            rf"${p['wall_s_mean']:.1f}$ \\".replace(",", r"{,}")
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="experiments/eF_mixed_slo/results_vllm_multiturn")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    cells = load_cells(in_dir)
    if not cells:
        print(f"[multiturn-summary] no cells in {in_dir}")
        return 1
    summary = summarise(cells)
    gates_res = gates(summary)
    summary["gates"] = gates_res
    (in_dir / "multiturn_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    emit_tex(summary, gates_res, in_dir / "multiturn_summary.tex")
    print(f"[multiturn-summary] wrote "
          f"{in_dir/'multiturn_summary.json'} and "
          f"{in_dir/'multiturn_summary.tex'}")
    print("Per-policy:")
    for p, d in summary["per_policy"].items():
        print(f"  {p:10s} step_miss={d['chat_miss_step_level']:.4f}  "
              f"CI=[{d['chat_miss_wilson_lo']:.4f},"
              f"{d['chat_miss_wilson_hi']:.4f}]  "
              f"n_load={d['n_load_total']}  "
              f"fwd_fires={d['forward_hook_fires']}")
    print("Acceptance gates:")
    for k, v in gates_res.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
