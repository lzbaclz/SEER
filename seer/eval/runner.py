"""CLI: run a benchmark with a given policy + model under an SLO.

Example
-------

    python -m seer.eval.runner \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --policy seer --lap checkpoints/lap_llama3_8b.onnx \\
        --workload sharegpt \\
        --slo P99=50ms \\
        --hbm_budget 0.2 \\
        --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \\
        --out results/eA_seer.json

The runner produces a single JSON file containing per-request results,
aggregated quality, RT percentiles, miss ratio, and the analytical
schedulability bound from Lemma 2 (when ``--lap`` is supplied so we can
estimate ε).

Backwards-compat: the legacy NeurIPS-style flags (``--benchmark``,
``--budget``, ``--lap_ckpt``) still work; the runner aliases them onto
the RTSS-style flags.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SEER benchmark runner.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", required=True,
                    choices=["full", "streaming", "h2o", "snapkv", "quest",
                             "recency", "random", "seer", "seer-safe",
                             "seer-warmup", "infinigen"])
    # both --lap (RTSS spelling) and --lap_ckpt (NeurIPS) accepted
    ap.add_argument("--lap", "--lap_ckpt", dest="lap", default=None,
                    help="LAP checkpoint (.onnx / .plan / .pt). Required for "
                         "--policy seer.")
    # workload selection
    ap.add_argument("--workload", "--benchmark", dest="workload", default="ruler",
                    choices=["ruler", "longbench", "sharegpt", "synthetic",
                             "mooncake"])
    ap.add_argument("--context_length", type=int, default=8192)
    # both --num_requests / --num_prompts
    ap.add_argument("--num_requests", "--num_prompts", dest="num_requests",
                    type=int, default=50)
    # SLO + budget
    ap.add_argument("--slo", default="P99=50ms",
                    help="SLO spec (e.g. 'P99=50ms', 'chat-50ms').")
    ap.add_argument("--hbm_budget", "--budget", dest="hbm_budget",
                    type=float, default=0.2,
                    help="HBM block budget as a fraction of full cache.")
    ap.add_argument("--metrics", default="tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality",
                    help="Comma-separated metric tags to include in the summary "
                         "(everything is always computed; this is just for filtering).")
    # tail latency target eC parameters
    ap.add_argument("--ell_bar_us", type=float, default=200.0,
                    help="Avg miss-tier IO latency for the analytical bound.")
    ap.add_argument("--sigma_residual_us", type=float, default=100.0,
                    help="Sub-Gaussian proxy of residual cost for Lemma 2.")
    ap.add_argument("--io_mode", default="measured-dma",
                    choices=["analytical", "measured-dma"],
                    help="IO penalty model. 'measured-dma' (default since "
                         "RTSS reviewer round M1/M3) times a real "
                         "cudaMemcpyAsync HtoD on a pinned-host scratch "
                         "for each miss. 'analytical' uses the Lemma-1 "
                         "first-moment form (misses*ell_bar/period); "
                         "kept for cross-check ablation only. The "
                         "measured-dma path addresses the OSDI/RTSS "
                         "reviewer concern that the analytical bound is "
                         "validated against its own form (circular).")
    # generation knobs
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0,
                    help="Master seed seeded into torch / numpy / Python "
                         "random; the runner output filename is suffixed by "
                         "the seed so multi-seed sweeps can write to the "
                         "same out dir without colliding.")
    ap.add_argument("--decision_period", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device_map", default=None,
                    help="HF device_map for multi-GPU sharding (e.g. 'auto'). "
                         "Use for 70B+ models on TP=2/4 hosts. When set, "
                         "--device is ignored (HF places shards itself).")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", required=True)
    # safe-fallback wrapper config
    ap.add_argument("--fallback", default=None,
                    help="Optional baseline name to wrap SEER with a confidence-"
                         "based safe fallback (e.g. 'h2o').")
    ap.add_argument("--fallback_threshold", type=float, default=0.4)
    # eD adversarial path: load prompts from a JSONL file
    ap.add_argument("--skip_prewarm", action="store_true",
                    help="Skip the cold-start mitigation (JIT pre-warm of "
                         "decode kernels). Default behaviour pre-warms once "
                         "before the first request so step 0 of the first "
                         "user prompt does not pay the JIT cost.")
    ap.add_argument("--prompts_file", default=None,
                    help="Optional JSONL file with one record per line: "
                         "{prompt, answer, family?, sigma?}. When set, "
                         "overrides --workload prompt loading. The 'answer' "
                         "field is used for substring/F1 quality scoring.")
    return ap


def _build_policy(args, model_meta: dict | None = None):
    from seer.policy import build_policy
    from seer.policy.fallback import SafeFallbackPolicy
    from seer.timing import parse_slo

    kwargs: dict = {}
    if args.policy in ("seer", "seer-warmup"):
        if not args.lap:
            raise SystemExit(f"--lap is required for --policy {args.policy}")
        from seer.lap.infer import LAPPredictor
        predictor = LAPPredictor.from_path(args.lap, device=args.device)
        kwargs["lap_predictor"] = predictor
        kwargs["slo"] = parse_slo(args.slo)
    primary = build_policy(args.policy, **kwargs)
    if args.fallback and args.policy == "seer":
        fb = build_policy(args.fallback)
        return SafeFallbackPolicy(
            primary=primary, fallback=fb,
            conf_threshold=args.fallback_threshold,
        )
    return primary


def _select_metrics(summary: dict, requested: list[str]) -> dict:
    """Return ``summary`` filtered to only the requested keys (debug aid)."""
    if not requested:
        return summary
    keep = set(requested)
    return {k: v for k, v in summary.items() if k in keep or k == "results"}


def main():
    args = _build_argparser().parse_args()

    import random

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from seer.eval.metrics import (
        LatencyStats,
        bound_pessimism,
        exact_match,
        f1_score,
        substring_match,
    )
    from seer.eval.sim import simulate_attention_mask
    from seer.timing import (
        derive_deadline_us,
        lemma2_miss_prob_bound,
        parse_slo,
    )
    from seer.trace.datasets import load_prompts_with_refs

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    print(f"[runner] loading {args.model} (policy={args.policy}, "
          f"budget={args.hbm_budget}, slo={args.slo})")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation="eager",
            device_map=args.device_map,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(args.device).eval()

    # --- Build policy
    policy = _build_policy(args)

    # --- Cold-start mitigation: JIT-warm the model decode kernels
    # before the first real prompt so step 0 of the user's first
    # request doesn't pay the JIT compile cost. The eA/eC steady-state
    # vs warm-up-included split (P999 49.9 ms vs 37.9 ms) is dominated
    # by this first-step outlier; pre-warming pulls the warm-up bound
    # into line with the steady-state one. See seer/eval/prewarm.py.
    if not getattr(args, "skip_prewarm", False):
        try:
            from seer.eval.prewarm import prewarm_model_for_decode
            warm = prewarm_model_for_decode(model, tok, device=args.device,
                                            prompt_tokens=64, decode_tokens=8)
            if warm["did_run"]:
                print(f"[runner] pre-warm took {warm['warmup_us']/1000:.1f} ms "
                      "(JIT decode kernels + CUDA Graph capture; first user "
                      "step will not pay this cost)")
        except Exception as e:  # noqa: BLE001
            print(f"[runner] pre-warm failed ({e}); first decode step may "
                  "include JIT cost — see eA warm-up vs steady-state split")

    # --- Resolve SLO + deadline
    slo = parse_slo(args.slo)
    deadline_us = derive_deadline_us(slo)
    print(f"[runner] slo={slo.name}  deadline={deadline_us:.0f}µs  "
          f"miss_target={slo.miss_target:.4f}")

    # --- Load prompts (from --prompts_file if given, else --workload)
    answers: list[str] = []
    family = None
    sigma = None
    if args.prompts_file:
        prompts = []
        with open(args.prompts_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompts.append(rec["prompt"])
                answers.append(rec.get("answer", ""))
                family = family or rec.get("family")
                sigma  = sigma  if sigma is not None else rec.get("sigma")
        print(f"[runner] loaded {len(prompts)} prompts from "
              f"{args.prompts_file} (family={family}, sigma={sigma})")
    else:
        # For Mooncake the SQuAD-stitched synthesiser (opt-in via
        # SEER_MOONCAKE_SYNTHESISER=squad) ships parallel reference
        # answers so the F1 column becomes meaningful. Other workloads
        # fall back to empty references / the RULER-needle regex baked
        # into seer.eval.sim._extract_ruler_reference.
        prompts, answers = load_prompts_with_refs(
            args.workload, [args.context_length], args.num_requests, tok
        )
        synth = os.environ.get("SEER_MOONCAKE_SYNTHESISER", "legacy")
        print(f"[runner] running {len(prompts)} prompts "
              f"(workload={args.workload}, synthesiser={synth})")

    # --- Run
    results = []
    all_latencies_us: list[float] = []
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        policy.reset()
        r = simulate_attention_mask(
            model, tok, prompt, policy,
            budget_frac=args.hbm_budget,
            max_new_tokens=args.max_new_tokens,
            decision_period=args.decision_period,
            ell_bar_us=args.ell_bar_us,
            io_mode=args.io_mode,
        )
        r["id"] = i
        # When the prompt source ships its own reference (prompts_file
        # for eD adversarial, or load_prompts_with_refs for Mooncake +
        # SQuAD synthesiser), prefer that over the RULER-needle regex
        # baked into _extract_ruler_reference in seer.eval.sim.
        if i < len(answers) and answers[i]:
            r["ref"] = answers[i]
        r["f1"] = f1_score(r["pred"], r["ref"])
        r["em"] = exact_match(r["pred"], r["ref"])
        r["substring"] = substring_match(r["pred"], r["ref"])
        # Aggregate per-step latencies for this request → tail-latency stats
        r["latency_stats"] = LatencyStats.from_vector(
            r.get("per_step_us", []), deadline_us=deadline_us
        ).to_dict()
        all_latencies_us.extend(r.get("per_step_us", []))
        results.append(r)
        if (i + 1) % 10 == 0:
            cur_f1 = sum(x["f1"] for x in results) / len(results)
            cur_p99 = LatencyStats.from_vector(
                all_latencies_us, deadline_us=deadline_us
            ).p99_us
            print(f"[runner]   [{i+1}/{len(prompts)}] mean F1={cur_f1:.3f}  "
                  f"P99 TPOT={cur_p99/1000:.1f}ms")
    dt = time.time() - t0

    # --- Aggregate RT metrics across all requests
    overall = LatencyStats.from_vector(all_latencies_us, deadline_us=deadline_us)

    # --- Analytical Lemma 2 bound (compute when we have the inputs)
    bound2 = None
    pessimism = None
    base_cost_us_used = None
    B_t_avg_used = None
    B_t_source: str | None = None
    epsilon: float | None = None
    eps_source: str = "n/a (non-seer)"
    if args.lap and args.policy == "seer":
        # T1-B fix (May 2026 reviewer round R1-N1/C3, R2-#3):
        # the previous path called ``_estimate_epsilon_from_lap`` with
        # ``workload=None`` which silently returned a hardcoded 0.15
        # fallback inside schedulability.py — meaning every "calibrated"
        # bound in the paper actually used the same ε regardless of
        # workload, budget, or LAP. We now compute ε directly from this
        # cell's runtime: each decode step produced a
        # ``per_step_eps_measured`` entry =
        #   |top-K wanted \ kept_set| / |top-K wanted|
        # i.e. the per-step false-negative rate of the deployed policy
        # against the schema-consistent top-K wanted set (T1-A).
        # The mean of those entries is the per-cell measured ε; we keep
        # the legacy estimator only as a last-resort fallback when the
        # simulator did not emit the oracle pass.
        eps_pool: list[float] = []
        for r in results:
            eps_pool.extend(float(e) for e in r.get("per_step_eps_measured", []))
        if eps_pool:
            epsilon = sum(eps_pool) / len(eps_pool)
            eps_source = "runner_measured_topk"
        else:
            try:
                from seer.timing.schedulability import (
                    _estimate_epsilon_from_lap,  # noqa: PLC2701
                )
                epsilon = _estimate_epsilon_from_lap(
                    args.lap, workload=None, hbm_budget=args.hbm_budget,
                )
                eps_source = "lap_estimator_fallback"
            except Exception:  # noqa: BLE001
                # Honest failure: no measured ε + estimator unavailable.
                # Bound is unreliable; mark it so callers don't quote it.
                epsilon = float("nan")
                eps_source = "missing"
        # B_t: working-set size required by the schedulability bound.
        # P0-6 fix (review round May 2026): prefer the ORACLE working-set
        # size (``per_step_B_t_oracle``) — the count of blocks the
        # unmasked attention layer attended to at each step. The legacy
        # ``per_step_block_count`` is the size of the policy's kept
        # set (bounded by the budget by construction), which makes
        # B_t structurally smaller than what the analytical bound
        # expects under any non-trivial eviction. Fall back to
        # ``per_step_block_count`` only when the simulator did not run
        # the oracle pass (e.g. ``compute_oracle=False`` smoke tests
        # or the forward-once ``full`` path).
        oracle_counts: list[int] = []
        for r in results:
            oracle_counts.extend(
                int(b) for b in r.get("per_step_B_t_oracle", [])
            )
        if oracle_counts:
            B_t_avg = sum(oracle_counts) / len(oracle_counts)
            B_t_source = "oracle_attention_top_k"
        else:
            block_counts: list[int] = []
            for r in results:
                block_counts.extend(
                    int(b) for b in r.get("per_step_block_count", [])
                )
            if block_counts:
                B_t_avg = sum(block_counts) / len(block_counts)
                B_t_source = "kept_set_size_legacy"
            else:
                B_t_avg = 64.0
                B_t_source = "default_64"
        # base_cost: mean of the IO-free residual (per_step_base_us) so the
        # bound's μ = base + ε·B·ℓ̄ does not double-count the IO term.
        # Falls back to overall.mean_us when the simulator did not emit the
        # decomposition (legacy path) — flagged in the summary so analyses
        # downstream know which calibration was used.
        base_us_pool: list[float] = []
        for r in results:
            base_us_pool.extend(float(b) for b in r.get("per_step_base_us", []))
        if base_us_pool:
            base_cost_us = sum(base_us_pool) / len(base_us_pool)
            base_cost_source = "io_free_mean"
        else:
            base_cost_us = overall.mean_us
            base_cost_source = "total_mean_legacy"
        base_cost_us_used = base_cost_us
        B_t_avg_used = B_t_avg
        bound2 = lemma2_miss_prob_bound(
            epsilon=epsilon,
            ell_bar_us=args.ell_bar_us,
            B_t=B_t_avg,
            deadline_us=deadline_us,
            sigma_residual_us=args.sigma_residual_us,
            base_cost_us=base_cost_us,
        )
        pessimism = bound_pessimism(bound2, overall.miss_ratio)

    summary = {
        "model": args.model,
        "policy": args.policy,
        "fallback": args.fallback,
        "hbm_budget": args.hbm_budget,
        "workload": args.workload,
        "prompts_file": args.prompts_file,
        "family": family,
        "sigma": sigma,
        "slo": {
            "name": slo.name, "kind": slo.kind,
            "percentile": slo.percentile, "threshold_ms": slo.threshold_ms,
            "miss_target": slo.miss_target,
        },
        "context_length": args.context_length,
        "num_requests": args.num_requests,
        "wall_time_s": dt,
        # Quality
        "f1_mean": sum(r["f1"] for r in results) / len(results),
        "em_mean": sum(r["em"] for r in results) / len(results),
        "substring_mean": sum(r["substring"] for r in results) / len(results),
        # Throughput
        "tokens_per_second": sum(r["n_gen_tokens"] for r in results) / dt,
        # RT (aggregated)
        "tpot_p50_us": overall.p50_us,
        "tpot_p90_us": overall.p90_us,
        "tpot_p99_us": overall.p99_us,
        "tpot_p999_us": overall.p999_us,
        "tpot_max_us": overall.max_us,
        "miss_ratio": overall.miss_ratio,
        "miss_count": overall.miss_count,
        "deadline_us": deadline_us,
        # Schedulability
        "bound_lemma2_miss": bound2,
        "pessimism": pessimism,
        "bound_inputs": {
            "base_cost_us": base_cost_us_used,
            "base_cost_source": ("io_free_mean" if base_cost_us_used is not None
                                  and B_t_avg_used is not None else None),
            "B_t_avg": B_t_avg_used,
            "B_t_source": B_t_source,
            "ell_bar_us": args.ell_bar_us,
            "sigma_residual_us": args.sigma_residual_us,
            # T1-B (May 2026 reviewer round): per-cell measured ε
            # against the schema top-K wanted set (T1-A). The previous
            # silent 0.15 fallback is no longer accepted; if the
            # simulator did not emit the oracle pass and the estimator
            # fallback also fails, ``epsilon`` is NaN and ``eps_source``
            # is "missing" so aggregators can drop the cell rather
            # than silently quote a fabricated number.
            "epsilon": epsilon,
            "eps_source": eps_source,
        },
        "io_mode": args.io_mode,
        "policy_stats": (policy.stats() if hasattr(policy, "stats") and
                          callable(getattr(policy, "stats", None)) else None),
        "results": results,
    }

    requested = [s.strip() for s in args.metrics.split(",") if s.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[runner] saved → {args.out}")
    print(
        f"[runner] F1={summary['f1_mean']:.3f}  "
        f"P50={overall.p50_us/1000:.2f}ms  "
        f"P99={overall.p99_us/1000:.2f}ms  "
        f"P999={overall.p999_us/1000:.2f}ms  "
        f"miss={overall.miss_ratio:.3%}  "
        + (f"bound={bound2:.3e}  pessimism={pessimism:.2f}×  "
           if bound2 is not None else "")
        + f"t={dt:.1f}s",
    )
    # Print tag-filtered top-level summary as a final reminder
    if requested:
        keys_in_summary = {
            "tpot_p50": "tpot_p50_us", "tpot_p99": "tpot_p99_us",
            "tpot_p999": "tpot_p999_us", "miss_ratio": "miss_ratio",
            "quality": "f1_mean",
        }
        wanted = {tag: summary.get(keys_in_summary.get(tag, tag)) for tag in requested}
        print("[runner] requested metrics:", json.dumps(wanted))


if __name__ == "__main__":
    main()
