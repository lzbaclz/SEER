"""R36f vLLM hot-path measured end-to-end comparison (cycle-2 deliverable).

§VI.D currently says "decode-step hook shipped (vllm_hot_path_hook.py,
11 tests pass), measured delta queued (todo_atc.md A2.4-A2.5)".  This
script produces the **measured delta**: same workload, same SLO, same
seed -- run once WITH the hook + a recency-based router installed,
run once WITHOUT.  Compare chat-miss + P99 TPOT.

Why this matters: the framework's §VI.D currently labels the vLLM
patch a "diagnosed non-binding".  If the hot-path hook produces a
measurable chat-miss delta on a workload that exercises prefix-cache
hits, the "non-binding" framing can be retracted (or qualified) in
the paper text.

Design:

  1. Mooncake-24-style multi-turn workload (n_turns=5) with shared
     prefix structure between turns -> each turn 2..n hits the
     prefix-cache and the hook's KVCacheManager.get_computed_blocks
     wrapper fires.
  2. Recency-based scorer: later block_ids = more important (decode
     blocks weighted over prompt blocks).  Simple but realistic.
  3. SLO = P99=200ms (relaxed band, same as §VI.A).
  4. Tight slack budget configured in HotPathRouter (ell_bar = 200us)
     so the hook actually truncates blocks under load.
  5. Two-run protocol:
       a) baseline (hook OFF) -> baseline.json
       b) with-hook (hook ON, router truncates under slack)
          -> with_hook.json
       Same seed, same workload, same vLLM init.
  6. Output a comparison JSON + a small TeX table.

Runs on GPU 0 (GPU 1 may be busy with operator's other work).
~10-15 min wall total.

Output:
  paper/hot_path_measured.{json,tex}
  experiments/eF_mixed_slo/results_hot_path/{baseline,with_hook}.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments/eF_mixed_slo/results_hot_path"
PAPER_JSON = ROOT / "paper/hot_path_measured.json"
PAPER_TEX = ROOT / "paper/hot_path_measured.tex"

# Optional: bump vLLM V1's multiproc executor execute_model RPC
# timeout (default 40s) via VLLM_EXECUTE_MODEL_TIMEOUT_S env var.
# Useful for TP-2/TP-4 on shared hosts where the profile-pass
# forward can be slow under cross-tenant contention.
_VLLM_EXECUTE_TIMEOUT_S = int(os.environ.get(
    "VLLM_EXECUTE_MODEL_TIMEOUT_S", "0"))


def _build_workload(num_prompts: int, n_turns: int, ctx_len: int):
    """Synthetic multi-turn shared-prefix workload.

    Each "conversation" has a fixed prefix (1024 tokens) + a small
    turn-specific suffix (32 tokens).  After turn 1 vLLM's prefix-
    cache has the shared prefix; turn 2..n hit it and trigger the
    KVCacheManager.get_computed_blocks hook.
    """
    import random
    rng = random.Random(0)
    base_vocab = list(range(1000, 30000))
    convs = []
    for i in range(num_prompts):
        # Shared prefix per conversation (32 token-ids per block;
        # 1024 tokens = 32 blocks -> 32 chances for hook to fire).
        prefix = [rng.choice(base_vocab) for _ in range(ctx_len)]
        turns = []
        for t in range(n_turns):
            suffix = [rng.choice(base_vocab) for _ in range(32)]
            turns.append(prefix + suffix)
        convs.append(turns)
    return convs


def _run_one(label: str, with_hook: bool, args) -> dict:
    """Run one vLLM sweep, optionally with the hot-path hook active."""
    # Bookkeeping for the hook firings (captured via a wrapped router).
    router_fire_count = {"n": 0, "truncations": 0,
                         "kept_block_count_sum": 0}

    if with_hook:
        # Install hook + register a recency-based router.
        from seer.integration.vllm_hot_path_hook import (
            install_hot_path_hook, register_router,
            set_request_slo_us, set_request_last_forward_us,
        )
        from seer.integration.hot_path_router import HotPathRouter

        # Wrap the scorer to count firings.
        def recency_scorer(request_id: str, block_ids):
            router_fire_count["n"] += 1
            if router_fire_count["n"] <= 3 or router_fire_count["n"] % 50 == 0:
                print(f"[hot-path/scorer] fire #{router_fire_count['n']} "
                      f"req={request_id} n_blocks={len(block_ids)} "
                      f"pid={os.getpid()}", flush=True)
            # Later block_ids are more recent (later in the trace);
            # we weight them higher so the router keeps recent blocks
            # under tight slack.
            return [float(bid) for bid in block_ids]

        router = HotPathRouter(score_fn=recency_scorer,
                               ell_bar_us=args.ell_bar_us)
        install_hot_path_hook()
        register_router(router)
        # Wrap max_blocks_under_budget to detect actual truncations.
        _orig_max = router.max_blocks_under_budget
        def _wrapped_max(slack_us):
            v = _orig_max(slack_us)
            router_fire_count["kept_block_count_sum"] += v
            return v
        router.max_blocks_under_budget = _wrapped_max
        print(f"[hot-path/{label}] hook installed; ell_bar_us={args.ell_bar_us}")
    else:
        print(f"[hot-path/{label}] baseline (no hook)")

    import vllm
    if _VLLM_EXECUTE_TIMEOUT_S > 0:
        try:
            import vllm.v1.executor.multiproc_executor as _mp_exec
            _mp_exec.EXECUTE_MODEL_TIMEOUT_S = _VLLM_EXECUTE_TIMEOUT_S
            print(f"[hot-path/{label}] patched "
                  f"EXECUTE_MODEL_TIMEOUT_S={_VLLM_EXECUTE_TIMEOUT_S}s")
        except Exception as _e:
            print(f"[hot-path/{label}] timeout-patch skipped: {_e}")
    print(f"[hot-path/{label}] vllm {vllm.__version__}; loading {args.model}")
    llm_kwargs = dict(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,  # essential for hook to fire
        disable_log_stats=True,
    )
    if args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        # vLLM 0.8.5 custom_all_reduce often deadlocks on TP-2 startup
        # when the host's NCCL setup is fresh; the standard all_reduce
        # path is slower but reliable.
        llm_kwargs["disable_custom_all_reduce"] = True
    llm = vllm.LLM(**llm_kwargs)

    sp = vllm.SamplingParams(max_tokens=args.max_tokens,
                              temperature=0.0)

    convs = _build_workload(args.num_prompts, args.n_turns,
                            args.ctx_len)
    # Flatten by turn order so prefix-cache cumulates as we iterate.
    per_request_tpot_us = []
    per_request_ttft_us = []
    deadline_us = float(args.slo_p99_ms) * 1000.0
    miss_us_counts = {"chat_miss_request": 0, "total_requests": 0}

    t_total_0 = time.time()
    for t in range(args.n_turns):
        # Set per-request SLO + last-forward stats for the hook's
        # slack estimator.  Since vLLM batches, we approximate
        # last_forward_us = 0 on turn 0 and the previous turn's
        # mean TPOT thereafter.
        if with_hook and per_request_tpot_us:
            last_fwd = statistics.median(per_request_tpot_us)
        else:
            last_fwd = 0.0

        prompts = [conv[t] for conv in convs]
        # We pass token-ids directly via TokensPrompt; vLLM treats
        # this as already-tokenised input.  prompt_token_ids is the
        # supported field in vllm 0.8.
        token_prompts = [vllm.inputs.TokensPrompt(prompt_token_ids=p)
                          for p in prompts]
        t0 = time.time()
        outs = llm.generate(token_prompts, sp, use_tqdm=False)
        t1 = time.time()
        wall_us = (t1 - t0) * 1e6
        # Per-request TPOT proxy: wall / n_gen_tokens.  vLLM's
        # internal per-step timing isn't easily exposed via the
        # public LLM API in 0.8; this is the same proxy the eF
        # driver uses for the multi-tenant headline.
        for o in outs:
            n_gen = len(o.outputs[0].token_ids)
            if n_gen <= 0:
                continue
            tpot_us = wall_us / max(1, n_gen)
            per_request_tpot_us.append(tpot_us)
            miss_us_counts["total_requests"] += 1
            if tpot_us > deadline_us:
                miss_us_counts["chat_miss_request"] += 1
        print(f"[hot-path/{label}] turn {t}: wall {wall_us/1e3:.0f} ms "
              f"avg TPOT {wall_us/len(outs)/1e3:.1f} ms")

    wall_total_s = time.time() - t_total_0

    if with_hook:
        from seer.integration.vllm_hot_path_hook import uninstall_hot_path_hook
        uninstall_hot_path_hook()
        print(f"[hot-path] final router_fire_count: {router_fire_count}", flush=True)

    p50 = statistics.median(per_request_tpot_us) if per_request_tpot_us else None
    p99 = (sorted(per_request_tpot_us)[int(0.99 * len(per_request_tpot_us))]
           if per_request_tpot_us else None)
    chat_miss = (miss_us_counts["chat_miss_request"]
                  / max(1, miss_us_counts["total_requests"]))
    return {
        "label": label,
        "with_hook": with_hook,
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "num_prompts": args.num_prompts,
        "n_turns": args.n_turns,
        "max_tokens": args.max_tokens,
        "ctx_len": args.ctx_len,
        "slo_p99_ms": args.slo_p99_ms,
        "ell_bar_us": args.ell_bar_us,
        "wall_total_s": wall_total_s,
        "n_requests": miss_us_counts["total_requests"],
        "chat_miss_request": chat_miss,
        "tpot_p50_us": p50,
        "tpot_p99_us": p99,
        "router_firings": router_fire_count["n"],
        "router_kept_blocks_sum": router_fire_count["kept_block_count_sum"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--num_prompts", type=int, default=24)
    ap.add_argument("--n_turns", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--ctx_len", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--slo_p99_ms", type=float, default=200.0)
    ap.add_argument("--ell_bar_us", type=float, default=200.0)
    ap.add_argument("--mode", choices=["baseline", "with_hook", "both"],
                    default="both")
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="TP size; 2 = H100 substitute on 2xA100")
    ap.add_argument("--out_subdir", type=str, default="",
                    help="Sub-directory under results_hot_path/ for "
                         "tagged runs (e.g. 'tp2', '13b'). Empty = "
                         "default output paths.")
    args = ap.parse_args()

    out_dir = (OUT_DIR / args.out_subdir) if args.out_subdir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    if args.mode in ("baseline", "both"):
        results["baseline"] = _run_one("baseline", False, args)
        (out_dir / "baseline.json").write_text(
            json.dumps(results["baseline"], indent=2))

    if args.mode in ("with_hook", "both"):
        # Force a fresh Python process for with_hook? Actually no, since
        # uninstall_hot_path_hook restores original method state.
        results["with_hook"] = _run_one("with_hook", True, args)
        (out_dir / "with_hook.json").write_text(
            json.dumps(results["with_hook"], indent=2))

    if "baseline" in results and "with_hook" in results:
        b = results["baseline"]
        h = results["with_hook"]
        delta = {
            "delta_chat_miss": h["chat_miss_request"] - b["chat_miss_request"],
            "delta_tpot_p99_us": (
                (h["tpot_p99_us"] or 0) - (b["tpot_p99_us"] or 0)
                if (h["tpot_p99_us"] and b["tpot_p99_us"]) else None
            ),
            "delta_wall_s": h["wall_total_s"] - b["wall_total_s"],
            "router_firings": h["router_firings"],
        }
        comparison = {
            "baseline": b, "with_hook": h, "delta": delta,
            "verdict": (
                "HOOK_HAS_EFFECT"
                if abs(delta["delta_chat_miss"]) > 0.001
                   or (delta["delta_tpot_p99_us"] is not None
                       and abs(delta["delta_tpot_p99_us"]) > 5000)
                else "HOOK_TRANSPARENT"
            ),
        }
        PAPER_JSON.write_text(json.dumps(comparison, indent=2))
        # Render small comparison table
        with open(PAPER_TEX, "w") as f:
            f.write("% Generated by scripts/measure_hot_path.py\n")
            f.write(r"\begin{tabular}{lrrrr}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Variant & chat-miss & $P_{99}$ TPOT (ms) & wall (s) & router fires \\" + "\n")
            f.write(r"\midrule" + "\n")
            for k in ("baseline", "with_hook"):
                r = results[k]
                p99ms = (r["tpot_p99_us"] / 1000.0) if r["tpot_p99_us"] else 0
                f.write(rf"\texttt{{{k}}} & "
                        rf"{r['chat_miss_request']:.4f} & "
                        rf"{p99ms:.1f} & {r['wall_total_s']:.1f} & "
                        rf"{r['router_firings']} \\" + "\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
        print(f"[hot-path] wrote comparison → {PAPER_JSON} and {PAPER_TEX}")
        print(f"[hot-path] verdict: {comparison['verdict']}")
        print(f"[hot-path] router fires: {h['router_firings']}")
        print(f"[hot-path] delta chat_miss: {delta['delta_chat_miss']:+.4f}")
        if delta["delta_tpot_p99_us"] is not None:
            print(f"[hot-path] delta P99 TPOT: "
                  f"{delta['delta_tpot_p99_us']/1000:+.1f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
