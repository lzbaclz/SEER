"""CLI: run a benchmark with a given policy + model, dump metrics to JSON.

Example
-------
    python -m seer.eval.runner \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --policy seer --lap_ckpt checkpoints/lap_llama3_8b.onnx \\
        --benchmark ruler --context_length 8192 \\
        --budget 0.2 --num_prompts 50 \\
        --out results/ruler_seer_b20.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SEER benchmark runner.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", required=True,
                    choices=["full", "streaming", "h2o", "snapkv", "recency", "random", "seer"])
    ap.add_argument("--lap_ckpt", default=None,
                    help="path to .onnx (or .pt) checkpoint; required for --policy seer")
    ap.add_argument("--benchmark", default="ruler", choices=["ruler", "longbench"])
    ap.add_argument("--budget", type=float, default=0.2,
                    help="fraction of full cache (HBM blocks to keep)")
    ap.add_argument("--context_length", type=int, default=8192)
    ap.add_argument("--num_prompts", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--decision_period", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = _parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from seer.eval.metrics import exact_match, f1_score, substring_match
    from seer.eval.policies import build_policy
    from seer.eval.sim import simulate_attention_mask
    from seer.trace.datasets import load_prompts

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    print(f"[runner] loading {args.model} (policy={args.policy}, budget={args.budget})")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device).eval()

    # --- Build policy
    kwargs: dict = {}
    if args.policy == "seer":
        if not args.lap_ckpt:
            raise SystemExit("--lap_ckpt is required for --policy seer")
        from seer.lap.infer import LAPPredictor
        if str(args.lap_ckpt).endswith(".onnx"):
            predictor = LAPPredictor.from_onnx(args.lap_ckpt, device=args.device)
        else:
            predictor = LAPPredictor.from_torch_ckpt(args.lap_ckpt, device=args.device)
        kwargs["lap_predictor"] = predictor
    policy = build_policy(args.policy, **kwargs)

    # --- Load prompts
    prompts = load_prompts(args.benchmark, [args.context_length], args.num_prompts, tok)
    print(f"[runner] running {len(prompts)} prompts")

    # --- Run
    results = []
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        policy.reset()
        r = simulate_attention_mask(
            model, tok, prompt, policy,
            budget_frac=args.budget,
            max_new_tokens=args.max_new_tokens,
            decision_period=args.decision_period,
        )
        r["id"] = i
        r["f1"] = f1_score(r["pred"], r["ref"])
        r["em"] = exact_match(r["pred"], r["ref"])
        r["substring"] = substring_match(r["pred"], r["ref"])
        results.append(r)
        if (i + 1) % 10 == 0:
            cur_f1 = sum(x["f1"] for x in results) / len(results)
            print(f"[runner]   [{i+1}/{len(prompts)}] mean F1 = {cur_f1:.3f}")
    dt = time.time() - t0

    summary = {
        "model": args.model,
        "policy": args.policy,
        "budget": args.budget,
        "benchmark": args.benchmark,
        "context_length": args.context_length,
        "num_prompts": args.num_prompts,
        "wall_time_s": dt,
        "f1_mean": sum(r["f1"] for r in results) / len(results),
        "em_mean": sum(r["em"] for r in results) / len(results),
        "substring_mean": sum(r["substring"] for r in results) / len(results),
        "tokens_per_second": sum(r["n_gen_tokens"] for r in results) / dt,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[runner] saved → {args.out}")
    print(f"[runner] F1={summary['f1_mean']:.3f}  "
          f"EM={summary['em_mean']:.3f}  "
          f"sub={summary['substring_mean']:.3f}  "
          f"t={dt:.1f}s  tok/s={summary['tokens_per_second']:.1f}")


if __name__ == "__main__":
    main()
