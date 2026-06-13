"""eF EDF + admission driver (review M4 / P4).

The published eF numbers used a ThreadPoolExecutor with FIFO ordering
and a shared model on a single GPU. Long doc-prefill tasks pre-empted
the GPU for chat tasks → chat_miss = 100\\% on every policy except
``full``.

This driver replaces the FIFO pool with:
  * an :class:`EDFAdmissionController` that orders tasks by absolute
    deadline (chat tasks first by virtue of their tighter 50 ms target),
  * an admission predicate that defers a task if admitting would push
    the Lemma 2 bound above ``miss_target`` for the current population.

Output: chat / doc miss-ratios + bound-predicted miss + admission /
defer counts. Intended to be compared head-to-head against
``driver.py`` (FIFO) on the same workload.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lap", default=None)
    ap.add_argument("--policy", default="seer",
                    choices=["full", "seer", "h2o", "snapkv", "streaming"])
    ap.add_argument("--n_chat", type=int, default=10)
    ap.add_argument("--n_doc", type=int, default=2)
    ap.add_argument("--chat_slo", default="P99=50ms")
    ap.add_argument("--doc_slo",  default="TTFT-P99=1s")
    ap.add_argument("--hbm_budget", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--miss_target", type=float, default=0.01,
                    help="Admission predicate threshold for Lemma-2 bound.")
    ap.add_argument("--mode", default="edf",
                    choices=["edf", "fifo"],
                    help="edf = bound-aware EDF admission (P4); "
                         "fifo = legacy ThreadPoolExecutor for A/B baseline.")
    return ap.parse_args()


def _make_chat_prompt(i: int) -> str:
    return f"User: explain the value of P=NP problem in one paragraph. (req {i})"


def _make_doc_prompt(i: int) -> str:
    boilerplate = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                   "Sed do eiusmod tempor incididunt ut labore et dolore. ") * 80
    return f"Document {i}: {boilerplate}\nSummarize in 3 bullet points:"


def _build_policy(args):
    from seer.policy import build_policy
    from seer.timing import parse_slo
    kw = {}
    if args.policy == "seer":
        if not args.lap:
            raise SystemExit("--lap required for --policy seer")
        from seer.lap.infer import LAPPredictor
        kw["lap_predictor"] = LAPPredictor.from_path(args.lap, device=args.device)
        kw["slo"] = parse_slo(args.chat_slo)
    return build_policy(args.policy, **kw)


def main():
    args = _build_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from seer.eval.metrics import LatencyStats
    from seer.eval.sim import simulate_attention_mask
    from seer.policy.admission import EDFAdmissionController, SharedLambdaState
    from seer.timing import derive_deadline_us, parse_slo

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    policy = _build_policy(args)

    chat_slo = parse_slo(args.chat_slo)
    doc_slo  = parse_slo(args.doc_slo)
    chat_D = derive_deadline_us(chat_slo)
    doc_D  = derive_deadline_us(doc_slo)

    # Bound inputs for the admission predicate (mirrors §4.2 derivation).
    bound_inputs = {
        "epsilon_mean": 0.04, "epsilon_var": 0.005,
        "ell_bar_us": 200.0, "B_t": 12.8,
        "deadline_us": chat_D, "sigma_us": 1200.0,
        "base_cost_us": 36000.0,
    }
    admitter = EDFAdmissionController(miss_target=args.miss_target,
                                       bound_inputs=bound_inputs)
    shared_lambda = None
    if hasattr(policy, "controller") and policy.controller is not None:
        shared_lambda = SharedLambdaState(policy.controller)

    # Build task population
    tasks = []
    for i in range(args.n_chat):
        tasks.append(("chat", _make_chat_prompt(i), args.max_tokens, chat_D))
    for i in range(args.n_doc):
        tasks.append(("doc", _make_doc_prompt(i), max(32, args.max_tokens // 2), doc_D))

    # Admit all tasks initially (they all arrive at t=0)
    t0 = time.monotonic() * 1e6
    for idx, (kind, prompt, max_new, D) in enumerate(tasks):
        admitter.admit(
            task=(idx, kind, prompt, max_new),
            deadline_us=D,
            tenant=kind,
            arrival_us=t0,
        )
    admit_stats_initial = admitter.stats()

    # ----- Execution loop -----
    results = []
    t_exec_start = time.time()
    if args.mode == "edf":
        # Serial EDF: pop earliest-deadline task, run it, repeat. After
        # each completion, retry deferred to see if anything can be admitted.
        while True:
            item = admitter.pop_next()
            if item is None:
                admitter.retry_deferred()
                item = admitter.pop_next()
                if item is None:
                    break
            idx, kind, prompt, max_new = item.task
            # Do NOT reset the policy controller — it's shared cross-tenant.
            # Reset only the per-request stats buffers (handled inside policy).
            if hasattr(policy, "_conf_window"):
                policy._conf_window.clear()
            r = simulate_attention_mask(
                model, tok, prompt, policy,
                budget_frac=args.hbm_budget,
                max_new_tokens=max_new,
                decision_period=8,
            )
            per_step = r.get("per_step_us", [])
            ttft_us = r.get("prefill_us", 0.0)
            D = chat_D if kind == "chat" else doc_D
            stats = LatencyStats.from_vector(per_step, deadline_us=D)
            # Feed observed latencies back into the shared λ.
            if shared_lambda is not None and per_step:
                for lat in per_step:
                    shared_lambda.observe(lat, D, tenant=kind)
            results.append({
                "task_id": idx, "kind": kind,
                "stats": stats.to_dict(), "ttft_us": ttft_us,
            })
    else:  # fifo
        import concurrent.futures as cf
        def _run_one(item):
            idx, kind, prompt, max_new = item.task
            r = simulate_attention_mask(
                model, tok, prompt, policy,
                budget_frac=args.hbm_budget,
                max_new_tokens=max_new,
                decision_period=8,
            )
            per_step = r.get("per_step_us", [])
            ttft_us = r.get("prefill_us", 0.0)
            D = chat_D if kind == "chat" else doc_D
            stats = LatencyStats.from_vector(per_step, deadline_us=D)
            return {"task_id": idx, "kind": kind,
                    "stats": stats.to_dict(), "ttft_us": ttft_us}
        # Drain queue in FIFO order
        items = []
        while True:
            it = admitter.pop_next()
            if it is None:
                admitter.retry_deferred()
                it = admitter.pop_next()
                if it is None:
                    break
            items.append(it)
        # Sort by arrival, not deadline → FIFO
        items.sort(key=lambda it: it.arrival_us)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(_run_one, it) for it in items]
            for fut in cf.as_completed(futs):
                results.append(fut.result())

    dt = time.time() - t_exec_start

    chat_r = [r for r in results if r["kind"] == "chat"]
    doc_r  = [r for r in results if r["kind"] == "doc"]
    chat_miss = sum(r["stats"]["miss_ratio"] for r in chat_r) / max(1, len(chat_r))
    doc_violations = sum(1 for r in doc_r if r["ttft_us"] > doc_D)
    doc_miss = doc_violations / max(1, len(doc_r))

    summary = {
        "mode": args.mode,
        "model": args.model, "policy": args.policy,
        "wall_time_s": dt,
        "chat_slo": {"name": chat_slo.name, "deadline_us": chat_D},
        "doc_slo": {"name": doc_slo.name, "deadline_us": doc_D},
        "n_chat": len(chat_r), "n_doc": len(doc_r),
        "chat_miss_ratio": chat_miss,
        "doc_miss_ratio": doc_miss,
        "admission_initial": admit_stats_initial,
        "admission_final": admitter.stats(),
        "shared_lambda_stats": shared_lambda.stats() if shared_lambda else None,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eF-{args.mode}] {args.policy}: chat_miss={chat_miss:.4f} "
          f"doc_miss={doc_miss:.4f}  t={dt:.1f}s  → {args.out}")
    print(f"  admission: initial={admit_stats_initial}  final={admitter.stats()}")


if __name__ == "__main__":
    main()
