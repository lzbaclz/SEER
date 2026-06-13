r"""eA cross-check on real vLLM 0.8.5.

Re-measures the §6.3 headline tail-latency claim
(P99 = 35.8 ms, P999 = 49.9 ms at $\hbm = 0.20$) on real vLLM
instead of the simulator backend, to give the reviewer one
honest sanity-check that the simulator-derived numbers are not
an artefact of our hand-rolled mask-based driver.

We construct prompts that match eA's parameters (Mooncake-style
chat workload, ~1k input tokens, max_tokens 64), run them through
vllm.LLM with the SeerKVConnector plugin active, and report
per-step P50 / P99 / P999 / max from the LLMEngine.step() loop
(same per-step metric the simulator already records).

Output: experiments/eA_tail_latency/results_vllm_check/seer.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--lap", required=True)
    ap.add_argument("--num_requests", type=int, default=20,
                    help="Match eA's NUM_REQUESTS=20 default.")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--input_tokens", type=int, default=1000,
                    help="Mooncake median input length is ~1-2k tokens.")
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    return ap


def _build_chat_prompt(i: int, n_input_tokens: int) -> str:
    body = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * max(
        1, n_input_tokens // 5
    )
    return (
        f"You are a helpful assistant. Background context for "
        f"request {i}:\n\n{body}\n\n"
        f"Now please summarise the difference between solar PV and "
        f"solar thermal in two sentences.\nAnswer:"
    )


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    from seer.integration.vllm_connector import register_with_vllm_factory
    register_with_vllm_factory("SeerKVConnector")
    import vllm
    from vllm.config import KVTransferConfig

    print(f"[eA-vllm] vllm {vllm.__version__}, num_requests={args.num_requests}, "
          f"input~{args.input_tokens} tok, max_tokens={args.max_tokens}")

    kv_cfg = KVTransferConfig(
        kv_connector="SeerKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lap_path": args.lap,
            "budget_blocks": 200,  # consistent with eA's HBM budget = 0.20
            "horizon_steps": 4,
            "p_threshold": 0.4,
        },
    )
    llm = vllm.LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_log_stats=True,
        kv_transfer_config=kv_cfg,
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompts = [_build_chat_prompt(i, args.input_tokens) for i in range(args.num_requests)]

    # LLMEngine.step() loop for honest per-step timing.
    engine = llm.llm_engine
    for i, p in enumerate(prompts):
        try:
            engine.add_request(f"r{i}", p, sp)
        except TypeError:
            engine.add_request(request_id=f"r{i}", prompt=p, params=sp)

    per_step: dict[str, list[float]] = {f"r{i}": [] for i in range(args.num_requests)}
    t_start = time.perf_counter()
    while engine.has_unfinished_requests():
        t0 = time.perf_counter()
        outs = engine.step()
        step_us = (time.perf_counter() - t0) * 1e6
        for o in outs:
            if o.request_id in per_step:
                per_step[o.request_id].append(step_us)
    wall = time.perf_counter() - t_start

    # Decode-only metric: drop first ``n_prefill_steps`` per request,
    # since the simulator's published P99=35.8ms is decode TPOT not
    # including prefill. vLLM 0.8 chunks prefill across multiple
    # step() calls; we conservatively drop the first 8 steps per
    # request as "prefill / first-decode warm-up".
    n_prefill_steps = 8
    decode_steps = []
    for steps in per_step.values():
        decode_steps.extend(steps[n_prefill_steps:])
    all_steps = sorted(decode_steps)

    def _pctile(p: float) -> float:
        if not all_steps:
            return 0.0
        idx = int((len(all_steps) - 1) * p)
        return all_steps[idx]

    summary = {
        "vllm_version": vllm.__version__,
        "model": args.model,
        "num_requests": args.num_requests,
        "input_tokens_target": args.input_tokens,
        "max_tokens": args.max_tokens,
        "wall_time_s": wall,
        "per_step_count": len(all_steps),
        "p50_ms": _pctile(0.50) / 1000.0,
        "p90_ms": _pctile(0.90) / 1000.0,
        "p99_ms": _pctile(0.99) / 1000.0,
        "p999_ms": _pctile(0.999) / 1000.0,
        "max_ms": (max(all_steps) if all_steps else 0) / 1000.0,
        "mean_ms": (statistics.fmean(all_steps) if all_steps else 0) / 1000.0,
        "compare_published_p99_ms": 35.8,
        "compare_published_p999_ms": 49.9,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[eA-vllm] wall={wall:.1f}s | P50={summary['p50_ms']:.2f}ms | "
          f"P99={summary['p99_ms']:.2f}ms (vs published 35.8ms) | "
          f"P999={summary['p999_ms']:.2f}ms (vs published 49.9ms)")
    print(f"[eA-vllm] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
