# E3 · vLLM Baseline Benchmark Harness

This subdirectory now contains the **black-box vLLM baseline harness** that the
existing `README.md` flagged as TODO. It does NOT touch any Seer code; it just
produces the vanilla vLLM numbers we'll compare against.

## Files

| File | What it does |
|---|---|
| [`serve_vllm.sh`](serve_vllm.sh) | Launches a vLLM OpenAI-compatible HTTP server. Configurable model / port / TP / max_len via env vars. |
| [`make_trace.py`](make_trace.py) | Builds a JSONL trace of prompts. `--mode synthetic` for offline / no-network; `--mode sharegpt` to pull a small subset from a public ShareGPT mirror. |
| [`bench_vllm.py`](bench_vllm.py) | Replays the JSONL trace concurrently against the running server, records per-request TTFT / TPOT / total latency, and prints aggregate throughput + p50/p95/p99 percentiles. |
| `results/` | Output directory (created on first run). `bench.csv` per-request, `bench.json` aggregate. |

## Wiring (3 terminals — read first, run later)

```bash
# Terminal 1: start vLLM
cd experiments/e3_throughput
MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8000 MAX_LEN=4096 bash serve_vllm.sh

# Terminal 2: build a tiny trace (32 short synthetic prompts)
cd experiments/e3_throughput
python make_trace.py --mode synthetic --num 32 --prompt_len 512 \
                     --max_tokens 128 --out trace.jsonl

# Terminal 3: run the bench
cd experiments/e3_throughput
python bench_vllm.py \
    --trace trace.jsonl \
    --base_url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --concurrency 4 \
    --out_csv results/bench.csv \
    --out_json results/bench.json
```

## What you get

`results/bench.json` — one-shot summary:

```json
{
  "total_requests": 32,
  "successful_requests": 32,
  "wall_seconds": 14.213,
  "total_completion_tokens": 4096,
  "throughput_tokens_per_sec": 288.21,
  "ttft_ms": {"p50": 91.4, "p95": 224.7, "p99": 312.6},
  "tpot_ms": {"p50": 18.2, "p95": 22.0, "p99": 27.5},
  "total_ms": {"p50": 2410.0, "p95": 4128.0, "p99": 5701.0}
}
```

`results/bench.csv` — one row per request, columns:
`prompt_id, prompt_chars, prompt_tokens, completion_tokens, ttft_ms, tpot_ms,
total_ms, submit_ts, finish_ts, error`.

These are exactly the columns the Seer §5 main table will need (TTFT p99,
throughput, etc.).

## Why this design

- **Black-box** — `bench_vllm.py` only talks HTTP. When Seer modifies vLLM's
  KV-cache internals later, the same script measures the change without edits.
- **Streaming** — TTFT is meaningful only when we read the first generated
  token's timestamp; that requires the streaming OpenAI API. We use
  `stream_options={"include_usage": True}` so the final chunk carries an
  authoritative `prompt_tokens` / `completion_tokens` count from the server.
- **Concurrency via asyncio** — `--concurrency 4` puts up to 4 requests
  in-flight simultaneously, which is what triggers vLLM's continuous batching.
  Setting it to 1 measures sequential latency; setting it equal to number of
  prompts measures peak burst throughput.
- **No retries, no warmup loop yet** — keep it simple. Add those once we have
  numbers we trust.

## Next steps (after you've eyeballed the scripts)

1. Run the wiring above against a small model (Qwen2.5-7B or Llama-3-8B) to
   confirm the harness works end-to-end.
2. Sweep `--concurrency` ∈ {1, 2, 4, 8, 16} and plot throughput vs concurrency.
   This will be a Seer §5 figure: "vanilla vLLM throughput-latency Pareto curve."
3. Replace `--mode synthetic` with `--mode sharegpt` once network access /
   gated-model auth is sorted, so the workload distribution matches what the
   paper claims to evaluate on.
4. Add Mooncake / LongBench traces as additional `make_trace.py` modes.
5. When Seer's vLLM fork is ready, point `--base_url` at the Seer-instrumented
   server and re-run with the **identical** trace + concurrency — apples-to-apples.

## Things explicitly out of scope here

- vLLM internals (block manager, scheduler, attention kernels) — Seer's
  modifications live there, not in this directory.
- Quality metrics (LongBench F1, etc.) — that's `e2_pareto` / `e6_policy`'s
  job. E3 is purely throughput + latency.
- Multi-node serving — not needed for the small models we'll run baselines on.
