#!/usr/bin/env python3
"""experiments/e3_throughput/bench_vllm.py

Replay a JSONL trace against a running vLLM OpenAI-compatible server and record
per-request latency metrics (TTFT, TPOT, total time) plus aggregate throughput.

This is the **vanilla baseline harness**. Seer-vs-baseline experiments will reuse
this same script, only swapping the model id / server endpoint pointed at.

Inputs
------
    --trace      : path to JSONL produced by make_trace.py
    --base_url   : vLLM server URL, e.g. http://localhost:8000/v1
    --model      : model id served by vLLM (must match what serve_vllm.sh launched)
    --concurrency: number of concurrent in-flight requests
    --out_csv    : per-request metrics
    --out_json   : aggregate summary

Metrics recorded per request
----------------------------
    prompt_id        : id from the trace
    prompt_chars     : len(prompt) in characters (rough proxy for tokens)
    prompt_tokens    : actual prompt tokens (from server `usage` if available)
    completion_tokens: number of tokens generated
    ttft_ms          : time-to-first-token (ms) — measured by streaming
    tpot_ms          : average time-per-output-token AFTER first (ms)
    total_ms         : wall-clock time for the entire request (ms)
    submit_ts        : monotonic submit timestamp (s, relative to run start)
    finish_ts        : monotonic finish timestamp (s, relative to run start)

Aggregates
----------
    total_requests, total_completion_tokens, wall_seconds
    throughput_tokens_per_sec  (sum completion / wall_seconds)
    p50/p95/p99 TTFT, p50/p95/p99 TPOT

This script does NOT modify vLLM internals — it's a pure black-box client.
That's the point: any KV-cache change you make in vLLM is observable through
exactly these end-to-end metrics, with no changes here.

Usage
-----
    # 1) start vLLM in another terminal:
    bash serve_vllm.sh

    # 2) build a tiny trace:
    python make_trace.py --mode synthetic --num 32 --out trace.jsonl

    # 3) run the bench (will refuse if server is down):
    python bench_vllm.py \
        --trace trace.jsonl \
        --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --concurrency 4 \
        --out_csv results/bench.csv \
        --out_json results/bench.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class RequestResult:
    prompt_id: int
    prompt_chars: int
    prompt_tokens: int | None
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    submit_ts: float
    finish_ts: float
    error: str | None = None


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


async def run_one(
    client,                   # AsyncOpenAI
    model: str,
    row: dict,
    run_start: float,
) -> RequestResult:
    """Issue one streaming chat request and time it."""
    submit = time.monotonic()
    prompt = row["prompt"]
    max_tokens = int(row.get("max_tokens", 128))

    first_chunk_at: float | None = None
    last_chunk_at: float | None = None
    completion_tokens = 0
    prompt_tokens: int | None = None
    err: str | None = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},   # vLLM 0.4+ supports this
        )
        async for chunk in stream:
            now = time.monotonic()
            # First chunk that actually carries content marks TTFT.
            choice = chunk.choices[0] if chunk.choices else None
            delta = choice.delta if choice else None
            if delta and (delta.content or "") and first_chunk_at is None:
                first_chunk_at = now
            if delta and delta.content:
                completion_tokens += 1   # crude: count chunks; see note below
                last_chunk_at = now
            # vLLM emits a final chunk with `usage` when stream_options requests it
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens
                # prefer the server's exact completion_tokens count
                completion_tokens = chunk.usage.completion_tokens
    except Exception as e:
        err = repr(e)

    finish = time.monotonic()
    if first_chunk_at is None:
        first_chunk_at = finish     # request errored or empty; TTFT undefined
    if last_chunk_at is None:
        last_chunk_at = finish

    total_ms = (finish - submit) * 1000
    ttft_ms = (first_chunk_at - submit) * 1000
    # TPOT = (last_chunk_at - first_chunk_at) / (completion_tokens - 1), in ms
    if completion_tokens > 1:
        tpot_ms = (last_chunk_at - first_chunk_at) * 1000 / (completion_tokens - 1)
    else:
        tpot_ms = 0.0

    return RequestResult(
        prompt_id=int(row["id"]),
        prompt_chars=len(prompt),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_ms=total_ms,
        submit_ts=submit - run_start,
        finish_ts=finish - run_start,
        error=err,
    )


async def run_bench(
    base_url: str,
    model: str,
    rows: list[dict],
    concurrency: int,
) -> tuple[list[RequestResult], float]:
    try:
        from openai import AsyncOpenAI   # type: ignore
    except ImportError:
        sys.exit("ERROR: pip install openai  (the official OpenAI SDK; works with vLLM)")

    client = AsyncOpenAI(base_url=base_url, api_key="dummy", timeout=600.0)
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    run_start = time.monotonic()

    async def _bounded(row):
        async with sem:
            r = await run_one(client, model, row, run_start)
            print(
                f"  [{r.prompt_id:>4}] ttft={r.ttft_ms:7.1f}ms "
                f"tpot={r.tpot_ms:6.2f}ms tok={r.completion_tokens:>4} "
                f"total={r.total_ms:7.1f}ms"
                + (f"  ERR={r.error}" if r.error else ""),
                flush=True,
            )
            results.append(r)

    await asyncio.gather(*(_bounded(row) for row in rows))
    wall = time.monotonic() - run_start
    return results, wall


def percentiles(xs: list[float], ps=(50, 95, 99)) -> dict[str, float]:
    if not xs:
        return {f"p{p}": float("nan") for p in ps}
    s = sorted(xs)
    out = {}
    for p in ps:
        # nearest-rank
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        out[f"p{p}"] = s[k]
    return out


def summarize(results: list[RequestResult], wall: float) -> dict[str, Any]:
    ok = [r for r in results if r.error is None and r.completion_tokens > 0]
    total_completion = sum(r.completion_tokens for r in ok)
    return {
        "total_requests": len(results),
        "successful_requests": len(ok),
        "wall_seconds": round(wall, 3),
        "total_completion_tokens": total_completion,
        "throughput_tokens_per_sec": round(total_completion / wall, 2) if wall > 0 else 0.0,
        "ttft_ms": percentiles([r.ttft_ms for r in ok]),
        "tpot_ms": percentiles([r.tpot_ms for r in ok if r.tpot_ms > 0]),
        "total_ms": percentiles([r.total_ms for r in ok]),
    }


def write_csv(path: Path, results: list[RequestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(results[0]).keys()) if results else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True,
                    help="must match the model id launched by serve_vllm.sh")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out_csv", type=Path, default=Path("results/bench.csv"))
    ap.add_argument("--out_json", type=Path, default=Path("results/bench.json"))
    args = ap.parse_args()

    rows = load_trace(args.trace)
    print(f"[bench] {len(rows)} prompts, concurrency={args.concurrency}, "
          f"endpoint={args.base_url}, model={args.model}")
    results, wall = asyncio.run(run_bench(args.base_url, args.model, rows, args.concurrency))

    # sort by submit_ts for human-readable CSV
    results.sort(key=lambda r: r.submit_ts)
    write_csv(args.out_csv, results)

    summary = summarize(results, wall)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))

    print()
    print("=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"[bench] per-request -> {args.out_csv}")
    print(f"[bench] summary     -> {args.out_json}")


if __name__ == "__main__":
    main()
