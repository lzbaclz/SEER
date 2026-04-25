#!/usr/bin/env python3
"""experiments/e3_throughput/make_trace.py

Build a small benchmark trace and dump it as JSONL, one prompt per line:
    {"id": 0, "prompt": "...", "max_tokens": 256}

Two modes:

  --mode synthetic   : generate N random prompts of given length (default).
                       Reproducible (uses --seed). No network needed.

  --mode sharegpt    : pull a small subset from the public ShareGPT-style dataset
                       on HuggingFace. Requires `datasets` + network.

Usage:
    python make_trace.py --mode synthetic --num 64 --prompt_len 512 --out trace.jsonl
    python make_trace.py --mode sharegpt  --num 64 --out trace.jsonl

The trace is consumed by bench_vllm.py.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path


def make_synthetic(num: int, prompt_len: int, max_tokens: int, seed: int) -> list[dict]:
    """Random ASCII prompts. Length is approximate (in characters, not tokens)."""
    rng = random.Random(seed)
    rows = []
    vocab = string.ascii_lowercase + " "
    for i in range(num):
        # produce a 'prompt_len'-char nonsense paragraph wrapped in an instruction
        body = "".join(rng.choices(vocab, k=prompt_len))
        prompt = f"Summarize the following text in three sentences:\n\n{body}\n\nSummary:"
        rows.append({"id": i, "prompt": prompt, "max_tokens": max_tokens})
    return rows


def make_sharegpt(num: int, max_tokens: int, seed: int) -> list[dict]:
    """Pull `num` first-turn user messages from a ShareGPT-style HF dataset."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        sys.exit("ERROR: pip install datasets  (then re-run with --mode sharegpt)")

    # anon8231489123/ShareGPT_Vicuna_unfiltered is the historical canonical dump,
    # but availability varies. We try a couple of community mirrors in order.
    candidates = [
        ("RyokoAI/ShareGPT52K", "train"),
        ("anon8231489123/ShareGPT_Vicuna_unfiltered", "train"),
    ]
    last_err: Exception | None = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception as e:  # network / gated / removed
            last_err = e
            continue
    else:
        sys.exit(f"ERROR: could not load any ShareGPT mirror; last error:\n  {last_err!r}")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), k=min(num * 4, len(ds)))  # oversample, then filter

    rows: list[dict] = []
    for idx in indices:
        row = ds[idx]
        # Schema is roughly {"conversations": [{"from": "human", "value": "..."}, ...]}
        convs = row.get("conversations") or row.get("messages") or []
        first_human = next(
            (c.get("value") or c.get("content") for c in convs
             if (c.get("from") or c.get("role")) in ("human", "user")),
            None,
        )
        if not first_human or len(first_human) < 32:
            continue
        rows.append({"id": len(rows), "prompt": first_human, "max_tokens": max_tokens})
        if len(rows) >= num:
            break

    if len(rows) < num:
        print(f"WARN: only collected {len(rows)} / {num} usable prompts", file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "sharegpt"], default="synthetic")
    ap.add_argument("--num", type=int, default=64, help="number of prompts in the trace")
    ap.add_argument("--prompt_len", type=int, default=512,
                    help="(synthetic only) approximate prompt length in chars")
    ap.add_argument("--max_tokens", type=int, default=128,
                    help="max_tokens to ask the model to generate per request")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("trace.jsonl"))
    args = ap.parse_args()

    if args.mode == "synthetic":
        rows = make_synthetic(args.num, args.prompt_len, args.max_tokens, args.seed)
    else:
        rows = make_sharegpt(args.num, args.max_tokens, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[make_trace] wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
