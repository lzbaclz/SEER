"""Prompt loaders for benchmarks used to collect traces.

Falls back to synthetic prompts if the real dataset is unreachable.
"""
from __future__ import annotations

from typing import Any


def load_prompts(
    dataset: str,
    context_lengths: list[int],
    num_prompts: int,
    tokenizer: Any | None = None,
) -> list[str]:
    if dataset == "ruler":
        return _load_ruler_synthetic(context_lengths, num_prompts, tokenizer)
    if dataset == "longbench":
        return _load_longbench(num_prompts)
    if dataset == "pile":
        return _load_pile(num_prompts, context_lengths[0] if context_lengths else 8192, tokenizer)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
#  Synthetic RULER: needle-in-a-haystack style
# ---------------------------------------------------------------------------

def _load_ruler_synthetic(
    context_lengths: list[int],
    num_prompts: int,
    tokenizer: Any | None,
) -> list[str]:
    prompts = []
    filler = "The quick brown fox jumps over the lazy dog. "
    for i in range(num_prompts):
        ctx_len = context_lengths[i % len(context_lengths)]
        repeats = max(1, ctx_len // 10)
        haystack = filler * repeats
        needle = f"\nThe secret password is {7919 * (i + 1)}.\n"
        mid = len(haystack) // 2
        prompt = haystack[:mid] + needle + haystack[mid:]
        prompt += "\n\nQ: What is the secret password? Reply with only the number.\nA:"
        if tokenizer is not None:
            # Trim to context_length tokens, leaving room for the Q/A tail
            ids = tokenizer.encode(prompt)[:ctx_len]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
#  LongBench (Huggingface)
# ---------------------------------------------------------------------------

def _load_longbench(num_prompts: int) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", "narrativeqa", split=f"test[:{num_prompts}]")
        return [f"{r['context']}\n\nQuestion: {r['input']}\nAnswer:" for r in ds]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] LongBench unavailable, falling back to RULER synthetic: {e}")
        return _load_ruler_synthetic([8192], num_prompts, None)


# ---------------------------------------------------------------------------
#  The Pile (Neel Nanda's 10k subset is small + easy to fetch)
# ---------------------------------------------------------------------------

def _load_pile(num_prompts: int, ctx_len: int, tokenizer: Any | None) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{num_prompts}]")
        prompts = []
        for r in ds:
            text = r["text"]
            if tokenizer is not None:
                ids = tokenizer.encode(text)[:ctx_len]
                text = tokenizer.decode(ids, skip_special_tokens=True)
            prompts.append(text)
        return prompts
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Pile unavailable, falling back to RULER synthetic: {e}")
        return _load_ruler_synthetic([ctx_len], num_prompts, tokenizer)
