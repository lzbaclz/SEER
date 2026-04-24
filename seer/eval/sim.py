"""Offline simulator for policy evaluation.

This is NOT a production inference backend. Its purpose is to answer:
"If we had evicted these blocks according to policy P at each step, what
would the task quality have been?"

Two modes:

  ATTENTION-MASK mode (default)
  -----------------------------
  We run full HuggingFace generate() with `output_attentions=True`, then
  between steps, recompute the set of kept blocks per (layer, head) and
  apply a *mask* to the attention weights for blocks that should have
  been evicted. This is a quality estimator — it captures the effect of
  the policy on attention outputs without implementing actual cache
  eviction. Faithful to the end quality; NOT faithful to throughput.

  FORWARD-ONCE mode (placeholder)
  -------------------------------
  For throughput measurement we will need a true vLLM integration —
  stubbed here, filled in by `seer.integration.vllm` (TODO).

For NeurIPS E1 (predictability) we don't need this simulator at all;
we measure AUC on the trace. For E2 (quality-budget Pareto) we do need
it. For E3 (throughput-at-iso-quality) we need the real vLLM path.
"""
from __future__ import annotations

from typing import Any


def simulate_attention_mask(
    model: Any,
    tokenizer: Any,
    prompt: str,
    policy,
    budget_frac: float,
    max_new_tokens: int = 128,
    decision_period: int = 8,
) -> dict:
    """Run generation and evaluate a policy in attention-mask mode.

    Returns a dict with:
      - pred: str — generated text
      - ref: str — ground-truth (extracted from the RULER prompt; "" for others)
      - n_gen_tokens: int
      - evictions: int — number of blocks evicted across the run

    Implementation note: this function deliberately does NOT implement the
    full masked-attention pathway inline (that requires injecting a
    custom attention processor into HuggingFace). Instead, for the first
    MVP we fall back to *plain* greedy generation and return the same
    output regardless of the policy. The plumbing is in place so that a
    follow-up PR can swap in a masked attention processor.
    """
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(model.config, "max_position_embeddings", 32768),
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
    ref = _extract_ruler_reference(prompt)
    return {
        "pred": pred,
        "ref": ref,
        "n_gen_tokens": int(gen.shape[0]),
        "evictions": 0,   # filled in by the real masked-attn version
        "policy": policy.name,
        "budget_frac": budget_frac,
        "decision_period": decision_period,
    }


def _extract_ruler_reference(prompt: str) -> str:
    """Extract the needle from a RULER-synthetic prompt."""
    import re
    m = re.search(r"secret password is (\d+)", prompt)
    return m.group(1) if m else ""
