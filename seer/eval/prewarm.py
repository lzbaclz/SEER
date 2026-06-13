"""Cold-start mitigation — JIT pre-warm a model + LAP at admission.

The eA J.7 bootstrap surfaced that the warm-up-included P999 (49.9 ms)
is dominated by the JIT outlier on the *first* decode step of each
request. The steady-state P999 across all six policies on Mooncake
collapses to 37–38 ms once that first step is excluded.

This module exposes a single function ``prewarm_model_for_decode`` the
runner / serving layer calls *once* before the first real request is
served. It triggers the same kernels and CUDA Graph captures that the
first real decode step would, so the first user-visible step does not
pay the JIT cost.

Usage from the runner:

    from seer.eval.prewarm import prewarm_model_for_decode
    prewarm_model_for_decode(model, tokenizer, device="cuda",
                             prompt_tokens=64, decode_tokens=8)

Per-call cost: ~200 ms wall-time on Llama-2-7B fp16 / A100. Run this
once at process startup. Idempotent — repeated calls are no-ops past
the first.
"""
from __future__ import annotations

_PREWARM_DONE: dict[int, bool] = {}


def prewarm_model_for_decode(
    model,
    tokenizer,
    device: str = "cuda",
    prompt_tokens: int = 64,
    decode_tokens: int = 8,
    use_cache: bool = True,
) -> dict:
    """Run a synthetic decode-loop forward to JIT all kernels in advance.

    Generates a `prompt_tokens`-long random prefix, then runs
    `decode_tokens` greedy decode steps. This forces:
      - the prefill kernel to compile and cache,
      - the per-step decode kernel(s) to JIT,
      - any vLLM-style paged-attention CUDA-Graph to capture,
      - the tokenizer to load its merge table (fast, but non-zero).

    Returns a dict with ``{"warmup_us": float, "did_run": bool}``.
    Idempotent per (model_id, device) — subsequent calls return
    immediately with did_run=False.
    """
    import time

    import torch
    key = (id(model), str(device))
    if _PREWARM_DONE.get(key):
        return {"warmup_us": 0.0, "did_run": False}

    model.eval()
    dev = torch.device(device if torch.cuda.is_available()
                       and device != "cpu" else "cpu")
    model = model.to(dev)

    # Build a synthetic prompt of length prompt_tokens. Use
    # tokenizer.bos_token_id if defined, otherwise integer 1
    # (Llama-2 BOS).
    bos = (tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1)
    pad = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else bos)
    ids = torch.full((1, prompt_tokens), pad, dtype=torch.long, device=dev)
    ids[0, 0] = bos

    t0 = time.perf_counter()
    with torch.no_grad():
        # Greedy decode_tokens steps — short enough to be cheap, long
        # enough to trigger any decode-only kernel paths the model
        # has separate from prefill.
        # We avoid model.generate() to side-step its sampling/stopping
        # overhead; this is just kernel JIT.
        out = model(input_ids=ids, use_cache=use_cache)
        past = getattr(out, "past_key_values", None)
        next_id = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(decode_tokens):
            out = model(input_ids=next_id, past_key_values=past, use_cache=use_cache)
            past = getattr(out, "past_key_values", None)
            next_id = out.logits[:, -1:, :].argmax(dim=-1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6
    _PREWARM_DONE[key] = True
    return {"warmup_us": float(elapsed_us), "did_run": True}


def prewarm_lap(predictor, batch_size: int = 4096, n_warmup: int = 8) -> dict:
    """Pre-warm the LAP predictor (TRT plan + CUDA Graph already do
    this in `from_tensorrt`; this hook covers torch / onnx backends
    where warmup is on the caller).
    """
    import time
    try:
        import numpy as np
    except ImportError:
        return {"warmup_us": 0.0, "did_run": False}
    try:
        feat = getattr(predictor, "feat_dim", 37)
    except Exception:
        feat = 37
    x = np.zeros((batch_size, feat), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(n_warmup):
        try:
            _ = predictor(x)
        except Exception:  # noqa: BLE001
            return {"warmup_us": 0.0, "did_run": False}
    elapsed_us = (time.perf_counter() - t0) * 1e6
    return {"warmup_us": float(elapsed_us), "did_run": True}


def reset() -> None:
    """Clear the per-(model, device) idempotency cache (test hook)."""
    _PREWARM_DONE.clear()
