"""CLI: collect attention + per-step timing traces from a HuggingFace model.

Writes one parquet file per request under ``--out``.

Example
-------

    python -m seer.trace.collect \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --workload sharegpt --slo_class chat-50ms \\
        --num_requests 100 --out data/traces/llama3-8b/

RTSS pivot
----------
Beyond the NeurIPS-era attention scores, this collector also records a
per-step timing breakdown — the four cost terms that show up in the
schedulability analysis (Lemmas 1–2):

    c_lap_us   : LAP inference time (zero in trace collection — there is
                 no LAP yet; populated downstream by the simulator)
    c_io_us    : synchronous fetch from non-HBM tier (zero here — full
                 cache during trace collection)
    c_attn_us  : attention compute time (measured)
    c_ffn_us   : FFN + residual + layer norm time (computed as
                 ``step_latency - c_attn``)
    step_latency_us : total decode step latency (measured)

The collector remains usable in legacy "RULER + context_lengths" mode for
backwards compatibility with the e1 predictability experiment.

Note on step counting: HuggingFace ``generate()`` fires attention hooks
at each layer within a forward pass, but does not expose a per-step
hook. We wrap ``generate`` by running one-token-at-a-time via
``_step_generate()`` so that ``tracer.advance_step()`` and
``tracer.record_step_timing()`` can be called between forward passes.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm


def _step_generate(model, tokenizer, input_ids, attention_mask,
                   max_new_tokens: int, tracer):
    """Greedy one-token-at-a-time decode so the tracer can see per-step attn."""
    import torch
    device = input_ids.device
    past_key_values = None

    # --- Prefill (step=0). We still time attn vs everything-else even though
    #     prefill is one big forward; the breakdown isn't really attributable
    #     to any one step here, so we record it under step=0.
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize()
    t_pre0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
        )
    if use_cuda:
        torch.cuda.synchronize()
    t_pre1 = time.perf_counter()
    tracer.record_step_timing(step_latency_us=(t_pre1 - t_pre0) * 1e6)

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = out.past_key_values
    generated = torch.cat([input_ids, next_token], dim=1)

    # --- Decode steps
    for _step in range(1, max_new_tokens):
        tracer.advance_step()
        attn_mask = torch.ones(generated.shape, dtype=torch.long, device=device)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_us = (t1 - t0) * 1e6
        tracer.record_step_timing(step_latency_us=step_us)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = out.past_key_values
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return generated


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collect attention + timing traces.")
    ap.add_argument("--model", required=True)
    # Legacy NeurIPS-style args (still supported)
    ap.add_argument("--dataset", default=None,
                    choices=[None, "ruler", "longbench", "pile"],
                    help="Legacy dataset selector (NeurIPS pipeline). Use --workload "
                         "for the RTSS pipeline.")
    ap.add_argument("--context_lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--num_prompts", type=int, default=50,
                    help="Legacy alias for --num_requests")
    # RTSS pivot args
    ap.add_argument("--workload", default=None,
                    choices=[None, "ruler", "longbench", "sharegpt", "synthetic",
                             "mooncake"],
                    help="RTSS workload preset. Translates to a prompt loader.")
    ap.add_argument("--slo_class", default=None,
                    help="SLO tag stamped in trace metadata, e.g. 'chat-50ms'.")
    ap.add_argument("--num_requests", type=int, default=None,
                    help="Number of prompts to trace (RTSS spelling).")
    # Common
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device_map", default=None,
                    help="HF device_map for multi-GPU sharding (e.g. 'auto').")
    ap.add_argument("--n_head_groups", type=int, default=None,
                    help="Override GQA head-group count (default: num_key_value_heads)")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip first N prompts (for resume)")
    return ap


def _resolve_dataset(args: argparse.Namespace) -> tuple[str, int]:
    """Map (workload | dataset) → (prompt-loader name, num_prompts)."""
    n = args.num_requests if args.num_requests is not None else args.num_prompts
    if args.workload is not None:
        ds = args.workload
        if ds == "synthetic":
            ds = "ruler"
        return ds, n
    return (args.dataset or "ruler"), n


def main():
    args = _build_argparser().parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from seer.trace.datasets import load_prompts
    from seer.trace.hook import AttentionTracer

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    print(f"[collect] loading {args.model} (dtype={args.dtype}, attn=eager"
          f"{', device_map=' + str(args.device_map) if args.device_map else ''})")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kw: dict = {
        "torch_dtype": dtype,
        "attn_implementation": "eager",  # required for output_attentions
    }
    if args.device_map is not None:
        load_kw["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if args.device_map is None:
        model = model.to(args.device)
    model.eval()
    model_device = next(model.parameters()).device

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    n_head_groups = args.n_head_groups or n_kv_heads
    print(f"[collect] layers={n_layers}  heads={n_heads}  kv_heads={n_kv_heads}  "
          f"→ head_groups={n_head_groups}")

    tracer = AttentionTracer(n_layers=n_layers, n_head_groups=n_head_groups)
    tracer.attach(model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name, num = _resolve_dataset(args)
    prompts = load_prompts(dataset_name, args.context_lengths, num, tok)
    prompts = prompts[args.start:]

    if args.slo_class:
        print(f"[collect] SLO tag = {args.slo_class}")

    for offset, prompt_text in enumerate(tqdm(prompts, desc="trace")):
        req_id = args.start + offset
        out_path = out_dir / f"req_{req_id:06d}.parquet"
        if out_path.exists():
            continue
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        tracer.begin_request(req_id)
        inputs = tok(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max(args.context_lengths),
        ).to(model_device)
        try:
            _step_generate(
                model, tok, inputs["input_ids"], inputs["attention_mask"],
                args.max_new_tokens, tracer,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] req {req_id} failed: {e}")
            tracer.clear()
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        tracer.finalize_request()
        tracer.to_parquet(str(out_path))
        tracer.clear()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    tracer.detach()
    print(f"[collect] saved traces → {out_dir}")


if __name__ == "__main__":
    main()
