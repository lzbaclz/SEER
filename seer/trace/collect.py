"""CLI: collect attention traces from a HuggingFace model on benchmark data.

Writes one parquet file per request under --out.

Example
-------
    python -m seer.trace.collect \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset ruler \\
        --context_lengths 4096 8192 \\
        --num_prompts 100 \\
        --out data/traces/llama3-8b/

Note on step counting: HuggingFace `generate()` fires attention hooks at
each layer within a forward pass, but does not expose a per-step hook. We
wrap `generate` by running one-token-at-a-time via `_step_generate()` so
that `tracer.advance_step()` can be called between forward passes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm


def _step_generate(model, tokenizer, input_ids, attention_mask, max_new_tokens: int, tracer):
    """Greedy one-token-at-a-time decode so the tracer can see per-step attn."""
    import torch
    device = input_ids.device
    past_key_values = None
    generated = input_ids
    # --- Prefill step (step=0)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
        )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = out.past_key_values
    generated = torch.cat([generated, next_token], dim=1)
    # --- Decode steps
    for step in range(1, max_new_tokens):
        tracer.advance_step()
        attn_mask = torch.ones(generated.shape, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = out.past_key_values
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return generated


def main():
    ap = argparse.ArgumentParser(description="Collect attention traces.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ruler", choices=["ruler", "longbench", "pile"])
    ap.add_argument("--context_lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--num_prompts", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_head_groups", type=int, default=None,
                    help="Override GQA head-group count (default: num_key_value_heads)")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip first N prompts (for resume)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from seer.trace.datasets import load_prompts
    from seer.trace.hook import AttentionTracer

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    print(f"[collect] loading {args.model} (dtype={args.dtype}, attn=eager)")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",  # required for output_attentions
    ).to(args.device).eval()

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

    prompts = load_prompts(args.dataset, args.context_lengths, args.num_prompts, tok)
    prompts = prompts[args.start:]

    for offset, prompt_text in enumerate(tqdm(prompts, desc="trace")):
        req_id = args.start + offset
        out_path = out_dir / f"req_{req_id:06d}.parquet"
        if out_path.exists():
            continue
        tracer.begin_request(req_id)
        inputs = tok(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max(args.context_lengths),
        ).to(args.device)
        try:
            _step_generate(
                model, tok, inputs["input_ids"], inputs["attention_mask"],
                args.max_new_tokens, tracer,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] req {req_id} failed: {e}")
            tracer.clear()
            continue
        tracer.finalize_request()
        tracer.to_parquet(str(out_path))
        tracer.clear()

    tracer.detach()
    print(f"[collect] saved traces → {out_dir}")


if __name__ == "__main__":
    main()
