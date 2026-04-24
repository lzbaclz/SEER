# E7: Cross-Model Generalization

Train LAP on Llama-3-8B traces, test zero-shot on:
- Qwen2.5-7B
- Mistral-7B-v0.3
- Llama-3-70B (if compute allows)

Metric: AUC drop vs. in-domain LAP. Secondary: end-to-end task quality
delta when deploying the Llama-3-8B LAP as SEER policy for other models.

## Run

```bash
# Assumes ../../data/traces/{llama3-8b, qwen2.5-7b, mistral-7b}/ exist.

# 1. Train LAP only on Llama traces
python -m seer.lap.train \
    --traces ../../data/traces/llama3-8b \
    --model tiny_mlp --epochs 10 \
    --out results/lap_llama_only.pt

# 2. Evaluate cross-model AUC
python analyze_cross_model.py \
    --ckpt results/lap_llama_only.pt \
    --test_traces ../../data/traces/qwen2.5-7b ../../data/traces/mistral-7b \
    --out results/cross_model.json
```
