# E2: Quality–Budget Pareto Frontier

## Purpose

For each (model × benchmark), sweep the HBM budget over
`{10%, 20%, 40%, 80%}` and run every policy. Plot the Pareto frontier of
task quality vs. memory budget; SEER should dominate.

## Matrix

- Models: Llama-3-8B, Qwen2.5-7B, Mistral-7B (main paper)
- Benchmarks: LongBench (narrative_qa, multi_news, gov_report), RULER (4K/8K/16K/32K)
- Policies: `full`, `streaming`, `h2o`, `snapkv`, `recency`, `random`, `seer`
- Budgets: 0.10, 0.20, 0.40, 0.80 (fraction of full cache blocks)

Total runs: ~3 × 8 × 7 × 4 ≈ 672. Prefer to run in parallel shards.

## Run

```bash
# Full sweep (requires pre-trained LAP checkpoint)
LAP_CKPT=../../checkpoints/lap_llama3_8b.onnx bash run.sh

# Single cell
python -m seer.eval.runner \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --policy h2o --benchmark ruler --budget 0.2 \
    --context_length 8192 --num_prompts 50 \
    --out results/ruler_h2o_b20.json
```

## Analyze

```bash
python analyze.py results/ --out figures/e2_pareto.pdf
```

## Status

⏳ Skeleton only — fill in once E1 GO decision is made.
