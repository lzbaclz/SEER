# E4: LAP Architecture Ablation

Compare `tiny_mlp` / `block_rnn` / `block_transformer` at matched training
budget. Report val AUC + inference latency per architecture.

## Run

```bash
for arch in tiny_mlp block_rnn block_transformer; do
  python -m seer.lap.train \
      --traces ../../data/traces/llama3-8b \
      --model $arch --epochs 10 \
      --out results/lap_${arch}.pt
done
python analyze.py results/
```

Expected result: tiny_mlp already captures > 90% of the AUC gain; RNN /
Transformer add a few tenths of a point but ~10× latency.
