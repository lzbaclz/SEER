#!/usr/bin/env bash
set -uo pipefail
cd /home/lzq/codes/SEER
export CUDA_VISIBLE_DEVICES=1
PY=/home/lzq/miniconda3/envs/seer/bin/python

# Step 1: collect attention traces from Qwen via synthetic workload
mkdir -p data/traces/qwen25_train
if ! ls data/traces/qwen25_train/*.parquet >/dev/null 2>&1; then
  echo "[Qwen] Step 1/4: collect 12 traces"
  $PY -m seer.trace.collect --model Qwen/Qwen2.5-7B \
    --workload synthetic --num_requests 12 --max_new_tokens 64 \
    --out data/traces/qwen25_train 2>&1 | tail -3
fi
ls data/traces/qwen25_train/ | head

# Step 2: train TinyMLP on Qwen traces
if [ ! -f checkpoints/lap_qwen25.pt ]; then
  echo "[Qwen] Step 2/4: train LAP"
  $PY -m seer.lap.train \
    --traces data/traces/qwen25_train \
    --arch tiny_mlp --history 32 --horizons 1 4 16 64 \
    --wcet_budget_us 200 --epochs 6 \
    --out checkpoints/lap_qwen25.pt 2>&1 | tail -8
fi

# Step 3: export ONNX
if [ ! -f checkpoints/lap_qwen25.onnx ]; then
  echo "[Qwen] Step 3/4: export ONNX"
  $PY -m seer.lap.export --ckpt checkpoints/lap_qwen25.pt \
    --backend onnx --static_shape --batch 4096 \
    --out checkpoints/lap_qwen25.onnx 2>&1 | tail -5
fi

# Step 4: re-run SEER (and the 3 baselines we lack) on Qwen
mkdir -p experiments/eA_tail_latency/results_qwen_retrained
echo "[Qwen] Step 4/4: eA on Qwen with retrained LAP"
for policy in streaming h2o quest seer; do
  out="experiments/eA_tail_latency/results_qwen_retrained/${policy}_b020.json"
  if [ -f "$out" ]; then echo "  skip $policy"; continue; fi
  extra=""
  [ "$policy" = "seer" ] && extra="--lap checkpoints/lap_qwen25.onnx"
  echo "  [Qwen-eA] $policy"
  $PY -m seer.eval.runner --model Qwen/Qwen2.5-7B \
    --policy "$policy" $extra --workload mooncake --context_length 2048 \
    --num_requests 20 --max_new_tokens 32 --decision_period 8 \
    --slo P99=50ms --hbm_budget 0.20 --out "$out" 2>&1 | tail -2
done

echo "=== Qwen retrain DONE ==="
date
