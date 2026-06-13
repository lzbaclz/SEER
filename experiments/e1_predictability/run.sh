#!/usr/bin/env bash
# E1: Predictability Study — collect traces, train LAP, compute AUC vs. baselines.
# Outputs: results/predictability.json with GO/NO-GO verdict.
set -euo pipefail

cd "$(dirname "$0")"

# ---------- Config (override via env) ----------
MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
TRACE_DIR=${TRACE_DIR:-../../data/traces/llama3-8b}
OUT_DIR=${OUT_DIR:-results}
NUM_PROMPTS=${NUM_PROMPTS:-100}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"4096 8192"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
EPOCHS=${EPOCHS:-10}
MODEL_ARCH=${MODEL_ARCH:-tiny_mlp}

mkdir -p "$OUT_DIR" "$TRACE_DIR"

# ---------- 1. Collect attention traces ----------
if [ -z "$(ls -A "$TRACE_DIR" 2>/dev/null | grep parquet)" ]; then
  echo "[E1] (1/3) collecting traces → $TRACE_DIR"
  python -m seer.trace.collect \
      --model "$MODEL" \
      --dataset ruler \
      --context_lengths $CONTEXT_LENGTHS \
      --num_prompts "$NUM_PROMPTS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --out "$TRACE_DIR"
else
  echo "[E1] (1/3) traces already in $TRACE_DIR (skipping collect)"
fi

# ---------- 2. Train LAP ----------
CKPT="$OUT_DIR/lap_${MODEL_ARCH}.pt"
echo "[E1] (2/3) training LAP ($MODEL_ARCH) → $CKPT"
python -m seer.lap.train \
    --traces "$TRACE_DIR" \
    --model "$MODEL_ARCH" \
    --epochs "$EPOCHS" \
    --out "$CKPT" \
    --log_dir "$OUT_DIR/train_log"

# ---------- 3. Analyze ----------
echo "[E1] (3/3) analyzing predictability"
python analyze.py \
    --traces "$TRACE_DIR" \
    --ckpt "$CKPT" \
    --out "$OUT_DIR/predictability.json"

echo "[E1] done."
echo "[E1] verdict:"
python -c "import json; d=json.load(open('$OUT_DIR/predictability.json')); print(json.dumps(d['summary'], indent=2))"
