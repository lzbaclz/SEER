#!/usr/bin/env bash
# E2: Quality-Budget Pareto frontier sweep.
# Expects a trained LAP checkpoint at $LAP_CKPT (default: ../../checkpoints/lap.onnx).
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
MODEL_TAG=${MODEL_TAG:-llama3-8b}
LAP_CKPT=${LAP_CKPT:-../../checkpoints/lap_${MODEL_TAG}.onnx}
BENCHMARK=${BENCHMARK:-ruler}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
NUM_PROMPTS=${NUM_PROMPTS:-50}
OUT_DIR=${OUT_DIR:-results/${MODEL_TAG}_${BENCHMARK}_ctx${CONTEXT_LENGTH}}
mkdir -p "$OUT_DIR"

POLICIES=("full" "streaming" "h2o" "snapkv" "recency" "random" "seer")
BUDGETS=(0.10 0.20 0.40 0.80)

for policy in "${POLICIES[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    # full cache is the same for any budget; only run once
    [ "$policy" = "full" ] && [ "$budget" != "0.80" ] && continue

    tag="${policy}_b$(echo "$budget" | tr -d .)"
    out="$OUT_DIR/$tag.json"
    if [ -f "$out" ]; then
      echo "[E2] skip $tag (exists)"
      continue
    fi

    extra=()
    [ "$policy" = "seer" ] && extra=(--lap_ckpt "$LAP_CKPT")

    echo "[E2] running $tag"
    python -m seer.eval.runner \
        --model "$MODEL" \
        --policy "$policy" \
        --benchmark "$BENCHMARK" \
        --context_length "$CONTEXT_LENGTH" \
        --num_prompts "$NUM_PROMPTS" \
        --budget "$budget" \
        --out "$out" \
        "${extra[@]}"
  done
done

echo "[E2] sweep complete → $OUT_DIR"
