#!/usr/bin/env bash
# E3 (SKELETON): Throughput at iso-quality across context lengths.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
LAP_CKPT=${LAP_CKPT:-../../checkpoints/lap_llama3-8b.onnx}
OUT_DIR=${OUT_DIR:-results}
mkdir -p "$OUT_DIR"

CONTEXTS=(8192 32768)
POLICIES=("full" "h2o" "seer")
BUDGET=${BUDGET:-0.20}

for ctx in "${CONTEXTS[@]}"; do
  for p in "${POLICIES[@]}"; do
    out="$OUT_DIR/ctx${ctx}_${p}.json"
    [ -f "$out" ] && { echo "[E3] skip $out"; continue; }
    extra=()
    [ "$p" = "seer" ] && extra=(--lap_ckpt "$LAP_CKPT")
    echo "[E3] ctx=$ctx policy=$p"
    python -m seer.eval.runner \
        --model "$MODEL" --policy "$p" \
        --benchmark ruler --context_length "$ctx" \
        --budget "$BUDGET" --num_prompts 30 \
        --max_new_tokens 128 \
        --out "$out" "${extra[@]}"
  done
done

echo "[E3] (placeholder) — throughput numbers here are HF wall-time, not vLLM."
