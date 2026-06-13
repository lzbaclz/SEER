#!/usr/bin/env bash
# eF: mixed-SLO chat+doc concurrent driver — RTSS 2027 follow-up sweep.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-Qwen/Qwen2.5-7B}
LAP=${LAP:-../../checkpoints/lap_smoke.onnx}
N_CHAT=${N_CHAT:-10}
N_DOC=${N_DOC:-2}
BUDGET=${BUDGET:-0.2}
OUT_DIR=${OUT_DIR:-results}
WORKERS=${WORKERS:-2}
POLICIES=${POLICIES:-"h2o seer"}

mkdir -p "$OUT_DIR"

for p in $POLICIES; do
  out="$OUT_DIR/${p}_chat${N_CHAT}_doc${N_DOC}.json"
  [ -f "$out" ] && { echo "[eF] skip $out"; continue; }
  extra=()
  [ "$p" = "seer" ] && extra=(--lap "$LAP")
  echo "[eF] running policy=$p"
  python driver.py \
      --model "$MODEL" \
      --policy "$p" \
      --n_chat "$N_CHAT" \
      --n_doc "$N_DOC" \
      --hbm_budget "$BUDGET" \
      --max_workers "$WORKERS" \
      --out "$out" "${extra[@]}"
done

python analyze.py "$OUT_DIR" || true
