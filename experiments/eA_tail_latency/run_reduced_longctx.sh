#!/usr/bin/env bash
# Reduced long-context driver (M9 fix).
# Full sweep at 32K prefill / 1024 decode is ~8 GPU hours; this driver
# runs 16K prefill / 512 decode × 3 prompts × 5 policies × 2 budgets
# (≈45 min on A100) so the long-context regime is exercised within
# one session. The full 32K/1024 sweep remains as `run_longctx.sh`.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${SEER_PY:-/home/lzq/miniconda3/envs/seer/bin/python}
MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_prod_tiny.pt}
OUT_DIR=experiments/eA_tail_latency/results_longctx_reduced
mkdir -p "$OUT_DIR"

CTX=${SEER_CTX:-16384}
POLICIES=(seer h2o streaming snapkv full)
BUDGETS=(0.10 0.20)
MAX_NEW=${SEER_MAX_NEW:-512}
NUM_REQ=${SEER_NUM_REQ:-3}
WORKLOAD=${SEER_WORKLOAD:-longbench}
LONGBENCH_TASK=${SEER_LONGBENCH_TASK:-qmsum}

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for pol in "${POLICIES[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    btag=$(echo "$budget" | tr -d '.')
    out="$OUT_DIR/${pol}_b${btag}_ctx${CTX}.json"
    if [ -f "$out" ]; then
      echo "[skip] $out"
      continue
    fi
    echo "[run]  pol=$pol budget=$budget ctx=$CTX"
    PYTHONPATH=. "$PY" -m seer.eval.runner \
      --model "$MODEL" \
      --workload "$WORKLOAD" \
      ${LONGBENCH_TASK:+--longbench_task "$LONGBENCH_TASK"} \
      --num_requests "$NUM_REQ" \
      --max_new_tokens "$MAX_NEW" \
      --policy "$pol" \
      --hbm_budget "$budget" \
      --io_mode measured-dma \
      --slo "P99=50ms" \
      --context_length "$CTX" \
      ${LAP:+--lap "$LAP"} \
      --out "$out" || {
        echo "[warn] $pol@b${btag} failed; continuing"
        continue
      }
  done
done

PYTHONPATH=. "$PY" experiments/eA_tail_latency/aggregate_longctx.py \
  --in_dir "$OUT_DIR" --out_tex "$OUT_DIR/table.tex" \
  --out_json "$OUT_DIR/summary.json" || true
echo "[done] $OUT_DIR/table.tex"
