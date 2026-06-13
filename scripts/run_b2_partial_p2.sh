#!/bin/bash
# B2 (May 2026 reviewer round): partial P2 scaleup rerun with current
# T1-A/T1-B pipeline so reviewers can inspect a cell JSON containing
# per_step_eps_measured / eps_source=runner_measured_topk /
# B_t_source=oracle_attention_top_k.
#
# Reduced scope: 4 policies × 1 budget (φ=0.20) × 1 seed × 200 prompts ×
# 64 max-tokens (P2 scale-up uses 1000×64×3-seed). ~80 min on a quiesced
# A100. Outputs to results_p2_t1_audit/ (NOT the same dir as
# results_p2_scaleup/ — the headline P999/miss is still that earlier
# 192K-step run; this is the audit that the new pipeline produces the
# expected fields).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${SEER_PY:-/home/lzq/miniconda3/envs/seer/bin/python}
MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_prod_tiny.pt}
OUT_DIR=experiments/eA_tail_latency/results_p2_t1_audit
mkdir -p "$OUT_DIR"

POLICIES=(seer h2o streaming full)
BUDGET=0.20
SEED=0
NUM_REQ=${SEER_NUM_REQ:-200}
MAX_NEW=${SEER_MAX_NEW:-64}

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for pol in "${POLICIES[@]}"; do
  btag=$(echo "$BUDGET" | tr -d '.')
  out="$OUT_DIR/${pol}_b${btag}_s${SEED}.json"
  if [ -f "$out" ]; then
    echo "[b2] skip $out"
    continue
  fi
  echo "[b2] pol=$pol budget=$BUDGET seed=$SEED nreq=$NUM_REQ"
  PYTHONPATH=. "$PY" -m seer.eval.runner \
    --model "$MODEL" \
    --workload mooncake \
    --num_requests "$NUM_REQ" \
    --max_new_tokens "$MAX_NEW" \
    --policy "$pol" \
    --hbm_budget "$BUDGET" \
    --io_mode measured-dma \
    --slo "P99=50ms" \
    --seed "$SEED" \
    ${LAP:+--lap "$LAP"} \
    --out "$out"
done

echo "[b2] done"
