#!/bin/bash
# Multi-seed B2 audit (R1-Cut3 fully close): 3 seeds × 4 policies × phi=0.20
# × n=200 × max_new=64. ~80 min per seed on a quiesced A100 → ~4h total.
# Bootstrap CI on the pessimism factor itself (not just on measured
# miss-ratio) is computed by scripts/recompute_p2_t1_audit_bound.py after
# this script's outputs land in results_p2_t1_audit/.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${SEER_PY:-/home/lzq/miniconda3/envs/seer/bin/python}
MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_prod_tiny.pt}
OUT_DIR=experiments/eA_tail_latency/results_p2_t1_audit
mkdir -p "$OUT_DIR"

POLICIES=(seer h2o streaming full)
BUDGET=0.20
SEEDS=(0 1 2)
NUM_REQ=${SEER_NUM_REQ:-200}
MAX_NEW=${SEER_MAX_NEW:-64}

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for SEED in "${SEEDS[@]}"; do
  for pol in "${POLICIES[@]}"; do
    btag=$(echo "$BUDGET" | tr -d '.')
    out="$OUT_DIR/${pol}_b${btag}_s${SEED}.json"
    if [ -f "$out" ]; then
      echo "[b2-ms] skip $out (exists)"
      continue
    fi
    echo "[b2-ms] pol=$pol budget=$BUDGET seed=$SEED nreq=$NUM_REQ"
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
done

echo "[b2-ms] all seeds + policies complete"
echo "[b2-ms] now: PYTHONPATH=. $PY scripts/recompute_p2_t1_audit_bound.py"
