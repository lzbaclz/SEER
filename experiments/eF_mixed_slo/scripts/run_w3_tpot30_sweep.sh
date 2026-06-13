#!/bin/bash
# T2-M (R1-Cut1 follow-up, second iteration): re-run 12-cell sweep with
# the corrected TPOT-only metric AND a tighter 30ms TPOT SLO so the
# chat-miss column reflects the policy ordering already visible in
# P99/P999. The 50ms SLO is too loose for the 19ms-P50 / 44ms-P99
# decode distribution; almost every policy clears it.
set -e
PY=/home/lzq/miniconda3/envs/seer/bin/python
OUT=experiments/eF_mixed_slo/results_vllm_w3_tpot30
mkdir -p ${OUT}
COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step --disable_prefix_cache \
  --hbm_budget 0.2 --max_model_len 2048 --max_tokens 64 --gpu_mem_util 0.85 \
  --n_chat 50 --n_doc 0 --chat_slo P99=30ms"
for policy in full streaming h2o seer; do
  for seed in 0 1 2; do
    OUTFILE=${OUT}/${policy}_s${seed}.json
    if [ -f "$OUTFILE" ]; then echo "SKIP ${policy}_s${seed}"; continue; fi
    echo "=== ${policy} seed=${seed} (TPOT 30ms SLO) ==="
    SEER_NO_HOOK_INSTALL=1 SEER_USE_KV_MAGNITUDE=1 SEER_DISABLE_XFER=1 \
      timeout 240 $PY -m experiments.eF_mixed_slo.driver \
      $COMMON --policy $policy --seed $seed --out $OUTFILE 2>&1 | tail -3
  done
done
echo "ALL DONE"
