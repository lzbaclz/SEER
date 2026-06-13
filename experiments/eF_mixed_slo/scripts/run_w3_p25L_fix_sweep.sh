#!/bin/bash
# Phase 2.5L-fix re-sweep: re-run the 4-policy x 3-seed matrix with the
# corrected K-magnitude path (centroids computed on slot_mapping-touched
# blocks only; candidate set bounded to ever-touched blocks; GPU-side
# topk; single sync at end of each save_kv_layer call).
#
# Previous P2.5L commit (b06bf52) computed centroids on the unfilled
# 6744-slot static pool and "refreshed" the last slot of the pool, so
# all magnitudes stayed near zero. Ranking-by-magnitude degenerated to
# ranking-by-block-id, which is recency-correlated → SEER/h2o/streaming
# all gave the same chat-miss. This re-sweep is the certificate that
# the corrected ranking matters at the chat-miss level (or, if all
# four policies still cluster, that the substrate effect dominates,
# which is also a paper-worthy finding).
set -e
PY=/home/lzq/miniconda3/envs/seer/bin/python
OUT=experiments/eF_mixed_slo/results_vllm_w3_p25L_fix
mkdir -p ${OUT}
COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step --disable_prefix_cache \
  --hbm_budget 0.2 --max_model_len 2048 --max_tokens 64 --gpu_mem_util 0.85 \
  --n_chat 50 --n_doc 0 --chat_slo P99=50ms"
for policy in full streaming h2o seer; do
  for seed in 0 1 2; do
    OUTFILE=${OUT}/${policy}_s${seed}.json
    if [ -f "$OUTFILE" ]; then echo "SKIP ${policy}_s${seed}"; continue; fi
    echo "=== ${policy} seed=${seed} ==="
    SEER_NO_HOOK_INSTALL=1 SEER_USE_KV_MAGNITUDE=1 SEER_DISABLE_XFER=1 \
      timeout 240 $PY -m experiments.eF_mixed_slo.driver \
      $COMMON --policy $policy --seed $seed --out $OUTFILE 2>&1 | tail -3
  done
done
echo "ALL DONE"
