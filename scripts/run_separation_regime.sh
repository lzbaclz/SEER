#!/bin/bash
# J.5 separation regime: high pressure + vLLM prefix cache OFF + plugin path,
# 4 policies. The previous benign regime had vLLM's prefix cache absorbing
# the policy gap (chat_miss identical to 4 decimals across SEER/H2O/Streaming/Full).
# Disabling prefix cache exposes whatever per-policy difference the connector
# can drive on its own (currently: scheduler-side bookkeeping; worker-side
# transport is no-op pending J.6).
#
# Run on a single empty GPU. ~5 min per policy.
set -euo pipefail

OUT=experiments/eF_mixed_slo/results_vllm_separation
mkdir -p "$OUT"
LAP=checkpoints/lap_prod_b1024.onnx

for POLICY in seer h2o streaming full; do
    echo "=== policy=$POLICY ==="
    python -m experiments.eF_mixed_slo.driver \
        --backend vllm \
        --vllm_plugin_seer \
        --use_engine_step \
        --policy "$POLICY" \
        --lap "$LAP" \
        --model meta-llama/Llama-2-7b-hf \
        --n_chat 20 --n_doc 0 \
        --hbm_budget 0.10 \
        --long_context 400 \
        --max_tokens 128 \
        --max_model_len 3072 \
        --gpu_mem_util 0.85 \
        --tensor_parallel 1 \
        --disable_prefix_cache \
        --out "$OUT/${POLICY}_nopfx_chat20.json"
done
echo "=== done ==="
