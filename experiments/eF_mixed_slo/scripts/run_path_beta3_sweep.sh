#!/bin/bash
# Path beta3 (v-beta-3-real-eviction): SEER eviction-policy hook in
# the vLLM source path. Uses a forked vLLM 0.8.5.post1 with a
# 31-line patch on `vllm.v1.core.kv_cache_manager.KVCacheManager.free`
# that consults the registered KVConnector_V1 for the per-request
# eviction order. The fork lives at /home/lzq/codes/vllm-seer-fork
# on branch `seer-eviction-hook`; the patch file is in
# `seer/integration/vllm_patches/0001-add-eviction-order-hook.patch`
# (upstream-PR-ready diff).
#
# Same operating point as path_beta2_routed: 50 prompts (10
# threads x 5 turns), --disable_prefix_cache, gpu_mem_util=0.20,
# chat_slo P99=200ms. Wall budget per cell ~80s + load ~22s.
set -e

PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}
FORK=${FORK:-/home/lzq/codes/vllm-seer-fork}
OUT=${OUT:-experiments/eF_mixed_slo/results_vllm_path_beta3}
N_THREADS=${N_THREADS:-10}
N_TURNS=${N_TURNS:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
HBM=${HBM:-0.2}
GPU=${CUDA_VISIBLE_DEVICES:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.20}
SLO=${SLO:-P99=200ms}

mkdir -p ${OUT}

COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step \
  --workload multi_turn_chat --n_turns ${N_TURNS} \
  --disable_prefix_cache \
  --hbm_budget ${HBM} --max_model_len ${MAX_MODEL_LEN} --max_tokens ${MAX_TOKENS} \
  --gpu_mem_util ${GPU_MEM_UTIL} \
  --n_chat ${N_THREADS} --n_doc 0 --chat_slo ${SLO}"

# Random execution order to spread per-seed warmup variance across
# policies (not sequential per-policy).
for cell in "full 0" "h2o 0" "seer 0" "streaming 0" \
            "h2o 1" "seer 1" "streaming 1" "full 1" \
            "streaming 2" "seer 2" "full 2" "h2o 2"; do
  POL=$(echo $cell | cut -d' ' -f1)
  SEED=$(echo $cell | cut -d' ' -f2)
  OUTFILE=${OUT}/${POL}_s${SEED}.json
  if [ -f "$OUTFILE" ]; then
    echo "SKIP ${POL}_s${SEED}"; continue
  fi
  echo "=== ${POL} seed=${SEED} ==="
  PYTHONPATH=${FORK}:${PYTHONPATH} \
    CUDA_VISIBLE_DEVICES=${GPU} \
    SEER_USE_KV_MAGNITUDE=1 SEER_HOOK_LAYER_STRIDE=1 \
    SEER_BLOCK_TABLE_LOAD=1 SEER_BLOCK_TABLE_LOAD_COPY=1 \
    SEER_POLICY_ROUTE_WORKER=1 SEER_LAP_DECISION_PERIOD=64 \
    timeout 600 $PY -m experiments.eF_mixed_slo.driver \
    $COMMON --policy $POL --seed $SEED --out $OUTFILE 2>&1 | tail -5
done

$PY experiments/eF_mixed_slo/scripts/multiturn_summary.py --in_dir ${OUT}
echo "ALL DONE -- ${OUT}/multiturn_summary.tex"
