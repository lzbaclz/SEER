#!/bin/bash
# Mixed-tenant stress: extends beta3 (chat-only) with concurrent doc tenants.
# Co-tenant pressure: 10 chat threads (50ms TPOT-P99) + N_DOC TTFT tenants
# sharing the same hbm/gpu_mem_util, all running simultaneously.
# Probes whether the 6-way chat-miss tie holds under co-tenant interference.
set -e

PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}
FORK=${FORK:-/home/lzq/codes/vllm-seer-fork}
OUT=${OUT:-experiments/eF_mixed_slo/results_vllm_mixed_tenant}
N_CHAT=${N_CHAT:-10}
N_DOC=${N_DOC:-4}
N_TURNS=${N_TURNS:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
HBM=${HBM:-0.2}
GPU=${CUDA_VISIBLE_DEVICES:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.20}
SLO=${SLO:-P99=200ms}
DOC_SLO=${DOC_SLO:-TTFT-P99=1s}

mkdir -p ${OUT}

COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step \
  --workload multi_turn_chat --n_turns ${N_TURNS} \
  --disable_prefix_cache \
  --hbm_budget ${HBM} --max_model_len ${MAX_MODEL_LEN} --max_tokens ${MAX_TOKENS} \
  --gpu_mem_util ${GPU_MEM_UTIL} \
  --n_chat ${N_CHAT} --n_doc ${N_DOC} --chat_slo ${SLO} --doc_slo ${DOC_SLO}"

for cell in "full 0" "streaming 0" "h2o 0" "snapkv 0" "quest 0" "seer 0" \
            "full 1" "streaming 1" "h2o 1" "snapkv 1" "quest 1" "seer 1"; do
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
    timeout 900 $PY -m experiments.eF_mixed_slo.driver \
    $COMMON --policy $POL --seed $SEED --out $OUTFILE 2>&1 | tail -5
done

$PY experiments/eF_mixed_slo/scripts/multiturn_summary.py --in_dir ${OUT}
$PY experiments/eF_mixed_slo/scripts/per_seed_table.py --in_dir ${OUT}
$PY experiments/eF_mixed_slo/scripts/multiturn_slo_sweep.py --in_dir ${OUT}
echo "ALL DONE -- ${OUT}/multiturn_summary.tex"
