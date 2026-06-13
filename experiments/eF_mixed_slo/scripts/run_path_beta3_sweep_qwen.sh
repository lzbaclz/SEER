#!/bin/bash
# Qwen2.5-7B cross-model multi-turn beta3 sweep.
# 5 policies (full,streaming,h2o,snapkv,quest,seer) x 2 seeds, same op-point
# as the Llama-2-7B beta3 (50 prompts/seed = 10 threads x 5 turns,
# disable_prefix_cache, gpu_mem_util=0.20, P99=200ms).
set -e

PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}
FORK=${FORK:-/home/lzq/codes/vllm-seer-fork}
OUT=${OUT:-experiments/eF_mixed_slo/results_vllm_path_beta3_qwen}
LAP=${LAP:-checkpoints/lap_qwen25.pt}
N_THREADS=${N_THREADS:-10}
N_TURNS=${N_TURNS:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
HBM=${HBM:-0.2}
GPU=${CUDA_VISIBLE_DEVICES:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.40}
# Qwen2.5-7B is heavier than Llama-2-7B (656ms P50 TPOT vs 100ms);
# loosened SLO so per-policy comparison is non-trivial.
SLO=${SLO:-P99=1000ms}

mkdir -p ${OUT}

COMMON="--backend vllm --model Qwen/Qwen2.5-7B \
  --lap ${LAP} \
  --vllm_plugin_seer --use_engine_step \
  --workload multi_turn_chat --n_turns ${N_TURNS} \
  --disable_prefix_cache \
  --hbm_budget ${HBM} --max_model_len ${MAX_MODEL_LEN} --max_tokens ${MAX_TOKENS} \
  --gpu_mem_util ${GPU_MEM_UTIL} \
  --n_chat ${N_THREADS} --n_doc 0 --chat_slo ${SLO}"

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
    timeout 600 $PY -m experiments.eF_mixed_slo.driver \
    $COMMON --policy $POL --seed $SEED --out $OUTFILE 2>&1 | tail -5
done

$PY experiments/eF_mixed_slo/scripts/multiturn_summary.py --in_dir ${OUT}
$PY experiments/eF_mixed_slo/scripts/per_seed_table.py --in_dir ${OUT}
$PY experiments/eF_mixed_slo/scripts/multiturn_slo_sweep.py --in_dir ${OUT}
echo "ALL DONE -- ${OUT}/multiturn_summary.tex"
