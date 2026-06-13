#!/bin/bash
# Path-β4: scaled sweep on patched vLLM fork to address W5
# (statistical power). N_TURNS=10 (default) × N_THREADS=10 × 64 tokens
# × 3 seeds = ~18.8K decode steps per seed per policy (rotated chat
# prompts; raw step counts in *_s{0,1,2}.json:chat_miss_step_total_n),
# enough for Wilson-95% CI of width ~0.0013 at a 0.002 base rate.
# Six policies × 3 seeds = 18 cells. Default cell list runs
# {full,h2o,seer,streaming,snapkv,quest} × seeds {0,1,2}. To run only
# the four-policy core (legacy behaviour), pass POLICIES="full h2o seer streaming".
#
# Wall budget: ~150s per cell at 10 turns x 10 threads (see
# multiturn_summary.json:wall_s_mean), 18 cells = ~45 min + load
# overhead. Set N_TURNS=25 to match the earlier (~48K steps/cell)
# sizing comment.
set -e

PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}
FORK=${FORK:-${SEER_VLLM_FORK:-/home/lzq/codes/vllm-seer-fork}}
OUT=${OUT:-experiments/eF_mixed_slo/results_vllm_path_beta4_scaled}
N_THREADS=${N_THREADS:-10}
N_TURNS=${N_TURNS:-10}
MAX_TOKENS=${MAX_TOKENS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
HBM=${HBM:-0.2}
GPU=${CUDA_VISIBLE_DEVICES:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.20}
SLO=${SLO:-P99=200ms}
SEEDS=${SEEDS:-"0 1 2"}
POLICIES=${POLICIES:-"full streaming h2o snapkv quest seer"}

mkdir -p ${OUT}

COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step \
  --workload multi_turn_chat --n_turns ${N_TURNS} \
  --disable_prefix_cache \
  --hbm_budget ${HBM} --max_model_len ${MAX_MODEL_LEN} --max_tokens ${MAX_TOKENS} \
  --gpu_mem_util ${GPU_MEM_UTIL} \
  --n_chat ${N_THREADS} --n_doc 0 --chat_slo ${SLO}"

CELLS=""
for POL in $POLICIES; do
  for SEED in $SEEDS; do
    CELLS="${CELLS}${POL}:${SEED} "
  done
done

for cell in $CELLS; do
  POL=$(echo $cell | cut -d':' -f1)
  SEED=$(echo $cell | cut -d':' -f2)
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
    timeout 1200 $PY -m experiments.eF_mixed_slo.driver \
    $COMMON --policy $POL --seed $SEED --out $OUTFILE 2>&1 | tail -5
done

$PY experiments/eF_mixed_slo/scripts/multiturn_summary.py --in_dir ${OUT}
echo "ALL DONE -- ${OUT}/multiturn_summary.tex"
