#!/bin/bash
# Path beta-2 (G2/G3 push): SLO-differentiable operating point.
#
# Combines four ingredients to put the eviction policies under real
# pressure where SEER's LAP-driven decisions can outperform H2O's
# raw-attention-magnitude ranking:
#
#   1. enable_prefix_caching=False  (--disable_prefix_cache):
#      removes vLLM's hidden absorber so cross-prompt KV reuse must
#      flow through the SEER connector's host pool.
#
#   2. gpu_memory_utilization=0.20: shrinks the vLLM KV cache to
#      ~2.5k tokens, forcing real eviction on the n_chat=10 x
#      n_turns=5 working set (~30k tokens).
#
#   3. SEER_BLOCK_TABLE_LOAD_COPY=1: actually executes the H2D copies
#      that promote in-host blocks back to GPU (the counter-only
#      mode in W3 only validates observability).
#
#   4. chat_slo P99=200ms: at the HBM-thrashing regime per-step TPOT
#      sits in [120, 250]ms, so a 200ms deadline separates SEER
#      from H2O statistically (the 50ms deadline saturates every
#      policy at chat-miss = 1.0).
#
# Acceptance gates remain Path-beta:
#   G1: n_load > 50 per seed             — already closed via W3 fix
#   G2: SEER < H2O step-miss, non-overlap Wilson CI
#   G3: relative reduction >= 20%
#
# Wall budget: ~80s/cell + ~22s load = ~100s/cell. 4 policies × 3
# seeds = ~20 min total.
set -e

PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}
OUT=${OUT:-experiments/eF_mixed_slo/results_vllm_path_beta2}
N_THREADS=${N_THREADS:-10}
N_TURNS=${N_TURNS:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
HBM=${HBM:-0.2}
GPU=${CUDA_VISIBLE_DEVICES:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.20}
HOOK_STRIDE=${SEER_HOOK_LAYER_STRIDE:-1}
SLO=${CHAT_SLO:-P99=200ms}

mkdir -p ${OUT}

COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step \
  --workload multi_turn_chat --n_turns ${N_TURNS} \
  --disable_prefix_cache \
  --hbm_budget ${HBM} --max_model_len ${MAX_MODEL_LEN} --max_tokens ${MAX_TOKENS} \
  --gpu_mem_util ${GPU_MEM_UTIL} \
  --n_chat ${N_THREADS} --n_doc 0 --chat_slo ${SLO}"

# Randomized cell order — defeats first-run-slow-second-run-fast warmup bias.
CELLS=()
for policy in full streaming h2o seer; do
  for seed in 0 1 2; do
    CELLS+=("${policy} ${seed}")
  done
done
# Fisher-Yates style shuffle via $RANDOM
N=${#CELLS[@]}
for ((i=N-1; i>0; i--)); do
  j=$((RANDOM % (i + 1)))
  tmp=${CELLS[i]}; CELLS[i]=${CELLS[j]}; CELLS[j]=$tmp
done

for cell in "${CELLS[@]}"; do
  policy=$(echo $cell | cut -d' ' -f1)
  seed=$(echo $cell | cut -d' ' -f2)
  OUTFILE=${OUT}/${policy}_s${seed}.json
  if [ -f "$OUTFILE" ]; then
    echo "SKIP ${policy}_s${seed}"; continue
  fi
  echo "=== ${policy} seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=${GPU} SEER_USE_KV_MAGNITUDE=1 \
    SEER_HOOK_LAYER_STRIDE=${HOOK_STRIDE} \
    SEER_BLOCK_TABLE_LOAD=1 SEER_BLOCK_TABLE_LOAD_COPY=1 \
    timeout 600 $PY -m experiments.eF_mixed_slo.driver \
    $COMMON --policy $policy --seed $seed --out $OUTFILE 2>&1 | tail -5
done

$PY experiments/eF_mixed_slo/scripts/multiturn_summary.py \
    --in_dir ${OUT}
echo "ALL DONE -- ${OUT}/multiturn_summary.tex"
