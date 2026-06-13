#!/bin/bash
# Path beta (R4 W3) low-concurrency complement.
#
# At n_chat=50 the 50ms TPOT deadline is structurally unmet because
# every vLLM engine.step() over a 50-prompt concurrent batch takes
# ~70-100ms wall on Llama-2-7B / A100 (the per-step batched wall
# dominates the per-token cost on its own), so chat_miss=1.0 for
# every policy and no policy claim is recoverable. This sweep
# complements run_w_multiturn_sweep.sh with n_chat=8 (8 concurrent
# threads * 5 turns = 40 prompts total) -- the concurrency regime
# where the deadline is reachable and per-policy differentiation
# is observable.
#
# Wall budget: 4 policies x 3 seeds x ~15s/cell ~= 3 minutes plus
# per-cell engine startup (~20s) so ~5 minutes total.
set -e

PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}
OUT=${OUT:-experiments/eF_mixed_slo/results_vllm_multiturn_lowconc}
N_THREADS=${N_THREADS:-10}
N_TURNS=${N_TURNS:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
HBM=${HBM:-0.2}
GPU=${CUDA_VISIBLE_DEVICES:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55}
HOOK_STRIDE=${SEER_HOOK_LAYER_STRIDE:-1}

mkdir -p ${OUT}

COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step \
  --workload multi_turn_chat --n_turns ${N_TURNS} \
  --hbm_budget ${HBM} --max_model_len ${MAX_MODEL_LEN} --max_tokens ${MAX_TOKENS} \
  --gpu_mem_util ${GPU_MEM_UTIL} \
  --n_chat ${N_THREADS} --n_doc 0 --chat_slo P99=50ms"

for policy in full streaming h2o seer; do
  for seed in 0 1 2; do
    OUTFILE=${OUT}/${policy}_s${seed}.json
    if [ -f "$OUTFILE" ]; then
      echo "SKIP ${policy}_s${seed}"; continue
    fi
    echo "=== ${policy} seed=${seed} ==="
    CUDA_VISIBLE_DEVICES=${GPU} SEER_USE_KV_MAGNITUDE=1 \
      SEER_HOOK_LAYER_STRIDE=${HOOK_STRIDE} \
      timeout 600 $PY -m experiments.eF_mixed_slo.driver \
      $COMMON --policy $policy --seed $seed --out $OUTFILE 2>&1 | tail -5
  done
done

$PY -m experiments.eF_mixed_slo.aggregate_w3 \
    --in_dir ${OUT} --out_dir ${OUT}
echo "ALL DONE -- ${OUT}/vllm_w3_aggregate.tex"
