#!/bin/bash
# Path beta (R4 W3 follow-through): multi-turn KV-reuse workload sweep.
# This sweep is the closure of the n_load=0 namespace gap reported in
# results_vllm_w3_xfer_on/: by switching to a workload with cross-prompt
# KV reuse (multi-turn chat with a shared system prompt) and enabling
# vLLM's prefix-cache absorber, the scheduler-side decision and the
# worker-side save keys live in the same cache-slot namespace and the
# substrate load events fire from real prefetch decisions.
#
# Target metrics:
#   n_load > 50 per seed (proves the namespace gap is closed).
#   SEER chat-miss < H2O chat-miss with non-overlapping Wilson 95% CI.
#   Relative chat-miss reduction >= 20%.
#
# This script is queued for GPU run; the multi_turn_chat workload
# generator is unit-testable without a GPU
# (`python -m experiments.eF_mixed_slo.workloads.multi_turn_chat`),
# but the actual vLLM sweep below requires 1x A100 SXM4-80GB and
# ~6 hours of wall time (4 policies x 3 seeds x ~30 min/cell).

set -e

# Requires that experiments/eF_mixed_slo/driver.py be patched to
# accept --workload multi_turn_chat (see workloads/multi_turn_chat.py
# docstring). The driver currently hard-codes a templated chat prompt;
# the multi-turn integration is a 30-line change pending GPU access:
#
#   1. driver.py:128 — replace _make_chat_prompt with a call to
#      workloads.multi_turn_chat.generate(n_threads=args.n_chat,
#      n_turns=args.n_turns).
#   2. driver.py CLI: add --workload {default,multi_turn_chat} and
#      --n_turns (default 5).
#   3. The driver should write the (thread_id, turn_id) of each
#      result so the aggregator can compute per-thread vs per-turn
#      metrics.
#
# Until that integration lands, this script is parked but reproducible
# at the workload-generator level.

PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}
OUT=${OUT:-experiments/eF_mixed_slo/results_vllm_multiturn}
N_THREADS=${N_THREADS:-50}
N_TURNS=${N_TURNS:-5}
MAX_TOKENS=${MAX_TOKENS:-64}
HBM=${HBM:-0.2}
GPU=${CUDA_VISIBLE_DEVICES:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-3072}
HOOK_STRIDE=${SEER_HOOK_LAYER_STRIDE:-1}

mkdir -p ${OUT}

# Common flags. prefix_caching is left at vLLM default (enabled) --
# this is the structural difference vs the w3_p25L_fix sweep that
# we hope closes the namespace gap. SEER_HOOK_LAYER_STRIDE=1 hooks
# all 32 Attention layers (vs the 4-of-32 default at stride=8) so
# the LAP attention proxy fires at every layer.
COMMON="--backend vllm --model meta-llama/Llama-2-7b-hf \
  --lap checkpoints/lap_prod_tiny.pt \
  --vllm_plugin_seer --use_engine_step \
  --workload multi_turn_chat --n_turns ${N_TURNS} \
  --hbm_budget ${HBM} --max_model_len ${MAX_MODEL_LEN} --max_tokens ${MAX_TOKENS} \
  --gpu_mem_util ${GPU_MEM_UTIL} \
  --n_chat ${N_THREADS} --n_doc 0 --chat_slo P99=50ms"

# Xfer ENABLED in this sweep (SEER_DISABLE_XFER unset).
for policy in full streaming h2o seer; do
  for seed in 0 1 2; do
    OUTFILE=${OUT}/${policy}_s${seed}.json
    if [ -f "$OUTFILE" ]; then
      echo "SKIP ${policy}_s${seed} (output exists)"
      continue
    fi
    echo "=== ${policy} seed=${seed} ==="
    CUDA_VISIBLE_DEVICES=${GPU} SEER_USE_KV_MAGNITUDE=1 \
      SEER_HOOK_LAYER_STRIDE=${HOOK_STRIDE} \
      timeout 1800 $PY -m experiments.eF_mixed_slo.driver \
      $COMMON --policy $policy --seed $seed --out $OUTFILE 2>&1 | tail -5
  done
done

# Aggregate (uses the existing aggregator; the multi-turn driver
# should emit the same JSON schema with optional thread_id/turn_id
# fields so the aggregator gives both per-turn and per-thread stats).
$PY -m experiments.eF_mixed_slo.aggregate_w3 \
    --in_dir ${OUT} --out_dir ${OUT}

echo "ALL DONE -- inspect ${OUT}/vllm_w3_aggregate.{csv,tex,json}"
echo "Target gates:"
echo "  - n_load > 50 per seed (namespace gap closed)"
echo "  - SEER chat-miss < H2O chat-miss with non-overlapping Wilson CI"
echo "  - Relative chat-miss reduction >= 20%"
