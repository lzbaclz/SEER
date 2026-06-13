#!/usr/bin/env bash
# R36e B: multi-tenant production-serving multi-seed sweep
# (reviewer's "真实 multi-tenant production serving" cycle-2 ask;
# upgrades the existing eF_mixed_slo single-seed transparency note
# to a 5-seed reviewer-defending sweep at chat-tier deadlines).
#
# Configuration:
#   - Workload mix: chat (TPOT-bound) + doc (TTFT-bound)
#   - Arrival: configurable Poisson rate per priority class
#   - SLO: chat TPOT P99 = 200 ms (relaxed-$D$ band on A100),
#          doc TTFT = 1 s
#   - Policies: full / h2o / seer (the three that survive R32-R36
#     in-tree-baseline scoping; snapkv/quest are static-budget
#     reference)
#   - Seeds: 5 (closes reviewer's n_seeds >= 5 ask)
#
# Total wall time: ~30-40 min on a quiet A100 with vLLM warm
# (3 policies × 5 seeds = 15 cells, each ~2 min).
#
# Output:
#   results_multitenant/{policy}_s{seed}.json
#   paper/multitenant_pareto.tex (rendered post-hoc by
#     scripts/gen_multitenant_table.py)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if ! python -c "import seer, torch" 2>/dev/null; then
  if [[ -f /home/lzq/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/lzq/miniconda3/etc/profile.d/conda.sh
    conda activate seer
  fi
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Workaround for stale MPS socket: see comment in
# experiments/eA_tail_latency/run_baseline_parity.sh for context.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$(mktemp -d -t seer-no-mps-XXXX)}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$CUDA_MPS_PIPE_DIRECTORY}"

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-$REPO_ROOT/checkpoints/lap_prod_dyn.onnx}
N_CHAT=${N_CHAT:-30}                   # 30 chat / 6 doc — 5:1 mix
N_DOC=${N_DOC:-6}
BUDGET=${BUDGET:-0.20}
WORKERS=${WORKERS:-4}                   # 4 concurrent vLLM workers
SEEDS=${SEEDS:-"0 1 2 3 4"}
POLICIES=${POLICIES:-"full h2o seer"}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/results_multitenant}

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

run_cell () {
  local policy=$1 seed=$2
  local out=$OUT_DIR/${policy}_s${seed}.json
  if [[ -s $out ]]; then
    echo "[multitenant] skip $policy seed=$seed (exists)"
    return
  fi
  echo "[multitenant] running policy=$policy seed=$seed -> $out"
  # vllm backend always needs --lap (even for non-seer policies it
  # is used as the connector's bootstrap path). For simulator backend
  # --lap is only used by seer.
  local extra=(--lap "$LAP")
  PYTHONHASHSEED=$seed \
    python driver.py \
      --model "$MODEL" \
      --backend "${BACKEND:-vllm}" \
      --chat_slo "${CHAT_SLO:-P99=200ms}" \
      --doc_slo "${DOC_SLO:-TTFT-P99=1s}" \
      --policy "$policy" \
      --n_chat "$N_CHAT" \
      --n_doc "$N_DOC" \
      --hbm_budget "$BUDGET" \
      --max_workers "$WORKERS" \
      --max_model_len "${MAX_MODEL_LEN:-2048}" \
      --gpu_mem_util "${GPU_MEM_UTIL:-0.85}" \
      --seed "$seed" \
      --out "$out" "${extra[@]}"
}

for p in $POLICIES; do
  for s in $SEEDS; do
    run_cell "$p" "$s"
  done
done

echo "[multitenant] done. Rendering table:"
python "$REPO_ROOT/scripts/gen_multitenant_table.py"
