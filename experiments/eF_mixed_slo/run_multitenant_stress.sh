#!/usr/bin/env bash
# R36e-stress: real multi-tenant production-serving stress sweep.
#
# The R36e-measured baseline run (30 chat / 6 doc / max_tokens=64)
# completed each cell in 1.2 s with chat_miss = 0 across all
# policies — too light to differentiate at any SLO.  This stress
# variant scales the workload until vLLM is actually saturated:
#   - n_chat = 120 (4x baseline)
#   - n_doc  = 24  (4x baseline)
#   - max_tokens = 512 (8x baseline)
#   - max_workers = 8 (2x baseline)
#   - chat SLO at the relaxed D=200ms operating point
#
# Target wall: ~30-60 s per cell (vs 1.2 s baseline), enough to
# saturate the BlockPool and surface per-policy tail differences.
# 3 policies x 5 seeds = 15 cells, ~10-15 min total.
#
# Output: experiments/eF_mixed_slo/results_multitenant_stress/
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
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$(mktemp -d -t seer-no-mps-XXXX)}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$CUDA_MPS_PIPE_DIRECTORY}"

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-$REPO_ROOT/checkpoints/lap_prod_dyn.onnx}

# Stressed workload: 4x prompts, 8x decode length, 2x workers.
N_CHAT=${N_CHAT:-120}
N_DOC=${N_DOC:-24}
BUDGET=${BUDGET:-0.20}
WORKERS=${WORKERS:-8}
MAX_TOKENS=${MAX_TOKENS:-512}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
CHAT_SLO=${CHAT_SLO:-P99=200ms}
DOC_SLO=${DOC_SLO:-TTFT-P99=1s}

SEEDS=${SEEDS:-"0 1 2 3 4"}
POLICIES=${POLICIES:-"full h2o seer"}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/results_multitenant_stress}

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

run_cell () {
  local policy=$1 seed=$2
  local out=$OUT_DIR/${policy}_s${seed}.json
  if [[ -s $out ]]; then
    echo "[stress] skip $policy seed=$seed (exists)"
    return
  fi
  echo "[stress] running policy=$policy seed=$seed n_chat=$N_CHAT n_doc=$N_DOC max_tokens=$MAX_TOKENS -> $out"
  PYTHONHASHSEED=$seed \
    python driver.py \
      --model "$MODEL" \
      --backend vllm \
      --chat_slo "$CHAT_SLO" \
      --doc_slo "$DOC_SLO" \
      --policy "$policy" \
      --n_chat "$N_CHAT" \
      --n_doc "$N_DOC" \
      --hbm_budget "$BUDGET" \
      --max_workers "$WORKERS" \
      --max_tokens "$MAX_TOKENS" \
      --max_model_len "$MAX_MODEL_LEN" \
      --gpu_mem_util "${GPU_MEM_UTIL:-0.85}" \
      --seed "$seed" \
      --lap "$LAP" \
      --out "$out"
}

for p in $POLICIES; do
  for s in $SEEDS; do
    run_cell "$p" "$s"
  done
done

echo "[stress] done. Rendering table:"
OUT_DIR_ABS=$OUT_DIR python "$REPO_ROOT/scripts/gen_multitenant_table.py"
