#!/usr/bin/env bash
# R36d baseline parity audit (reviewer's "完成 upstream SnapKV/Quest/H2O
# parity" item). Reuses the same Mooncake-24 + Llama-2-7B-hf + A100 +
# P99=200ms harness the R32 H2O parity used; env-var dispatch picks
# the upstream vs in-tree variant. Result: two cells per baseline
# (intree, upstream) for chat-miss + P99 TPOT parity comparison.
#
# Configuration:
#   - SnapKV: SEER_SNAPKV_VARIANT={intree,upstream}
#       upstream = FasterDecoding/SnapKV sliding obs-window avg-pool
#   - Quest:  SEER_QUEST_VARIANT={intree,upstream}
#       upstream = sub-page max-attention with recent_floor=8
#   - H2O is already covered by R32 (run_h2o_upstream_parity.sh)
#
# Total wall time: ~10 min on a quiet A100 (24 prompts x 2 variants
# x 2 baselines = 4 cells; 24 prompts is the headline Mooncake size).
#
# Output:
#   results_baseline_parity/{snapkv,quest}/{intree,upstream}/b020.json
# which scripts/gen_baseline_parity_table.py renders to
# paper/baseline_parity.tex.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Activate seer conda env if not already (so `python -m seer.eval.runner`
# can find torch + seer together; `seer` may be importable in base via
# `pip install -e .` but base lacks torch, so we MUST require both).
if ! python -c "import seer, torch" 2>/dev/null; then
  if [[ -f /home/lzq/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/lzq/miniconda3/etc/profile.d/conda.sh
    conda activate seer
  fi
fi

# Offline-friendly default (Mooncake-24 is a synthetic mirror, not HF-gated).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Disable MPS routing for our CUDA inits. If a previous MPS daemon
# wedged and was killed without cleaning /tmp/nvidia-mps, new CUDA
# clients still try to connect to the dead control socket and hit
# "Error 805: MPS client failed to connect to the MPS control
# daemon". Pointing CUDA_MPS_PIPE_DIRECTORY at an empty dir makes
# the runtime fall back to direct (non-MPS) CUDA contexts.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$(mktemp -d -t seer-no-mps-XXXX)}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$CUDA_MPS_PIPE_DIRECTORY}"

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-$REPO_ROOT/checkpoints/lap_prod_dyn.onnx}
WORKLOAD=${WORKLOAD:-mooncake}
CONTEXT=${CONTEXT_LENGTH:-4096}
SLO=${SLO:-P99=200ms}
NUM=${NUM_REQUESTS:-24}
BUDGET=${BUDGET:-0.20}

ROOT_OUT=${ROOT_OUT:-$SCRIPT_DIR/results_baseline_parity}
mkdir -p "$ROOT_OUT"

run_cell () {
  local policy=$1 variant=$2 outdir=$ROOT_OUT/$policy/$variant
  mkdir -p "$outdir"
  local out=$outdir/b020.json
  if [[ -s $out ]]; then
    echo "[parity] skip $policy $variant (exists: $out)"
    return
  fi
  echo "[parity] running $policy $variant -> $out"
  if [[ $policy == snapkv ]]; then
    SEER_SNAPKV_VARIANT=$variant \
      python -m seer.eval.runner \
        --model "$MODEL" --policy snapkv --workload "$WORKLOAD" \
        --context_length "$CONTEXT" --num_requests "$NUM" \
        --slo "$SLO" --hbm_budget "$BUDGET" \
        --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
        --out "$out"
  elif [[ $policy == quest ]]; then
    SEER_QUEST_VARIANT=$variant \
      python -m seer.eval.runner \
        --model "$MODEL" --policy quest --workload "$WORKLOAD" \
        --context_length "$CONTEXT" --num_requests "$NUM" \
        --slo "$SLO" --hbm_budget "$BUDGET" \
        --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
        --out "$out"
  else
    echo "[parity] unknown policy $policy"; exit 1
  fi
}

for policy in snapkv quest; do
  for variant in intree upstream; do
    run_cell "$policy" "$variant"
  done
done
echo "[parity] done. Rendering table:"
python "$REPO_ROOT/scripts/gen_baseline_parity_table.py"
