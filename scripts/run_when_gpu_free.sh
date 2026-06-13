#!/usr/bin/env bash
# Polls GPU 1 utilisation; when quiet (util<10%, mem>60GB free, only this
# script's own process or nothing on the card) for 3 consecutive 30-second
# samples, kicks off the work3.md §1 queued work in this order:
#
#   G   - Card-1 unlocked WCET baseline (no sudo). Saves to a separate
#         file `lap_wcet_A100_card1_unlocked_<date>.json` so the
#         pre-existing `lap_wcet_A100_card1_*` row stays untouched.
#         If the unlocked-quiet number is within 5% of Card 0 (P50
#         ≈ 26-28 µs), §6.7 Cross-card Asymmetry can be downgraded
#         from "we suspect contention" to "confirmed contention,
#         resolved by quiescing the card". Otherwise the gap is
#         hardware/firmware and we punt to RTSS-2028 with a more
#         specific ask.
#   E   - `make eF-multimix` (3 mix ratios; auto-renders 4x3 table).
#   H   - 24-request e1 retrain into separate checkpoint paths so the
#         12-request production checkpoint stays in place.
#   I.1 - eA re-run with SEER_MOONCAKE_SYNTHESISER=squad into the
#         results_mooncake_squad/ directory. Does NOT overwrite
#         results_mooncake_full/ (the §6.3 headline numbers).
#
# Authorised by user 2026-05-09. Logs to /tmp/seer_gpu_queue.log; touches
# /tmp/seer_gpu_queue.<step>.done after each step. Aborts the queue on
# the first failure so subsequent steps don't pile up bad data.
#
# Override polling defaults via:
#   QUIET_UTIL_PCT=10 QUIET_FREE_MIB=60000 QUIET_STREAK=3 SLEEP_S=30 ...
set -uo pipefail

cd /home/lzq/codes/SEER

LOG=/tmp/seer_gpu_queue.log
exec >> "$LOG" 2>&1

QUIET_UTIL_PCT=${QUIET_UTIL_PCT:-10}
QUIET_FREE_MIB=${QUIET_FREE_MIB:-60000}
QUIET_STREAK=${QUIET_STREAK:-3}
SLEEP_S=${SLEEP_S:-30}
GPU_INDEX=${GPU_INDEX:-1}

source ~/miniconda3/bin/activate seer

# HuggingFace HEAD-probe goes through a 503-flaky proxy on this host;
# all required models (Llama-2-7b-hf, Qwen2.5-7B, Mistral-7B-v0.1) are
# already in ~/.cache/huggingface/hub/, so disable the network probe.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

quiet_streak=0
print_state() {
    local why=$1
    util=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    free=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    nproc=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | wc -l)
    echo "$(date +%H:%M:%S) gpu${GPU_INDEX} util=${util}% free=${free}MiB nproc=${nproc} streak=${quiet_streak} (${why})"
}

wait_for_quiet() {
    quiet_streak=0
    while true; do
        util=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        free=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        nproc=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | wc -l)
        if [[ "$util" -lt "$QUIET_UTIL_PCT" && "$free" -gt "$QUIET_FREE_MIB" && "$nproc" -eq 0 ]]; then
            quiet_streak=$((quiet_streak + 1))
            print_state "quiet"
            if [[ "$quiet_streak" -ge "$QUIET_STREAK" ]]; then
                echo "$(date) GPU ${GPU_INDEX} confirmed quiet -- proceeding"
                return 0
            fi
        else
            quiet_streak=0
            print_state "busy"
        fi
        sleep "$SLEEP_S"
    done
}

run_step() {
    local name=$1
    shift
    local marker=/tmp/seer_gpu_queue.${name}.done
    if [[ -f "$marker" ]]; then
        echo "[$(date)] $name already done (marker exists), skipping"
        return 0
    fi
    echo "===== [$(date)] $name START ====="
    wait_for_quiet
    echo "===== [$(date)] $name LAUNCH: $* ====="
    if "$@"; then
        touch "$marker"
        echo "===== [$(date)] $name DONE ====="
    else
        echo "===== [$(date)] $name FAILED -- aborting queue ====="
        return 1
    fi
}

echo ""
echo "===== [$(date)] seer GPU queue start (poll gpu${GPU_INDEX}) ====="

# G — Card-1 unlocked WCET baseline (no sudo, no clock lock)
run_step G bash -c '
  CUDA_VISIBLE_DEVICES=1 GPU_TAG=A100_card1_unlocked_$(date +%Y%m%d) \
      PLAN=checkpoints/lap_prod_b4096.plan BATCH=4096 REPS=3000 \
      bash scripts/smoke_tensorrt.sh
' || exit 1

# E — eF multimix (3 chat:doc ratios). Override MODEL to Llama-2-7b-hf
# so the new chat3:doc5 / chat8:doc0 rows match the existing chat6:doc2
# row's model in the committed JSONs.
run_step E bash -c '
  for ratio_chat in 3 6 8; do
    case "$ratio_chat" in
      3) ratio_doc=5 ;;
      6) ratio_doc=2 ;;
      8) ratio_doc=0 ;;
    esac
    CUDA_VISIBLE_DEVICES=1 \
        MODEL=meta-llama/Llama-2-7b-hf \
        LAP="$(pwd)/checkpoints/lap_prod_dyn.onnx" \
        OUT_DIR="$(pwd)/experiments/eF_mixed_slo/results" \
        N_CHAT="$ratio_chat" N_DOC="$ratio_doc" \
        bash experiments/eF_mixed_slo/run.sh
  done
  python experiments/eF_mixed_slo/analyze.py \
      experiments/eF_mixed_slo/results
  cp experiments/eF_mixed_slo/results/miss_matrix.pdf \
     paper/figures/eF_miss_matrix.pdf
' || exit 1

# H — 24-request e1 retrain (separate paths from production 12-req ckpts)
run_step H bash -c '
  CUDA_VISIBLE_DEVICES=1 make collect-trace \
      PROD_NUM_REQS=24 PROD_TRACE_DIR=data/traces/llama2_train_24 &&
  CUDA_VISIBLE_DEVICES=1 make train-lap \
      PROD_TRACE_DIR=data/traces/llama2_train_24 \
      PROD_LAP_PT=checkpoints/lap_prod_tiny_24.pt &&
  mkdir -p experiments/e1_predictability/results_24req
  python experiments/e1_predictability/analyze.py \
      --ckpt checkpoints/lap_prod_tiny_24.pt \
      --traces data/traces/llama2_train_24 \
      --out experiments/e1_predictability/results_24req/predictability_24req.json
' || exit 1

# I.1 — eA SQuAD re-run, separate output dir to preserve headline numbers
run_step I1 bash -c '
  make squad-cache &&
  CUDA_VISIBLE_DEVICES=1 SEER_MOONCAKE_SYNTHESISER=squad \
      MODEL=meta-llama/Llama-2-7b-hf WORKLOAD=mooncake \
      CONTEXT_LENGTH=2048 NUM_REQUESTS=20 SLO=P99=100ms \
      BUDGETS="0.10 0.20 0.40 0.80" \
      LAP="$(pwd)/checkpoints/lap_prod_dyn.onnx" \
      OUT_DIR="$(pwd)/experiments/eA_tail_latency/results_mooncake_squad" \
      bash experiments/eA_tail_latency/run.sh &&
  python experiments/eA_tail_latency/analyze.py \
      experiments/eA_tail_latency/results_mooncake_squad \
      --plot_budget 0.20 --slo_ms 50
' || exit 1

# I.1.fill -- supplementary fill-in for eF (streaming + full at the
# 2 new mix ratios). E's default policies were "h2o seer" only, so the
# table is asymmetric (4 policies at chat6:doc2, only 2 at the new
# ratios). Four 25-second runs make it 4x3.
run_step I1_fill bash -c '
  for ratio in "3 5" "8 0"; do
    set -- $ratio
    n_chat=$1; n_doc=$2
    for policy in full streaming; do
      echo "=== [I.1.fill] $policy chat${n_chat}:doc${n_doc} ==="
      CUDA_VISIBLE_DEVICES=1 \
          MODEL=meta-llama/Llama-2-7b-hf \
          LAP="$(pwd)/checkpoints/lap_prod_dyn.onnx" \
          OUT_DIR="$(pwd)/experiments/eF_mixed_slo/results" \
          N_CHAT="$n_chat" N_DOC="$n_doc" \
          POLICIES="$policy" \
          bash experiments/eF_mixed_slo/run.sh
    done
  done
  python experiments/eF_mixed_slo/analyze.py \
      experiments/eF_mixed_slo/results
  cp experiments/eF_mixed_slo/results/miss_matrix.pdf \
     paper/figures/eF_miss_matrix.pdf
' || exit 1

echo ""
echo "===== [$(date)] seer GPU queue COMPLETE ====="
touch /tmp/seer_gpu_queue.ALL.done
