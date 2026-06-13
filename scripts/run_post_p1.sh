#!/usr/bin/env bash
# Sequential post-P1 batch:
#   1) Aggregate P1 (real-DMA grid).
#   2) F1 + F5 fault-mode characterization (LAP corrupt + NVMe burst).
#   3) eF EDF admission comparison (--mode edf vs --mode fifo).
#   4) Optional: P2 scale-up (1000 prompts × 64 × 3 seeds; ~10 h).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_post_p1.sh [--skip-p2]
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_P2=0
for arg in "$@"; do
    case "$arg" in
        --skip-p2) SKIP_P2=1 ;;
    esac
done

source /home/lzq/miniconda3/etc/profile.d/conda.sh
conda activate seer

step() { printf "\033[1;36m[post-P1] %s\033[0m\n" "$*"; }

# 1. Aggregate P1
step "1/4 aggregate P1 measured-DMA results"
python experiments/eA_tail_latency/aggregate_p1.py || true

# 2. Fault-mode F1 + F5
step "2/4 F1 (LAP corrupt) + F5 (NVMe burst)"
python experiments/eFault_characterization/run_faults.py --faults F1,F5 || true

# 3. eF EDF admission
step "3/4 eF EDF admission (mode=edf and mode=fifo, 3 policies × 2 modes)"
mkdir -p experiments/eF_mixed_slo/results_p4
for POL in seer h2o streaming; do
    for MODE in edf fifo; do
        TAG="${POL}_${MODE}"
        if [[ -f experiments/eF_mixed_slo/results_p4/${TAG}.json ]]; then
            step "  skip $TAG"
            continue
        fi
        LAP_ARG=""
        [[ "$POL" == "seer" ]] && LAP_ARG="--lap checkpoints/lap_prod_tiny.pt"
        step "  $TAG"
        python experiments/eF_mixed_slo/driver_edf.py \
            --model meta-llama/Llama-2-7b-hf \
            --policy "$POL" $LAP_ARG \
            --n_chat 10 --n_doc 2 \
            --chat_slo "P99=50ms" --doc_slo "TTFT-P99=1s" \
            --hbm_budget 0.20 \
            --max_tokens 64 \
            --mode "$MODE" \
            --out "experiments/eF_mixed_slo/results_p4/${TAG}.json" \
            2>&1 | tail -3
    done
done
python experiments/eF_mixed_slo/aggregate_p4.py || true

# 4. Optional: P2 scale-up
if [[ "$SKIP_P2" == "0" ]]; then
    step "4/4 P2 scale-up (1000 prompts × 64 × 3 seeds ≈ 10 h)"
    bash experiments/eA_tail_latency/run_p2_scaleup.sh
else
    step "4/4 P2 skipped (--skip-p2)"
fi

step "all done"
