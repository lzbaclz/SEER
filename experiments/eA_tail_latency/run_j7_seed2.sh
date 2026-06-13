#!/usr/bin/env bash
# Finish the J.7 scale-up: only seed=2 of full/streaming/h2o/seer
# (seeds 0 and 1 are already in results_j7_scaleup/).
#
# Run on the same 1× A100 the rest of J.7 used. Wall time ≈ 30 min.
#
#   CUDA_VISIBLE_DEVICES=1 bash experiments/eA_tail_latency/run_j7_seed2.sh
#
# After it finishes, refresh CIs and the eA narrative:
#
#   PYTHONPATH=. python seer/eval/bootstrap.py \
#       experiments/eA_tail_latency/results_j7_scaleup/*_b020_s*.json \
#       --n_resamples 3000 \
#       --out_csv experiments/eA_tail_latency/results_j7_scaleup/bootstrap_ci.csv
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=2 bash experiments/eA_tail_latency/run_j7_scaleup.sh
