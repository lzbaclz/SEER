#!/usr/bin/env bash
# R31 advisor P0-2: ShareGPT-200 headline sweep (paired with the
# Mooncake-24 headline for the cross-workload-generalisation column).
#
# Configuration:
#   - Model:         Llama-2-7B-hf (headline; same as run_mooncake_headline.sh)
#   - Workload:      ShareGPT, 200 prompts
#   - Context:       4096 (ShareGPT tends to land short; ctx covers
#                    the 95%-ile prompt + 128-token expected response)
#   - SLO:           P99=200ms (relaxed-D band per §I; the headline
#                    rig's P50 TPOT ≈ 110ms makes <=100ms infeasible)
#   - Budgets:       0.10 / 0.20 / 0.40 / 0.80
#   - Policies:      full / streaming / h2o / snapkv / quest / recency / seer
#
# Total wall time: ~5h on a quiet A100 (200 prompts x 7 policies x
# 4 budgets x ~30 decode steps; LAP forward at 33.8us/step is
# amortised away).
#
# Output:
#   experiments/eA_tail_latency/results_sharegpt200/{policy}_b{NN}.json
# which scripts/gen_paper_numbers.py + scripts/gen_sharegpt200_table.py
# consume to render the §VI.A-bis paired-headline table.
#
# Reviewer asked: "Mooncake-24 is too small; ShareGPT-200 is the
# concurrent-work standard scale." This shim closes that gap without
# touching the Mooncake-24 §VI.A headline numbers.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-../../checkpoints/lap_prod_dyn.onnx}
WORKLOAD=${WORKLOAD:-sharegpt}
CONTEXT=${CONTEXT_LENGTH:-4096}
SLO=${SLO:-P99=200ms}
NUM=${NUM_REQUESTS:-200}
OUT_DIR=${OUT_DIR:-results_sharegpt200}
POLICIES=${POLICIES:-"full streaming h2o snapkv quest recency seer"}
BUDGETS=${BUDGETS:-"0.10 0.20 0.40 0.80"}

MODEL=$MODEL LAP=$LAP WORKLOAD=$WORKLOAD \
    CONTEXT_LENGTH=$CONTEXT SLO=$SLO NUM_REQUESTS=$NUM \
    OUT_DIR=$OUT_DIR POLICIES="$POLICIES" BUDGETS="$BUDGETS" \
    bash run.sh
