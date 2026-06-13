#!/usr/bin/env bash
# eA paper-headline configuration: Llama-2-7B + Mooncake-24 + P99=100ms.
#
# This is the exact config the §VI tables come from. It is what
# `make eA` runs (PROD_MODEL=meta-llama/Llama-2-7b-hf,
# EA_WORKLOAD=mooncake, EA_NUM=20, EA_SLO=P99=100ms in the top-level
# Makefile); this standalone script lets reviewers invoke the
# headline config without going through `make`.
#
# For the extended Llama-3-8B + ShareGPT-200 sweep (future work, not
# paper-headline), invoke run.sh directly.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-../../checkpoints/lap_prod_dyn.onnx}
WORKLOAD=${WORKLOAD:-mooncake}
CONTEXT=${CONTEXT_LENGTH:-2048}
SLO=${SLO:-P99=100ms}
NUM=${NUM_REQUESTS:-20}
OUT_DIR=${OUT_DIR:-results_mooncake_full}
POLICIES=${POLICIES:-"full streaming h2o snapkv quest recency seer"}
BUDGETS=${BUDGETS:-"0.10 0.20 0.40 0.80"}

MODEL=$MODEL LAP=$LAP WORKLOAD=$WORKLOAD \
    CONTEXT_LENGTH=$CONTEXT SLO=$SLO NUM_REQUESTS=$NUM \
    OUT_DIR=$OUT_DIR POLICIES="$POLICIES" BUDGETS="$BUDGETS" \
    bash run.sh
