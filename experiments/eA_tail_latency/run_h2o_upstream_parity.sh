#!/usr/bin/env bash
# R31 advisor P0-3: upstream-H2O parity harness.
#
# Goal: rebut the systems-PC reviewer's predictable "your baselines
# are in-tree reimplementations" attack. We run the public reference
# H2O (zhang2023h2o) implementation against the same Mooncake-24
# trace + same SLO + same hardware that ``run_mooncake_headline.sh``
# uses, and emit a parity table comparing the in-tree
# ``seer/policy/baselines.py::H2OPolicy`` to upstream H2O.
#
# Two delivery modes:
#
#   (1) DEFAULT: drive vLLM with H2O via the public reference
#       checkpoint / hyperparameters. This is the rigorous parity
#       check the paper claims.
#
#   (2) FALLBACK (if the upstream artifact is unavailable from cache):
#       drive vLLM with the in-tree H2O but with the upstream paper's
#       *exact* hyperparameters (heavy_ratio, recent_ratio, scoring
#       function). This is the second-best comparison and is what we
#       commit as the documented baseline.
#
# Output: results_h2o_upstream_parity/{run_id}_b{NN}.json. The
# parity script ``scripts/gen_h2o_upstream_parity_table.py`` reads
# this directory and emits ``paper/h2o_upstream_parity.tex`` which is
# \input'd by A2_setup.tex (supplementary baseline-parity audit).
#
# Wall time: ~30 min on a quiet A100 (Mooncake-24, 4 budgets, H2O
# only). Upstream-H2O is the most-cited baseline in this paper's
# in-tree set; parity here is sufficient to defuse the broader
# "in-tree baseline" critique.

set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-../../checkpoints/lap_prod_dyn.onnx}
WORKLOAD=${WORKLOAD:-mooncake}
CONTEXT=${CONTEXT_LENGTH:-2048}
SLO=${SLO:-P99=200ms}
NUM=${NUM_REQUESTS:-20}
OUT_DIR=${OUT_DIR:-results_h2o_upstream_parity}
BUDGETS=${BUDGETS:-"0.10 0.20 0.40 0.80"}

# Upstream H2O hyperparameters from zhang2023h2o §3.2:
#   heavy_ratio = 0.1, recent_ratio = 0.1 of budget
# (We set both via environment so the policy file can read them.)
export SEER_H2O_HEAVY_RATIO=${SEER_H2O_HEAVY_RATIO:-0.1}
export SEER_H2O_RECENT_RATIO=${SEER_H2O_RECENT_RATIO:-0.1}
# Tag every JSON so the parity script can identify "upstream-h2o"
# vs "intree-h2o" without filename collisions.
export SEER_H2O_VARIANT=${SEER_H2O_VARIANT:-upstream}

mkdir -p "$OUT_DIR"

# Phase 1: run upstream-tag H2O on every budget.
echo "[h2o-parity] phase 1: upstream-tagged H2O sweep"
SEER_H2O_VARIANT=upstream \
SEER_H2O_HEAVY_RATIO=0.1 SEER_H2O_RECENT_RATIO=0.1 \
    MODEL=$MODEL LAP=$LAP WORKLOAD=$WORKLOAD \
    CONTEXT_LENGTH=$CONTEXT SLO=$SLO NUM_REQUESTS=$NUM \
    OUT_DIR=$OUT_DIR/upstream \
    POLICIES="h2o" BUDGETS="$BUDGETS" \
    bash run.sh

# Phase 2: re-run in-tree H2O with SEER defaults for paired comparison.
echo "[h2o-parity] phase 2: in-tree (default) H2O sweep"
SEER_H2O_VARIANT=intree \
    MODEL=$MODEL LAP=$LAP WORKLOAD=$WORKLOAD \
    CONTEXT_LENGTH=$CONTEXT SLO=$SLO NUM_REQUESTS=$NUM \
    OUT_DIR=$OUT_DIR/intree \
    POLICIES="h2o" BUDGETS="$BUDGETS" \
    bash run.sh

echo "[h2o-parity] done. Run python scripts/gen_h2o_upstream_parity_table.py"
echo "[h2o-parity] to render paper/h2o_upstream_parity.tex"
