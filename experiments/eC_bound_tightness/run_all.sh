#!/usr/bin/env bash
# R28 advisor: single entry point for the eC bound-tightness scripts.
#
# eC has accumulated ~35 standalone scripts; this driver groups them
# into three tiers so a reviewer / new contributor has a single
# command to run instead of inferring the right ordering from
# Makefile's reproduce-smoke-cpu.
#
# Usage:
#   bash run_all.sh smoke        # ~60s CPU-only analytical pipeline
#   bash run_all.sh substrate    # GPU substrate / A7 microbenches
#   bash run_all.sh paper        # smoke + raw-trace audit + admission
#                                # map figure + claim-evidence table
#                                # (paper-headline subset)
#
# The substrate tier requires an A100 (or compatible CUDA-12 GPU);
# scripts that need the MPS daemon print a guidance message if it
# is not detected. Every harness refuses to overwrite an existing
# measured artifact (status=MEASUREMENT_COMPLETE) -- pass
# --force-stub-overwrite to clobber.
set -euo pipefail
TIER=${1:-smoke}
cd "$(dirname "$0")/../.."

case "$TIER" in
  smoke)
    # Pure-CPU analytical pipeline (matches `make reproduce-smoke-cpu`
    # minus the few non-eC steps).
    python -m experiments.eC_bound_tightness.q_base_holdout \
        --iofree_dir experiments/eA_tail_latency/results_iofree \
        --bootstrap_n 1000
    python experiments/eC_bound_tightness/a1_correlation_check.py
    python -m experiments.eC_bound_tightness.dense_deployed_eps
    python -m experiments.eC_bound_tightness.sizing_deployed_eps
    python -m experiments.eC_bound_tightness.sizing_predictor_compare
    python -m experiments.eC_bound_tightness.random_task_sets
    python -m experiments.eC_bound_tightness.operator_value
    python -m experiments.eC_bound_tightness.sizing_design_ablation
    python -m experiments.eC_bound_tightness.admission_map
    python -m experiments.eC_bound_tightness.admission_map_figure
    python -m experiments.eC_bound_tightness.measured_substrate
    python -m experiments.eC_bound_tightness.interference_aware_bound
    python -m experiments.eC_bound_tightness.held_out_replay
    python -m experiments.eC_bound_tightness.raw_trace_replay
    python -m experiments.eC_bound_tightness.raw_trace_replay \
        --io_mode recorded --out_suffix _recorded
    python -m experiments.eC_bound_tightness.raw_trace_io_mode_compare
    python -m experiments.eC_bound_tightness.raw_trace_replay \
        --B_t 51 --out_suffix _b51
    python -m experiments.eC_bound_tightness.bound_variant_compare
    python -m experiments.eC_bound_tightness.full_tail_guard_envelope
    python -m experiments.eC_bound_tightness.contractual_qstar
    python -m experiments.eC_bound_tightness.per_step_envelope
    python -m experiments.eC_bound_tightness.q_base_refit_llama2prod40
    python -m experiments.eC_bound_tightness.q_base_refit_sharegpt \
        --num_prompts 200 --workload sharegpt
    echo "[run_all.sh] smoke tier done"
    ;;
  substrate)
    python -m experiments.eC_bound_tightness.substrate_pcie_nvme --cuda
    python -m experiments.eC_bound_tightness.substrate_a7_contended --cuda
    python -m experiments.eC_bound_tightness.substrate_a7_moderate --cuda
    python -m experiments.eC_bound_tightness.a7_probe_substrate_vs_integrated
    # R26 heavy (1 MiB contender) and R27 cross-contention (0.016 MiB
    # at the R25 sharp-transition size). Matches `make reproduce-measurement`.
    python -m experiments.eC_bound_tightness.substrate_a7_mps_isolated \
        --cuda --contender-mb 1.0 --out-stem substrate_a7_mps_isolated
    python -m experiments.eC_bound_tightness.substrate_a7_mps_isolated \
        --cuda --contender-mb 0.016 --block-sizes-kb 4 16 \
        --out-stem substrate_a7_mps_cross_0p016
    echo "[run_all.sh] substrate tier done"
    ;;
  paper)
    bash "$(dirname "$0")/run_all.sh" smoke
    python scripts/gen_claim_evidence_table.py
    python scripts/build_main_figure.py
    echo "[run_all.sh] paper tier done"
    ;;
  *)
    echo "usage: bash run_all.sh {smoke|substrate|paper}"
    exit 1
    ;;
esac
