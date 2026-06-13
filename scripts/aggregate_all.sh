#!/usr/bin/env bash
# Aggregate every experiment's analyze.py and report what's there.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=$(pwd)
echo "=== Aggregating all experiments ==="

run() {
    echo
    echo "--- $1 ---"
    if [ -d "$2" ] && ls "$2"/*.json >/dev/null 2>&1; then
        eval "$3"
    else
        echo "  (no results yet at $2)"
    fi
}

run "e1 predictability" \
    "experiments/e1_predictability/results" \
    "ls experiments/e1_predictability/results/*.json"

run "eA tail latency (Llama-2-7B)" \
    "experiments/eA_tail_latency/results_llama2" \
    "python experiments/eA_tail_latency/analyze.py experiments/eA_tail_latency/results_llama2 2>&1 | head -40"

run "eA tail latency (Qwen2.5-7B)" \
    "experiments/eA_tail_latency/results_qwen" \
    "python experiments/eA_tail_latency/analyze.py experiments/eA_tail_latency/results_qwen 2>&1 | head -20"

run "eB SLO sweep" \
    "experiments/eB_slo_sweep/results" \
    "python experiments/eB_slo_sweep/analyze.py experiments/eB_slo_sweep/results --with_bound --no_plot 2>&1 | head -30"

run "eC bound tightness" \
    "experiments/eC_bound_tightness/results" \
    "python experiments/eC_bound_tightness/analyze.py \
        --grid experiments/eC_bound_tightness/results/grid.csv \
        --measured experiments/eA_tail_latency/results_llama2/seer_*.json experiments/eB_slo_sweep/results/seer_*.json \
        --eps_band 0.05 0.30 \
        --out_dir experiments/eC_bound_tightness/results 2>&1 | head -10"

run "eD adversarial" \
    "experiments/eD_adversarial/results" \
    "python experiments/eD_adversarial/analyze.py experiments/eD_adversarial/results --no_plot 2>&1 | head -20"

run "eE LAP WCET" \
    "experiments/eE_lap_wcet/results" \
    "python experiments/eE_lap_wcet/analyze.py experiments/eE_lap_wcet/results --no_plot 2>&1 | head -15"

run "eF mixed-SLO" \
    "experiments/eF_mixed_slo/results" \
    "python experiments/eF_mixed_slo/analyze.py experiments/eF_mixed_slo/results --no_plot 2>&1 | head -10"

echo
echo "=== file inventory ==="
find experiments -type f \( -name "*.json" -o -name "*.csv" -o -name "*.tex" -o -name "*.pdf" \) | sort
