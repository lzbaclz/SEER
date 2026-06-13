#!/usr/bin/env bash
#
# Long-context decode experiment (P1-5, M9 fix).
#
# Reviewer feedback (May 2026): paper title says "long-context", but
# experiments use max_new_tokens=32-64 and context_length=8192 — that's
# decode-heavy, not long-context (≥ 32K prefill, ≥ 1K decode). The eF
# / eA decode-time KV-cache management story doesn't exercise the
# regime the title advertises.
#
# This driver re-runs eA-style policies at 32K-prefill / 1K-decode and
# reports per-step P99/P999/miss alongside the headline 8K/64
# operating point. If 32K is too tight at φ=0.20 (predictor head-room
# matters), the driver will OOM cleanly with a clear "reduce
# context_length" message.
#
# Cost: 32K prefill is ~10× the 8K prefill cost. Plan for ~8 hours
# A100 to cover 5 policies × 2 budgets × 10 prompts.
#
# Outputs: experiments/eA_tail_latency/results_longctx/
#   {policy}_{budget}_ctx{N}.json
#   summary.json (aggregated)
#
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_tinymlp_a100.pt}
OUT_DIR=experiments/eA_tail_latency/results_longctx
mkdir -p "$OUT_DIR"

# 32K prefill + 1024 decode is the marquee long-context regime.
# Fall back to 16K if 32K OOMs on smaller-than-A100 GPUs.
CTX_LENGTHS=(${SEER_CTX_LENGTHS:-32768 16384})
POLICIES=(seer h2o streaming snapkv full)
BUDGETS=(0.10 0.20)
MAX_NEW=${SEER_MAX_NEW:-1024}
NUM_REQ=${SEER_NUM_REQ:-10}

# Long-context workload preference: LongBench qmsum (real long-context
# QA) → falls back to mooncake with synthesised long prompts.
WORKLOAD=${SEER_WORKLOAD:-longbench}
LONGBENCH_TASK=${SEER_LONGBENCH_TASK:-qmsum}

for ctx in "${CTX_LENGTHS[@]}"; do
  for pol in "${POLICIES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      btag=$(echo "$budget" | tr -d '.')
      out="$OUT_DIR/${pol}_b${btag}_ctx${ctx}.json"
      if [ -f "$out" ]; then
        echo "[skip] $out exists"
        continue
      fi
      echo "[run]  pol=$pol budget=$budget ctx=$ctx"
      # io_mode default is measured-dma since P1-1; we set it
      # explicitly to be self-documenting.
      PYTHONPATH=. python -m seer.eval.runner \
        --model "$MODEL" \
        --workload "$WORKLOAD" \
        ${LONGBENCH_TASK:+--longbench_task "$LONGBENCH_TASK"} \
        --num_requests "$NUM_REQ" \
        --max_new_tokens "$MAX_NEW" \
        --policy "$pol" \
        --hbm_budget "$budget" \
        --io_mode measured-dma \
        --slo "P99=50ms" \
        --context_length "$ctx" \
        ${LAP:+--lap "$LAP"} \
        --out "$out" || {
          rc=$?
          if grep -q "CUDA out of memory" "$out" 2>/dev/null; then
            echo "[OOM] $pol@b${btag}@ctx${ctx} — reduce SEER_CTX_LENGTHS"
          fi
          continue
        }
    done
  done
done

# Aggregate (script: build_longctx_table.py — generated below).
PYTHONPATH=. python experiments/eA_tail_latency/aggregate_longctx.py \
  --in_dir "$OUT_DIR" --out_tex "$OUT_DIR/table.tex" \
  --out_json "$OUT_DIR/summary.json" \
  || echo "[warn] aggregator missing; raw JSONs are in $OUT_DIR"
