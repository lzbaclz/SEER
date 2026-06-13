#!/usr/bin/env bash
# e1 saturation curve at 48 requests (paper §6.2 reviewer ask).
#
# The published e1 GO verdict is on 12 training requests. The
# 24-request ablation (results_24req/) shifted AUC 0.941→0.944
# (negligible, indicating saturation). A reviewer who is not yet
# convinced will want to see the full saturation curve — this script
# trains LAP at {12, 24, 36, 48} requests on the same fixed
# held-out test set and reports AUC vs n_train.
#
# Wall time on Llama-2-7B fp16 + 1× A100:
#   - Trace collection:  ~25 min total (idempotent; reuses cached parquet).
#   - LAP training:      ~5 min × 4  ≈ 20 min.
#   - Eval:              ~1 min × 4  ≈  4 min.
# Total: ~50 min.
#
#   CUDA_VISIBLE_DEVICES=1 bash experiments/e1_predictability/run_saturation_48req.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
TRACE_DIR=${TRACE_DIR:-data/traces/llama2-7b}
OUT_DIR=${OUT_DIR:-experiments/e1_predictability/results_saturation}
MAX_TRAIN=${MAX_TRAIN:-48}
EPOCHS=${EPOCHS:-6}
ARCH=${ARCH:-tiny_mlp}

mkdir -p "$OUT_DIR" "$TRACE_DIR"

# 1. Collect 48 requests of trace if needed (idempotent).
if [ "$(ls -A "$TRACE_DIR" 2>/dev/null | grep -c parquet)" -lt 48 ]; then
  echo "[e1-sat] collecting traces (target: 48 requests) → $TRACE_DIR"
  python -m seer.trace.collect \
      --model "$MODEL" \
      --dataset ruler \
      --context_lengths 1024 2048 \
      --num_prompts 48 \
      --max_new_tokens 64 \
      --out "$TRACE_DIR"
else
  echo "[e1-sat] traces already in $TRACE_DIR (skipping collect)"
fi

# 2. Train LAP at 12, 24, 36, 48 requests; eval each.
for N in 12 24 36 48; do
    CKPT="$OUT_DIR/lap_${ARCH}_n${N}.pt"
    JSON="$OUT_DIR/predictability_n${N}.json"
    if [ -f "$JSON" ]; then
        echo "[e1-sat] reuse $JSON"; continue
    fi
    echo "[e1-sat] train n=$N"
    python -m seer.lap.train \
        --traces "$TRACE_DIR" \
        --model "$ARCH" \
        --epochs "$EPOCHS" \
        --max_train_requests "$N" \
        --out "$CKPT"
    echo "[e1-sat] eval n=$N"
    python -m experiments.e1_predictability.analyze \
        --traces "$TRACE_DIR" \
        --ckpt "$CKPT" \
        --out "$JSON"
done

echo "[e1-sat] done. Saturation CSV:"
echo "  python -c '"
echo "import json, glob, csv, sys"
echo "rows=[]"
echo "for f in sorted(glob.glob(\"$OUT_DIR/predictability_n*.json\")):"
echo "    d=json.load(open(f)); n=int(f.split(\"_n\")[-1].split(\".\")[0])"
echo "    rows.append({\"n\":n, \"auc\":d.get(\"lap_mean_auc\"), \"gap\":d.get(\"gap_vs_best_baseline\")})"
echo "w=csv.DictWriter(sys.stdout, fieldnames=[\"n\",\"auc\",\"gap\"]); w.writeheader(); w.writerows(rows)'"
echo "    > $OUT_DIR/saturation.csv"
