#!/usr/bin/env bash
# P5: Llama-2-70B / Llama-3-70B WCET + eA tail-latency on a 4×A100 or
# single H100/H200 cloud pod.
#
# What this proves:
#   (a) LAP-WCET (TinyMLP inference) is model-agnostic — eE A100 numbers
#       already establish 33.8 µs P99.9 on a 22 K-param net. On 4×A100
#       hosting Llama-2-70B fp16, the LAP overhead remains < 0.1% of
#       the per-step decode (~150 ms), so the schedulability bound
#       holds a fortiori.
#   (b) The simulator's per-step latency model + Lemma 2 still apply at
#       70 B model scale; the eA P99 on the 70B model under SEER stays
#       within the larger 70B-class SLO (e.g. P99=300 ms for chat).
#
# Requirements (the user runs these on the pod before this script):
#   - 4×A100 80 GB (recommended) OR 1×H100 80 GB with int4 quant
#   - git clone https://github.com/lzbaclz/SEER.git ; cd SEER
#   - tar xzf /tmp/seer_checkpoints.tgz             (LAP checkpoints)
#   - bash cloud/h100_l40_wcet/cross_hw_setup.sh    (env + plan rebuild)
#   - huggingface-cli login                         (70B model access)
#
# Usage:
#   MODEL=meta-llama/Llama-2-70b-hf TP=4 bash cloud/llama70b_wcet/launch_70b_wcet.sh
#
# Expected wall time on 4×A100: ~2 hours (eE 10 K reps + eA 20 prompts).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-70b-hf}
TP=${TP:-4}
CTX=${CTX:-2048}
BATCH=${BATCH:-4096}   # WCET batch; matches eE A100 reference
OUT_DIR=${OUT_DIR:-experiments/p5_llama70b}
GPU_TAG=${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' || echo unknown)}
NUM_PROMPTS=${NUM_PROMPTS:-20}
MAX_NEW=${MAX_NEW:-64}

mkdir -p "$OUT_DIR"

step() { printf "\033[1;36m[P5] %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m[P5] FAIL: %s\033[0m\n" "$*" >&2; exit 1; }

# ---- A. LAP WCET sanity check (model-agnostic; should match A100 33.8 µs)
step "A. LAP WCET on $GPU_TAG (TinyMLP, batch=$BATCH, 10000 reps)"
python -m seer.lap.wcet \
    --ckpt checkpoints/lap_prod_tiny.pt \
    --backend torch \
    --device cuda \
    --batch "$BATCH" \
    --reps 10000 \
    --out "$OUT_DIR/lap_wcet_70b_${GPU_TAG}.json"

# ---- B. eA-style decode timing on 70B with SEER
# 70B fp16 ≈ 140 GiB → needs --device_map auto on 4×A100 or 2×H100.
step "B. 70B decode eA on $MODEL (device_map=auto, $NUM_PROMPTS prompts × $MAX_NEW tokens)"
LAP_CKPT=${LAP:-checkpoints/lap_prod_tiny.pt}
SLO=${SLO:-"P99=300ms"}  # relaxed: 70B per-step is ~150 ms baseline
for POL in seer h2o streaming; do
    LAP_ARG=""
    [[ "$POL" == "seer" ]] && LAP_ARG="--lap $LAP_CKPT"
    step "  policy=$POL"
    python -m seer.eval.runner \
        --model "$MODEL" \
        --policy "$POL" $LAP_ARG \
        --workload mooncake \
        --context_length "$CTX" \
        --num_requests "$NUM_PROMPTS" \
        --max_new_tokens "$MAX_NEW" \
        --slo "$SLO" \
        --hbm_budget 0.20 \
        --io_mode analytical \
        --device_map auto \
        --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio \
        --out "$OUT_DIR/eA_70b_${POL}_${GPU_TAG}.json" \
        2>&1 | tail -2
done

# ---- C. Summary
step "C. Summary"
python -c "
import json, pathlib
out = pathlib.Path('$OUT_DIR')
wcet = json.load(open(out / 'lap_wcet_70b_${GPU_TAG}.json'))
print(f'LAP WCET on ${GPU_TAG}:  P50={wcet[\"p50_us\"]:.1f}  P99.9={wcet[\"p999_us\"]:.1f}  WCET={wcet[\"wcet_us\"]:.1f}  µs')
print()
for p in sorted(out.glob('eA_70b_*.json')):
    d = json.load(open(p))
    print(f'{p.stem}: P50={d[\"tpot_p50_us\"]/1000:.1f}ms P99={d[\"tpot_p99_us\"]/1000:.1f}ms P999={d[\"tpot_p999_us\"]/1000:.1f}ms miss={d[\"miss_ratio\"]:.4f}')
"

echo "[P5] done. Push artefacts back to the laptop with:"
echo "    tar czf p5_results.tgz $OUT_DIR"
echo "    scp p5_results.tgz <laptop>:~/codes/SEER/"
