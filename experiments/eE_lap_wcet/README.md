# eE: LAP WCET on the target hardware

## Purpose

Empirically establish the upper bound on ``C_LAP`` (Lemma 2) on each
target GPU SKU. The schedulability claim is only as strong as the
distribution of measured LAP inference latencies.

A100 + H100 + L40 sweep per Phase 3 (P3.6) and Phase 6 (P6.12). The
canonical deployment path is TensorRT plan + CUDA Graph replay; ONNX
and torch backends are kept as sanity oracles.

## Multi-GPU NVLink note

This repo's reference machine has 2× A100 SXM4 connected via NV12
(12-lane NVLink). The single-card WCET below is what the schedulability
bound consumes for $C_\text{LAP}$. The cross-card NVLink P2P probe
(`nvlink_probe.py`) supplies the empirical $\bar\ell$ for the §3
"NVLink HBM" fourth tier.

## Run

Single SKU, TRT canonical path:

```bash
python -m seer.lap.export --ckpt ../../checkpoints/lap_prod_tiny.pt \
    --backend tensorrt --batch 4096 \
    --out ../../checkpoints/lap_prod_b4096.plan

python -m seer.lap.wcet \
    --plan ../../checkpoints/lap_prod_b4096.plan \
    --backend tensorrt --device cuda \
    --batch 4096 --reps 10000 --warmup 200 \
    --out results/lap_wcet_a100_trt_b4096.json
```

ONNX path (sanity oracle before TensorRT lands):

```bash
python -m seer.lap.wcet \
    --onnx ../../checkpoints/lap_prod_dyn.onnx \
    --backend onnx --device cuda \
    --batch 4096 --reps 10000 \
    --out results/lap_wcet_a100_onnx_b4096.json
```

NVLink P2P probe:

```bash
python nvlink_probe.py --out results/nvlink.json
```

## Output schema

JSON with: ``backend``, ``device``, ``batch_size``, ``feat_dim``,
``n_reps``, ``n_warmup``, ``p50_us``, ``p90_us``, ``p99_us``,
``p999_us``, ``max_us``, ``mean_us``, ``std_us``.

The §6.7 figure is a horizontal bar chart of P99.9 across SKUs and
backends, with the budget line (default 200 µs) overlaid.
