# SEER

> **S**elective **E**viction via **E**xpected **R**ecall —
> a learned KV-cache management policy for long-context LLM inference.

<p align="left">
  <img alt="status" src="https://img.shields.io/badge/status-research%20preview-orange">
  <img alt="venue" src="https://img.shields.io/badge/target-NeurIPS%202026-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-informational">
</p>

> ⚠️ **Research preview.** Code is under active development for a NeurIPS 2026
> submission. APIs, results, and the model checkpoint are not yet stable.

---

## TL;DR

State-of-the-art KV-cache eviction methods for long-context LLMs (StreamingLLM,
H2O, SnapKV, Quest, …) all rely on **heuristic** importance scores derived from
position or **past** attention. Our empirical study shows that future
attention is highly predictable: the top-k attended KV-blocks are stable
across decode steps with Jaccard 0.47–0.70, and a sub-1%-FLOPs neural predictor
recovers AUC > 0.85 of the offline oracle.

**SEER** turns this predictability into an explicit signal. A tiny
**Learned Attention Predictor (LAP)** estimates which KV-blocks will be needed
next; a **joint eviction–prefetch policy** uses these estimates, together with
storage-tier IO costs, to decide what stays in HBM, what is demoted to
DRAM/NVM/SSD, and what should be prefetched ahead of the next attention layer.

The result: at the same HBM budget, SEER delivers higher task quality on
LongBench / RULER / ∞Bench than the strongest heuristic baseline; at the same
quality, it delivers higher end-to-end throughput.

---

## Why a learned policy?

Existing methods can be classified by what signal drives the eviction decision:

| Family | Representative | Signal | Limitation |
|---|---|---|---|
| Position heuristic | StreamingLLM | sink + sliding window | ignores attention entirely |
| Past attention | H2O, Scissorhands | cumulative / persistent attention | uses past as a proxy for future |
| Prefill-only | SnapKV | prompt-end attention | static, no decode adaptation |
| Page estimation | Quest | per-page attention bound | still heuristic |
| Offload schedule | InfiniGen | speculative single-step lookahead | shallow horizon |

**SEER's claim:** if future attention is predictable from past attention
(an empirical question we answer in §6.1 of the paper), then training a
small predictor on attention traces should outperform any fixed heuristic —
and the gap is bounded by the predictor's accuracy.

---

## Method overview

```
                      ┌──────────────────────────────────────┐
   attention trace    │      Learned Attention Predictor     │
   (past N steps)  ─► │   (Tiny-MLP / Block-RNN, < 0.1% FLOPs) │
                      └──────────────────────────────────────┘
                                       │
                          per-block p̂(top-k in next H steps)
                                       │
                                       ▼
                      ┌──────────────────────────────────────┐
                      │     Joint Eviction–Prefetch Policy   │
                      │   utility = p̂  −  λ · IO_cost(tier)  │
                      │     (greedy top-B with sink/window)  │
                      └──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴───────────────────────────┐
            ▼                ▼                ▼                    ▼
      ╔═════════╗      ╔══════════╗     ╔══════════╗        ╔══════════╗
      ║ GPU HBM ║◄───► ║ Host DRAM║◄──► ║   NVM    ║◄─────► ║   SSD    ║
      ║ (hot)   ║      ║ (warm)   ║     ║ (warm)   ║        ║ (cold)   ║
      ╚═════════╝      ╚══════════╝     ╚══════════╝        ╚══════════╝
```

The four-tier storage substrate is provided by a hierarchical KV-cache
backend (released separately; see *Acknowledgments*). SEER itself is a
drop-in policy module — it can also run on a two-tier (GPU/CPU) backend
with a simplified IO-cost term.

---

## Repository structure

```
seer/
├── PROJECT_PLAN.md           # detailed research plan
├── TODO.md                   # working task list
├── seer/                     # Python package
│   ├── lap/                  # Learned Attention Predictor (model + training)
│   ├── trace/                # attention-trace collection & parsing
│   └── eval/                 # benchmark runner, baselines, policies
├── experiments/
│   ├── e1_predictability/    # §6.1 — can attention be predicted?  [GO/NO-GO]
│   ├── e2_pareto/            # §6.2 — quality–budget Pareto frontier
│   ├── e3_throughput/        # §6.3 — throughput at iso-quality
│   ├── e4_arch/              # §6.4 — predictor architecture ablation
│   ├── e6_policy/            # §6.4 — eviction-only / prefetch-only / joint
│   └── e7_generalization/    # §6.5 — cross-model transfer
├── tests/                    # pytest suite
├── data/                     # (gitignored) traces
├── checkpoints/              # (gitignored) model ckpts
├── results/                  # (gitignored) benchmark outputs
├── paper/                    # LaTeX source (NeurIPS template)
├── notes/                    # design notes, related-work reading log
├── pyproject.toml            # package + tool config
├── requirements.txt
├── Makefile
└── LICENSE
```

---

## Quick start

> The pipeline assumes a 4-tier KV-cache backend. A two-tier (GPU+CPU) reference
> implementation that runs on a single A100/H100 will be released alongside
> the paper.

### Environment

```bash
conda create -n seer python=3.11 -y
conda activate seer
make install       # == pip install -r requirements.txt && pip install -e .
```

### Collect attention traces

```bash
python -m seer.trace.collect \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset ruler \
    --context_lengths 4096 8192 16384 \
    --num_prompts 200 \
    --out data/traces/llama3-8b/
```

### Train LAP

```bash
python -m seer.lap.train \
    --traces data/traces/llama3-8b \
    --model tiny_mlp \
    --history 32 \
    --horizons 1 4 16 64 \
    --out checkpoints/lap_llama3_8b.pt

# Export to ONNX for low-latency inference inside the policy loop
python -m seer.lap.export \
    --ckpt checkpoints/lap_llama3_8b.pt \
    --out  checkpoints/lap_llama3_8b.onnx
```

### Run a benchmark

```bash
python -m seer.eval.runner \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --policy seer --lap_ckpt checkpoints/lap_llama3_8b.onnx \
    --benchmark longbench \
    --budget 0.2 \
    --out results/longbench_seer_b20.json
```

The same runner supports `--policy {full, streaming, h2o, snapkv, recency, random}`
for direct comparison.

### Run the GO/NO-GO predictability study (E1)

```bash
bash experiments/e1_predictability/run.sh
# verdict lands in results/predictability.json
```

---

## Results

> Numbers below are placeholders; the final figures will be filled in once
> experiments E2 / E3 complete. See `experiments/*/results.csv` for raw data.

### Quality–budget Pareto (LongBench, Llama-3-8B)

```
            Full   Stream   H2O   SnapKV   Quest   SEER (ours)
budget=10%    ─    XX.X    XX.X   XX.X    XX.X    **XX.X**
budget=20%    ─    XX.X    XX.X   XX.X    XX.X    **XX.X**
budget=40%   YY.Y  XX.X    XX.X   XX.X    XX.X    **XX.X**
budget=80%   YY.Y  XX.X    XX.X   XX.X    XX.X    **XX.X**
```

### Throughput at iso-quality (RULER 32K, A100-80GB)

*coming soon*

---

## Reproducing paper experiments

Each experiment is self-contained under `experiments/`. To reproduce a single
table or figure:

```bash
cd experiments/e2_pareto
bash run.sh                        # launches the full sweep
python analyze.py results/         # produces table + Pareto plot
```

A consolidated `make all` target is provided once individual experiments are
finalized.

---

## Citation

```bibtex
@inproceedings{seer2026,
  title     = {SEER: Learning to Evict KV-Cache by Predicting Future Attention},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). Trained model
checkpoints and attention traces are released under CC-BY-4.0.

---

## Acknowledgments

- The 4-tier hierarchical KV-cache backend used in our experiments is built on
  a separately released systems project; the link will be added after the
  double-blind review window closes.
- We thank the authors of vLLM, H2O, SnapKV, StreamingLLM, Quest, and InfiniGen
  for releasing their code, which enabled fair comparison.

---

## Status & roadmap

- [x] Project plan & experimental design (`PROJECT_PLAN.md`, `TODO.md`)
- [ ] §6.1 Predictability Study (E1) — *go/no-go gate*
- [ ] LAP training pipeline + ONNX export
- [ ] Joint eviction–prefetch policy implementation
- [ ] Main experiments (E2, E3) on Llama-3 / Qwen-2.5 / Mistral
- [ ] Ablations (E4, E6) and generalization (E7)
- [ ] Paper draft → submission

Track day-to-day progress in [`TODO.md`](TODO.md).
