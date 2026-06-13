# eB: Miss ratio vs. SLO tightness

## Purpose

Sweep the per-token deadline ``D`` and report deadline-miss ratio for
every policy. Produces the §6.4 plot: x = SLO threshold (ms), y =
miss ratio, one curve per policy.

Status: Active per Phase 6 (P6.6–P6.7).

## Pipeline

For each ``D ∈ {25, 35, 50, 75, 100, 200}`` ms:
  for each policy ∈ {full, streaming, h2o, snapkv, quest, recency,
                     random, seer}:
    run :mod:`seer.eval.runner` with ``--slo P99=D ms``

Then plot miss ratio vs D, expecting:
* SEER's curve below all heuristics across the entire D range.
* H2O catches up at large D (loose deadlines).
* StreamingLLM catastrophic at small D.

## Optimization: baseline reuse from eA

Per-step latency for non-SEER baselines is SLO-independent (only the
miss-ratio post-aggregation depends on D). The driver fans out
baseline `_b020.json` files from `experiments/eA_tail_latency/results_llama2/`
across all D values by recomputing miss_ratio from `per_step_us`,
saving 6×5 = 30 GPU runs.

## Run

```bash
bash run.sh
python analyze.py results --with_bound
```
