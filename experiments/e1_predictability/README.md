# E1: Predictability Study

## Purpose

Empirically answer the foundational research question of SEER:

> **Can a small neural predictor (< 1% model FLOPs) beat heuristic
> estimators at predicting which KV-blocks will be attended to in the
> next H decode steps?**

This is the **GO / NO-GO gate** for the entire SEER project — we run it
first, and if the answer is no, we pivot to a different paper direction
or submission venue (see `PROJECT_PLAN.md` §11).

## Decision criterion

Let
- `lap_mean_auc` = mean over horizons H ∈ {1, 4, 16, 64} of LAP's test AUC
- `best_baseline_mean_auc` = the same for the best heuristic (persistence / last-attn / recency)

We go to **GO** iff:

```
lap_mean_auc >= 0.80   AND   lap_mean_auc - best_baseline_mean_auc >= 0.05
```

Otherwise **NO-GO**: re-scope (larger LAP, or drop to workshop track).

## Pipeline

```
 RULER / LongBench prompts
          │
          ▼
 seer.trace.collect    ──►  data/traces/<model>/req_*.parquet
          │
          ▼
 seer.lap.train        ──►  checkpoints/lap_<model>.pt
          │
          ▼
 analyze.py            ──►  results/predictability.json
```

## Run

```bash
# Single command that does all three stages
bash run.sh
```

Configurable via env vars:
- `MODEL`          default `meta-llama/Meta-Llama-3-8B-Instruct`
- `TRACE_DIR`      default `../../data/traces/llama3-8b`
- `OUT_DIR`        default `results`
- `NUM_PROMPTS`    default `100`
- `CONTEXT_LENGTHS`  default `"4096 8192"`

## Inspect results

```bash
cat results/predictability.json | python -m json.tool | grep -A 5 '"summary"'
```

## Interpreting AUC per horizon

A common pattern in our pilot runs:
- **h=1** (next-step): easy; AUC > 0.90 for nearly everyone including `last_attn`
- **h=4**: this is where LAP should separate itself from heuristics
- **h=16**: harder; AUC gap should still be ≥ 0.03
- **h=64**: fundamental limit of how far ahead one can see from local signal

The headline number for the GO/NO-GO decision is **mean AUC across the four
horizons**, not any single horizon.
