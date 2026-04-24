# E3: Throughput at Iso-Quality

## Purpose

Fix a quality target (e.g. "mean F1 on LongBench ≥ 0.70"); for each policy
find the **minimum** HBM budget it needs to hit that target, then measure
end-to-end throughput (tokens/sec, TTFT, TPOT) at that budget.

## Status

⏳ Skeleton only. True throughput measurement requires vLLM integration
(see `seer.integration.vllm`, TODO). The MVP here records generation
wall-time from HuggingFace `generate()`, which is **not representative**
of serving throughput. Report these numbers only as "simulated wall-time"
in the paper appendix until the vLLM path lands.

## Runbook (placeholder)

```bash
bash run.sh    # calls runner.py with context_length ∈ {8192, 32768, 131072}
python analyze.py results/ --target_f1 0.70 --out figures/e3_throughput.pdf
```
