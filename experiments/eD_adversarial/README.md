# eD: Adversarial — robustness under attention shift

## Purpose

Test Lemma 3 empirically. Construct prompts that exhibit attention
non-stationarity (topic shift, multi-document interleave, persona
shift, instruction injection, mid-CoT topic switch) and measure each
policy's P99 TPOT and miss ratio.

Expected outcome (Lemma 3): heuristic policies degrade roughly
linearly with shift strength σ, while SEER's degradation is bounded by
its predictor error ε.

Status: Active per Phase 6 (P6.10–P6.11). Initial 5×4 = 20 runs landed;
RTSS 2027 expansion adds streaming/snapkv/quest/recency to reach 5×7.

## Run

```bash
python generate_prompts.py --out_dir prompts --n 30
bash run.sh
python analyze.py results
```
