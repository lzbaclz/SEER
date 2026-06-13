# eC: Schedulability bound vs. measured miss ratio

## Purpose

Test how tight Lemma 2's closed-form deadline-miss bound is in practice.
Sweep ``(epsilon, deadline_us, hbm_budget)`` on a grid and, for every
point, compute:

* The analytical bound from :func:`seer.timing.schedulability.lemma2_miss_prob_bound`
* The empirical miss ratio from :mod:`seer.eval.runner`
* Their ratio (the *pessimism factor*)

Then heat-map the pessimism in the ``(epsilon, deadline)`` plane and
overlay the operator-acceptable region.

Status: Active per Phase 6 (P6.8–P6.9). The single-point pessimism
factor produced by `seer.eval.runner` (in eA results) is retained as
a sanity check.

## How to run

```bash
python sweep.py --epsilons 0.05 0.10 0.15 0.20 0.30 \
                --deadlines_ms 25 35 50 75 100 \
                --budgets 0.1 0.2 0.4 0.8 \
                --out results/grid.csv
python analyze.py --grid results/grid.csv \
                  --measured ../eA_tail_latency/results_*/seer_*.json \
                             ../eB_slo_sweep/results/seer_*.json \
                  --eps_band 0.05 0.10 \
                  --out_dir results
```
