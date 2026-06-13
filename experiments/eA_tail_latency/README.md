# eA: Tail Latency under SLO — primary main figure

## Purpose

The headline experiment of the RTSS 2026 submission. Fix a per-token
TPOT SLO at the relaxed-$D$ band (``P99=200ms`` on the headline
A100/Llama-2-7B rig; ``P99=50ms`` is structurally infeasible on this
hardware per §I two-regime split since $P_{50}\approx 110$\,ms),
run a fixed workload, and report **P50 / P99 / P999 TPOT,
deadline-miss ratio, and quality (F1)** for every policy.

Two workloads are now populated:

* **Headline (Mooncake-24)**: synthetic Mooncake mirror, $n=24$
  prompts, **6 in-tree policies** at 4 budgets — this is the §VI.A
  paired-headline.
* **Cross-workload sanity audit (ShareGPT-200)**: $n=80$ prompts/cell,
  **7 policies × 4 budgets = 25 cells** + a $\phi{=}0.05$ tighter-budget
  probe (R36 streaming_b005). Demoted from "paired-headline" to
  "cross-workload sanity audit" in R34 because chat-miss saturates
  across all policies in $[0.0004, 0.0007]$; the deployable-policy
  differentiation axis on ShareGPT is F1 quality, not chat-miss or
  $P_{99}$ TPOT. See `paper/sections/A2_setup.tex`.

The full 7-policy zoo is: `full streaming h2o snapkv quest recency
seer`. R32 added the `SEER_H2O_VARIANT={upstream,intree}` env-var
dispatch so the H2O parity audit (`run_h2o_upstream_parity.sh`)
can run zhang2023h2o's cumulative-from-step-0 spec against the
in-tree rolling-window `H2OPolicy`.

The matrix may expand to {H100/TP-2 chat-50ms, 13B/70B, multi-tenant
serving} in a cycle-2 ATC follow-up (see `todo_atc.md` A1–D);
RTSS 2026 stays at A100/Llama-2-7B/relaxed-$D$.

## Inputs

* `bench_vllm.py` — black-box concurrent OpenAI-API client used to
  produce the *baseline vLLM* numbers. **Does not** drive a SEER policy
  — it talks to whatever vLLM endpoint is at ``--base_url``.
* `make_trace.py` — generates a JSONL prompt trace (synthetic or
  ShareGPT) consumed by ``bench_vllm.py``.
* `serve_vllm.sh` — convenience launcher for a vanilla vLLM server.
* `run.sh` — drives :mod:`seer.eval.runner` for the in-process,
  policy-driven sim. Used by both `run_mooncake_headline.sh` (Mooncake-24)
  and `run_sharegpt200_headline.sh` (ShareGPT cross-workload).

## Pipeline

```
 Mooncake-24 / ShareGPT-200 / synthetic prompts
          │
          ▼
   make_trace.py → trace.jsonl
          │
   ┌──────┴──────┐
   │             │
   ▼             ▼
black-box     in-process
serving       sim (runner.py)
   │             │
   └──────┬──────┘
          ▼
   results_{mooncake_full,sharegpt200,h2o_upstream_parity}/*.json
          │
          ▼
       scripts/gen_paper_numbers.py → paper/numbers.tex
       scripts/gen_sharegpt200_table.py → paper/sharegpt200_headline.tex
       scripts/gen_h2o_upstream_parity_table.py → paper/h2o_upstream_parity.tex
```

## Runbook

Headline Mooncake-24 (one-shot from cache, ~30 min on A100):

```bash
cd experiments/eA_tail_latency
bash run_mooncake_headline.sh        # writes results_mooncake_full/*.json
```

Cross-workload sanity audit (ShareGPT-200, ~2.5h on A100 at NUM=80):

```bash
NUM_REQUESTS=80 POLICIES="full streaming h2o snapkv quest recency seer" \
    bash run_sharegpt200_headline.sh
python ../../scripts/gen_sharegpt200_table.py
```

Upstream H2O parity (~30 min):

```bash
SEER_H2O_VARIANT=upstream SEER_H2O_HEAVY_RATIO=0.5 SEER_H2O_RECENT_RATIO=0.5 \
    bash run_h2o_upstream_parity.sh
python ../../scripts/gen_h2o_upstream_parity_table.py
```

## Outputs

* `results_mooncake_full/<policy>_b<budget>.json` — per-policy headline
* `results_sharegpt200/<policy>_b<budget>.json` — 25 cross-workload cells
* `results_h2o_upstream_parity/{upstream,intree}/h2o_b<NN>.json` — 8 cells
* `paper/figures/eA_cdf.pdf` — tail-latency CDF figure
* `paper/figures/a7_step_shape.pdf` — R25+R35 pooled A7 step figure

## Status

* `bench_vllm.py` / `make_trace.py` / `serve_vllm.sh` migrated from
  the legacy `e3_throughput` directory unmodified.
* `run.sh` is RTSS-pivot specific (uses `--workload`, `--slo`,
  `--metrics` flags from the new runner).
* `run_mooncake_headline.sh` / `run_sharegpt200_headline.sh` /
  `run_h2o_upstream_parity.sh` are the per-paper-section drivers.
* Known issue (R36): at $\phi=0.05$, h2o/snapkv/quest/recency/seer
  cells hang in model-load under the runner; streaming_b005 alone
  populated. Queued for follow-up; the saturation claim does not
  require the remaining cells.
