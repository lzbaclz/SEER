# SEER

SEER (**S**chedulable **E**viction via **E**xpected **R**etrieval) is a
Python research prototype for schedulable KV-cache management in
long-context LLM decoding.

This repository is scoped as a code repository. Local drafts, writing notes,
large generated outputs, and private measurement artifacts are kept out of
version control.

## What Is Included

- `seer/`: core Python package for policies, timing analysis, trace loading,
  LAP models, and vLLM integration hooks.
- `experiments/`: experiment drivers, analysis scripts, and reproduction
  harnesses.
- `scripts/`: utility scripts for setup, measurement, aggregation, and
  generated outputs.
- `tests/`: unit and contract tests for the package and experiment helpers.
- `docs/`: operational notes and runbooks for selected measurement paths.
- `cloud/`: cloud/HPC helper scripts for GPU measurements.

Large local datasets, checkpoints, logs, generated plots, and local writeups
are intentionally ignored.

## Installation

Python 3.11+ is required. A CUDA-capable Linux host is needed for the GPU and
vLLM paths; CPU-only tests and analytical checks can run on a normal
development machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For TensorRT/CUDA-specific paths, install the extra NVIDIA packages:

```bash
pip install --extra-index-url https://pypi.nvidia.com -r requirements-trt.txt
```

The conda workflow remains available:

```bash
bash scripts/setup_env.sh
conda activate seer
```

## Basic Checks

Run the unit tests:

```bash
pytest tests/ -q
```

Run the repository verification target:

```bash
make verify
```

`make verify` performs the code-facing checks: tests, ruff linting, and a
CPU-only schedulability CLI sanity check.

## Useful Commands

```bash
make install        # install runtime deps and editable package
make dev            # install dev tooling
make test           # run pytest
make lint           # run ruff
make format         # run ruff format
make clean          # remove Python/build caches
```

Experiment targets are exposed through the Makefile and through scripts under
`experiments/`. GPU-heavy targets assume model weights, traces, and hardware
availability are configured locally.

## Package Overview

- `seer.policy`: KV-cache selection policies and baselines.
- `seer.timing`: SLO parsing, schedulability bounds, sigma estimation, and
  substrate measurement helpers.
- `seer.trace`: trace schema, dataset loading, and collection helpers.
- `seer.lap`: learned attention predictor models, features, training, export,
  and inference helpers.
- `seer.eval`: simulation and metrics utilities.
- `seer.integration`: vLLM connector and hot-path hook prototypes.

## Notes On Generated Files

The codebase still contains experiment scripts that can emit tables, plots,
or summaries. Those generated outputs are ignored by default. Keep source
code, tests, and reproducibility harnesses in Git; keep large traces and
measurement outputs out of Git unless they are intentionally published
separately.
