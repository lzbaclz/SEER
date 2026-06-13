# SEER Installation

This guide installs the SEER Python package and the tools needed to run tests,
CPU-only analyses, and optional GPU/vLLM measurements.

## Requirements

| Component | Requirement | Notes |
| --- | --- | --- |
| Python | 3.11+ | Tested with Python 3.11/3.12 style environments. |
| OS | Linux recommended | CUDA and vLLM paths require Linux. |
| GPU | Optional NVIDIA GPU | Needed only for TensorRT, vLLM, and substrate measurements. |
| CUDA | 12.x runtime | Usually supplied by PyTorch wheels. |
| Disk | Varies | Model weights, traces, and checkpoints are not stored in Git. |

## Virtualenv Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Install development tools:

```bash
pip install 'pytest>=8.0' 'ruff>=0.4'
```

Run checks:

```bash
pytest tests/ -q
make verify
```

## Conda Install

The repository includes a setup script for the original conda workflow:

```bash
bash scripts/setup_env.sh
conda activate seer
pip install -e .
```

If you already have an environment, you can install from the exported file:

```bash
conda env create -f environment.yml
conda activate seer
pip install -e .
```

## Optional TensorRT / NVIDIA Dependencies

TensorRT paths require the extra NVIDIA package index:

```bash
pip install --extra-index-url https://pypi.nvidia.com -r requirements-trt.txt
```

If GPU imports fail, first check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Common Commands

```bash
make install        # runtime dependencies + editable package
make dev            # install dev tooling
make test           # pytest
make lint           # ruff check
make format         # ruff format
make verify         # tests + lint + CPU schedulability sanity
```

## Data And Outputs

The following are intentionally local-only and ignored by Git:

- `data/`
- `checkpoints/`
- `logs/`
- `results/`
- generated experiment plots and tables
- local drafts and writeups

Create those directories as needed when running experiments. Do not commit
model weights, private traces, local PDFs, or generated measurement dumps to
the code repository.
