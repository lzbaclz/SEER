.PHONY: help env env-yml install install-trt dev test verify lint format clean

help:
	@echo "Environment"
	@echo "  env          Create the conda env with scripts/setup_env.sh"
	@echo "  env-yml      Refresh environment.yml from conda env 'seer'"
	@echo "  install      Install Python runtime deps + editable package"
	@echo "  install-trt  Install optional TensorRT/NVIDIA deps"
	@echo "  dev          Install runtime deps plus pytest/ruff"
	@echo ""
	@echo "Code checks"
	@echo "  test         Run pytest"
	@echo "  lint         Run ruff"
	@echo "  format       Run ruff format"
	@echo "  verify       Run tests, lint, and CPU schedulability sanity"
	@echo "  clean        Remove local Python/build caches"
	@echo ""
	@echo "Experiments"
	@echo "  Run experiment harnesses directly under experiments/*/run.sh"
	@echo "  Generated results stay local and are ignored by Git."

env:
	bash scripts/setup_env.sh

env-yml:
	conda env export -n seer --no-builds | sed '/^prefix:/d' > environment.yml
	@echo "[env-yml] refreshed environment.yml ($$(wc -l < environment.yml) lines)"

install:
	pip install -r requirements.txt
	pip install -e .

install-trt:
	pip install --extra-index-url https://pypi.nvidia.com -r requirements-trt.txt

dev: install
	pip install 'pytest>=8.0' 'ruff>=0.4'

test:
	pytest tests/ -v

lint:
	ruff check seer tests experiments --output-format=concise

format:
	ruff format seer tests experiments

verify:
	@PY=$$(command -v python 2>/dev/null || command -v python3 2>/dev/null); \
	  test -n "$$PY" || { echo "FAIL: neither python nor python3 on PATH"; exit 1; }; \
	  $$PY -c "import pytest, ruff" 2>/dev/null || { echo "FAIL: pytest and/or ruff missing; run 'pip install pytest ruff'"; exit 1; }
	@PY=$$(command -v python 2>/dev/null || command -v python3 2>/dev/null); \
	  PYTEST=$$(command -v pytest 2>/dev/null || echo "$$PY -m pytest"); \
	  echo "[verify] 1/3 unit tests ($$PYTEST)"; \
	  $$PYTEST tests/ --tb=line -q || exit 1
	@PY=$$(command -v python 2>/dev/null || command -v python3 2>/dev/null); \
	  RUFF=$$(command -v ruff 2>/dev/null || echo "$$PY -m ruff"); \
	  echo "[verify] 2/3 ruff lint ($$RUFF)"; \
	  $$RUFF check seer tests experiments --output-format=concise || exit 1
	@echo "[verify] 3/3 schedulability CPU sanity"
	@PY=$$(command -v python 2>/dev/null || command -v python3 2>/dev/null); \
	  $$PY -m seer.timing.schedulability --epsilon 0.07 --slo P99=50ms --hbm_budget 0.20 \
	    --ell_bar_us 200 --sigma_residual_us 100 2>/dev/null | grep -q '"bound_lemma2_miss"' \
	    && echo "  [verify] CPU bound returns valid output"
	@echo ""
	@echo "[verify] ALL CHECKS PASSED"

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
