.PHONY: install dev test format lint clean help

help:
	@echo "Targets:"
	@echo "  install    Install deps + the seer package (editable)"
	@echo "  dev        install + ruff + pytest dev extras"
	@echo "  test       Run pytest suite"
	@echo "  format     Run ruff formatter"
	@echo "  lint       Run ruff linter"
	@echo "  clean      Remove Python + build caches"

install:
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install 'ruff>=0.4' 'pytest>=8.0'

test:
	pytest tests/ -v

format:
	ruff format seer tests experiments

lint:
	ruff check seer tests experiments

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
