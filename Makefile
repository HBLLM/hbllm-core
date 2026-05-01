VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(PYTHON) -m pytest

.PHONY: test test-v test-fast lint format typecheck install clean

## ── Testing ──────────────────────────────────────────────────────────────

test: ## Run full test suite
	$(PYTEST) tests/ -q --tb=short

test-v: ## Run tests with verbose output
	$(PYTEST) tests/ -v --tb=short

test-fast: ## Run tests excluding slow/heavy tests
	$(PYTEST) tests/ -q --tb=short --timeout=15 \
		--ignore=tests/test_cognitive_training.py \
		--ignore=tests/test_lora_loop.py

test-%: ## Run a specific test file (e.g., make test-brain_factory)
	$(PYTEST) tests/test_$*.py -v --tb=short

## ── Code Quality ─────────────────────────────────────────────────────────

lint: ## Run ruff linter
	$(PYTHON) -m ruff check hbllm/ tests/

format: ## Auto-format code
	$(PYTHON) -m ruff format hbllm/ tests/

typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy hbllm/

## ── Setup ────────────────────────────────────────────────────────────────

install: $(VENV) ## Install project in development mode
	$(PIP) install -e ".[dev]"

$(VENV):
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

clean: ## Remove build artifacts and caches
	rm -rf .pytest_cache __pycache__ .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

## ── Help ─────────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
