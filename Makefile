VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

.PHONY: test test-v test-fast test-unit test-integration lint format typecheck install clean \
       rust-test rust-clippy rust-build test-all pre-commit install-hooks \
       db-migrate db-upgrade db-downgrade db-history \
       docker-build docker-up docker-down

## ── Testing ──────────────────────────────────────────────────────────────

test: ## Run full test suite
	$(PYTEST) -q --tb=short

test-v: ## Run tests with verbose output
	$(PYTEST) -v --tb=short

test-fast: ## Run unit tests excluding slow/heavy tests
	$(PYTEST) tests/unit/ -q --tb=short --timeout=60 \
		--ignore=tests/unit/brain/test_cognitive_training.py \
		--ignore=tests/unit/ml/test_lora_loop.py

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ tests/modules/ -q --tb=short

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v --tb=short --timeout=120

test-%: ## Run a specific test file (e.g., make test-brain_factory)
	$(PYTEST) -k "test_$*" -v --tb=short

## ── Code Quality ─────────────────────────────────────────────────────────

lint: ## Run ruff linter
	$(PYTHON) -m ruff check .

format: ## Auto-format code and docs
	$(PYTHON) -m ruff format .

typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy hbllm/

## ── Git & Pre-commit ─────────────────────────────────────────────────────

pre-commit: ## Run all pre-commit quality gates (Ruff & Cargo)
	@echo "🔍 Running Ruff linter..."
	@if [ -x "$(PYTHON)" ]; then $(PYTHON) -m ruff check .; else ruff check .; fi
	@echo "🎨 Running Ruff formatter check..."
	@if [ -x "$(PYTHON)" ]; then $(PYTHON) -m ruff format --check .; else ruff format --check .; fi
	@echo "🦀 Checking Cargo formatting..."
	@cargo fmt --all -- --check
	@echo "🦀 Checking Cargo compilation..."
	@cargo check --workspace
	@echo "🦀 Running Cargo Clippy..."
	@cargo clippy --workspace -- -D warnings
	@echo "✅ All pre-commit quality gates passed!"

install-hooks: ## Configure git hooks for this repository
	@git config core.hooksPath .githooks
	@chmod +x .githooks/pre-commit
	@if [ -d "../.git/modules/core/hooks" ]; then \
		cp .githooks/pre-commit ../.git/modules/core/hooks/pre-commit; \
		chmod +x ../.git/modules/core/hooks/pre-commit; \
	fi
	@echo "✅ Pre-commit hooks installed successfully!"


## ── Setup ────────────────────────────────────────────────────────────────

install: $(VENV) ## Install project in development mode
	$(PIP) install -e ".[dev]"

$(VENV):
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

clean: ## Remove build artifacts and caches
	rm -rf .pytest_cache __pycache__ .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

## ── Rust ─────────────────────────────────────────────────────────────────

rust-test: ## Run all Rust unit tests
	cargo test --workspace --no-default-features

rust-clippy: ## Run clippy with deny warnings across all crates
	cargo clippy --workspace --lib -- -D warnings

rust-build: ## Build all Rust crates as Python modules (requires maturin)
	@for dir in rust/*/; do \
		if [ -f "$$dir/Cargo.toml" ]; then \
			echo "Building $$dir..."; \
			cd "$$dir" && maturin develop --release && cd ../..; \
		fi; \
	done

test-all: rust-test test ## Run Rust tests then Python tests

## ── Database Migrations ─────────────────────────────────────────────────

db-migrate: ## Create a new migration (usage: make db-migrate msg="add users table")
	$(PYTHON) -m alembic revision --autogenerate -m "$(msg)"

db-upgrade: ## Apply all pending migrations
	$(PYTHON) -m alembic upgrade head

db-downgrade: ## Rollback the last migration
	$(PYTHON) -m alembic downgrade -1

db-history: ## Show migration history
	$(PYTHON) -m alembic history --verbose

## ── Docker ──────────────────────────────────────────────────────────────

docker-build: ## Build production Docker image
	docker build -t hbllm-core:latest .

docker-up: ## Start all services (PostgreSQL + Redis + HBLLM)
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-prod: ## Start production stack (multi-replica)
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

## ── Help ─────────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
