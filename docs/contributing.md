---
title: "Contributing to HBLLM Core"
description: "Guidelines for contributing to the HBLLM open-source cognitive architecture project."
---

# Contributing

We welcome contributions to push AGI forward! HBLLM Core is an open-source project and we value every contribution.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork and create a feature branch
3. Install development dependencies:

```bash
cd HBLLM/core
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_network.py -v

# With coverage
python -m pytest tests/ --cov=hbllm
```

### Linting

```bash
# Check for errors
ruff check .

# Auto-fix safe issues
ruff check --fix .

# Format code
ruff format .
```

### Type Checking

```bash
mypy hbllm/
```

### Rust Components

```bash
# Format
cargo fmt --manifest-path rust/compute_kernel/Cargo.toml

# Lint
cargo clippy --manifest-path rust/compute_kernel/Cargo.toml --workspace -- -D warnings

# Build
cargo check --manifest-path rust/compute_kernel/Cargo.toml
```

## Key Areas for Contribution

| Area | Description | Difficulty |
|---|---|---|
| 🧠 **New Cognitive Nodes** | Emotion modeling, temporal reasoning | Advanced |
| 📱 **Edge Optimization** | Raspberry Pi 5, Jetson Orin Nano patches | Intermediate |
| 🌐 **Starter Zones** | Pre-trained LoRAs for Medicine, Law, Creative Writing | Intermediate |
| 📖 **Documentation** | Tutorials, examples, API docs | Beginner |
| 🧪 **Testing** | Expand test coverage, fuzz testing | Beginner |
| 🔧 **Rust Kernels** | New SIMD optimizations, quantization formats | Advanced |

## Pull Request Guidelines

1. **One concern per PR** — Keep changes focused and reviewable.
2. **Tests required** — All new features must include tests.
3. **Lint clean** — PRs must pass `ruff check`, `ruff format --check`, and `mypy`.
4. **Descriptive commits** — Use conventional commit format (`feat:`, `fix:`, `docs:`).

## Code Style

- Python: Enforced by `ruff` (configured in `pyproject.toml`)
- Rust: Enforced by `cargo fmt` and `cargo clippy`
- Markdown: Standard GitHub-Flavored Markdown

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
