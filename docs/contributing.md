---
title: "Contributing to HBLLM Core"
description: "Guidelines, invariants, testing, and tooling workflow for contributing to the Pure-HCIR HBLLM Core repository."
---

# Contributing

We welcome contributions to **HBLLM Core**! Please read `AGENTS.md` in the repository root alongside this guide so every contribution preserves our Pure-HCIR invariants, architectural boundaries, and test gates.

---

## Getting Started

1. Fork the repository on GitHub and clone your fork.
2. Install development dependencies inside `HBLLM/core`:

```bash
cd HBLLM/core
pip install -e ".[dev]"
```

*(Optional)* Build native Rust extensions:

```bash
pip install maturin
for crate in rust/*/; do
  [ -f "$crate/Cargo.toml" ] && maturin develop --manifest-path "$crate/Cargo.toml" --release
done
```

---

## Development Workflow

### Running Tests

```bash
# Architecture & Pure-HCIR invariant checks (required)
pytest tests/unit/architecture/ -q

# Fast HCIR & world-model test suites
pytest tests/unit/hcir/ tests/unit/world/ -q

# Plugin test suites
pytest plugins/*/tests/ -q --timeout=30

# Docs strict build check
mkdocs build --strict -d /tmp/hbllm_docs && rm -rf /tmp/hbllm_docs
```

### Linting & Formatting

```bash
ruff check .
ruff format --check .
```

### Type Checking

```bash
mypy hbllm/
```

### Knowledge Graph Sync (`graphify`)

After substantial Python code changes or adding files, refresh the local knowledge graph:

```bash
graphify update .
```

### Rust Components

```bash
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

---

## Key Areas for Contribution

| Area | Description | Difficulty |
|---|---|---|
| 🧠 **Cortical World Model (`hbllm.hcir.world`)** | Probabilistic transition beliefs, stochastic MDP/EFE planning, active inference | Advanced |
| 👁️ **Symbol Grounding (`hbllm.hcir.grounding`)** | SNN & learned morphology-invariant `FeatureExtractor`s, tracking speedups | Intermediate |
| 🗣️ **Language & Reading (`hbllm.hcir.language`)** | New `LanguageProfile`s, parser/realizer coverage, Track T4 knowledge ingestion | Intermediate |
| 🔧 **Rust Kernels (`rust/*`)** | Wiring `hbllm_simulation_engine` & `hbllm_structure_matcher`, Rust unit tests | Advanced |
| 📖 **Documentation & Tests** | Tutorials, API examples, Hypothesis property tests | Beginner |

---

## Pull Request Guidelines

1. **Respect Core Invariants:** Zero external LLM imports in `hbllm/hcir/**`, mutate state only via `TransactionManager`, keep `hbllm/` strictly domain-agnostic (put task adapters in `plugins/`).
2. **One concern per PR:** Keep changes focused and reviewable.
3. **Tests & Docs required:** Include unit tests and update relevant `docs/` pages if public architecture or APIs change.
4. **Lint & CI clean:** Pass `ruff check`, `ruff format --check`, `mypy`, `pytest tests/unit/architecture/`, and `mkdocs build --strict`.

---

## License

By contributing, you agree that your contributions will be licensed under the **GNU General Public License v3.0 (GPLv3)**, the same license as the project.
