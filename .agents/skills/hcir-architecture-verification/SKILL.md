---
name: hcir-architecture-verification
description: >-
  Use this skill before completing any task in `HBLLM/core` to run
  architectural invariant gates, structural-debt ratchets, rule-size
  checks, PyO3 parity checks, and `graphify` knowledge-graph synchronization.
---

# Skill: HCIR Architecture & Invariant Verification

## Scope
Run this verification workflow before finishing any code, configuration, or documentation change in `/Users/Dumith_Salinda/Projects/HBLLM/core`.

## Verification Checklist

1. **Run Full Architecture Invariant Suite:**
   ```bash
   pytest tests/unit/architecture/ -q
   ```
   Verifies:
   - Zero LLM imports in `hbllm/hcir/**`, zero `__path__.append` shadow-tree shims, and shipped-module import resolution (`test_loc_budget.py`).
   - Layering boundaries (`hcir ↛ brain`, `core ↛ plugins`, `test_layering.py`).
   - Single composition root (`build_cognitive_system`, `test_single_composition_root.py`).
   - Fail-closed `TransactionManager` & raw graph mutation ratchets (`test_state_ownership_invariants.py`).
   - `hasattr` / `getattr` / `except: pass` / `sqlite3.connect` / blacklisted-attr ratchets (`test_structural_debt_ratchet.py`).
   - Rust ABI crate parity (`test_rust_abi_version.py`) and no duplicate faculties (`test_no_duplicate_faculties.py`).

2. **Check Workspace Rule Byte Limits & Skill Frontmatter (when editing `AGENTS.md`, `.agents/`, or `docs/`):**
   - `AGENTS.md` and every `.agents/rules/*.md` file **must stay strictly under 24,000 bytes** (`wc -c AGENTS.md .agents/rules/*.md`) to avoid silent truncation in Antigravity.
   - Every `.agents/skills/<name>/SKILL.md` must start with valid YAML frontmatter containing `name: <name>` and a non-empty `description`.
   - Every path in `mkdocs.yml` and `docs/adr/README.md` must exist on disk.

3. **Run Targeted Functional Tests & Runtime Smoke Checks (after code edits):**
   ```bash
   pytest tests/unit/hcir/ -q
   python -m hbllm diagnostics
   ```

4. **Synchronize Knowledge Graph (after `.py` code edits):**
   ```bash
   graphify update .
   ```
   *(Skip `graphify update .` when only Markdown/YAML docs were modified.)*

