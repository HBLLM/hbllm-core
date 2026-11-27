# ADR 006 — Fuzzy Symbolic Cognition & Adaptive Semantics: 2×2 Experimental Matrix Report

- **Date:** 2026-11-26
- **Branch:** `feat/fuzzy-symbolic-cognition` (base Phase F0+F1 commit `05b719a9` + Phase F1 hot-path fix, Phase F2, Phase F3)
- **Specification:** [ADR 006](../../docs/adr/0006-fuzzy-symbolic-cognition-and-adaptive-semantics.md), [Migration Plan](../../docs/architecture/fuzzy-cognition-migration-plan.md)
- **LLM Tokens Used:** `0` across all 4 cells (100% deterministic Layer-0 substrate)

---

## 1. 2×2 Experimental Design (`A` / `B` / `C` / `D`)

Following ADR 006 §7, fuzzy grounding (`enable_fuzzy_grounding`, Phase F1) and fuzzy symbolic reasoning (`enable_fuzzy_reasoning`, Phase F2) are evaluated orthogonally over the same symbolic `CognitiveGraph` world representation, keeping the downstream crisp decision threshold frozen at `α = 0.65` in Cells **A** and **B**:

| Cell | Grounding (`F1`) | Reasoning (`F2`) | Dev Zero-Shot (`n=20`) | Dev Cross-Seed Transfer (`n=16`) | ZS Deaths | 16×16 → 32×32 Scale/Appearance Shift (`n=20`) | Soft Unification, Graded Expr & Scale-Guarded Compose (`n=20`) | False-Positive Compositions (`fork NEAR warehouse`) | 32×32 CPU p50 / p95 (ms) | Gate (`<= 45.0 ms`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Crisp | Crisp | `19/20` (`0.9500`) | `16/16` (`1.0000`) | `1` | `0/20` (`0.0000`) | `0/20` (`0.0000`) | `0` | `24.73 / 34.36` | ✓ PASS |
| **B** | **Fuzzy** | Crisp (`α=0.65`) | `19/20` (`0.9500`) | `16/16` (`1.0000`) | `1` | **`20/20` (`1.0000`)** | `0/20` (`0.0000`) | `0` | `30.78 / 41.92` | ✓ PASS |
| **C** | Crisp | **Fuzzy** | `19/20` (`0.9500`) | `16/16` (`1.0000`) | `1` | `0/20` (`0.0000`) | **`20/20` (`1.0000`)** | `0` | `25.55 / 32.53` | ✓ PASS |
| **D** | **Fuzzy** | **Fuzzy** | `19/20` (`0.9500`) | `16/16` (`1.0000`) | `1` | **`20/20` (`1.0000`)** | **`20/20` (`1.0000`)** | `0` | `30.01 / 43.72` | ✓ PASS |

---

## 2. Per-Task Closed-Loop `dev` Breakdown (`5` seeds/task × `3` replay episodes)

All four cells achieved identical zero-regression closed-loop accuracy, zero runtime errors, and identical consolidated bytecode skill counts:

| Task Family (`benchmarks/suites/dev.yaml`) | Zero-Shot (`n=5`) | Cross-Seed Transfer (`n=4`) | Same-Layout Replay | Skills Consolidated | Errors (A/B/C/D) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `grid_nav_dev` | `4/5` (`0.800`) | `4/4` (`1.000`) | `1.000` | `231` | `0` |
| `grid_push_dev` | `5/5` (`1.000`) | `4/4` (`1.000`) | `1.000` | `176` | `0` |
| `continuous_reach_dev` | `5/5` (`1.000`) | `4/4` (`1.000`) | `1.000` | `147` | `0` |
| `symbolic_rules_dev` | `5/5` (`1.000`) | `4/4` (`1.000`) | `1.000` | `188` | `0` |

---

## 3. Separated Scalar ALU vs. VM Dispatch Latency (`n=200` / `n=100` calls)

Direct Layer-0 ALU calls (`hbllm.hcir.capabilities.math_capabilities`) are measured independently of `HCIRInterpreter` `Opcode.EXECUTE` routing (`hbllm.hcir.interpreter`):

| Component | p50 (µs) | p95 (µs) |
| :--- | :--- | :--- |
| `math.fuzzy.membership` (Gaussian scalar ALU) | `4.96` | `5.33` |
| `math.fuzzy.eval` (`FuzzyExpr` Product `AND` ALU) | `19.65` | `26.41` |
| `math.fuzzy.compose` (`RelationCompositionPolicy` `NEAR ∘ INSIDE`) | `55.85` | `112.76` |
| `math.fuzzy.bipolar` (Correlation-normalized cluster aggregator) | `40.00` | `70.83` |
| `HCIRInterpreter` `Opcode.EXECUTE` dispatch (`math.fuzzy.membership`) | `79.81` | `149.56` |

---

## 4. Key Architectural Findings

1. **Orthogonal Gains Confirmed Across B and C → Composed Cleanly in D:**
   - Enabling `enable_fuzzy_grounding=True` (Cell **B**) raises `16×16 → 32×32` cross-scale & recolored hazard/goal prior transfer from `0/20` (`0.0000`) to `20/20` (`1.0000`) even under frozen crisp `α = 0.65`, via dimensionless extent/Mahalanobis prototype matching persisted as `GroundedConceptNode` + `AdaptiveMembershipNode` in `CognitiveGraph`.
   - Enabling `enable_fuzzy_reasoning=True` (Cell **C**) raises perturbed numeric/concept unification, smooth `SymbolicExpr.evaluate_degree`, and declarative scale-guarded `SpatialOperator` composition (`INSIDE(cup, tray) ∘ NEAR(tray, plate) → NEAR(cup, plate)`) from `0/20` to `20/20`, with `0` false-positive compositions (`fork NEAR box ∘ box INSIDE warehouse` cleanly blocked by `max_scale_ratio`, and `NEAR ∘ NEAR` cleanly forbidden by policy).
   - Cell **D** combines both capabilities (`20/20` transfer and `20/20` soft reasoning/composition) with zero regressions and `0` errors.
2. **Hot-Path Optimization & Transactional Integrity:**
   - Profiling (`32×32` `GridNavEnvironment`) surfaced (a) redundant `cached_entity_priors` invalidation in `CorticalGroundingBridge.sync_theory_roles` when comparing role concept IDs against feature-cluster IDs, and (b) `UPSERT_EDGE` before `UPSERT_NODE` ordering in `HCIRCommitter.commit_fuzzy_assertions`. Fixing both eliminated every `Dangling edge reference` rollback and reduced `32×32` Cell **D** CPU `p95` from `61.70 ms` / `49.36 ms` down to **`43.72 ms`** (`p50 = 30.01 ms`, well under the `45.0 ms` constitutional budget).
3. **Suite & Verification Snapshot:**
   - Fuzzy suites (`F0`–`F3`) + `test_structural_debt_ratchet.py`: `29 passed`.
   - `mypy .` (`1274` source files): `0` issues; `cargo check` & `cargo clippy -- -D warnings` (all `13` Rust crates): `0` warnings.
   - `hbllm bench profile --strict`: Combined Decisional Cycle `p95 = 0.310 ms` (`<= 10.0 ms`); `32×32` E2E CPU `p50 = 18.54 ms`, `p95 = 20.62 ms`.

