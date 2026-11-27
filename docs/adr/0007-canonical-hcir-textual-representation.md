---
title: "ADR 007: Canonical .hcir Textual Representation & Lossless S-Expression Codec"
description: "Architectural decision defining .hcir as a canonical, lossless S-expression representation for HBLLM's homoiconic symbolic VM, patterns, expressions, and cognitive nodes."
---

# ADR 007: Canonical `.hcir` Textual Representation & Lossless S-Expression Codec

- **Status**: Implemented (Phases 1–7 Verified — 2026-11-27)
- **Deciders**: HBLLM Core Architecture Team
- **Date**: 2026-11-27
- **Technical Area**: HCIR Serialization & Inspection (`hbllm/hcir/`), Seed Loader (`hbllm/hcir/seed/`), CLI (`hbllm`)
- **Builds On**: [ADR 005: Homoiconic Symbolic VM & Experience-First Cognition](0005-homoiconic-symbolic-vm-and-experience-first-cognition.md), [ADR 006: Fuzzy Symbolic Cognition & Adaptive Semantics](0006-fuzzy-symbolic-cognition-and-adaptive-semantics.md)

---

## Context and Problem Statement

ADR 005 and ADR 006 established HBLLM Core as a **homoiconic graph-based symbolic cognitive VM with unification and transactional state execution**:

- Executable programs (`InstructionStream`, 12 ISA v2 opcodes over `$r0..$r7` registers and `?var` logic variables) live inside `SkillNode`, `ProcedureNode`, `ConstructionNode`, and `LearnedRuleNode` graph entries in `TieredWorkspace`.
- Declarative subgraph matching (`GraphPattern` with crisp and fuzzy constraints) and graded predicate trees (`SymbolicExpr`, `FuzzyExpr`) govern unification, preconditions, and alpha-cut branching.
- Wake–sleep consolidation (`hbllm/hcir/skills/library_learner.py`) anti-unifies successful `ExecutionReceipt` traces into parameterized `SkillNode`s.

What HBLLM still lacks is a **human-readable, machine-readable canonical textual representation** for these symbolic structures:

1. **Authoring Friction Drives Procedural Drift:**
   Hand-authoring ISA v2 streams or declarative priors in nested YAML (`hbllm/hcir/seed/core_priors.hcir.yaml` is 609 LOC) or raw Python Pydantic dicts is verbose and error-prone, creating constant pressure to fall back to procedural Python inside `hbllm/hcir/world/`.
2. **Learned Programs Are Hard to Inspect & Diff:**
   Experience-induced `SkillNode`s, `LearnedRuleNode`s, and `ConstructionNode`s serialize as dense JSON/dict payloads. Reviewing diffs across commits or auditing what wake–sleep anti-unification actually learned requires a clean, deterministic, structural text projection.
3. **Why Not Rewrite in Lisp, Symbolica, or Rust?**
   - **Not Lisp:** Conventional Lisp centers on expression evaluation over cons cells, whereas HBLLM's computational center is **unification (`UNIFY`), counterfactual branching (`FORK`/`MERGE`/`ROLLBACK`), graded fuzzy matching, and transactional state commits**. Replacing the VM with a Lisp evaluator would discard those primitives.
   - **Not Symbolica:** Computer algebra term-rewriting does not replace a cognitive execution VM with working-memory registers, epistemic posteriors, and constitutional verification.
   - **Not a Full Rust VM Rewrite Yet:** Fixed `32×32` kernel profiling already achieves `Decisional Cycle p95 = 0.182 ms` and `CPU E2E p95 = 14.58 ms <= 45 ms` with the 13 PyO3 Rust crates handling hot graph/matching/simulation inner loops. Moving the 1,093-line Python interpreter (`hbllm/hcir/interpreter.py`) to Rust without profiling evidence of dispatch bottleneck is premature.

---

## Decision

HBLLM Core adopts **`.hcir`** — a **canonical, lossless S-expression textual representation** of existing HCIR symbolic data and program structures, defined normatively in [Canonical `.hcir` Format Specification](../architecture/hcir-canonical-format.md).

### 1. Naming & Positioning

- The format and file extension are **`.hcir`** (avoiding `HCL`, which collides with HashiCorp Configuration Language).
- **Phase 1–7 Scope:** `.hcir` v1 is **not a separate programming language**; it is a canonical textual projection of existing HCIR models (`InstructionStream`, `GraphPattern`, `SymbolicExpr`, `FuzzyExpr`, `SkillNode`, `ProcedureNode`, `LearnedRuleNode`, `ConstructionNode`, `ConceptNode`, `AdaptiveMembershipNode`, `RelationCompositionPolicy`, `HCIREdge`, and seed bundles).

### 2. Rule 0 — Zero Hidden Semantics

> **Every `.hcir` v1 construct must correspond 1:1 to an existing HCIR semantic structure. The textual format must not introduce cognitive semantics, implicit runtime behaviors, or syntactic sugar that do not exist in the underlying IR.**

- Surface conveniences (such as `(if ...)` sugar compiling to `BRANCH` or macros/REPL) are deferred to **Phase 8 (v2)** and must lower strictly to canonical v1 forms.
- Four distinct syntactic families preserve semantic boundaries rather than collapsing them into one generic form:
  1. **Instruction streams** (`(stream ...)`, 12 ISA v2 opcodes, `(label ...)`)
  2. **Graph patterns** (`(pattern ...)`, `(pattern-map ...)`, `(expr-guard ...)`, explicit `(fuzzy ...)` and `(tolerances ...)`)
  3. **Symbolic & fuzzy expressions** (`SymbolicExpr` forms and `(fuzzy-ast ...)` for `FuzzyExpr`)
  4. **Cognitive nodes & seed bundles** (`(skill ...)`, `(procedure ...)`, `(rule ...)`, `(construction ...)`, `(concept ...)`, `(adaptive-membership ...)`, `(composition-policy ...)`, `(seed-bundle ...)`)

### 3. Self-Describing Envelopes & Serialization Modes

- Top-level documents carry `(hcir :version 1 :type <doc-type> ...)` (while the reader also accepts bare forms for embedding).
- **Two explicit formatting modes** resolve ephemeral runtime metadata (`Instruction.id`, `InstructionStream.id`, `Provenance.timestamp`):
  - `mode="canonical"` (default for Git diffs, seed authoring, CLI inspection): omits ephemeral random IDs/timestamps and unchanged `HCIRNode` base defaults while preserving stable node IDs, logic, and ordering.
  - `mode="exact"` (for lossless checkpoint/model equality): emits `(meta ...)` and `:id` fields so Pydantic models round-trip with strict field equality.

### 4. Four Binding Verification Invariants

Every implementation of the `.hcir` codec (`hbllm/hcir/sexpr.py`) must satisfy:

1. **Structural Round-Trip:** `parse_hcir(format_hcir(x, mode="exact")) == x`
2. **Canonical Idempotence:** `format_hcir(parse_hcir(text)) == format_hcir(parse_hcir(format_hcir(parse_hcir(text))))`
3. **Semantic Identity:** `semantic_digest(parse_hcir(format_hcir(x))) == semantic_digest(x)` (64-hex BLAKE3 digest via `hbllm_hcir_graph.blake3_hash`, stripping ephemeral IDs/timestamps/runtime counters while `exact_digest()` preserves them) and `InstructionStream.semantic_hash()` equality.
4. **Execution Equivalence:** `execute(x, state) == execute(parse_hcir(format_hcir(x)), state)` across nested `FORK`, `UNIFY` (crisp + fuzzy), `BRANCH` (`ALPHA_CUT`), `CALL`, and `ROLLBACK` programs.

### 5. Substrate Boundary

- The reader and pretty-printer live in zero-dependency Python (`hbllm/hcir/sexpr.py`) at authoring, seed-loading, and inspection boundaries.
- Runtime execution (`HCIRInterpreter`) continues to execute `InstructionStream` objects directly with zero parse overhead during decisional cycles.

---

## Phased Implementation Roadmap

| Phase | Deliverable | Exit Gate |
| :--- | :--- | :--- |
| **Phase 1 — Specification** | This ADR + [Canonical `.hcir` Format Spec](../architecture/hcir-canonical-format.md) | 100% field coverage across all 4 families, docs & `mkdocs.yml` checks pass |
| **Phase 2 — Reader** | Zero-dependency tokenizer & recursive-descent parser in `hbllm/hcir/sexpr.py` | Parses all spec forms with line/column diagnostics |
| **Phase 3 — Canonical Printer** | Deterministic `format_hcir(obj, *, mode="canonical" \| "exact")` | Lexicographically sorted maps, deterministic float/indent formatting |
| **Phase 4 — Round-Trip Tests** | Structural round-trip, canonical idempotence, and `semantic_digest` test suites | 100% pass across hand-written and seeded HCIR objects |
| **Phase 5 — Execution & Property Tests** | Hypothesis/fuzz property tests comparing `execute(x)` vs `execute(parse(format(x)))` | Zero behavioral divergence on `FORK`/`UNIFY`/fuzzy/`BRANCH`/`CALL`/`ROLLBACK` |
| **Phase 6 — Seed Loader Integration** | Dual `.hcir` + `.hcir.yaml` support in `hbllm/hcir/seed/loader.py` with domain purity gate | Identical hydrated `TransactionOperation`s from `.hcir` and `.hcir.yaml` |
| **Phase 7 — Inspection CLI** | `hbllm hcir inspect`, `print`, `validate`, `roundtrip` subcommands | Inspects learned `SkillNode`s and traces directly as `.hcir` |
| **Phase 8 — v2 Extensions (Deferred)** | Sugar (`if`), macros, interactive REPL | Must desugar strictly to canonical v1 |

---

## Consequences

- **Positive:** Learned skills, rules, constructions, and innate priors become human-readable, Git-diffable, and reviewable without introducing a second runtime VM or breaking ADR 005/006 invariants.
- **Guardrail:** Any future `.hcir` syntax proposal that cannot be mapped losslessly to existing `hbllm/hcir/` models requires an explicit IR/VM ADR first (Rule 0).

---

## References

- [Canonical `.hcir` Format Specification (v1)](../architecture/hcir-canonical-format.md)
- [ADR 005: Homoiconic Symbolic VM & Experience-First Cognition](0005-homoiconic-symbolic-vm-and-experience-first-cognition.md)
- [ADR 006: Fuzzy Symbolic Cognition & Adaptive Semantics](0006-fuzzy-symbolic-cognition-and-adaptive-semantics.md)
- [Homoiconic VM Migration Plan](../architecture/homoiconic-vm-migration-plan.md)

