# Rule: Homoiconic Symbolic VM & Experience-First Cognition ("Code = CPU, Graph = Mind")

Every modification touching `hbllm/hcir/**` or `hbllm/runtime/**` must uphold ADR 005 (`docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md`) and advance the `M0`–`M6` migration plan (`docs/architecture/homoiconic-vm-migration-plan.md`).

## 1. Classify Before Coding
Before writing any code, classify what you are building:
- **Layer 0 (Fixed Hardware CPU & ALU — Rust / SNN / Python):** Persistent graph storage, BLAKE3 hashing, the register-unification VM (`$r0..$r7`, `CallStack`), sensory/SNN encoders, stateless math/physics kernels (geodesics, Bellman backups, Dirichlet updates, subgraph isomorphism, collision cones), two-tier sandboxing, and constitutional transactions. Layer-0 code **must remain ontology-blind, domain-blind, and language-blind**.
- **Layer 1 (Innate Bootstrap Seed — `hbllm/hcir/seed/`):** Declarative `.hcir` priors strictly bounded by `docs/architecture/priors.md`, loaded transactionally at genesis.
- **Layer 2 (Homoiconic Symbolic Mind — `TieredWorkspace`):** Concepts, linguistic constructions (`ConstructionNode`), causal laws (`CausalPredicate` / `LearnedRuleNode`), reasoning strategies, and parameterized executable skills (`SkillNode` / `ProcedureNode` with `bytecode_stream`).

## 2. Hard Prohibitions (Phase 3)
1. **No Hardcoded Lexical or Rule Tables in Python:** Never add new English/multilingual verb lists, color maps, preposition dicts, or domain regexes to `compiler.py`, `reading_curriculum.py`, or `world/`. Model linguistic patterns as graph-stored `ConstructionNode`s.
2. **No High-Level Cognitive Escapes in `Opcode.EXECUTE`:** `EXECUTE` is strictly for Layer-0 hardware syscalls (`motor.*`, `sensor.*`, `math.*`, `tool.sandboxed.*`, `web.sandboxed.*`, `memory.*`). Never hide planning, decomposition, or reasoning heuristics inside Python capability handlers.
3. **No Metadata-Only `SkillNode`s:** Any learned, induced, or consolidated `SkillNode` must carry typed parameter slots and an executable `bytecode_stream` (`InstructionStream` v2) so future tasks can `UNIFY` and `CALL` it.
4. **No Python-Only Domain Predicate Subclasses:** Do not add new procedural `SymbolicPredicate` subclasses in `declarative.py` for domain checks; compose serializable `GraphPattern` / `SymbolicExpr` ASTs over `math.*` ALUs.
5. **No Direct State or Engine Attribute Mutations:** Propose `TransactionOperation`s / `StageResult.operations` over `FrozenGraphView`; never write `view.state.<attr> = ...` or `engine._last_*`.

