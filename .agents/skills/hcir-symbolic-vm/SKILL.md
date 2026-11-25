---
name: hcir-symbolic-vm
description: >-
  Use this skill when designing, modifying, or debugging the HCIR bytecode ISA
  (`bytecode.py`), `HCIRInterpreter` (`interpreter.py`), working-memory registers
  and unification engine, `stdlib.py`, `declarative.py`, homoiconic node schemas
  (`nodes.py`), or executing Phases M1-M2 of the Homoiconic Symbolic VM migration.
---

# Skill: HCIR Symbolic VM (ISA v2)

## Scope & Authoritative References
- [ADR 005: Homoiconic Symbolic VM](../../../docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md)
- [Migration Plan — Phases M1 & M2](../../../docs/architecture/homoiconic-vm-migration-plan.md)
- Key files: `hbllm/hcir/bytecode.py`, `hbllm/hcir/interpreter.py`, `hbllm/hcir/nodes.py`, `hbllm/hcir/stdlib.py`, `hbllm/hcir/skills/declarative.py`, `hbllm/hcir/replay_debugger.py`.

## Design Rules & Execution Steps

1. **Maintain the 12-Opcode Register-Unification ISA:**
   - Core v1 opcodes (`ASSERT`, `RETRACT`, `QUERY`, `EXECUTE`, `WAIT`, `FORK`, `MERGE`, `ROLLBACK`) must remain backward-compatible with literal v1 `InstructionStream`s while resolving `RegisterRef` (`$r0`..`$r7`) and `SymbolicVar` (`?var`) bindings.
   - Composition/control opcodes (`UNIFY`, `BRANCH`, `CALL`, `RETURN`) execute within a bounded `CallStack` (`max_depth=16`, cycle guard, and instruction fuel budget) to guarantee deterministic termination.

2. **Enforce Strict Hardware `SyscallAllowlist` on `sys_execute`:**
   - `Opcode.EXECUTE` may only invoke irreducible Layer-0 substrate syscalls (`motor.*`, `sensor.*`, `math.*`, `tool.sandboxed.*`, `web.sandboxed.*`, `memory.*`).
   - Do **not** implement high-level planning, goal decomposition, or reasoning strategies inside Python `EXECUTE` handlers (`stdlib.py` macros must compile to multi-opcode `QUERY -> UNIFY -> BRANCH -> FORK -> CALL -> MERGE` streams).

3. **Keep `SkillNode`, `ProcedureNode`, and `LearnedRuleNode` Homoiconic:**
   - Every executable program node carries `parameters`, `precondition_pattern`, `postcondition_pattern`, Dirichlet reliability `(alpha_success, beta_failure)`, and a serialized `bytecode_stream` (`InstructionStream`).
   - Express predicates as serializable `GraphPattern` / `SymbolicExpr` ASTs rather than Python-only `SymbolicPredicate` subclasses.

4. **Preserve Determinism & Fail-Closed Rollback:**
   - `InstructionStream.compute_hash()` must produce canonical deterministic hashes independent of ephemeral IDs when computing semantic digests.
   - `sys_rollback` must fail closed if restored graph content hash disagrees with the target snapshot hash.
   - `CognitiveGraph.fork()` / `HCIRNode` branching must isolate `properties` mutations via frozen models or copy-on-write.

## Verification
Run from `HBLLM/core/`:
```bash
pytest tests/unit/hcir/test_hcir_interpreter.py tests/unit/hcir/test_declarative_skills.py tests/unit/hcir/test_hcir_observability_and_optimization.py -q
pytest tests/unit/architecture/test_state_ownership_invariants.py -q
```

