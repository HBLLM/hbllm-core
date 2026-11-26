---
name: wake-sleep-skill-consolidation
description: >-
  Use this skill when modifying `learning_loop.py`, `skills/induction.py`,
  `skills/library_learner.py`, `world/option_discovery.py`,
  `world/episodic/replay.py`, or `ExecutiveRuntime` offline idle consolidation
  and wake-sleep library learning (Phases M2 and M4).
---

# Skill: Wake–Sleep Anti-Unification & Skill Consolidation

## Scope & Authoritative References
- [ADR 005](../../../docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md)
- [Migration Plan — Phases M2 & M4](../../../docs/architecture/homoiconic-vm-migration-plan.md)
- Key files: `hbllm/hcir/learning_loop.py`, `hbllm/hcir/skills/induction.py`, `hbllm/hcir/world/option_discovery.py`, `hbllm/hcir/world/episodic/replay.py`, `hbllm/hcir/kernel/executive_runtime.py`, `hbllm/runtime/bootstrap.py`.

## Design Rules & Execution Steps

1. **Always Persist Parameterized Executable Bytecode:**
   - Never emit metadata-only `SkillNode`s (`skill_name` + `description` without `bytecode_stream`) and never generate dynamic Python classes at runtime (e.g. `AutonomousOptionSkill`).
   - Induced and consolidated skills must store typed `parameters`, `precondition_pattern`, `postcondition_pattern`, Dirichlet counts `(alpha_success, beta_failure)`, and a valid `InstructionStream` v2 (`bytecode_stream`) executable via `Opcode.CALL`.

2. **Anti-Unify Across Episodes to Prevent Surface-ID Poisoning:**
   - When consolidating traces in `LearningLoopEngine`, `SkillInductionEngine`, `HippocampalReplayConsolidator`, or `ExecutiveRuntime` offline sleep:
     - Replace episode-local entity IDs, surface colors, and absolute coordinates with typed `?role` / `?entity` parameters (`SymbolicVar`).
     - Strip layout-local bindings at `EpisodeBoundary(same_level=False)` so seed *k* never poisons seed *k+1* in `partial_observability` or `language_goals`.

3. **Gate Promotion by MDL Compression & Transaction Safety:**
   - Only promote a candidate macro `SkillNode` when Minimum Description Length improves (`compressed_cost < sum(raw_trace_costs)`) or empirical support exceeds threshold.
   - Commit all promoted skills and schemas atomically through `TransactionManager` / `TieredWorkspace` (`Brain` -> `Persistent`), strictly scoped by `tenant_id` — never use raw `sqlite3.connect`.

4. **Ensure Benchmark Observability:**
   - Keep `bootstrap.py` and `benchmarks/suites/runner.py` wired to the live `TieredWorkspace` skill library so `skills_consolidated` accurately reflects promoted homoiconic skills (`> 0`).

## Verification
Run from `HBLLM/core/`:
```bash
pytest tests/unit/hcir/test_declarative_skills.py tests/unit/architecture/test_no_direct_sqlite.py -q
pytest tests/unit/ -k "replay or consolidation or skill_induction" -q
```

