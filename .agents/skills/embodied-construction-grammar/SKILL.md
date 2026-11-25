---
name: embodied-construction-grammar
description: >-
  Use this skill when modifying `hbllm/hcir/compiler.py`, `hbllm/hcir/language/`
  (`en`, `si`, `ta`, `HCIRGateway`, `ReadingCurriculumIngestor`,
  `LexiconAcquisitionLoop`), grounding natural language instructions, or
  implementing compositional unknown-instruction resolution via prior knowledge
  (Phase M3).
---

# Skill: Embodied Construction Grammar & Unknown-Instruction Resolution

## Scope & Authoritative References
- [ADR 005](../../../docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md)
- [Migration Plan — Phase M3](../../../docs/architecture/homoiconic-vm-migration-plan.md)
- [Language & Reading Architecture](../../../docs/architecture/language-and-reading.md)
- Key files: `hbllm/hcir/compiler.py`, `hbllm/hcir/language/runtime.py`, `hbllm/hcir/language/reading_curriculum.py`, `hbllm/hcir/language/core/gateway.py`, `hbllm/hcir/language/core/grounding.py`, `hbllm/hcir/language/acquisition/lexicon_acquisition_loop.py`.

## Design Rules & Execution Steps

1. **Zero Hardcoded Lexical/Verb Tables in Python:**
   - Never add new hardcoded English/Sinhala/Tamil verb tuples, color-to-int maps, or regex domain rules in `compiler.py` or `reading_curriculum.py`.
   - Seed functional grammar constructions declaratively in `hbllm/hcir/seed/` and store learned `(Surface Pattern <-> Parameterized HCIR Program)` mappings as graph nodes (`ConstructionNode` / `LearnedRuleNode(domain="construction_grammar")`) in `TieredWorkspace`.

2. **Compile Expository & Procedural Text Directly to Executable Graph Nodes:**
   - When `ReadingCurriculumIngestor` or `ExpositoryProseProcessor` encounters definitional or procedural prose (*"to X a Y means..."*, *"if P then Q"*), commit both relational `BeliefNode`s and executable `ConstructionNode` / `SkillNode` proposals via `TransactionManager`.

3. **Execute the 4-Step Unknown-Instruction Resolution Pipeline:**
   When an utterance contains an unseen verb, modifier, or construction:
   1. **Partial Unification:** Bind all recognized concepts, spatial relations, and affordances into registers (`$r0..$rk`); isolate the unseen span as an `UnknownNode` with an inferred `(InputTypes -> GoalConstraint)` signature.
   2. **Memory & Reading Recall:** `QUERY` `TieredWorkspace` (`Brain` + `Persistent`) and `LexicalOntologyStore` for definitions, hypernyms, or prior episodes linked to the stem.
   3. **Structural Analogy:** Use `hbllm_structure_matcher` / `AnalogyOperator` to unify existing `SkillNode` `(precondition_pattern -> postcondition_pattern)` signatures with current scene affordances.
   4. **Active Inquiry & Consolidation:** If ambiguity remains, `FORK` a counterfactual probe or emit `REQUEST_EVIDENCE` (`EpistemicCuriosityController`), check `PredictionErrorNode`, and commit the grounded `ConstructionNode` + `SkillNode` so future compositional instructions resolve zero-shot.

4. **Enforce State Ownership & Hermeticity:**
   - Route all mutations in `gateway.py` and `grounding.py` through `commit_operations` (never raw `.add_node()` / `.upsert_node()`).
   - Keep `hbllm/hcir/language/**` 100% LLM-free and ensure `test_pure_hcir_graduation_gate.py` runs hermetically without untracked `checkpoints/`.

## Verification
Run from `HBLLM/core/`:
```bash
pytest tests/unit/hcir/test_pure_hcir_graduation_gate.py tests/unit/architecture/test_loc_budget.py -q
pytest tests/unit/ -k "language or compiler or grounding" -q
```

