---
title: "API Reference — Reasoning Operators & Execution Runtime"
description: "API reference for HBLLM's Cognitive Execution Runtime (hbllm.hcir.reasoning): ReasoningOperator protocol, 14 default operators, OperatorRegistry, FrozenGraphView, and UnifiedReasoningRuntime."
---

# Reasoning Operators & Execution Runtime API

The **Cognitive Execution Runtime** provides formal execution semantics and composable reasoning operators over immutable graph snapshots (`FrozenGraphView`), proposing atomic state transactions without mutating live cognitive state directly.

**Canonical Package:** `hbllm.hcir.reasoning` / `hbllm.hcir.reasoning.operators` *(compatibility re-exports at `hbllm.brain.reasoning`)*

---

## Subsystem & Operator Index

### Infrastructure & Runtime

| Class / Protocol | Module | Role |
|---|---|---|
| `ReasoningOperator` | `operators.base` | Runtime protocol implemented by all reasoning engines |
| `UnifiedReasoningRuntime` | `unified_runtime` | Orchestrates context building, operator dispatch, trace recording, and transaction assembly |
| `RuntimeConfig` | `unified_runtime` | Configuration (`max_operators_per_pass`, `min_selection_score`, `stop_on_no_result`, `merge_results`) |
| `OperatorRegistry` | `operators.registry` | Operator discovery, applicability/reliability scoring, and `create_default_operator_registry()` |
| `FrozenGraphView` | `operators.base` | Read-only, content-hashed snapshot of the HCIR `CognitiveGraph` |
| `CognitiveContext` | `operators.base` | Immutable execution scope provided to operators |
| `ReasoningProblem` | `operators.base` | Formal definition of the reasoning goal, focus nodes, and `ProblemType` |
| `CognitiveResult` | `operators.base` | Standardized reasoning output containing conclusions, provenance, and proposed mutations |
| `OperatorTrace` | `operators.base` | Complete deterministic audit trail of a multi-operator reasoning cycle |
| `ReasoningBudget` | `operators.base` | Resource limits (wall-clock ms, node exploration caps) |

### The 14 Default Reasoning Operators (`create_default_operator_registry()`)

| Operator Class | Module | Problem Types Handled |
|---|---|---|
| `DeductionOperator` | `operators.deduction` | `EXPLANATION`, `CONSTRAINT` |
| `InductionOperator` | `operators.induction` | `GENERALIZATION`, `CLASSIFICATION` |
| `AbductionOperator` | `operators.abduction` | `EXPLANATION`, `DIAGNOSIS` |
| `SpatialOperator` | `operators.spatial` | `SPATIAL`, `PLANNING` |
| `TemporalOperator` | `operators.temporal` | `TEMPORAL`, `CAUSAL` |
| `CausalOperator` | `operators.causal` | `CAUSAL`, `EXPLANATION`, `DIAGNOSIS`, `PREDICTION` |
| `AnalogyOperator` | `operators.analogy` | `ANALOGY`, `GENERALIZATION` |
| `CounterfactualOperator` | `operators.counterfactual` | `COUNTERFACTUAL`, `EXPLANATION`, `DIAGNOSIS`, `CAUSAL` |
| `ContradictionOperator` | `operators.contradiction` | `CONTRADICTION`, `DIAGNOSIS`, `EXPLANATION` |
| `PredictionOperator` | `operators.prediction` | `PREDICTION`, `TEMPORAL`, `PLANNING` |
| `SimulationOperator` | `operators.simulation` | `PLANNING`, `CONSTRAINT`, `PREDICTION` |
| `ActiveInferenceOperator` | `operators.active_inference` | `PLANNING`, `PREDICTION`, `CONSTRAINT` |
| `EmbodiedCausalOperator` | `operators.embodied_causal` | `PLANNING`, `CAUSAL`, `SPATIAL` |
| `SNNReasoningOperator` | `operators.snn_reasoning` | `CAUSAL`, `EXPLANATION`, `DIAGNOSIS`, `PREDICTION` |

> **Auxiliary Schema & Verification Engines:** `RelationalSchemaInducer` (`operators.relational_schema`), `SkillMacroChunker` (`operators.skill_chunker`), `TemporalVelocityModel` (`operators.temporal_dynamics`), and `SourceVerifier` (`source_verifier`).

---

## Core Types Reference

### `ReasoningOperator` Protocol

```python
class ReasoningOperator(Protocol):
    @property
    def operator_id(self) -> str: ...

    @property
    def operator_name(self) -> str: ...

    @property
    def prerequisites(self) -> tuple[str, ...]: ...

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float: ...

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost: ...

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult: ...
```

---

## Unified Reasoning Runtime

### Initialization & Execution

```python
from hbllm.hcir.reasoning.operators import (
    DeductionOperator,
    OperatorRegistry,
    ReasoningBudget,
    TemporalOperator,
    create_default_operator_registry,
)
from hbllm.hcir.reasoning.unified_runtime import RuntimeConfig, UnifiedReasoningRuntime

# 1. Use default 14-operator registry or customize
registry = create_default_operator_registry()
# Or: registry = OperatorRegistry(); registry.register(DeductionOperator()); registry.register(TemporalOperator())

# 2. Configure runtime
config = RuntimeConfig(
    max_operators_per_pass=5,
    min_selection_score=0.001,
    stop_on_no_result=False,
    merge_results=True,
)
runtime = UnifiedReasoningRuntime(registry=registry, config=config)

# 3. Execute reasoning over a CognitiveGraph
trace = runtime.reason(
    graph=live_cognitive_graph,
    problem=reasoning_problem,
    budget=ReasoningBudget(compute_ms=100.0, max_node_reads=500),
)

# 4. Access deterministic trace & proposed transaction
print(trace.trace_id, trace.context_hash)
if trace.proposed_transaction:
    transaction_manager.commit(trace.proposed_transaction)
```

### Operator Registry Selection

```python
scores = registry.select(problem=problem, context=context, max_candidates=3)
for s in scores:
    print(f"Operator: {s.operator_id}, Composite: {s.composite_score:.3f}")

registry.record_execution(
    operator_id="deduction",
    success=True,
    wall_clock_ms=3.4,
)
```
