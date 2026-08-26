---
title: "API Reference — Reasoning Operators & Execution Runtime"
description: "API reference for HBLLM's Cognitive Execution Runtime: ReasoningOperator protocol, 13 composable operators, OperatorRegistry, FrozenGraphView, and UnifiedReasoningRuntime."
---

# Reasoning Operators & Execution Runtime API

The **Cognitive Execution Runtime** provides the formal execution semantics and composable reasoning operator substrate for HBLLM. Operators execute over immutable graph snapshots and propose atomic state transactions to HCIR without owning or modifying live cognitive state.

**Package:** `hbllm.brain.reasoning` / `hbllm.brain.reasoning.operators`

---

## Subsystem & Operator Index

### Infrastructure & Runtime

| Class / Protocol | Module | Role |
|---|---|---|
| `ReasoningOperator` | `operators.base` | Runtime protocol implemented by all 13 reasoning engines |
| `UnifiedReasoningRuntime` | `unified_runtime` | Orchestrates context building, operator dispatch, trace recording, and transaction assembly |
| `RuntimeConfig` | `unified_runtime` | Configuration for budget enforcement, pipeline sequencing, and stop conditions |
| `OperatorRegistry` | `operators.registry` | Multi-dimensional operator discovery, scoring, and reliability tracking |
| `FrozenGraphView` | `operators.base` | Read-only, content-hashed snapshot of the HCIR `CognitiveGraph` |
| `CognitiveContext` | `operators.base` | Immutable execution scope provided to operators |
| `ReasoningProblem` | `operators.base` | Formal definition of the reasoning goal, focus nodes, and problem type |
| `CognitiveResult` | `operators.base` | Standardized reasoning output containing conclusions, provenance, and proposed mutations |
| `OperatorTrace` | `operators.base` | Complete deterministic audit trail of a multi-operator reasoning cycle |
| `ReasoningBudget` | `operators.base` | Resource limits (wall-clock ms, token caps, node exploration limits) |

### The 13 Reasoning Operators

| Operator Class | Module | Problem Types Handled |
|---|---|---|
| `DeductionOperator` | `operators.deduction` | `EXPLANATION`, `CONSTRAINT` |
| `InductionOperator` | `operators.induction` | `GENERALIZATION`, `CLASSIFICATION` |
| `AbductionOperator` | `operators.abduction` | `EXPLANATION`, `DIAGNOSIS` |
| `TemporalOperator` | `operators.temporal` | `TEMPORAL`, `CAUSAL` |
| `SpatialOperator` | `operators.spatial` | `SPATIAL`, `PLANNING` |
| `AnalogyOperator` | `operators.analogy` | `ANALOGY`, `GENERALIZATION` |
| `PredictionOperator` | `operators.prediction` | `PREDICTION`, `TEMPORAL`, `PLANNING` |
| `ContradictionOperator` | `operators.contradiction` | `CONTRADICTION`, `DIAGNOSIS`, `EXPLANATION` |
| `CounterfactualOperator` | `operators.counterfactual` | `COUNTERFACTUAL`, `EXPLANATION`, `DIAGNOSIS`, `CAUSAL` |
| `CausalOperator` | `operators.causal` | `CAUSAL`, `EXPLANATION`, `DIAGNOSIS`, `PREDICTION` |
| `ActiveInferenceOperator` | `operators.active_inference` | `PLANNING`, `PREDICTION`, `CONSTRAINT` |
| `SimulationOperator` | `operators.simulation` | `PLANNING`, `CONSTRAINT`, `PREDICTION` |
| `SNNReasoningOperator` | `operators.snn_reasoning` | `CAUSAL`, `EXPLANATION`, `DIAGNOSIS`, `PREDICTION` |

---

## Core Types Reference

### `ReasoningOperator` Protocol

Every operator implements this runtime protocol:

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

### `ReasoningProblem`

```python
class ReasoningProblem(BaseModel):
    problem_type: ProblemType
    problem_id: str = Field(default_factory=...)
    description: str = ""
    focus_node_ids: list[str] = Field(default_factory=list)
    scope: Scope = Scope.LOCAL
    constraints: dict[str, Any] = Field(default_factory=dict)
```

### `CognitiveResult`

```python
class CognitiveResult(BaseModel):
    status: ResultStatus = ResultStatus.SUCCESS  # SUCCESS | PARTIAL | NO_RESULT | FAILED | TIMEOUT
    conclusions: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    assumptions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    proposed_transitions: list[TransactionOperation] = Field(default_factory=list)
    provenance_chains: list[ProvenanceChain] = Field(default_factory=list)
    operator_id: str = ""
    resource_cost: ResourceCost = Field(default_factory=ResourceCost)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

## Unified Reasoning Runtime

### Initialization & Execution

```python
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime, RuntimeConfig
from hbllm.brain.reasoning.operators import OperatorRegistry, DeductionOperator, TemporalOperator

# 1. Register operators
registry = OperatorRegistry()
registry.register(DeductionOperator())
registry.register(TemporalOperator())

# 2. Configure runtime
config = RuntimeConfig(
    max_operators_per_cycle=5,
    stop_on_no_result=False,
    min_confidence_threshold=0.2,
)
runtime = UnifiedReasoningRuntime(registry=registry, config=config)

# 3. Execute reasoning
trace = runtime.reason(
    graph=live_cognitive_graph,
    problem=reasoning_problem,
    budget=ReasoningBudget(compute_ms=100.0, max_node_reads=500),
)

# 4. Access deterministic trace & proposed transaction
print(trace.trace_id)
print(trace.context_hash)
if trace.proposed_transaction:
    # Submit proposed transaction to HCIR TransactionManager
    transaction_manager.submit(trace.proposed_transaction)
```

### Pipeline Composition

The runtime can compose multi-operator sequences where outputs of prior operators inform subsequent evaluations within the same cycle:

```python
trace = runtime.reason_pipeline(
    graph=live_cognitive_graph,
    problem=reasoning_problem,
    operator_ids=["temporal", "causal", "prediction"],
    budget=ReasoningBudget(compute_ms=250.0),
)
```

---

## Operator Registry

The `OperatorRegistry` tracks registration, applicability scoring, and operator reliability:

```python
registry = OperatorRegistry()

# Register single operator or batch
registry.register(DeductionOperator())

# Query ranked operators for a specific problem and context
scores = registry.select(problem=problem, context=context, max_candidates=3)
for s in scores:
    print(
        f"Operator: {s.operator_id}, Score: {s.composite_score:.3f} (Applicability: {s.applicability:.2f})"
    )

# Record execution outcome for dynamic reliability adaptation
registry.record_execution(
    operator_id="deduction",
    success=True,
    wall_clock_ms=3.4,
)
```
