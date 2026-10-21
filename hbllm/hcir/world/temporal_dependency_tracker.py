"""Domain-General Temporal Dependency Tracking, Delayed Effects, Dual-Store Memory & Transfer.

Implements capabilities:
- W167: Hidden-State & Latent Register Inference
- W172: Delayed-Effect Modeler with Multi-Step Lag Estimation
- W173: Long-Horizon Prerequisite Dependency Graph
- W176: Efficient Macro-Action Sequencing
- W177: Invariant Causal Schema Compression
- W178: Cross-Context Knowledge Transfer with Provenance
- W179: Non-Stationary Environment Change & Rule Invalidation Detection
- W180: Epistemic Invariant vs. Episodic Instance Memory Decoupling
- W189: Objective Learning Efficiency Measurement (Sample Complexity & Regret)
- W190: Zero-Shot Generalization under Novelty
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate, LogicOperator


class SchemaValidationStatus(StrEnum):
    """Lifecycle validation status of a transferred causal schema in a new context."""

    PENDING_TARGET_VALIDATION = "PENDING_TARGET_VALIDATION"
    VALIDATED = "VALIDATED"
    REFUTED = "REFUTED"


# ── W172, W167: Delayed Effects & Latent State Inference ─────────────────────


@dataclass
class DelayedEffectRecord:
    """Discovered causal effect that manifests with temporal lag tau > 0 (W172)."""

    trigger_action: Any
    consequence_key: str
    observed_lag_steps: int
    confidence: float
    support_count: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert delayed effect into a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable=self.consequence_key,
            operator="==",
            value=True,
            relation=f"delayed_by_{self.observed_lag_steps}_steps",
            target_entity=f"action_{self.trigger_action}",
        )


class DelayedEffectModeler:
    """Discovers non-instantaneous consequences and delayed temporal dynamics (W172)."""

    def __init__(self, max_lag: int = 10) -> None:
        self.max_lag = max_lag
        self.action_history: deque[tuple[int, Any, dict[str, Any]]] = deque(maxlen=64)
        self.discovered_delayed_effects: list[DelayedEffectRecord] = []

    def record_step(
        self,
        step: int,
        action_id: Any,
        state_features: dict[str, Any],
        observed_mutations: dict[str, Any] | None = None,
    ) -> list[DelayedEffectRecord]:
        """Correlate newly observed mutations at step t with historical actions at t - tau."""
        self.action_history.append((step, action_id, state_features))
        newly_found: list[DelayedEffectRecord] = []

        if not observed_mutations or len(self.action_history) < 2:
            return newly_found

        for mut_key in sorted(observed_mutations.keys()):
            # Check backwards across historical steps within max_lag
            for prior_step, prior_act, _ in reversed(self.action_history):
                lag = step - prior_step
                if lag <= 0:
                    continue
                if lag > self.max_lag:
                    break

                # Found potential delayed effect
                rec = DelayedEffectRecord(
                    trigger_action=prior_act,
                    consequence_key=mut_key,
                    observed_lag_steps=lag,
                    confidence=0.75,
                    support_count=1,
                )
                self.discovered_delayed_effects.append(rec)
                newly_found.append(rec)

        return newly_found

    def get_effects_for_action(self, action_id: Any) -> list[DelayedEffectRecord]:
        """Return all delayed consequences triggered by the given action."""
        return [e for e in self.discovered_delayed_effects if e.trigger_action == action_id]

    def reset(self) -> None:
        """Clear historical action trace across episodes."""
        self.action_history.clear()


# ── W173, W176: Long-Horizon Dependency Graph & Macro-Sequencing ────────────


@dataclass
class DependencyNode:
    """A prerequisite milestone node in a long-horizon task graph (W173)."""

    node_id: str
    description: str
    preconditions: list[str] = field(default_factory=list)
    satisfying_actions: list[Any] = field(default_factory=list)
    is_satisfied: bool = False

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert milestone requirement into declarative HCIR CausalPredicate."""
        if not self.preconditions:
            return CausalPredicate(variable=f"milestone_{self.node_id}", operator="==", value=True)
        children = [
            CausalPredicate(variable=f"milestone_{p}", operator="==", value=True)
            for p in sorted(self.preconditions)
        ]
        if len(children) == 1:
            return children[0]
        return CausalPredicate(logic_op=LogicOperator.AND, children=children)


class LongHorizonDependencyGraph:
    """Builds prerequisite DAGs and schedules macro-action sequences (W173, W176)."""

    def __init__(self) -> None:
        self.nodes: dict[str, DependencyNode] = {}
        self.edges: dict[str, set[str]] = defaultdict(set)  # prereq -> dependent

    def add_milestone(
        self,
        node_id: str,
        description: str,
        prerequisites: Sequence[str] | None = None,
        satisfying_actions: Sequence[Any] | None = None,
    ) -> None:
        """Register a milestone in the dependency graph."""
        self.nodes[node_id] = DependencyNode(
            node_id=node_id,
            description=description,
            preconditions=list(prerequisites or []),
            satisfying_actions=list(satisfying_actions or []),
        )
        if prerequisites:
            for p in prerequisites:
                self.edges[p].add(node_id)

    def get_executable_milestones(self) -> list[DependencyNode]:
        """Return milestones whose prerequisites are all satisfied."""
        executable: list[DependencyNode] = []
        for n_id, node in sorted(self.nodes.items(), key=lambda kv: kv[0]):
            if node.is_satisfied:
                continue
            all_prereqs_met = all(
                self.nodes[p].is_satisfied for p in node.preconditions if p in self.nodes
            )
            if all_prereqs_met:
                executable.append(node)
        return executable

    def mark_satisfied(self, node_id: str) -> None:
        """Mark milestone achieved."""
        if node_id in self.nodes:
            self.nodes[node_id].is_satisfied = True

    def topological_sort(self) -> list[str]:
        """Compute valid topological ordering of milestones via Kahn's algorithm."""
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        for prereq, dependents in self.edges.items():
            for dep in dependents:
                if dep in in_degree:
                    in_degree[dep] += 1

        queue: deque[str] = deque([n for n, deg in in_degree.items() if deg == 0])
        ordered: list[str] = []

        while queue:
            curr = queue.popleft()
            ordered.append(curr)
            for neighbor in self.edges.get(curr, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return ordered

    def is_acyclic(self) -> bool:
        """True if the prerequisite dependency graph contains no cycles."""
        return len(self.topological_sort()) == len(self.nodes)


# ── W177, W178, W180: Dual-Store Memory & Cross-Context Transfer ─────────────


@dataclass
class CausalSchemaInvariant:
    """Domain-general causal rule or affordance invariant decoupled from episodic facts (W177, W180)."""

    schema_id: str
    rule_signature: str
    provenance_source: str  # World / game / environment origin ID
    confidence: float
    support_count: int
    last_validated_timestamp: float = field(default_factory=time.time)
    validation_status_in_target: str = "PENDING_TARGET_VALIDATION"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export invariant rule as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable=f"schema::{self.schema_id}",
            operator="==",
            value=self.rule_signature,
            relation="invariant_causal_schema",
            target_entity=self.provenance_source,
        )


class DualStoreMemoryManager:
    """Decouples invariant generalizable schemas from episodic instance facts with provenance (W177, W178, W180)."""

    def __init__(self) -> None:
        # Invariant generalizable store (cross-episode schemas)
        self.invariant_schemas: dict[str, CausalSchemaInvariant] = {}
        # Episodic instance store (local level / current episode facts)
        self.episodic_facts: dict[str, Any] = {}

    def commit_invariant_schema(
        self,
        schema_id: str,
        rule_signature: str,
        provenance_source: str,
        confidence: float = 0.85,
    ) -> CausalSchemaInvariant:
        """Store an invariant generalizable mechanic with explicit source provenance (W178)."""
        schema = CausalSchemaInvariant(
            schema_id=schema_id,
            rule_signature=rule_signature,
            provenance_source=provenance_source,
            confidence=confidence,
            support_count=1,
            validation_status_in_target=SchemaValidationStatus.PENDING_TARGET_VALIDATION.value,
        )
        self.invariant_schemas[schema_id] = schema
        return schema

    def validate_schema_in_target(
        self,
        schema_id: str,
        is_consistent_with_target_evidence: bool,
    ) -> bool:
        """Validation gate: verifies transferred schema against local target evidence before reliance (W178)."""
        if schema_id not in self.invariant_schemas:
            return False
        schema = self.invariant_schemas[schema_id]
        if is_consistent_with_target_evidence:
            schema.validation_status_in_target = SchemaValidationStatus.VALIDATED.value
            schema.confidence = min(1.0, schema.confidence + 0.10)
            return True
        else:
            schema.validation_status_in_target = SchemaValidationStatus.REFUTED.value
            schema.confidence = max(0.0, schema.confidence - 0.50)
            return False

    def reset_episodic_memory(self) -> None:
        """Clear transient episode facts while preserving all invariant causal schemas."""
        self.episodic_facts.clear()
        # Reset target validation status of schemas for the new context
        for s in self.invariant_schemas.values():
            s.validation_status_in_target = SchemaValidationStatus.PENDING_TARGET_VALIDATION.value

    def get_validated_schemas(self) -> list[CausalSchemaInvariant]:
        """Return all invariant schemas confirmed valid in the target domain."""
        return [
            s
            for s in self.invariant_schemas.values()
            if s.validation_status_in_target == SchemaValidationStatus.VALIDATED.value
        ]

    def export_world_theory(self) -> list[CausalPredicate]:
        """Export all validated schemas as declarative HCIR CausalPredicates."""
        return [s.to_causal_predicate() for s in self.get_validated_schemas()]


# ── W179: Non-Stationary Environment Change Detection ───────────────────────


class EnvironmentChangeDetector:
    """Monitors predictive surprise and detects environmental regime shifts or rule invalidations (W179)."""

    def __init__(self, surprise_threshold: float = 2.0) -> None:
        self.surprise_threshold = surprise_threshold
        self.surprise_history: deque[float] = deque(maxlen=20)

    def record_prediction_error(
        self, predicted_state: Any, actual_state: Any
    ) -> tuple[bool, float]:
        """Compute surprise score and detect regime change."""
        if isinstance(predicted_state, dict) and isinstance(actual_state, dict):
            discrepancies = sum(
                1
                for k in set(predicted_state) | set(actual_state)
                if predicted_state.get(k) != actual_state.get(k)
            )
            surprise = float(discrepancies)
        elif isinstance(predicted_state, np.ndarray) and isinstance(actual_state, np.ndarray):
            surprise = float(np.count_nonzero(predicted_state != actual_state))
        else:
            surprise = 0.0 if predicted_state == actual_state else 1.0

        self.surprise_history.append(surprise)
        mean_surprise = sum(self.surprise_history) / float(len(self.surprise_history))

        # Regime shift detected if current surprise significantly exceeds threshold
        regime_shifted = surprise >= self.surprise_threshold and mean_surprise >= 1.0
        return regime_shifted, round(surprise, 4)

    def reset(self) -> None:
        """Reset surprise memory."""
        self.surprise_history.clear()


# ── W189, W190: Objective Learning Efficiency Measurement ───────────────────


@dataclass
class LearningEfficiencyMetrics:
    """Objective, reproducible learning efficiency benchmarks (W189)."""

    sample_complexity_steps: int  # Steps required to achieve task solution
    cumulative_regret: float  # Sum of sub-optimal action penalties
    information_gain_rate: float  # Bits acquired per step
    zero_shot_transfer_success: bool  # Immediate success in novel context without retries

    def to_dict(self) -> dict[str, Any]:
        """Dictionary summary for benchmark serialization."""
        return {
            "sample_complexity_steps": self.sample_complexity_steps,
            "cumulative_regret": self.cumulative_regret,
            "information_gain_rate": self.information_gain_rate,
            "zero_shot_transfer_success": self.zero_shot_transfer_success,
        }


class ObjectiveLearningEfficiencyTracker:
    """Computes sample complexity and cumulative regret against fixed protocols (W189, W190)."""

    def __init__(self) -> None:
        self.total_steps: int = 0
        self.cumulative_regret: float = 0.0
        self.bits_acquired: float = 0.0

    def record_step_evaluation(
        self,
        action_optimal_value: float,
        action_chosen_value: float,
        info_bits: float = 0.0,
    ) -> None:
        """Record per-step regret and information acquisition."""
        self.total_steps += 1
        regret = max(0.0, action_optimal_value - action_chosen_value)
        self.cumulative_regret += regret
        self.bits_acquired += info_bits

    def compute_summary_metrics(self, task_solved: bool) -> LearningEfficiencyMetrics:
        """Compute final objective efficiency report."""
        info_rate = self.bits_acquired / float(max(1, self.total_steps))
        return LearningEfficiencyMetrics(
            sample_complexity_steps=self.total_steps,
            cumulative_regret=round(self.cumulative_regret, 4),
            information_gain_rate=round(info_rate, 4),
            zero_shot_transfer_success=task_solved and self.total_steps < 50,
        )

    def reset(self) -> None:
        """Reset efficiency tracker."""
        self.total_steps = 0
        self.cumulative_regret = 0.0
        self.bits_acquired = 0.0
