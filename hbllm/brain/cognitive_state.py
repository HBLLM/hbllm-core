"""Cognitive State — Immutable, versioned working memory and evidence ledger.

Subsystems consume snapshots of CognitiveState and return updated, derived snapshots.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from hbllm.brain.autonomy.task_graph import Goal


@dataclass(frozen=True)
class Evidence:
    """Provenance and context tracking for facts and beliefs."""

    source: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    generated_by: str = ""  # Subsystem or Node ID
    reasoning_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "generated_by": self.generated_by,
            "reasoning_path": self.reasoning_path,
        }


@dataclass(frozen=True)
class CandidatePlan:
    """A first-class candidate execution or reasoning plan."""

    plan_id: str = field(default_factory=lambda: f"pln_{uuid.uuid4().hex[:12]}")
    graph: dict[str, Any] = field(default_factory=dict)
    origin: str = "planner"  # "planner", "analogy", "fallback"
    confidence: float = 1.0
    predicted_reward: float = 0.0
    predicted_cost: dict[str, float] = field(default_factory=dict)
    analogy_used: str | None = None
    simulation_result: dict[str, Any] | None = None
    execution_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "graph": self.graph,
            "origin": self.origin,
            "confidence": self.confidence,
            "predicted_reward": self.predicted_reward,
            "predicted_cost": self.predicted_cost,
            "analogy_used": self.analogy_used,
            "simulation_result": self.simulation_result,
            "execution_trace": self.execution_trace,
        }


@dataclass(frozen=True)
class CognitiveBudget:
    """Multidimensional resource allocation parameters."""

    attention_budget: float | None = None
    memory_budget: int | None = None
    simulation_budget: int | None = None
    reasoning_budget: int | None = None
    verification_budget: int | None = None
    planning_budget: float | None = None  # seconds
    tool_budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_budget": self.attention_budget,
            "memory_budget": self.memory_budget,
            "simulation_budget": self.simulation_budget,
            "reasoning_budget": self.reasoning_budget,
            "verification_budget": self.verification_budget,
            "planning_budget": self.planning_budget,
            "tool_budget": self.tool_budget,
        }


DEFAULT_COGNITIVE_BUDGET = CognitiveBudget(
    attention_budget=1.0,
    memory_budget=10,
    simulation_budget=5,
    reasoning_budget=1000,
    verification_budget=3,
    planning_budget=30.0,
    tool_budget=5,
)


@dataclass(frozen=True)
class CognitivePolicy:
    """Hierarchical policy rules configuring system behaviors."""

    reasoning_strategy: str | None = None  # "direct", "CoT", "GoT", "analogical"
    simulation_depth: int | None = None
    verification_budget: int | None = None
    retrieval_budget: int | None = None
    planner_type: str | None = None
    memory_budget: int | None = None
    model_selection: str | None = None
    reflection_enabled: bool | None = None
    budget: CognitiveBudget | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_strategy": self.reasoning_strategy,
            "simulation_depth": self.simulation_depth,
            "verification_budget": self.verification_budget,
            "retrieval_budget": self.retrieval_budget,
            "planner_type": self.planner_type,
            "memory_budget": self.memory_budget,
            "model_selection": self.model_selection,
            "reflection_enabled": self.reflection_enabled,
            "budget": self.budget.to_dict() if self.budget else None,
        }


DEFAULT_COGNITIVE_POLICY = CognitivePolicy(
    reasoning_strategy="direct",
    simulation_depth=1,
    verification_budget=2,
    retrieval_budget=5,
    planner_type="graph",
    memory_budget=10,
    model_selection="default",
    reflection_enabled=True,
    budget=DEFAULT_COGNITIVE_BUDGET,
)


@dataclass(frozen=True)
class HierarchicalCognitivePolicy:
    """A cascade of overrides from Global to Task level."""

    global_policy: CognitivePolicy
    conversation_policy: CognitivePolicy | None = None
    goal_policy: CognitivePolicy | None = None
    task_policy: CognitivePolicy | None = None

    def resolve(self) -> CognitivePolicy:
        """Resolve effective CognitivePolicy by traversing overrides."""
        resolved_fields: dict[str, Any] = {}
        for field_name in [
            "reasoning_strategy",
            "simulation_depth",
            "verification_budget",
            "retrieval_budget",
            "planner_type",
            "memory_budget",
            "model_selection",
            "reflection_enabled",
        ]:
            if (
                field_name == "reflection_enabled"
                and self.global_policy
                and self.global_policy.reflection_enabled
            ):
                resolved_fields[field_name] = True
                continue

            resolved_val = None
            for p in [
                self.task_policy,
                self.goal_policy,
                self.conversation_policy,
                self.global_policy,
                DEFAULT_COGNITIVE_POLICY,
            ]:
                if p is not None:
                    val = getattr(p, field_name)
                    if val is not None:
                        resolved_val = val
                        break
            resolved_fields[field_name] = resolved_val

        budget_fields: dict[str, Any] = {}
        for b_field in [
            "attention_budget",
            "memory_budget",
            "simulation_budget",
            "reasoning_budget",
            "verification_budget",
            "planning_budget",
            "tool_budget",
        ]:
            resolved_val = None
            for p in [
                self.task_policy,
                self.goal_policy,
                self.conversation_policy,
                self.global_policy,
                DEFAULT_COGNITIVE_POLICY,
            ]:
                if p is not None:
                    b_obj = getattr(p, "budget", None)
                    if b_obj is not None:
                        val = getattr(b_obj, b_field, None)
                        if val is not None:
                            resolved_val = val
                            break
            budget_fields[b_field] = resolved_val

        resolved_fields["budget"] = CognitiveBudget(**budget_fields)
        return CognitivePolicy(**resolved_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_policy": self.global_policy.to_dict(),
            "conversation_policy": self.conversation_policy.to_dict()
            if self.conversation_policy
            else None,
            "goal_policy": self.goal_policy.to_dict() if self.goal_policy else None,
            "task_policy": self.task_policy.to_dict() if self.task_policy else None,
            "effective": self.resolve().to_dict(),
        }


@dataclass(frozen=True)
class CognitiveState:
    """Immutable working memory snapshot. Any change derives a new versioned state."""

    goal: Goal
    policy: CognitivePolicy | HierarchicalCognitivePolicy
    state_id: str = field(default_factory=lambda: f"state_{uuid.uuid4().hex[:12]}")
    version: int = 1
    parent_state_id: str | None = None
    retrieved_memory: list[dict[str, Any]] = field(default_factory=list)
    simulations: list[dict[str, Any]] = field(
        default_factory=list
    )  # Serialized counterfactual scenarios

    @property
    def effective_policy(self) -> CognitivePolicy:
        if isinstance(self.policy, HierarchicalCognitivePolicy):
            return self.policy.resolve()
        return HierarchicalCognitivePolicy(global_policy=self.policy).resolve()

    candidate_plans: list[CandidatePlan] = field(default_factory=list)
    active_skills: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    evidence_ledger: dict[str, Evidence] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)

    def derive_state(self, **mutations: Any) -> CognitiveState:
        """Derive a new version of the CognitiveState with bumped version number."""
        next_version = self.version + 1
        new_state_id = f"state_{uuid.uuid4().hex[:12]}"

        # Override parent links
        mutations["version"] = next_version
        mutations["parent_state_id"] = self.state_id
        mutations["state_id"] = new_state_id
        mutations["created_at"] = time.time()

        return replace(self, **mutations)

    def fork(self) -> CognitiveState:
        """Fork current state to begin a branching future (maintains parent link)."""
        return self.derive_state()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "version": self.version,
            "parent_state_id": self.parent_state_id,
            "goal": self.goal.to_dict(),
            "policy": self.policy.to_dict(),
            "retrieved_memory": self.retrieved_memory,
            "simulations": self.simulations,
            "candidate_plans": [plan.to_dict() for plan in self.candidate_plans],
            "active_skills": self.active_skills,
            "reflections": self.reflections,
            "beliefs": self.beliefs,
            "evidence_ledger": {k: v.to_dict() for k, v in self.evidence_ledger.items()},
            "working_memory": self.working_memory,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }
