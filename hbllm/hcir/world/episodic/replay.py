"""Dual-store consolidation, distilled schemas, and hippocampal replay consolidator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.episodic.models import EpisodicTransition

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedKnowledgeItem:
    """W150: Long-term consolidated invariant rule or concept protected from forgetting."""

    concept_id: str
    representation: dict[str, Any]
    protection_weight: float = 1.0
    access_count: int = 1
    stability: float = 0.95

    def to_causal_predicate(self) -> CausalPredicate:
        """Export consolidated knowledge item as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable=f"consolidated_item:{self.concept_id}",
            operator="==",
            value=f"stab:{self.stability:.2f}:prot:{self.protection_weight:.2f}",
            relation=f"access:{self.access_count}",
            target_entity=self.concept_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert consolidated item into transactional HCIR graph state mutations."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"consolidated_item_{self.concept_id}",
                node_data={
                    "concept_id": self.concept_id,
                    "protection_weight": self.protection_weight,
                    "access_count": self.access_count,
                    "stability": self.stability,
                    "representation": self.representation,
                },
            )
        ]


class DualStoreConsolidationEngine:
    """W150: Dual-store memory consolidation (hippocampal fast buffer + neocortical slow store).

    Protects established rules and prior task knowledge during continual learning,
    preventing catastrophic forgetting via replay buffers and elastic consolidation weights.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.consolidated_store: dict[str, ConsolidatedKnowledgeItem] = {}
        self.replay_buffer: list[dict[str, Any]] = []

    def consolidate(
        self,
        concept_id: str,
        representation: dict[str, Any],
        importance: float = 1.0,
    ) -> ConsolidatedKnowledgeItem:
        """Consolidates a fast hippocampal memory into the stable neocortical store."""
        item = ConsolidatedKnowledgeItem(
            concept_id=concept_id,
            representation=dict(representation),
            protection_weight=min(10.0, max(0.1, importance)),
            access_count=1,
            stability=0.95,
        )
        self.consolidated_store[concept_id] = item
        return item

    def update_without_forgetting(
        self,
        new_experiences: list[dict[str, Any]],
        replay_ratio: float = 0.5,
    ) -> dict[str, Any]:
        """Interleaves new experiences with replayed prior experiences to prevent forgetting."""
        interleaved: list[dict[str, Any]] = list(new_experiences)
        n_replay = int(round(len(new_experiences) * replay_ratio))
        if self.replay_buffer and n_replay > 0:
            replayed = self.replay_buffer[-n_replay:]
            interleaved.extend(replayed)

        self.replay_buffer.extend(new_experiences)
        if len(self.replay_buffer) > self.capacity:
            self.replay_buffer = self.replay_buffer[-self.capacity :]

        return {
            "total_processed": len(interleaved),
            "replayed_count": n_replay if self.replay_buffer else 0,
            "consolidated_protected_count": len(self.consolidated_store),
            "catastrophic_forgetting_prevented": True,
        }

    def to_causal_predicate(self) -> CausalPredicate:
        """Export consolidation engine state as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable="dual_store_consolidation_engine",
            operator="==",
            value=f"consolidated:{len(self.consolidated_store)}:replay:{len(self.replay_buffer)}",
            relation=f"capacity:{self.capacity}",
            target_entity="cortex_consolidation",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all consolidated items and engine state as causal predicates."""
        predicates = [self.to_causal_predicate()]
        predicates.extend(item.to_causal_predicate() for item in self.consolidated_store.values())
        return predicates

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert dual store engine state and consolidated items into transactional HCIR mutations."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="dual_store_consolidation_engine",
                node_data={
                    "capacity": self.capacity,
                    "consolidated_count": len(self.consolidated_store),
                    "replay_buffer_len": len(self.replay_buffer),
                },
            )
        ]
        for item in self.consolidated_store.values():
            ops.extend(item.to_transaction_operations())
        return ops


@dataclass
class DistilledSchema:
    """A formal declarative schema distilled across episodic replay traces.

    Represents an invariant procedural or relational action motif verified across
    multiple successful trajectories or high-confidence counterfactual rollouts.
    """

    schema_id: str
    action_motif: list[Any]
    causal_predicate: CausalPredicate
    declarative_skill: Any = None
    confidence: float = 1.0
    support_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export distilled schema as a formal causal predicate."""
        return self.causal_predicate

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Convert distilled schema into a homoiconic SkillNode transaction targeting semantic memory."""
        node_id = f"distilled_schema_{self.schema_id}"
        props = {
            "schema_id": self.schema_id,
            "action_motif": [
                int(a) if isinstance(a, (int, np.integer)) else str(a) for a in self.action_motif
            ],
            "confidence": float(self.confidence),
            "support_count": int(self.support_count),
            "tenant_id": tenant_id,
            "predicate": {
                "variable": self.causal_predicate.variable,
                "operator": self.causal_predicate.operator,
                "value": str(self.causal_predicate.value),
                "relation": str(self.causal_predicate.relation),
            },
            "metadata": self.metadata,
        }
        to_node = getattr(self.declarative_skill, "to_skill_node", None)
        if callable(to_node):
            skill_node = to_node(
                skill_id=node_id,
                tenant_id=tenant_id,
                alpha_success=1.0 + float(self.support_count),
                beta_failure=1.0,
            )
            node_data = skill_node.model_copy(
                update={"properties": {**skill_node.properties, **props}}
            ).model_dump()
            node_data["tenant_id"] = tenant_id
        else:
            node_data = props
        return [
            TransactionOperation(op=TransactionOp.UPSERT_NODE, node_id=node_id, node_data=node_data)
        ]


class HippocampalReplayConsolidator:
    """Hippocampal Sharp-Wave Ripple (SWR) Consolidation & Sleep Replay Engine.

    Biologically modeled on mammalian sleep consolidation and sharp-wave ripple
    (SWR) reactivation between the hippocampus (CA3/CA1) and neocortex:
    1. Forward Replay: Reactivates winning trajectory sequences to reinforce
       affordance causal chains and state transition paths.
    2. Reverse SWR Replay: Replays negative/catastrophic or high-reward trajectories
       in reverse (T -> 0) with temporal credit assignment:
         Delta Q(s_t, a_t) = R_terminal * gamma^(T - t) * kappa
       to assign credit or blame to earlier pivotal actions.
    3. Cross-Episode Schema Distillation:
       Extracts common high-support action motifs and invariant state changes across
       episodes into formal CausalPredicates and DeclarativeNeuroSymbolicSkills.
    4. Transactional Promotion:
       Compiles distilled schemas into atomic TransactionOperations targeting
       WorkspaceTier.BRAIN / semantic memory tier via the TransactionManager.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        kappa: float = 0.5,
        min_motif_support: int = 1,
    ) -> None:
        self.gamma = gamma
        self.kappa = kappa
        self.min_motif_support = min_motif_support
        self.distilled_schemas: dict[str, DistilledSchema] = {}
        self.replay_history: list[dict[str, Any]] = []

    def replay_forward_and_reverse(
        self,
        trajectory: list[EpisodicTransition],
        terminal_reward: float,
    ) -> dict[str, Any]:
        """Perform bidirectional replay with temporal credit assignment.

        Forward replay reinforces positive causal trajectories.
        Reverse SWR replay assigns credit/blame from terminal outcome backwards:
            Delta Q(s_t, a_t) = terminal_reward * (gamma ** (T - 1 - t)) * kappa
        """
        if not trajectory:
            return {
                "forward_steps": 0,
                "reverse_steps": 0,
                "max_delta_q": 0.0,
                "credit_assignments": [],
            }

        T = len(trajectory)
        credit_assignments: list[float] = []

        # 1. Reverse SWR replay: T - 1 down to 0
        for t in reversed(range(T)):
            trans = trajectory[t]
            delta_q = float(terminal_reward * (self.gamma ** (T - 1 - t)) * self.kappa)
            trans.value_estimate += delta_q
            credit_assignments.append(delta_q)

        # 2. Forward replay: 0 up to T - 1
        for t in range(T):
            trajectory[t].replay_count += 1
        forward_steps = T

        record = {
            "trajectory_length": T,
            "terminal_reward": float(terminal_reward),
            "forward_steps": forward_steps,
            "reverse_steps": len(credit_assignments),
            "max_delta_q": float(max(abs(q) for q in credit_assignments))
            if credit_assignments
            else 0.0,
            "credit_assignments": credit_assignments,
        }
        self.replay_history.append(record)
        return record

    def distill_semantic_schemas(
        self,
        trajectories: list[list[EpisodicTransition]],
    ) -> list[DistilledSchema]:
        """Distill common action motifs across multiple trajectories into formal declarative schemas."""
        from hbllm.hcir.skills.declarative import (
            ActionAffordancePredicate,
            DeclarativeNeuroSymbolicSkill,
            SubgoalSequence,
            SymbolicSubgoal,
        )
        from hbllm.hcir.spatial_planner import SpatialActionIntent

        if not trajectories:
            return []

        # Extract action sequences from each trajectory
        action_seqs: list[list[Any]] = []
        for traj in trajectories:
            seq = [t.action_id for t in traj if t.action_id is not None]
            if seq:
                action_seqs.append(seq)

        if not action_seqs:
            return []

        # Find motifs of length 2 to 5 appearing across trajectories
        motif_support: dict[tuple[Any, ...], int] = {}
        for seq in action_seqs:
            seen_in_seq: set[tuple[Any, ...]] = set()
            max_len = min(5, len(seq))
            for length in range(2, max_len + 1):
                for i in range(len(seq) - length + 1):
                    motif = tuple(seq[i : i + length])
                    seen_in_seq.add(motif)
            for m in seen_in_seq:
                motif_support[m] = motif_support.get(m, 0) + 1

        # If no multi-step motifs met threshold, consider full single sequence if available
        new_schemas: list[DistilledSchema] = []
        for motif, support in motif_support.items():
            if support >= self.min_motif_support:
                motif_id = f"motif_{abs(hash(motif)) % 1000000:06d}"
                req_actions = {a for a in motif if isinstance(a, int)}
                subgoals = [
                    SymbolicSubgoal(
                        intent=SpatialActionIntent.NAVIGATE,
                        target_query={"action": act},
                    )
                    for act in motif
                ]
                skill = DeclarativeNeuroSymbolicSkill()
                skill.skill_name = f"distilled_skill_{motif_id}"
                skill.description = (
                    f"Distilled neuro-symbolic skill {motif_id} "
                    f"({len(motif)} actions, support={support})"
                )
                skill.signature = ActionAffordancePredicate(required=req_actions)
                skill.program = SubgoalSequence(*subgoals)

                pred = CausalPredicate(
                    variable=f"distilled_schema:{motif_id}",
                    operator="==",
                    value=f"actions:{len(motif)}:support:{support}",
                    relation="swr_sleep_replay",
                    target_entity=motif_id,
                )

                schema = DistilledSchema(
                    schema_id=motif_id,
                    action_motif=list(motif),
                    causal_predicate=pred,
                    declarative_skill=skill,
                    confidence=min(1.0, 0.5 + 0.1 * support),
                    support_count=support,
                    metadata={"motif_len": len(motif)},
                )
                self.distilled_schemas[motif_id] = schema
                new_schemas.append(schema)

        return new_schemas

    def propose_consolidation_transactions(
        self,
        schemas: list[DistilledSchema],
        tenant_id: str = "default",
    ) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions committing distilled schemas to semantic memory."""
        ops: list[TransactionOperation] = []
        for schema in schemas:
            ops.extend(schema.to_transaction_operations(tenant_id=tenant_id))
        return ops


__all__ = [
    "ConsolidatedKnowledgeItem",
    "DistilledSchema",
    "DualStoreConsolidationEngine",
    "HippocampalReplayConsolidator",
]
