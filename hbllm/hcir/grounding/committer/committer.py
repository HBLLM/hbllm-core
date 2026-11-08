from __future__ import annotations

import hashlib
import logging
import uuid

from hbllm.hcir.graph import HCIREdge
from hbllm.hcir.grounding.concepts.lifecycle import ConceptPromotionProposal
from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.contracts.beliefs import BeliefState
from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.decisions import GroundingDecision
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle, DecisionOutcome
from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedEntityRef,
    LatentRelationRef,
    LiteralValueRef,
    PredicateRef,
)
from hbllm.hcir.grounding.tracking.operations import EntityReidentificationOp
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.nodes import (
    BeliefNode,
    GroundedConceptNode,
    HCIREdgeType,
    PhysicalEntityNode,
)
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOp,
    TransactionOperation,
    TransactionStatus,
)

logger = logging.getLogger(__name__)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class HCIRCommitter:
    """The authoritative bridge committing grounded state into the Canonical CognitiveGraph.

    Upholds Invariant 1 (State Ownership Invariant):
    The SGE Perception, Tracking, and Epistemics modules do not directly mutate graph state.
    Only the HCIRCommitter translates validated GroundingDecisions and ConceptPromotionProposals
    into formal TransactionOperations committed atomically via TransactionManager.
    """

    def __init__(self, transaction_manager: TransactionManager) -> None:
        self._tx_manager = transaction_manager

    @property
    def transaction_manager(self) -> TransactionManager:
        return self._tx_manager

    def commit_decision(
        self,
        decision: GroundingDecision,
        claim: GroundingClaim,
        belief: BeliefState | None = None,
        author: str = "SGE_Committer",
    ) -> HCIRTransaction:
        """Translates an accepted GroundingDecision and claim into an atomic HCIRTransaction."""
        if decision.outcome != DecisionOutcome.ACCEPT:
            raise ValueError(
                f"Cannot commit non-accepted decision (outcome={decision.outcome.value})"
            )

        operations: list[TransactionOperation] = []

        # 1. Ensure subject entity node exists
        subj_id = claim.subject_ref.id
        subj_node = PhysicalEntityNode(id=subj_id, entity_name=subj_id)
        operations.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=subj_id,
                node_data=subj_node.model_dump(),
            )
        )

        # 2. Handle object (entity vs literal)
        obj_ref = claim.object_ref
        target_node_id: str | None = None
        statement = f"{claim.predicate_ref}({subj_id}, {obj_ref})"

        if isinstance(obj_ref, (GroundedEntityRef, EntityCandidateRef)):
            target_node_id = obj_ref.id
            obj_node = PhysicalEntityNode(id=target_node_id, entity_name=target_node_id)
            operations.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=target_node_id,
                    node_data=obj_node.model_dump(),
                )
            )

            # Create typed relational edge with deterministic ID
            edge_type = self._map_predicate_to_edge_type(claim.predicate_ref)
            pred_str = str(
                claim.predicate_ref.name
                if isinstance(claim.predicate_ref, PredicateRef)
                else (
                    claim.predicate_ref.id
                    if isinstance(claim.predicate_ref, LatentRelationRef)
                    else claim.predicate_ref
                )
            )
            rel_edge_id = f"edge_rel_{hashlib.blake2b(f'{subj_id}:{pred_str}:{target_node_id}'.encode(), digest_size=8).hexdigest()}"
            rel_edge = HCIREdge(
                id=rel_edge_id,
                edge_type=edge_type,
                sources=[subj_id],
                targets=[target_node_id],
                properties={
                    "predicate": pred_str,
                    "polarity": claim.polarity.value,
                    "confidence": decision.posterior_belief,
                },
                weight=decision.posterior_belief,
            )
            operations.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_EDGE,
                    edge_id=rel_edge.id,
                    edge_data=rel_edge.model_dump(),
                )
            )

        elif isinstance(obj_ref, LiteralValueRef):
            # Property assertion stored on subject entity or belief
            statement = f"{claim.predicate_ref}({subj_id}) = {obj_ref.value} {obj_ref.unit}".strip()

        # 3. Create BeliefNode with deterministic ID for epistemic auditability
        belief_id = (
            belief.belief_id
            if (belief and belief.belief_id)
            else f"belief_{hashlib.blake2b(f'{subj_id}:{claim.predicate_ref}:{obj_ref}:{claim.polarity.value}'.encode(), digest_size=8).hexdigest()}"
        )
        belief_node = BeliefNode(
            id=belief_id,
            statement=statement,
            claim=statement,
            epistemic_confidence=decision.posterior_belief,
            evidence_sources=list(decision.supporting_evidence_ids),
        )
        operations.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=belief_id,
                node_data=belief_node.model_dump(),
            )
        )

        # Edge from Belief to Subject with deterministic ID
        target_str = target_node_id or ""
        support_edge_id = f"edge_sup_{hashlib.blake2b(f'{belief_id}:{subj_id}:{target_str}'.encode(), digest_size=8).hexdigest()}"
        support_edge = HCIREdge(
            id=support_edge_id,
            edge_type=HCIREdgeType.SUPPORTS,
            sources=[belief_id],
            targets=[subj_id] if not target_node_id else [subj_id, target_node_id],
            properties={"posterior": decision.posterior_belief},
            weight=decision.posterior_belief,
        )
        operations.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_EDGE,
                edge_id=support_edge.id,
                edge_data=support_edge.model_dump(),
            )
        )

        # 4. Assemble and commit transaction
        tx = HCIRTransaction(
            id=decision.transaction_proposal_id
            or f"tx_grounding_{hashlib.blake2b(f'{belief_id}:{decision.timestamp or 0}'.encode(), digest_size=8).hexdigest()}",
            author=author,
            operations=operations,
            metadata={"description": f"Grounded state assertion: {statement}"},
        )

        committed_tx = self._tx_manager.commit(tx)
        if committed_tx.status != TransactionStatus.COMMITTED:
            logger.error("Failed to commit grounding transaction: %s", committed_tx.status)
        return committed_tx

    def commit_concept_promotion(
        self,
        proposal: ConceptPromotionProposal,
        concept: GroundedConcept,
        author: str = "SGE_ConceptPromoter",
    ) -> HCIRTransaction:
        """Promotes a consolidated concept from Working Store into canonical CognitiveGraph."""
        concept_id = proposal.concept_ref.id
        operations: list[TransactionOperation] = []

        # 1. Create GroundedConceptNode
        concept_node = GroundedConceptNode(
            id=concept_id,
            concept_name=concept_id,
            feature_prototype=concept.prototypes[-1] if concept.prototypes else {},
            behavioral_regularities=list(proposal.affordances),
            causal_rule_ids=list(concept.predictive_models.keys()),
            prediction_accuracy=proposal.quality_summary.predictive_utility,
            confidence=proposal.quality_summary.predictive_utility,
            exemplar_count=proposal.exemplar_count,
            tags=["grounded_concept", "sge_promoted"],
        )
        operations.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=concept_id,
                node_data=concept_node.model_dump(),
            )
        )

        # 2. Attach validated exemplars via INSTANCE_OF edges
        for exemplar_id in concept.exemplar_entity_ids:
            # Ensure exemplar entity node exists
            exemplar_node = PhysicalEntityNode(id=exemplar_id, entity_name=exemplar_id)
            operations.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=exemplar_id,
                    node_data=exemplar_node.model_dump(),
                )
            )

            inst_edge_id = f"edge_inst_{hashlib.blake2b(f'{exemplar_id}:{concept_id}'.encode(), digest_size=8).hexdigest()}"
            inst_edge = HCIREdge(
                id=inst_edge_id,
                edge_type=HCIREdgeType.INSTANCE_OF,
                sources=[exemplar_id],
                targets=[concept_id],
                properties={"utility": proposal.quality_summary.predictive_utility},
                weight=proposal.quality_summary.predictive_utility,
            )
            operations.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_EDGE,
                    edge_id=inst_edge.id,
                    edge_data=inst_edge.model_dump(),
                )
            )

        tx = HCIRTransaction(
            id=proposal.proposal_id,
            author=author,
            operations=operations,
            metadata={
                "description": f"Promoted grounded concept {concept_id}: {proposal.rationale}"
            },
        )

        committed_tx = self._tx_manager.commit(tx)
        if committed_tx.status == TransactionStatus.COMMITTED:
            concept.transition_to(
                ConceptLifecycle.CANONICAL,
                reason=f"Committed to CognitiveGraph via tx {committed_tx.id}",
            )
        return committed_tx

    def commit_entity_reidentification(
        self,
        reid_op: EntityReidentificationOp,
        author: str = "SGE_EntityTracker",
    ) -> HCIRTransaction:
        """Commits confirmed entity identity equivalence (SAME_AS) into CognitiveGraph."""
        cand_id = reid_op.candidate_ref.id
        grounded_id = reid_op.grounded_ref.id

        cand_node = PhysicalEntityNode(id=cand_id, entity_name=cand_id)
        grounded_node = PhysicalEntityNode(id=grounded_id, entity_name=grounded_id)
        same_as_edge_id = f"edge_same_as_{hashlib.blake2b(f'{cand_id}:{grounded_id}'.encode(), digest_size=8).hexdigest()}"
        same_as_edge = HCIREdge(
            id=same_as_edge_id,
            edge_type=HCIREdgeType.SAME_AS,
            sources=[cand_id],
            targets=[grounded_id],
            properties=dict(reid_op.similarity_metrics),
            weight=reid_op.confidence,
        )

        operations = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=cand_id,
                node_data=cand_node.model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=grounded_id,
                node_data=grounded_node.model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_EDGE,
                edge_id=same_as_edge.id,
                edge_data=same_as_edge.model_dump(),
            ),
        ]

        tx = HCIRTransaction(
            id=reid_op.operation_id,
            author=author,
            operations=operations,
            metadata={"description": f"Confirmed identity: {cand_id} SAME_AS {grounded_id}"},
        )

        return self._tx_manager.commit(tx)

    @staticmethod
    def _map_predicate_to_edge_type(
        predicate: str | PredicateRef | LatentRelationRef,
    ) -> HCIREdgeType:
        """Maps semantic relational predicate strings to formal HCIREdgeTypes."""
        if isinstance(predicate, PredicateRef):
            name = predicate.name
        elif isinstance(predicate, LatentRelationRef):
            name = predicate.id
        else:
            name = str(predicate)
        pred = name.upper()
        if pred in ("IN", "INSIDE", "CONTAINED_IN", "LOCATED_IN"):
            return HCIREdgeType.LOCATED_IN
        elif pred in ("NEAR", "BESIDE", "CLOSE_TO", "LOCATED_NEAR"):
            return HCIREdgeType.NEAR
        elif pred in ("ON", "ABOVE", "ON_TOP_OF"):
            return HCIREdgeType.ABOVE
        elif pred in ("BELOW", "UNDER"):
            return HCIREdgeType.BELOW
        elif pred in ("TOUCHING", "CONTACT"):
            return HCIREdgeType.TOUCHING
        elif pred in ("OVERLAPPING",):
            return HCIREdgeType.OVERLAPPING
        elif pred in ("SAME_AS", "IDENTICAL"):
            return HCIREdgeType.SAME_AS
        elif pred in ("INSTANCE_OF", "TYPE_OF"):
            return HCIREdgeType.INSTANCE_OF
        return HCIREdgeType.CORRELATES_WITH
