from __future__ import annotations

import pytest

from hbllm.hcir.grounding.committer.committer import HCIRCommitter
from hbllm.hcir.grounding.concepts.lifecycle import ConceptPromotionProposal
from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.contracts.beliefs import BeliefState
from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.decisions import GroundingDecision
from hbllm.hcir.grounding.contracts.enums import (
    ClaimOrigin,
    ConceptLifecycle,
    DecisionOutcome,
    Polarity,
)
from hbllm.hcir.grounding.contracts.quality import GroundingQuality
from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedConceptRef,
    GroundedEntityRef,
    LiteralValueRef,
)
from hbllm.hcir.grounding.tracking.operations import (
    EntityReidentificationOp,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.nodes import (
    BeliefNode,
    GroundedConceptNode,
    HCIREdgeType,
    PhysicalEntityNode,
)
from hbllm.hcir.transactions import TransactionStatus
from hbllm.hcir.workspace import HCIRWorkspaceState


class TestHCIRCommitter:
    @pytest.fixture
    def workspace(self) -> HCIRWorkspaceState:
        return HCIRWorkspaceState()

    @pytest.fixture
    def tx_manager(self, workspace: HCIRWorkspaceState) -> TransactionManager:
        return TransactionManager(workspace)

    @pytest.fixture
    def committer(self, tx_manager: TransactionManager) -> HCIRCommitter:
        return HCIRCommitter(tx_manager)

    def test_commit_accepted_relational_decision(
        self,
        committer: HCIRCommitter,
        workspace: HCIRWorkspaceState,
    ) -> None:
        claim = GroundingClaim(
            claim_id="claim_on_1",
            subject_ref=GroundedEntityRef("e_cup_1"),
            predicate_ref="ON",
            object_ref=GroundedEntityRef("e_table_1"),
            polarity=Polarity.AFFIRMATIVE,
            origin=ClaimOrigin.DIRECT_SENSORY,
            evidence_ids=("ev_1", "ev_2"),
            temporal_scope={"start": 0.0, "end": 1.0},
            spatial_scope={"region": "kitchen"},
            generated_by="rel_detector",
            created_at=100.0,
        )

        decision = GroundingDecision(
            claim_id=claim.claim_id,
            outcome=DecisionOutcome.ACCEPT,
            posterior_belief=0.94,
            calibrated_evidence_strength=0.90,
            supporting_evidence_ids=("ev_1", "ev_2"),
            contradictions=(),
            transaction_proposal_id="tx_prop_1",
            decision_rationale="Strong visual and spatial support",
            timestamp=101.0,
        )

        belief = BeliefState(
            belief_id="belief_on_1",
            subject_ref=claim.subject_ref,
            predicate_ref=claim.predicate_ref,
            object_ref=claim.object_ref,
            polarity=Polarity.AFFIRMATIVE,
            posterior_probability=0.94,
            evidence_strength=0.90,
            supporting_evidence_ids=("ev_1", "ev_2"),
            contradicting_evidence_ids=(),
            provenance_claims=(claim.claim_id,),
            revision_count=0,
            last_updated=101.0,
        )

        tx = committer.commit_decision(decision, claim, belief)
        assert tx.status == TransactionStatus.COMMITTED

        # Verify physical entity nodes exist in workspace graph
        assert workspace.graph.has_node("e_cup_1")
        assert isinstance(workspace.graph.get_node("e_cup_1"), PhysicalEntityNode)
        assert workspace.graph.has_node("e_table_1")
        assert isinstance(workspace.graph.get_node("e_table_1"), PhysicalEntityNode)

        # Verify belief node exists
        assert workspace.graph.has_node("belief_on_1")
        assert isinstance(workspace.graph.get_node("belief_on_1"), BeliefNode)

        # Verify relational edge
        edges = list(workspace.graph.all_edges())
        rel_edges = [e for e in edges if e.edge_type == HCIREdgeType.ABOVE]
        assert len(rel_edges) == 1
        assert rel_edges[0].sources == ["e_cup_1"]
        assert rel_edges[0].targets == ["e_table_1"]

        # Verify support edge
        sup_edges = [e for e in edges if e.edge_type == HCIREdgeType.SUPPORTS]
        assert len(sup_edges) == 1
        assert "belief_on_1" in sup_edges[0].sources

    def test_commit_rejected_decision_raises(
        self,
        committer: HCIRCommitter,
    ) -> None:
        claim = GroundingClaim(
            claim_id="claim_rej_1",
            subject_ref=GroundedEntityRef("e_1"),
            predicate_ref="NEAR",
            object_ref=GroundedEntityRef("e_2"),
            polarity=Polarity.AFFIRMATIVE,
            origin=ClaimOrigin.INFERRED_BY_REASONING,
            evidence_ids=(),
            temporal_scope={},
            spatial_scope={},
            generated_by="test",
            created_at=100.0,
        )

        decision = GroundingDecision(
            claim_id=claim.claim_id,
            outcome=DecisionOutcome.REJECT,
            posterior_belief=0.12,
            calibrated_evidence_strength=0.85,
            supporting_evidence_ids=(),
            contradictions=("Physics violation",),
            transaction_proposal_id="tx_rej",
            decision_rationale="Contradicted by physics",
            timestamp=101.0,
        )

        with pytest.raises(ValueError, match="Cannot commit non-accepted decision"):
            committer.commit_decision(decision, claim)

    def test_commit_literal_value_decision(
        self,
        committer: HCIRCommitter,
        workspace: HCIRWorkspaceState,
    ) -> None:
        claim = GroundingClaim(
            claim_id="claim_color_1",
            subject_ref=GroundedEntityRef("e_cup_2"),
            predicate_ref="COLOR",
            object_ref=LiteralValueRef(value="red"),
            polarity=Polarity.AFFIRMATIVE,
            origin=ClaimOrigin.DIRECT_SENSORY,
            evidence_ids=("ev_col_1",),
            temporal_scope={},
            spatial_scope={},
            generated_by="vision_color",
            created_at=100.0,
        )

        decision = GroundingDecision(
            claim_id=claim.claim_id,
            outcome=DecisionOutcome.ACCEPT,
            posterior_belief=0.98,
            calibrated_evidence_strength=0.95,
            supporting_evidence_ids=("ev_col_1",),
            contradictions=(),
            transaction_proposal_id="tx_col_1",
            decision_rationale="High confidence RGB spectrum",
            timestamp=101.0,
        )

        tx = committer.commit_decision(decision, claim)
        assert tx.status == TransactionStatus.COMMITTED
        assert workspace.graph.has_node("e_cup_2")

        # Verify belief statement
        belief_nodes = [n for n in workspace.graph.all_nodes() if isinstance(n, BeliefNode)]
        assert len(belief_nodes) == 1
        assert "COLOR(e_cup_2) = red" in belief_nodes[0].statement

    def test_commit_concept_promotion(
        self,
        committer: HCIRCommitter,
        workspace: HCIRWorkspaceState,
    ) -> None:
        concept = GroundedConcept(
            concept_ref=GroundedConceptRef("concept_cargo_box"),
            lifecycle_state=ConceptLifecycle.CONSOLIDATED,
            prototypes=[{"mass": 10.0, "movable": 1.0}],
            exemplar_entity_ids={"e_box_1", "e_box_2"},
            affordances={"PUSHABLE", "STACKABLE"},
        )

        proposal = ConceptPromotionProposal(
            proposal_id="prop_cargo_1",
            concept_ref=GroundedConceptRef("concept_cargo_box"),
            concept_version=1,
            quality_summary=GroundingQuality.balanced(0.92),
            exemplar_count=2,
            affordances=("PUSHABLE", "STACKABLE"),
            linguistic_aliases={"box": 0.9, "cargo": 0.85},
            rationale="High predictive utility and stable exemplars",
            proposed_at=120.0,
        )

        tx = committer.commit_concept_promotion(proposal, concept)
        assert tx.status == TransactionStatus.COMMITTED

        # Verify GroundedConceptNode
        assert workspace.graph.has_node("concept_cargo_box")
        c_node = workspace.graph.get_node("concept_cargo_box")
        assert isinstance(c_node, GroundedConceptNode)
        assert c_node.feature_prototype == {"mass": 10.0, "movable": 1.0}
        assert "PUSHABLE" in c_node.behavioral_regularities

        # Verify exemplars and INSTANCE_OF edges
        assert workspace.graph.has_node("e_box_1")
        assert workspace.graph.has_node("e_box_2")

        edges = list(workspace.graph.all_edges())
        inst_edges = [e for e in edges if e.edge_type == HCIREdgeType.INSTANCE_OF]
        assert len(inst_edges) == 2
        sources = {e.sources[0] for e in inst_edges}
        assert sources == {"e_box_1", "e_box_2"}
        for e in inst_edges:
            assert e.targets == ["concept_cargo_box"]

    def test_commit_entity_reidentification(
        self,
        committer: HCIRCommitter,
        workspace: HCIRWorkspaceState,
    ) -> None:
        reid_op = EntityReidentificationOp(
            operation_id="reid_op_1",
            confidence=0.94,
            rationale="High visual cosine and spatial proximity",
            candidate_ref=EntityCandidateRef("ecand_entity_99"),
            grounded_ref=GroundedEntityRef("e_persistent_entity_42"),
            similarity_metrics={"visual_cosine": 0.96, "spatial_proximity": 0.91},
            timestamp=150.0,
        )

        tx = committer.commit_entity_reidentification(reid_op)
        assert tx.status == TransactionStatus.COMMITTED

        assert workspace.graph.has_node("ecand_entity_99")
        assert workspace.graph.has_node("e_persistent_entity_42")

        edges = list(workspace.graph.all_edges())
        same_as_edges = [e for e in edges if e.edge_type == HCIREdgeType.SAME_AS]
        assert len(same_as_edges) == 1
        assert same_as_edges[0].sources == ["ecand_entity_99"]
        assert same_as_edges[0].targets == ["e_persistent_entity_42"]
        assert same_as_edges[0].weight == 0.94
