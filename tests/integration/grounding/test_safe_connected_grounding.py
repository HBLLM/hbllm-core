from __future__ import annotations

import time

import pytest

from hbllm.hcir.grounding.concepts.model import GroundedConcept, GroundingQuality
from hbllm.hcir.grounding.contracts.enums import (
    ConceptLifecycle,
    ModalitySource,
    TrafficClass,
)
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef
from hbllm.hcir.grounding.features.feature_extractor import (
    SpatialGridFeatureExtractor,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.nodes import GroundedConceptNode
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge
from hbllm.hcir.world.surprise_engine import SurpriseEngine


def _build_test_bridge() -> tuple[CorticalGroundingBridge, HCIRWorkspaceState, SurpriseEngine]:
    workspace = HCIRWorkspaceState()
    tx_mgr = TransactionManager(workspace)
    surprise_eng = SurpriseEngine(surprise_threshold=0.15)
    bridge = CorticalGroundingBridge(
        workspace=workspace,
        transaction_manager=tx_mgr,
        surprise_engine=surprise_eng,
    )
    return bridge, workspace, surprise_eng


class TestSafeConnectedGrounding:
    """Safe, Bounded, and Connected Grounding Integration Suite."""

    def test_a_200_step_synthetic_loop_bounded_growth(self) -> None:
        """(a) 200-step synthetic loop: graph nodes and edges plateau (no unbounded sprawl)."""
        bridge, workspace, _ = _build_test_bridge()

        entities = [
            {
                "id": "e_box",
                "position": (2.0, 2.0),
                "bbox": (2.0, 2.0, 1.0, 1.0),
                "color": 2,
                "role": "box",
                "size": 1,
            },
            {
                "id": "e_goal",
                "position": (5.0, 5.0),
                "bbox": (5.0, 5.0, 1.0, 1.0),
                "color": 3,
                "role": "goal",
                "size": 1,
            },
        ]

        node_counts: list[int] = []
        edge_counts: list[int] = []

        for step in range(200):
            bridge.process_perception(extracted_entities=entities)
            node_counts.append(workspace.graph.node_count)
            edge_counts.append(workspace.graph.edge_count)

        # After warm-up (step 10), node count and edge count must be completely plateaued
        warmup_nodes = node_counts[10]
        warmup_edges = edge_counts[10]

        final_nodes = node_counts[-1]
        final_edges = edge_counts[-1]

        assert final_nodes == warmup_nodes, f"Nodes grew unbounded: {warmup_nodes} -> {final_nodes}"
        assert final_edges == warmup_edges, f"Edges grew unbounded: {warmup_edges} -> {final_edges}"
        assert final_nodes < 20, f"Expected bounded node count, got {final_nodes}"

    def test_b_permutation_invariance_preserves_candidate_ids(self) -> None:
        """(b) Permuting entity order between frames does not change candidate IDs."""
        bridge, _, _ = _build_test_bridge()

        e1 = {
            "position": (1.0, 1.0),
            "bbox": (1.0, 1.0, 1.0, 1.0),
            "color": 1,
            "role": "agent",
            "size": 1,
        }
        e2 = {
            "position": (4.0, 4.0),
            "bbox": (4.0, 4.0, 2.0, 2.0),
            "color": 2,
            "role": "target",
            "size": 4,
        }
        e3 = {
            "position": (7.0, 7.0),
            "bbox": (7.0, 7.0, 1.0, 1.0),
            "color": 3,
            "role": "wall",
            "size": 1,
        }

        # Frame 1: Order [e1, e2, e3]
        sig1 = bridge.process_perception(extracted_entities=[e1, e2, e3])
        states1 = sig1["sge_entity_states"]

        cand1_for_e1 = next(cid for cid, s in states1.items() if s["color"] == 1)
        cand1_for_e2 = next(cid for cid, s in states1.items() if s["color"] == 2)
        cand1_for_e3 = next(cid for cid, s in states1.items() if s["color"] == 3)

        # Frame 2: Order permuted [e3, e1, e2]
        sig2 = bridge.process_perception(extracted_entities=[e3, e1, e2])
        states2 = sig2["sge_entity_states"]

        cand2_for_e1 = next(cid for cid, s in states2.items() if s["color"] == 1)
        cand2_for_e2 = next(cid for cid, s in states2.items() if s["color"] == 2)
        cand2_for_e3 = next(cid for cid, s in states2.items() if s["color"] == 3)

        assert cand2_for_e1 == cand1_for_e1, "Permutation corrupted candidate ID for e1"
        assert cand2_for_e2 == cand1_for_e2, "Permutation corrupted candidate ID for e2"
        assert cand2_for_e3 == cand1_for_e3, "Permutation corrupted candidate ID for e3"

    def test_c_constant_velocity_motion_yields_zero_surprise_after_warmup(self) -> None:
        """(c) A moving entity with constant velocity yields zero surprise alerts after warm-up."""
        bridge, _, _ = _build_test_bridge()

        # Step 0: Initial position (1.0, 1.0)
        ent_0 = [{"id": "e_mover", "position": (1.0, 1.0), "color": 2, "role": "patroller"}]
        bridge.process_perception(extracted_entities=ent_0)
        bridge.process_motor(proposed_action="MOVE", action_data={"target_id": "e_mover"})

        # Step 1: Moved by (1.0, 0.0) -> position (2.0, 1.0)
        ent_1 = [{"id": "e_mover", "position": (2.0, 1.0), "color": 2, "role": "patroller"}]
        states_1 = bridge.process_perception(extracted_entities=ent_1)["sge_entity_states"]
        bridge.process_assimilation(current_entity_states=states_1)
        # Tracker now has trajectory [(1.0, 1.0), (2.0, 1.0)], velocity = (1.0, 0.0)
        bridge.process_motor(proposed_action="MOVE", action_data={"target_id": "e_mover"})

        # Step 2: Continues moving with constant velocity -> position (3.0, 1.0)
        ent_2 = [{"id": "e_mover", "position": (3.0, 1.0), "color": 2, "role": "patroller"}]
        states_2 = bridge.process_perception(extracted_entities=ent_2)["sge_entity_states"]
        assim_2 = bridge.process_assimilation(current_entity_states=states_2)

        # After warm-up, kinematic forward prediction (2.0 + 1.0 = 3.0) matches perfectly
        assert assim_2["sge_surprise_alerts"] == 0, (
            "Expected zero surprise alerts for constant velocity motion"
        )
        assert assim_2["sge_max_surprise"] == 0.0

    def test_d_surprise_engine_receives_grounding_prediction_error(self) -> None:
        """(d) SurpriseEngine receives grounding prediction error directly (no hasattr guard)."""
        bridge, _, surprise_eng = _build_test_bridge()

        # Step 0: Initial entity
        ent_0 = [{"id": "e_stationary", "position": (5.0, 5.0), "color": 4, "role": "beacon"}]
        bridge.process_perception(extracted_entities=ent_0)
        # Forward predictor predicts entity stays at (5.0, 5.0)
        bridge.process_motor(proposed_action="WAIT", action_data={"target_id": "e_stationary"})

        # Step 1: Sudden teleportation to (10.0, 10.0)
        ent_1 = [{"id": "e_stationary", "position": (10.0, 10.0), "color": 4, "role": "beacon"}]
        states_1 = bridge.process_perception(extracted_entities=ent_1)["sge_entity_states"]
        assim_1 = bridge.process_assimilation(current_entity_states=states_1)

        assert assim_1["sge_surprise_alerts"] > 0
        # Verify SurpriseEngine ledger received the grounding error
        assert "grounding" in surprise_eng._surprise_ledger
        ledger = surprise_eng._surprise_ledger["grounding"]
        assert len(ledger) > 0
        assert ledger[-1] >= surprise_eng.surprise_threshold

    def test_e_concept_promotion_to_grounded_concept_node(self) -> None:
        """(e) A concept with >= N exemplars and quality >= threshold appears as GroundedConceptNode."""
        bridge, workspace, _ = _build_test_bridge()

        # Create a consolidated concept with 3 exemplars and high predictive utility
        concept_ref = GroundedConceptRef("concept_promotable")
        concept = GroundedConcept(
            concept_ref=concept_ref,
            lifecycle_state=ConceptLifecycle.CONSOLIDATED,
            grounding_quality=GroundingQuality.balanced(0.85),
            exemplar_entity_ids={"ex_1", "ex_2", "ex_3"},
            affordances={"pushable", "navigable"},
        )
        bridge.concept_store.register_concept(concept)

        # Trigger perception cycle
        bridge.process_perception(extracted_entities=[])

        # Assert node committed into CognitiveGraph
        assert workspace.graph.has_node("concept_promotable"), (
            "Promoted concept node was not found in CognitiveGraph"
        )
        node = workspace.graph.get_node("concept_promotable")
        assert isinstance(node, GroundedConceptNode)
        assert node.concept_name == "concept_promotable"
        assert node.exemplar_count == 3
        assert "pushable" in node.behavioral_regularities

    def test_f_translation_invariance_assigns_same_concept_ids(self) -> None:
        """(f) Translation of the whole scene by (+5, +5) assigns the same concept IDs."""
        extractor = SpatialGridFeatureExtractor()

        now = time.time()
        mono = time.monotonic()

        # Object at (2, 2)
        env_original = ObservationEnvelope(
            observation_id="obs_orig",
            source_modality=ModalitySource.VISION,
            source_channel_id="grid_cam",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=1,
            wall_timestamp=now,
            monotonic_timestamp=mono,
            payload_ref="hcir://cam/orig",
            metadata={
                "entities": [
                    {
                        "id": "e_box",
                        "position": (2.0, 2.0),
                        "bbox": (2.0, 2.0, 3.0, 4.0),
                        "color": 5,
                        "size": 12,
                    }
                ]
            },
        )

        # Identical object translated by (+5, +5) to (7, 7)
        env_translated = ObservationEnvelope(
            observation_id="obs_trans",
            source_modality=ModalitySource.VISION,
            source_channel_id="grid_cam",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=2,
            wall_timestamp=now,
            monotonic_timestamp=mono,
            payload_ref="hcir://cam/trans",
            metadata={
                "entities": [
                    {
                        "id": "e_box_trans",
                        "position": (7.0, 7.0),
                        "bbox": (7.0, 7.0, 3.0, 4.0),
                        "color": 5,
                        "size": 12,
                    }
                ]
            },
        )

        feats_orig = extractor.extract(env_original)
        feats_trans = extractor.extract(env_translated)

        assert len(feats_orig) == 1
        assert len(feats_trans) == 1

        # The dense feature embeddings must be identical (translation invariant)
        assert feats_orig[0].embedding == pytest.approx(feats_trans[0].embedding, abs=1e-5)

        # Test through ConceptLearner: both must be assigned to the exact same concept ID
        from hbllm.hcir.grounding.concepts.concept_learner import ConceptLearner
        from hbllm.hcir.grounding.concepts.store import WorkingConceptStore

        store = WorkingConceptStore(similarity_threshold=0.5)
        learner = ConceptLearner(concept_store=store)

        assign_1 = learner.learn(feats_orig[0], exemplar_entity_id="e_box")
        assign_2 = learner.learn(feats_trans[0], exemplar_entity_id="e_box_trans")

        assert assign_1.concept_ref.id == assign_2.concept_ref.id, (
            f"Translation caused concept split: {assign_1.concept_ref.id} vs {assign_2.concept_ref.id}"
        )
