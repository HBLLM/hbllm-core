"""Unit and integration tests for Milestone A23.5 Scientific Battery (E2, E3, E4)."""

from __future__ import annotations

import pytest

from hbllm.brain.transfer.mapper import MappingStatus
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode
from plugins.developmental_adapter.a20_transfer_bridge import A20RelationalTransferBridge
from plugins.developmental_adapter.cohorts import (
    ActiveDevelopmentalHCIRCohort,
    NeuralLearnerCohort,
    PassiveDevelopmentalHCIRCohort,
    detect_target_variable,
)
from plugins.developmental_adapter.environment import BabyWorldEnvironment

# ═══════════════════════════════════════════════════════════════════════════
# A23.5-E2: Causal Invariance Across Multi-Modal Confounders
# ═══════════════════════════════════════════════════════════════════════════


class TestA23E2CausalInvariance:
    """Evaluates causal variable invariance under dynamic orthogonal confounders."""

    @pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_active_cohort_converges_on_mass_invariant(self, mode: int):
        env = BabyWorldEnvironment(seed=100 + mode)
        env.reset("randomized_confounded_world")
        env._setup_randomized_confounded_world(mode_override=mode)

        cohort = ActiveDevelopmentalHCIRCohort(seed=100 + mode)
        res = cohort.run_causal_discovery_trial(env=env, max_interventions=10)

        # Active cohort must ALWAYS identify mass_sensation regardless of superficial mode
        assert res.identified_causal_rule is True
        assert res.true_causal_variable == "mass_sensation"
        assert res.interventions_to_discovery <= 2
        assert res.level2_unseen_entities_accuracy == 1.0
        assert res.level3_unseen_world_accuracy == 1.0

    def test_neural_learner_trapped_by_correlation(self):
        env = BabyWorldEnvironment(seed=42)
        env.reset("randomized_confounded_world")
        env._setup_randomized_confounded_world(mode_override=0)

        neural_cohort = NeuralLearnerCohort(seed=42)
        res = neural_cohort.run_causal_discovery_trial(env=env, max_interventions=20)

        # Neural baseline remains trapped in observational correlation
        assert res.identified_causal_rule is False
        assert res.interventions_to_discovery == 20
        assert res.level2_unseen_entities_accuracy <= 0.70


# ═══════════════════════════════════════════════════════════════════════════
# A23.5-E3: Novel Physical Causal Mechanisms
# ═══════════════════════════════════════════════════════════════════════════


class TestA23E3NovelCausalMechanisms:
    """Evaluates domain-neutral discovery across diverse physical generative laws."""

    def test_mass_mechanism_discovery(self):
        env = BabyWorldEnvironment(seed=42)
        env.reset("mass_confounded_world")
        assert detect_target_variable(env) == "mass_sensation"

        cohort = ActiveDevelopmentalHCIRCohort(seed=42)
        res = cohort.run_causal_discovery_trial(env=env, max_interventions=10)
        assert res.identified_causal_rule is True
        assert res.true_causal_variable == "mass_sensation"
        assert res.level3_unseen_world_accuracy == 1.0

    def test_friction_mechanism_discovery(self):
        env = BabyWorldEnvironment(seed=42)
        env.reset("friction_confounded_world")
        assert detect_target_variable(env) == "surface_friction"

        cohort = ActiveDevelopmentalHCIRCohort(seed=42)
        res = cohort.run_causal_discovery_trial(env=env, max_interventions=10)
        assert res.identified_causal_rule is True
        assert res.true_causal_variable == "surface_friction"
        assert res.level3_unseen_world_accuracy == 1.0

    def test_force_mechanism_discovery(self):
        env = BabyWorldEnvironment(seed=42)
        env.reset("force_confounded_world")
        assert detect_target_variable(env) == "static_threshold"

        cohort = ActiveDevelopmentalHCIRCohort(seed=42)
        res = cohort.run_causal_discovery_trial(env=env, max_interventions=10)
        assert res.identified_causal_rule is True
        assert res.true_causal_variable == "static_threshold"
        assert res.level3_unseen_world_accuracy == 1.0

    def test_aperture_clearance_mechanism_discovery(self):
        env = BabyWorldEnvironment(seed=42)
        env.reset("aperture_confounded_world")
        assert detect_target_variable(env) == "clearance_diameter"

        cohort = ActiveDevelopmentalHCIRCohort(seed=42)
        res = cohort.run_causal_discovery_trial(env=env, max_interventions=10)
        assert res.identified_causal_rule is True
        assert res.true_causal_variable == "clearance_diameter"
        assert res.level3_unseen_world_accuracy == 1.0

    def test_passive_vs_active_efficiency_contrast(self):
        env = BabyWorldEnvironment(seed=123)
        env.reset("friction_confounded_world")

        active_cohort = ActiveDevelopmentalHCIRCohort(seed=123)
        active_res = active_cohort.run_causal_discovery_trial(env=env, max_interventions=20)

        passive_cohort = PassiveDevelopmentalHCIRCohort(seed=123)
        passive_res = passive_cohort.run_causal_discovery_trial(env=env, max_interventions=20)

        assert active_res.identified_causal_rule is True
        assert active_res.interventions_to_discovery <= 2
        assert passive_res.identified_causal_rule is True


# ═══════════════════════════════════════════════════════════════════════════
# A23.5-E4: Compositional Causal Transfer to A20 Relational Structures
# ═══════════════════════════════════════════════════════════════════════════


class TestA23E4A20RelationalTransfer:
    """Evaluates lifting developmental schemas into A20 and zero-shot analogical transfer."""

    def test_developmental_containment_transfer_to_hopper(self):
        bridge = A20RelationalTransferBridge()

        # 1. Lift developmental containment schema
        schema = bridge.lift_containment_schema()
        assert schema.name == "Developmental-Containment-Transport"
        assert schema.is_transferable is True

        # 2. Target domain: Industrial storage hopper and mechanical valve
        target_graph = CognitiveGraph()
        target_graph.add_node(
            PhysicalEntityNode(
                id="storage_hopper_01",
                entity_type="container",
                properties={"is_closed": False, "is_mobile": True},
            )
        )
        target_graph.add_node(
            PhysicalEntityNode(
                id="valve_part_42",
                entity_type="physical_entity",
                properties={"mass": 0.8},
            )
        )

        # 3. Transfer schema zero-shot
        result = bridge.transfer_to_target_domain(schema, target_graph)
        assert result["is_applicable"] is True
        assert result["mapping_status"] == MappingStatus.APPLICABLE.value
        assert result["role_mapping"]["Container"] == "storage_hopper_01"
        assert result["role_mapping"]["Payload"] == "valve_part_42"
        assert len(result["candidate_actions"]) >= 1
        assert result["candidate_actions"][0][0] == "PUT_IN"

    def test_developmental_containment_negative_transfer_rejection(self):
        bridge = A20RelationalTransferBridge()
        schema = bridge.lift_containment_schema()

        # Target domain with closed container (constraint violation)
        target_graph = CognitiveGraph()
        target_graph.add_node(
            PhysicalEntityNode(
                id="sealed_vault",
                entity_type="container",
                properties={"is_closed": True},  # Violates is_closed == False
            )
        )
        target_graph.add_node(
            PhysicalEntityNode(
                id="gold_ingot",
                entity_type="physical_entity",
                properties={"mass": 2.0},
            )
        )

        result = bridge.transfer_to_target_domain(schema, target_graph)
        assert result["is_applicable"] is False
        assert result["is_rejected"] is True
        assert len(result["violations"]) >= 1
        assert "is_closed" in result["violations"][0]

    def test_developmental_tool_transfer_to_mechanical_lever(self):
        bridge = A20RelationalTransferBridge()
        schema = bridge.lift_tool_reach_schema()
        assert schema.name == "Developmental-Tool-Reach-Extension"

        # Target domain: Robotic arm, crowbar, and heavy crate
        target_graph = CognitiveGraph()
        target_graph.add_node(
            PhysicalEntityNode(id="robot_manipulator", entity_type="agent", properties={})
        )
        target_graph.add_node(
            PhysicalEntityNode(
                id="crowbar_tool",
                entity_type="tool",
                properties={"is_rigid": True, "mass": 1.2},
            )
        )
        target_graph.add_node(
            PhysicalEntityNode(id="heavy_crate", entity_type="physical_entity", properties={})
        )

        result = bridge.transfer_to_target_domain(schema, target_graph)
        assert result["is_applicable"] is True
        assert result["role_mapping"]["Agent"] == "robot_manipulator"
        assert result["role_mapping"]["Tool"] == "crowbar_tool"
        assert result["role_mapping"]["Target"] == "heavy_crate"
        assert result["candidate_actions"][0][0] == "USE_TOOL"

    def test_developmental_causal_rule_transfer_to_pusher(self):
        bridge = A20RelationalTransferBridge()
        causal_rule = {
            "rule_id": "rule_push_friction",
            "action": "PUSH",
            "precondition": {"property": "surface_friction", "operator": "<", "value": 1.0},
            "consequence": "MOVES",
            "empirical_support_count": 8,
        }
        schema = bridge.lift_causal_rule_schema(causal_rule)
        assert "Developmental-Causal-Push" in schema.name

        target_graph = CognitiveGraph()
        target_graph.add_node(PhysicalEntityNode(id="conveyor_arm", entity_type="agent"))
        target_graph.add_node(
            PhysicalEntityNode(
                id="smooth_pallet",
                entity_type="physical_entity",
                properties={"surface_friction": 0.2},
            )
        )

        result = bridge.transfer_to_target_domain(schema, target_graph)
        assert result["is_applicable"] is True
        assert result["role_mapping"]["Agent"] == "conveyor_arm"
        assert result["role_mapping"]["Target"] == "smooth_pallet"
        assert result["candidate_actions"][0][0] == "PUSH"
