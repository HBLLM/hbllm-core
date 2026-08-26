"""A20 Relational Generalization & Analogical Transfer Benchmark Suite (21 Scenarios).

Evaluates RelationalSchema representation, grounded induction, Gentner Structure Mapping,
constraint-gated negative transfer rejection, zero-shot action synthesis, A18/A19 integration,
schema Bayesian reinforcement/specialization, and the Flagship Cross-Domain Industrial Transfer Trial.
"""

from __future__ import annotations

import sys

from hbllm.brain.decision import CandidateKind, DecisionCandidate, DecisionEngine, DecisionType
from hbllm.brain.simulation import MentalSandbox
from hbllm.brain.transfer import (
    AnalogicalTransferEngine,
    GroundedEpisode,
    MappingStatus,
    RelationalSchema,
    RelationalSchemaExtractor,
    SchemaConstraint,
    SchemaLifecycleStatus,
    SchemaRelation,
    SchemaRole,
    StructureMappingEngine,
)
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode

# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: A20-01 Relational Schema Representation
# ═══════════════════════════════════════════════════════════════════════════


class TestRelationalSchemaRepresentation:
    """RelationalSchema encapsulates roles, relations, constraints, and action templates."""

    def test_schema_role_and_relation_definition(self) -> None:
        schema = RelationalSchema(
            name="Support-Chain",
            roles=[
                SchemaRole(role_id="Base", type_requirement="physical_entity"),
                SchemaRole(role_id="Payload", type_requirement="physical_entity"),
            ],
            relations=[
                SchemaRelation(source_role="Payload", edge_type="LOCATED_ON", target_role="Base"),
            ],
            constraints=[
                SchemaConstraint(role_id="Base", property_key="geometry", expected_value="flat"),
            ],
        )
        assert len(schema.roles) == 2
        assert schema.confidence > 0.60
        assert schema.is_transferable


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: A20-02 Grounded Schema Extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestGroundedSchemaExtraction:
    """Extracts generalized SupportSchema from a recorded physical stacking episode."""

    def test_extract_support_schema_from_stacking_episode(self) -> None:
        extractor = RelationalSchemaExtractor()
        ep = GroundedEpisode(
            action_sequence=[("STACK", {"item_id": "cup_1", "base_id": "box_1"})],
            observed_consequences=["cup_1_stable_on_box_1"],
            is_success=True,
        )

        schema = extractor.extract_support_schema(ep)
        assert schema.name == "Support-Chain"
        assert len(schema.roles) == 2
        assert any(c.property_key == "geometry" for c in schema.constraints)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: A20-03 Structure Mapping 1-to-1 Alignment
# ═══════════════════════════════════════════════════════════════════════════


class TestStructureMappingAlignment:
    """StructureMappingEngine enforces strict 1-to-1 entity-to-role binding."""

    def test_exact_1_to_1_role_binding(self) -> None:
        schema = RelationalSchema(
            name="Support-Chain",
            roles=[SchemaRole(role_id="Base"), SchemaRole(role_id="Payload")],
            constraints=[
                SchemaConstraint(role_id="Base", property_key="geometry", expected_value="flat")
            ],
        )
        target_graph = CognitiveGraph()
        platform = PhysicalEntityNode(
            id="platform_x", entity_type="platform", properties={"geometry": "flat"}
        )
        block = PhysicalEntityNode(id="block_y", entity_type="block", properties={})
        target_graph.add_node(platform)
        target_graph.add_node(block)

        mapper = StructureMappingEngine()
        res = mapper.map_schema_to_target(schema, target_graph)

        assert res.status == MappingStatus.APPLICABLE
        assert res.role_bindings["Base"] == "platform_x"
        assert res.role_bindings["Payload"] == "block_y"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: A20-04 Relational Systematicity
# ═══════════════════════════════════════════════════════════════════════════


class TestRelationalSystematicity:
    """Connected higher-order relation chains receive systematicity bonus over unary pairs."""

    def test_connected_chains_receive_systematicity_bonus(self) -> None:
        schema_2tier = RelationalSchema(
            roles=[SchemaRole(role_id="R1"), SchemaRole(role_id="R2")],
            relations=[SchemaRelation(source_role="R1", edge_type="LOCATED_ON", target_role="R2")],
        )
        schema_3tier = RelationalSchema(
            roles=[SchemaRole(role_id="R1"), SchemaRole(role_id="R2"), SchemaRole(role_id="R3")],
            relations=[
                SchemaRelation(source_role="R1", edge_type="LOCATED_ON", target_role="R2"),
                SchemaRelation(source_role="R2", edge_type="LOCATED_ON", target_role="R3"),
            ],
        )

        target = CognitiveGraph()
        n1 = PhysicalEntityNode(id="n1", entity_type="node")
        n2 = PhysicalEntityNode(id="n2", entity_type="node")
        n3 = PhysicalEntityNode(id="n3", entity_type="node")
        target.add_node(n1)
        target.add_node(n2)
        target.add_node(n3)

        mapper = StructureMappingEngine()
        res2 = mapper.map_schema_to_target(schema_2tier, target)
        res3 = mapper.map_schema_to_target(schema_3tier, target)

        assert res3.systematicity_score > res2.systematicity_score


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: A20-05 Attribute Invariance
# ═══════════════════════════════════════════════════════════════════════════


class TestAttributeInvariance:
    """Transfers successfully across surface differences in color, size, and material."""

    def test_transfers_across_color_size_and_material_differences(self) -> None:
        schema = RelationalSchema(
            name="Support-Chain",
            roles=[SchemaRole(role_id="Base"), SchemaRole(role_id="Payload")],
            constraints=[
                SchemaConstraint(role_id="Base", property_key="geometry", expected_value="flat")
            ],
        )
        target_graph = CognitiveGraph()
        steel_plate = PhysicalEntityNode(
            id="steel_plate",
            entity_type="plate",
            properties={
                "color": "dark_grey",
                "material": "steel",
                "size": "large",
                "geometry": "flat",
            },
        )
        plastic_sensor = PhysicalEntityNode(
            id="plastic_sensor",
            entity_type="sensor",
            properties={"color": "bright_orange", "material": "plastic", "size": "tiny"},
        )
        target_graph.add_node(steel_plate)
        target_graph.add_node(plastic_sensor)

        mapper = StructureMappingEngine()
        res = mapper.map_schema_to_target(schema, target_graph)

        assert res.status == MappingStatus.APPLICABLE
        assert res.role_bindings["Base"] == "steel_plate"
        assert res.role_bindings["Payload"] == "plastic_sensor"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: A20-06 Cross-Domain Support Transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossDomainSupportTransfer:
    """Transfers tabletop stacking experience onto a mechanical platform domain."""

    def test_tabletop_stacking_to_mechanical_platform_transfer(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        bed = PhysicalEntityNode(
            id="work_bed", entity_type="machine_bed", properties={"geometry": "flat"}
        )
        motor = PhysicalEntityNode(id="motor_assembly", entity_type="motor", properties={})
        target_graph.add_node(bed)
        target_graph.add_node(motor)

        transfer, _, mapping = engine.transfer_schema_to_domain(schema, target_graph)

        assert transfer is not None
        assert mapping.status == MappingStatus.APPLICABLE
        assert transfer.role_mapping["Base"] == "work_bed"
        assert transfer.role_mapping["Payload"] == "motor_assembly"
        assert transfer.candidate_actions[0] == (
            "STACK",
            {"item_id": "motor_assembly", "base_id": "work_bed"},
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: A20-07 Cross-Domain Containment Transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossDomainContainmentTransfer:
    """Transfers box containment experience onto an industrial bin domain."""

    def test_box_containment_to_industrial_bin_transfer(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_containment_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        bin_obj = PhysicalEntityNode(
            id="storage_hopper", entity_type="hopper", properties={"is_closed": False}
        )
        valve = PhysicalEntityNode(id="valve_part", entity_type="part", properties={})
        target_graph.add_node(bin_obj)
        target_graph.add_node(valve)

        transfer, _, mapping = engine.transfer_schema_to_domain(schema, target_graph)

        assert transfer is not None
        assert mapping.status == MappingStatus.APPLICABLE
        assert transfer.candidate_actions[0] == (
            "PUT_IN",
            {"item_id": "valve_part", "container_id": "storage_hopper"},
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: A20-08 Cross-Domain Tool Transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossDomainToolTransfer:
    """Transfers stick-pusher tool experience onto a mechanical lever domain."""

    def test_pusher_tool_to_mechanical_lever_transfer(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_tool_use_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        robot = PhysicalEntityNode(id="robot_arm", entity_type="agent", properties={})
        lever = PhysicalEntityNode(id="crowbar", entity_type="tool", properties={"is_rigid": True})
        crate = PhysicalEntityNode(id="heavy_crate", entity_type="crate", properties={})
        target_graph.add_node(robot)
        target_graph.add_node(lever)
        target_graph.add_node(crate)

        transfer, _, mapping = engine.transfer_schema_to_domain(schema, target_graph)

        assert transfer is not None
        assert mapping.status == MappingStatus.APPLICABLE
        assert transfer.role_mapping["Tool"] == "crowbar"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: A20-09 Negative Transfer Rejection: Curved Base
# ═══════════════════════════════════════════════════════════════════════════


class TestNegativeTransferCurvedBase:
    """Rejects support transfer when target base violates the flatness constraint (convex/curved)."""

    def test_rejects_support_transfer_when_target_base_is_convex(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        curved_pipe = PhysicalEntityNode(
            id="pipe_1", entity_type="pipe", properties={"geometry": "convex"}
        )
        tool = PhysicalEntityNode(id="wrench", entity_type="tool", properties={})
        target_graph.add_node(curved_pipe)
        target_graph.add_node(tool)

        transfer, cond_hyp, mapping = engine.transfer_schema_to_domain(schema, target_graph)

        assert transfer is None
        assert mapping.status == MappingStatus.REJECTED
        assert any("geometry=flat" in v for v in mapping.violated_constraints)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: A20-10 Negative Transfer Rejection: Closed Container
# ═══════════════════════════════════════════════════════════════════════════


class TestNegativeTransferClosedContainer:
    """Rejects containment transfer when target bin is sealed/closed."""

    def test_rejects_containment_transfer_when_target_bin_is_closed(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_containment_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        sealed_safe = PhysicalEntityNode(
            id="safe_box", entity_type="safe", properties={"is_closed": True}
        )
        key = PhysicalEntityNode(id="keycard", entity_type="key", properties={})
        target_graph.add_node(sealed_safe)
        target_graph.add_node(key)

        transfer, _, mapping = engine.transfer_schema_to_domain(schema, target_graph)

        assert transfer is None
        assert mapping.status == MappingStatus.REJECTED
        assert any("is_closed" in v for v in mapping.violated_constraints)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: A20-11 Analogical Epistemic Provenance
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalogicalEpistemicProvenance:
    """Transferred projections carry provenance_source = 'ANALOGICAL_TRANSFER'."""

    def test_transferred_relations_carry_analogical_source(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        base = PhysicalEntityNode(id="p1", entity_type="platform", properties={"geometry": "flat"})
        item = PhysicalEntityNode(id="c1", entity_type="component", properties={})
        target_graph.add_node(base)
        target_graph.add_node(item)

        transfer, _, _ = engine.transfer_schema_to_domain(schema, target_graph)
        assert transfer is not None
        assert transfer.provenance_source == "ANALOGICAL_TRANSFER"
        assert transfer.source_schema_name == "Support-Chain"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: A20-12 Zero-Shot Plan Synthesis
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroShotPlanSynthesis:
    """Transferred schema instantiates valid operator action parameters for target entities."""

    def test_schema_synthesizes_executable_actions(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        bed = PhysicalEntityNode(id="anvil", entity_type="anvil", properties={"geometry": "flat"})
        hammer = PhysicalEntityNode(id="sledgehammer", entity_type="hammer", properties={})
        target_graph.add_node(bed)
        target_graph.add_node(hammer)

        transfer, _, _ = engine.transfer_schema_to_domain(schema, target_graph)
        assert transfer is not None
        assert len(transfer.candidate_actions) == 1
        op_name, params = transfer.candidate_actions[0]
        assert op_name == "STACK"
        assert params["item_id"] == "sledgehammer"
        assert params["base_id"] == "anvil"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 13: A20-13 A18 Simulation Verification of Transferred Plan
# ═══════════════════════════════════════════════════════════════════════════


class TestA18SimulationVerification:
    """Synthesized analogical plan simulates successfully in A18 MentalSandbox."""

    def test_transferred_plan_verified_in_mental_sandbox(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        fixture = PhysicalEntityNode(
            id="fixture_base", entity_type="fixture", properties={"geometry": "flat"}
        )
        bracket = PhysicalEntityNode(id="bracket_payload", entity_type="bracket", properties={})
        target_graph.add_node(fixture)
        target_graph.add_node(bracket)

        transfer, _, _ = engine.transfer_schema_to_domain(schema, target_graph)
        assert transfer is not None

        # Verify in A18 MentalSandbox
        sandbox = MentalSandbox()
        branch, results = sandbox.simulate_trajectory(target_graph, transfer.candidate_actions)

        assert len(results) == 1
        assert results[0].is_success
        assert branch.accumulated_risk < 0.10


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 14: A20-14 A19 Decision Engine Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestA19DecisionIntegration:
    """Transferred plan creates a high-utility DecisionCandidate evaluated by A19."""

    def test_transferred_plan_evaluated_by_decision_engine(self) -> None:
        decision_engine = DecisionEngine()
        transferred_cand = DecisionCandidate(
            candidate_kind=CandidateKind.GOAL_ACTION,
            description="Analogical Stack Plan",
            action_sequence=[("STACK", {"item_id": "bracket", "base_id": "fixture"})],
            goal_progress=1.0,
            predicted_risk=0.05,
            action_cost=0.10,
        )

        res = decision_engine.select_best_decision([transferred_cand])
        assert res.decision_type == DecisionType.ACTION
        assert res.selected_candidate == transferred_cand
        assert res.expected_utility > 0.80


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 15: A20-15 Partial Mapping Epistemic State
# ═══════════════════════════════════════════════════════════════════════════


class TestPartialMappingEpistemicState:
    """Missing target role produces ConditionalAnalogicalHypothesis instead of direct execution."""

    def test_missing_role_produces_conditional_hypothesis(self) -> None:
        schema = RelationalSchema(
            name="3-Tier Support",
            roles=[
                SchemaRole(role_id="Base"),
                SchemaRole(role_id="Mid"),
                SchemaRole(role_id="Top"),
            ],
            constraints=[
                SchemaConstraint(role_id="Base", property_key="geometry", expected_value="flat")
            ],
        )
        target_graph = CognitiveGraph()
        p1 = PhysicalEntityNode(id="p1", entity_type="platform", properties={"geometry": "flat"})
        b1 = PhysicalEntityNode(id="b1", entity_type="block", properties={})
        target_graph.add_node(p1)
        target_graph.add_node(b1)
        # Missing third entity!

        engine = AnalogicalTransferEngine()
        transfer, cond_hyp, mapping = engine.transfer_schema_to_domain(schema, target_graph)

        assert transfer is None
        assert cond_hyp is not None
        assert mapping.status == MappingStatus.PARTIALLY_APPLICABLE
        assert len(cond_hyp.missing_roles) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 16: A20-16 Bayesian Schema Reinforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestBayesianSchemaReinforcement:
    """Successful physical transfer outcome increases Bayesian reliability (alpha)."""

    def test_success_increases_alpha_confidence(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())
        initial_conf = schema.confidence

        engine = AnalogicalTransferEngine(extractor=extractor)
        engine.record_transfer_outcome(schema.schema_id, is_success=True)

        assert schema.confidence > initial_conf
        assert schema.alpha_success == 5.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 17: A20-17 Schema Specialization on Physical Contradiction
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaSpecialization:
    """Physical failure adds boundary constraint and updates status to SPECIALIZED."""

    def test_failure_adds_boundary_constraint_and_specializes(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        engine.record_transfer_outcome(
            schema.schema_id,
            is_success=False,
            failed_constraint="requires_high_friction_contact",
        )

        assert schema.beta_failure == 2.0
        assert schema.status == SchemaLifecycleStatus.SPECIALIZED
        assert "requires_high_friction_contact" in schema.specialization_rules


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 18: A20-18 The Flagship Cross-Domain Relational Transfer Trial
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagshipCrossDomainTransfer:
    """The Flagship Acceptance Gate: Learns schema from tabletop toys, encounters

    completely novel industrial machinery workspace with zero vocabulary overlap,
    maps relational structure, synthesizes action plan, verifies in A18, decides in A19,
    and executes physical goal plan.
    """

    def test_tabletop_to_unfamiliar_industrial_machinery_transfer(self) -> None:
        # 1. Source Experience: Learn Support-Chain from tabletop (cup on box)
        extractor = RelationalSchemaExtractor()
        source_ep = GroundedEpisode(
            action_sequence=[("STACK", {"item_id": "toy_cup", "base_id": "toy_box"})],
            observed_consequences=["stable_on_box"],
            is_success=True,
        )
        schema = extractor.extract_support_schema(source_ep)

        # 2. Target Domain: Unfamiliar industrial equipment (Zero vocabulary overlap!)
        target_graph = CognitiveGraph()
        platform = PhysicalEntityNode(
            id="industrial_gantry_bed", entity_type="gantry_bed", properties={"geometry": "flat"}
        )
        rotor = PhysicalEntityNode(
            id="high_torque_turbine_rotor", entity_type="rotor", properties={}
        )
        target_graph.add_node(platform)
        target_graph.add_node(rotor)

        # 3. Transfer Engine matches best schema
        engine = AnalogicalTransferEngine(extractor=extractor)
        transfer, mapping = engine.match_best_schema(target_graph)

        assert transfer is not None
        assert mapping is not None
        assert mapping.status == MappingStatus.APPLICABLE
        assert transfer.role_mapping["Base"] == "industrial_gantry_bed"
        assert transfer.role_mapping["Payload"] == "high_torque_turbine_rotor"

        # 4. A18 Mental Sandbox verification
        sandbox = MentalSandbox()
        branch, sim_results = sandbox.simulate_trajectory(target_graph, transfer.candidate_actions)
        assert len(sim_results) == 1
        assert sim_results[0].is_success

        # 5. A19 Decision Engine selects synthesized plan
        decision_engine = DecisionEngine()
        cand = DecisionCandidate(
            candidate_kind=CandidateKind.GOAL_ACTION,
            description="Execute Transferred Industrial Assembly Plan",
            action_sequence=transfer.candidate_actions,
            goal_progress=1.0,
            predicted_risk=branch.accumulated_risk,
            action_cost=0.1,
        )
        decision = decision_engine.select_best_decision([cand])
        assert decision.decision_type == DecisionType.ACTION
        assert decision.selected_candidate == cand

        # 6. Physical execution feedback reinforces schema
        engine.record_transfer_outcome(schema.schema_id, is_success=True)
        assert schema.confidence > 0.80


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 19: A20-19 Multi-Schema Competition
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiSchemaCompetition:
    """When multiple schemas are stored in library, matches best applicable schema for target."""

    def test_best_matching_schema_selected_among_alternatives(self) -> None:
        extractor = RelationalSchemaExtractor()
        extractor.extract_support_schema(GroundedEpisode())
        extractor.extract_containment_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)

        # Target has open container hopper and a part -> should match ContainmentSchema
        target_graph = CognitiveGraph()
        hopper = PhysicalEntityNode(
            id="hopper", entity_type="hopper", properties={"is_closed": False}
        )
        pellet = PhysicalEntityNode(id="pellet", entity_type="pellet", properties={})
        target_graph.add_node(hopper)
        target_graph.add_node(pellet)

        transfer, mapping = engine.match_best_schema(target_graph)

        assert transfer is not None
        assert transfer.source_schema_name == "Container-Payload"
        assert transfer.candidate_actions[0][0] == "PUT_IN"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 20: A20-20 Zero-LLM Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Structure mapping and relational transfer run with 100% deterministic code and zero neural/LLM imports."""

    def test_zero_llm_imports(self) -> None:
        import subprocess
        import sys

        check_code = """
import sys
import hbllm.brain.transfer

llm_markers = ["openai", "anthropic", "litellm", "langchain", "transformers"]
loaded = set(sys.modules.keys())
for marker in llm_markers:
    assert marker not in loaded, f"LLM module loaded in transfer runtime: {marker}"
"""
        res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
        assert res.returncode == 0, f"Zero-LLM verification failed:\n{res.stderr}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 21: A20-21 Analogical Prediction Before Action
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalogicalPredictionBeforeAction:
    """Transferred schema projects predicted consequences into target domain prior to execution."""

    def test_transferred_schema_predicts_outcome_prior_to_execution(self) -> None:
        extractor = RelationalSchemaExtractor()
        schema = extractor.extract_support_schema(GroundedEpisode())

        engine = AnalogicalTransferEngine(extractor=extractor)
        target_graph = CognitiveGraph()
        platform = PhysicalEntityNode(
            id="gantry", entity_type="gantry", properties={"geometry": "flat"}
        )
        gear = PhysicalEntityNode(id="cog", entity_type="cog", properties={})
        target_graph.add_node(platform)
        target_graph.add_node(gear)

        transfer, _, _ = engine.transfer_schema_to_domain(schema, target_graph)
        assert transfer is not None
        assert len(transfer.projected_predictions) == 1

        pred = transfer.projected_predictions[0]
        assert pred["consequence_type"] == "stable_support"
        assert pred["source_node"] == "cog"
        assert pred["target_node"] == "gantry"
