"""Unit tests for EpistemicSafetyGate.

Validates graph-connectivity-based affirmative claim overrides across:
1. 1-hop direct belief-to-entity edge
2. 2-hop spatial zone containment
3. 3-hop perception chain (Belief -> Evidence -> Observation -> PhysicalEntity)
4. Unrelated contradiction in different entity (zero false positives)
5. Resolved contradiction (permits affirmative execution)
6. Silent/absent context (no-op, preserving governance fail-closed baseline)
7. Perimeter authorization override
8. Language-independent non-English claims and novel phrasings
"""

from __future__ import annotations

from typing import Any

from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    ContradictionNode,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    ObservationNode,
    PhysicalEntityNode,
)
from hbllm.hcir.kernel.governance.epistemic_gate import EpistemicSafetyGate


def test_gate_1_hop_direct_belief_entity_override() -> None:
    """Active contradiction directly linked to target entity overrides affirmative clearance."""
    graph = CognitiveGraph()
    arm = PhysicalEntityNode(id="robot_arm", entity_name="arm", entity_type="robotic_arm")
    graph.upsert_node(arm)

    sensor_belief = BeliefNode(
        id="sensor_human",
        claim="Human detected 0.3m from manipulator axis",
    )
    user_claim = BeliefNode(
        id="user_claim_clear",
        claim="Workspace is verified safe and clear",
    )
    graph.upsert_node(sensor_belief)
    graph.upsert_node(user_claim)

    # 1-hop edge to entity
    graph.add_edge(
        HCIREdge(
            sources=[sensor_belief.id],
            targets=[arm.id],
            edge_type=HCIREdgeType.APPLIES_TO,
        )
    )

    contra = ContradictionNode(
        id="contra_arm_clearance",
        claim_a_id=sensor_belief.id,
        claim_b_id=user_claim.id,
        resolution_status="unresolved",
    )
    graph.upsert_node(contra)

    gate = EpistemicSafetyGate(graph=graph)
    affirmative_context: dict[str, Any] = {
        "workspace_cleared": True,
        "human_in_workspace": False,
    }

    overridden = gate.evaluate_overrides(
        target="arm",
        capability_name="actuator_control",
        context=affirmative_context,
    )

    assert overridden["workspace_cleared"] is False
    assert overridden["human_in_workspace"] is True
    assert overridden["epistemic_contradiction_active"] is True
    assert "contra_arm_clearance" in overridden["epistemic_conflict_ids"]


def test_gate_2_hop_spatial_zone_containment_override() -> None:
    """Contradiction linked to workspace zone overrides actuator located in that zone."""
    graph = CognitiveGraph()
    arm = PhysicalEntityNode(id="robot_arm", entity_name="arm", entity_type="robotic_arm")
    zone = PhysicalEntityNode(id="bay_1", entity_name="assembly_bay", entity_type="zone")
    graph.upsert_node(arm)
    graph.upsert_node(zone)

    # Arm is located in bay_1
    graph.add_edge(
        HCIREdge(
            sources=[arm.id],
            targets=[zone.id],
            edge_type=HCIREdgeType.LOCATED_IN,
        )
    )

    # Belief about bay_1
    hazard_belief = BeliefNode(id="bay_hazard", claim="Hazardous presence in zone")
    user_belief = BeliefNode(id="bay_clear", claim="Zone is safe")
    graph.upsert_node(hazard_belief)
    graph.upsert_node(user_belief)

    # Belief is located in bay_1
    graph.add_edge(
        HCIREdge(
            sources=[hazard_belief.id],
            targets=[zone.id],
            edge_type=HCIREdgeType.LOCATED_IN,
        )
    )

    contra = ContradictionNode(
        id="contra_bay",
        claim_a_id=hazard_belief.id,
        claim_b_id=user_belief.id,
        resolution_status="unresolved",
    )
    graph.upsert_node(contra)

    gate = EpistemicSafetyGate(graph=graph, max_hops=4)
    res = gate.evaluate_overrides(
        target="robot_arm",
        capability_name="actuator_control",
        context={"workspace_cleared": True, "human_in_workspace": False},
    )

    assert res["workspace_cleared"] is False
    assert res["human_in_workspace"] is True
    assert res["epistemic_contradiction_active"] is True


def test_gate_3_hop_perception_chain_override() -> None:
    """Belief -> Evidence -> Observation -> PhysicalEntity multi-hop path."""
    graph = CognitiveGraph()
    arm = PhysicalEntityNode(id="robot_arm", entity_name="arm", entity_type="robotic_arm")
    obs = ObservationNode(id="obs_depth_cam", modality="visual")
    evi = EvidenceNode(id="evi_depth_anomaly")
    bel_a = BeliefNode(id="bel_depth_human", claim="Depth map indicates human silhouette")
    bel_b = BeliefNode(id="bel_depth_empty", claim="Depth map empty")

    graph.upsert_node(arm)
    graph.upsert_node(obs)
    graph.upsert_node(evi)
    graph.upsert_node(bel_a)
    graph.upsert_node(bel_b)

    # Observation IDENTIFIES entity
    graph.add_edge(HCIREdge(sources=[obs.id], targets=[arm.id], edge_type=HCIREdgeType.IDENTIFIES))
    # Evidence DERIVED_FROM Observation
    graph.add_edge(
        HCIREdge(sources=[obs.id], targets=[evi.id], edge_type=HCIREdgeType.DERIVED_FROM)
    )
    # Belief SUPPORTS Evidence
    graph.add_edge(HCIREdge(sources=[bel_a.id], targets=[evi.id], edge_type=HCIREdgeType.SUPPORTS))

    contra = ContradictionNode(
        id="contra_depth",
        claim_a_id=bel_a.id,
        claim_b_id=bel_b.id,
    )
    graph.upsert_node(contra)

    gate = EpistemicSafetyGate(graph=graph, max_hops=4)
    res = gate.evaluate_overrides(
        target="arm",
        capability_name="actuator_control",
        context={"workspace_cleared": True},
    )

    assert res["workspace_cleared"] is False
    assert res["epistemic_contradiction_active"] is True


def test_gate_unrelated_contradiction_does_not_override() -> None:
    """Contradiction on unrelated entity does not affect actuator affirmative clearance."""
    graph = CognitiveGraph()
    arm = PhysicalEntityNode(id="robot_arm", entity_name="arm", entity_type="robotic_arm")
    weather_sensor = PhysicalEntityNode(
        id="weather_station", entity_name="weather", entity_type="sensor"
    )
    graph.upsert_node(arm)
    graph.upsert_node(weather_sensor)

    b1 = BeliefNode(id="rain_yes", claim="Rain detected")
    b2 = BeliefNode(id="rain_no", claim="No rain")
    graph.upsert_node(b1)
    graph.upsert_node(b2)

    graph.add_edge(
        HCIREdge(sources=[b1.id], targets=[weather_sensor.id], edge_type=HCIREdgeType.APPLIES_TO)
    )

    contra = ContradictionNode(id="contra_rain", claim_a_id=b1.id, claim_b_id=b2.id)
    graph.upsert_node(contra)

    gate = EpistemicSafetyGate(graph=graph)
    context = {"workspace_cleared": True, "human_in_workspace": False}
    res = gate.evaluate_overrides(
        target="robot_arm",
        capability_name="actuator_control",
        context=context,
    )

    # Unrelated contradiction does NOT touch context
    assert res["workspace_cleared"] is True
    assert res["human_in_workspace"] is False
    assert "epistemic_contradiction_active" not in res


def test_gate_resolved_contradiction_does_not_override() -> None:
    """Resolved contradiction permits affirmative clearance to remain active."""
    graph = CognitiveGraph()
    arm = PhysicalEntityNode(id="robot_arm", entity_name="arm", entity_type="robotic_arm")
    graph.upsert_node(arm)

    b1 = BeliefNode(id="b1", claim="presence detected")
    b2 = BeliefNode(id="b2", claim="presence absent")
    graph.upsert_node(b1)
    graph.upsert_node(b2)
    graph.add_edge(HCIREdge(sources=[b1.id], targets=[arm.id], edge_type=HCIREdgeType.APPLIES_TO))

    contra = ContradictionNode(
        id="contra_resolved",
        claim_a_id=b1.id,
        claim_b_id=b2.id,
        resolution_status="resolved",  # Already resolved
    )
    graph.upsert_node(contra)

    gate = EpistemicSafetyGate(graph=graph)
    context = {"workspace_cleared": True}
    res = gate.evaluate_overrides(
        target="arm",
        capability_name="actuator_control",
        context=context,
    )

    assert res["workspace_cleared"] is True
    assert "epistemic_contradiction_active" not in res


def test_gate_silent_context_is_noop() -> None:
    """When context has no affirmative clearance flags, gate performs no work."""
    graph = CognitiveGraph()
    arm = PhysicalEntityNode(id="robot_arm", entity_name="arm")
    graph.upsert_node(arm)

    b1 = BeliefNode(id="b1", claim="hazard")
    b2 = BeliefNode(id="b2", claim="clear")
    graph.upsert_node(b1)
    graph.upsert_node(b2)
    graph.add_edge(HCIREdge(sources=[b1.id], targets=[arm.id], edge_type=HCIREdgeType.APPLIES_TO))
    graph.upsert_node(ContradictionNode(id="c1", claim_a_id=b1.id, claim_b_id=b2.id))

    gate = EpistemicSafetyGate(graph=graph)
    context: dict[str, Any] = {"unrelated_key": "some_value"}
    res = gate.evaluate_overrides(
        target="arm",
        capability_name="actuator_control",
        context=context,
    )

    # Returns untouched dict
    assert res == context


def test_gate_perimeter_authorization_override() -> None:
    """Contradiction on perimeter barrier overrides authorized=True."""
    graph = CognitiveGraph()
    door = PhysicalEntityNode(id="front_door", entity_name="front_door", entity_type="door")
    graph.upsert_node(door)

    tamper_belief = BeliefNode(id="tamper_detected", claim="Tamper sensor triggered on lock")
    user_cred = BeliefNode(id="user_valid_cred", claim="Badge scan valid")
    graph.upsert_node(tamper_belief)
    graph.upsert_node(user_cred)

    graph.add_edge(
        HCIREdge(sources=[tamper_belief.id], targets=[door.id], edge_type=HCIREdgeType.APPLIES_TO)
    )

    graph.upsert_node(
        ContradictionNode(
            id="contra_door",
            claim_a_id=tamper_belief.id,
            claim_b_id=user_cred.id,
        )
    )

    gate = EpistemicSafetyGate(graph=graph)
    res = gate.evaluate_overrides(
        target="front_door",
        capability_name="perimeter_control",
        context={"authorized": True},
    )

    assert res["authorized"] is False
    assert res["epistemic_contradiction_active"] is True


def test_gate_language_invariance() -> None:
    """Non-English claims (Sinhala/Tamil) and novel phrasings are handled identically via graph topology."""
    graph = CognitiveGraph()
    arm = PhysicalEntityNode(id="robot_arm", entity_name="arm", entity_type="robotic_arm")
    graph.upsert_node(arm)

    # Sinhala claim: "බාධාවක් හමු විය" (Obstacle found)
    si_belief = BeliefNode(id="si_b1", claim="බාධාවක් හමු විය")
    # Sinhala claim: "ප්‍රදේශය ආරක්ෂිතයි" (Area is safe)
    si_b2 = BeliefNode(id="si_b2", claim="ප්‍රදේශය ආරක්ෂිතයි")
    graph.upsert_node(si_belief)
    graph.upsert_node(si_b2)

    graph.add_edge(
        HCIREdge(sources=[si_belief.id], targets=[arm.id], edge_type=HCIREdgeType.APPLIES_TO)
    )

    graph.upsert_node(
        ContradictionNode(
            id="contra_si",
            claim_a_id=si_belief.id,
            claim_b_id=si_b2.id,
        )
    )

    gate = EpistemicSafetyGate(graph=graph)
    res = gate.evaluate_overrides(
        target="arm",
        capability_name="actuator_control",
        context={"workspace_cleared": True},
    )

    # Zero English keywords in claims, yet overridden cleanly
    assert res["workspace_cleared"] is False
    assert res["epistemic_contradiction_active"] is True
