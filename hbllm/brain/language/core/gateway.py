"""HCIR Gateway for A16.

Connects Language SemanticFrames to HCIR cognitive operations:
- Assertions -> Ingested as EvidenceNodes (Language is an evidence modality).
- Queries -> Evaluated against A13 World Model & A11 Epistemics, returning CognitiveEpistemicState.
- Commands -> Converted to GoalNode / Intent proposals for A12 Execution & Planning.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.language.core.epistemic_policy import CognitiveEpistemicState
from hbllm.brain.language.core.semantic_frame import (
    GroundedSemanticFrame,
    ThematicRole,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    EvidenceNode,
    GoalNode,
    GroundedConceptNode,
    HCIRNodeType,
    PhysicalEntityNode,
)

logger = logging.getLogger(__name__)


class HCIRGateway:
    """Bridges grounded linguistic frames with HCIR cognition.

    Usage::

        gateway = HCIRGateway(graph)
        # Assertions
        evidence_node = gateway.process_assertion(grounded_frame)

        # Queries
        epistemic_state = gateway.process_query(grounded_frame)

        # Commands
        goal_node = gateway.process_command(grounded_frame)
    """

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph

    # ── Assertion: Language as Evidence ───────────────────────────────

    def process_assertion(
        self,
        grounded: GroundedSemanticFrame,
        speaker: str = "human",
    ) -> EvidenceNode:
        """Ingest a linguistic assertion as an EvidenceNode in HCIR.

        Language does NOT directly mutate truth; it is treated as a perception
        modality with explicit linguistic provenance.
        """
        frame = grounded.frame
        subject_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        object_ref = frame.get_role(ThematicRole.LOCATION) or frame.get_role(ThematicRole.PATIENT)

        subject_name: str = subject_ref.concept_name if (subject_ref and subject_ref.concept_name) else "entity"
        target_entity_id = grounded.grounded_entities.get(ThematicRole.THEME) or grounded.grounded_entities.get(ThematicRole.AGENT)

        obj_desc = ""
        if object_ref:
            if object_ref.concept_name:
                obj_desc = object_ref.concept_name
            elif object_ref.properties:
                obj_desc = "_".join(f"{k}_{v}" for k, v in sorted(object_ref.properties.items()))

        claim_str = f"{subject_name} {frame.predicate} {obj_desc}".strip()
        evidence = EvidenceNode(
            claim_id=claim_str,
            methodology=f"language_assertion:{frame.metadata.language}:{claim_str}",
            confidence=0.85,
            source_uri=f"language://{frame.metadata.language}/{speaker}/{frame.metadata.utterance_id}",
            tags=[
                "language_evidence",
                frame.metadata.language,
                speaker,
                frame.predicate,
                subject_name,
            ],
        )
        self._graph.add_node(evidence)

        # If entity exists, update observed property / spatial relation if supported
        if target_entity_id:
            ent = self._graph.get_node(target_entity_id)
            if isinstance(ent, PhysicalEntityNode):
                target_dict = ent.properties if hasattr(ent, "properties") and isinstance(ent.properties, dict) else ent.observed_properties
                if subject_ref and subject_ref.properties:
                    target_dict.update(subject_ref.properties)
                if object_ref and object_ref.properties:
                    target_dict.update(object_ref.properties)
                self._graph.upsert_node(ent)

        logger.debug(
            "HCIRGateway: Ingested language assertion as EvidenceNode %s (%s)",
            evidence.id,
            evidence.methodology,
        )
        return evidence

    # ── Query: Epistemic State Resolution ─────────────────────────────

    def process_query(self, grounded: GroundedSemanticFrame) -> CognitiveEpistemicState:
        """Query HCIR world model and epistemics for a grounded question."""
        frame = grounded.frame
        target_role = ThematicRole.THEME if ThematicRole.THEME in grounded.grounded_entities else ThematicRole.AGENT
        entity_id = grounded.grounded_entities.get(target_role)

        subject_ref = frame.get_role(target_role)
        subject_name: str = subject_ref.concept_name if (subject_ref and subject_ref.concept_name) else "entity"

        # If entity could not be grounded, return insufficient evidence
        if not entity_id:
            return CognitiveEpistemicState(
                target_predicate=frame.predicate,
                target_subject=subject_name,
                is_known=False,
                confidence=0.0,
                uncertainty=1.0,
                support_count=0,
                contradiction_count=0,
            )

        ent_node = self._graph.get_node(entity_id)
        ent_props: dict[str, Any] = {}
        if isinstance(ent_node, PhysicalEntityNode):
            ent_props = ent_node.properties if hasattr(ent_node, "properties") and isinstance(ent_node.properties, dict) else ent_node.observed_properties

        freshness_val = float(ent_props.get("freshness", 1.0))

        # 1. Spatial / Location Query ("Where is the X?")
        if frame.query_target == "location":
            # Search outbound edges from entity
            loc_found: str | None = None
            edge_type_found: str = "located_on"

            for edge in self._graph.edges_from(entity_id):
                for target_id in edge.targets:
                    target_node = self._graph.get_node(target_id)
                    if isinstance(target_node, PhysicalEntityNode):
                        loc_found = target_node.entity_type
                        edge_type_found = edge.edge_type.value
                        break
                if loc_found:
                    break

            # Fallback: check properties dictionary
            if not loc_found:
                loc_found = ent_props.get("location")

            if loc_found:
                return CognitiveEpistemicState(
                    target_predicate=edge_type_found,
                    target_subject=subject_name,
                    target_object=loc_found,
                    confidence=0.96,
                    uncertainty=0.04,
                    support_count=5,
                    freshness=freshness_val,
                    is_known=True,
                    raw_belief_value=loc_found,
                )
            else:
                # Check if concepts provide a default location
                concept_id = grounded.grounded_concepts.get(target_role)
                if concept_id:
                    concept_node = self._graph.get_node(concept_id)
                    if isinstance(concept_node, GroundedConceptNode) and "location" in concept_node.feature_prototype:
                        default_loc = str(concept_node.feature_prototype["location"])
                        return CognitiveEpistemicState(
                            target_predicate="located_on",
                            target_subject=subject_name,
                            target_object=default_loc,
                            confidence=0.65,  # Plausible inference from concept
                            uncertainty=0.35,
                            support_count=1,
                            freshness=freshness_val,
                            is_known=True,
                            raw_belief_value=default_loc,
                        )

                return CognitiveEpistemicState(
                    target_predicate="located_on",
                    target_subject=subject_name,
                    is_known=False,
                    confidence=0.10,
                    uncertainty=0.90,
                )

        # 2. Property / Color Query ("What color is the X?")
        elif frame.query_target == "property":
            prop_key = frame.predicate.replace("color_of", "color").replace("property_", "")

            # Check for conflicting language evidence in the graph
            evidence_nodes = [
                n for n in self._graph.nodes_by_type(HCIRNodeType.EVIDENCE)
                if isinstance(n, EvidenceNode) and ("language_evidence" in n.tags and subject_name in n.tags)
            ]
            distinct_claims = {ev.methodology for ev in evidence_nodes if ev.methodology}
            if len(distinct_claims) > 1:
                return CognitiveEpistemicState(
                    target_predicate=prop_key,
                    target_subject=subject_name,
                    is_known=True,
                    support_count=1,
                    contradiction_count=1,
                )

            prop_val = ent_props.get(prop_key)
            if prop_val:
                return CognitiveEpistemicState(
                    target_predicate=prop_key,
                    target_subject=subject_name,
                    target_object=str(prop_val),
                    confidence=0.98,
                    uncertainty=0.02,
                    support_count=4,
                    is_known=True,
                    raw_belief_value=str(prop_val),
                )
            return CognitiveEpistemicState(
                target_predicate=prop_key,
                target_subject=subject_name,
                is_known=False,
            )

        # 3. Verification / Yes-No Query ("Is the ball on the table?")
        elif frame.query_target == "verification":
            object_role = ThematicRole.LOCATION if ThematicRole.LOCATION in grounded.grounded_entities else ThematicRole.PATIENT
            object_id = grounded.grounded_entities.get(object_role)
            object_ref = frame.get_role(object_role)
            expected_object_name = object_ref.concept_name if object_ref else ""

            # Check if matching edge exists
            is_connected = False
            for edge in self._graph.edges_from(entity_id):
                if object_id and object_id in edge.targets:
                    is_connected = True
                    break
                for target_id in edge.targets:
                    target_node = self._graph.get_node(target_id)
                    if isinstance(target_node, PhysicalEntityNode) and target_node.entity_type == expected_object_name:
                        is_connected = True
                        break

            # Also check properties
            if not is_connected and expected_object_name:
                if str(ent_props.get("location", "")).lower() == expected_object_name.lower():
                    is_connected = True

            if is_connected:
                return CognitiveEpistemicState(
                    target_predicate=frame.predicate or "located_on",
                    target_subject=subject_name,
                    target_object=expected_object_name,
                    confidence=0.96,
                    uncertainty=0.04,
                    support_count=3,
                    is_known=True,
                    raw_belief_value=True,
                )
            else:
                return CognitiveEpistemicState(
                    target_predicate=frame.predicate or "located_on",
                    target_subject=subject_name,
                    target_object=expected_object_name,
                    confidence=0.92,
                    uncertainty=0.08,
                    support_count=2,
                    contradiction_count=0,
                    is_known=True,
                    raw_belief_value=False,
                )

        return CognitiveEpistemicState(
            target_predicate=frame.predicate,
            target_subject=subject_name,
            is_known=False,
        )

    # ── Command: Goal & Action Proposals ──────────────────────────────

    def process_command(self, grounded: GroundedSemanticFrame) -> GoalNode:
        """Convert a grounded command frame into a GoalNode in HCIR."""
        frame = grounded.frame
        action_verb = frame.predicate
        patient_ref = frame.get_role(ThematicRole.PATIENT) or frame.get_role(ThematicRole.THEME)
        destination_ref = frame.get_role(ThematicRole.DESTINATION) or frame.get_role(ThematicRole.LOCATION)

        patient_name = patient_ref.concept_name if patient_ref else "object"
        dest_name = destination_ref.concept_name if destination_ref else ""

        goal_spec = f"{action_verb} {patient_name} to {dest_name}".strip()
        goal = GoalNode(
            description=goal_spec,
            priority=0.8,
            tags=["language_command", frame.metadata.language, action_verb],
        )
        self._graph.add_node(goal)
        return goal
