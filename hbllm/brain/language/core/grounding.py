"""Grounding Resolver for A16.

Maps symbolic EntityReferences in SemanticFrames to physical entities (A13)
and grounded concepts (A15) stored in the HCIR CognitiveGraph.

Detects explicit epistemic error states:
- GROUNDING_FAILED: No entity in HCIR matches the reference criteria.
- AMBIGUOUS_REFERENCE: Multiple candidate entities match a definite reference.
"""

from __future__ import annotations

import logging

from hbllm.brain.language.core.reference import ReferenceResolver
from hbllm.brain.language.core.semantic_frame import (
    EntityReference,
    GroundedSemanticFrame,
    LanguageErrorType,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    GroundedConceptNode,
    HCIRNodeType,
    PhysicalEntityNode,
)

logger = logging.getLogger(__name__)


class GroundingResolver:
    """Resolves linguistic entity references to HCIR nodes and concepts.

    Usage::

        resolver = GroundingResolver(graph, reference_resolver)
        grounded_frame = resolver.ground_frame(semantic_frame)
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        reference_resolver: ReferenceResolver | None = None,
    ) -> None:
        self._graph = graph
        self._ref_resolver = reference_resolver or ReferenceResolver()

    def ground_frame(self, frame: SemanticFrame) -> GroundedSemanticFrame:
        """Ground all entity references within a SemanticFrame against HCIR state."""
        grounded = GroundedSemanticFrame(frame=frame)

        # Discourse priority: locations/obliques resolved first, primary theme/agent resolved last
        # so primary arguments stay at top of discourse stack.
        role_priority = {
            ThematicRole.LOCATION: 1,
            ThematicRole.DESTINATION: 2,
            ThematicRole.SOURCE: 3,
            ThematicRole.INSTRUMENT: 4,
            ThematicRole.RECIPIENT: 5,
            ThematicRole.PATIENT: 6,
            ThematicRole.THEME: 7,
            ThematicRole.AGENT: 8,
        }
        sorted_args = sorted(
            frame.arguments.items(),
            key=lambda item: role_priority.get(item[0], 0),
        )

        for role, ref in sorted_args:
            if ref is None:
                continue

            # 1. Check for anaphora ("it", "that", "this")
            if ref.specifier == "anaphoric":
                resolved_mention = self._ref_resolver.resolve_anaphor(ref)
                if resolved_mention and self._graph.get_node(resolved_mention.entity_id):
                    grounded.grounded_entities[role] = resolved_mention.entity_id
                    continue
                else:
                    grounded.grounding_success = False
                    grounded.grounding_error = LanguageErrorType.GROUNDING_FAILED
                    grounded.grounding_error_detail = (
                        f"Anaphoric reference '{ref.raw_text}' could not be resolved in discourse."
                    )
                    return grounded

            # 2. Match Grounded Concepts in A15
            if ref.concept_name:
                for node in self._graph.nodes_by_type(HCIRNodeType.GROUNDED_CONCEPT):
                    if isinstance(node, GroundedConceptNode):
                        if (
                            node.concept_name.lower() == ref.concept_name.lower()
                            or ref.concept_name.lower() in [tag.lower() for tag in node.tags]
                            or ref.concept_name.lower() == node.domain.lower()
                        ):
                            grounded.grounded_concepts[role] = node.id
                            break

            # 3. Match Physical Entities in A13
            matching_entities: list[PhysicalEntityNode] = []
            for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
                if isinstance(node, PhysicalEntityNode):
                    if self._entity_matches_reference(node, ref):
                        matching_entities.append(node)

            grounded.candidate_entities[role] = [e.id for e in matching_entities]

            # 4. Evaluate Definite vs Indefinite matching
            if ref.specifier == "definite":  # "the red ball"
                if len(matching_entities) == 0:
                    # If this is an assertion about a new entity, grounding might introduce it,
                    # but for queries/commands, it's a grounding failure.
                    grounded.grounding_success = False
                    grounded.grounding_error = LanguageErrorType.GROUNDING_FAILED
                    grounded.grounding_error_detail = (
                        f"No entity matching '{ref.raw_text}' found in world state."
                    )
                    return grounded
                elif len(matching_entities) > 1:
                    # Ambiguous reference
                    grounded.grounding_success = False
                    grounded.grounding_error = LanguageErrorType.AMBIGUOUS_REFERENCE
                    grounded.grounding_error_detail = (
                        f"Reference '{ref.raw_text}' is ambiguous; matches {len(matching_entities)} entities."
                    )
                    return grounded
                else:
                    chosen_entity = matching_entities[0]
                    grounded.grounded_entities[role] = chosen_entity.id
                    # Register mention in discourse
                    self._ref_resolver.register_mention(
                        entity_id=chosen_entity.id,
                        concept_name=ref.concept_name or chosen_entity.entity_type,
                        properties=ref.properties,
                    )

            elif ref.specifier in ("indefinite", "generic"):  # "a ball" / "balls"
                if matching_entities:
                    chosen_entity = matching_entities[0]
                    grounded.grounded_entities[role] = chosen_entity.id
                    self._ref_resolver.register_mention(
                        entity_id=chosen_entity.id,
                        concept_name=ref.concept_name or chosen_entity.entity_type,
                        properties=ref.properties,
                    )

        return grounded

    def _entity_matches_reference(
        self,
        entity: PhysicalEntityNode,
        ref: EntityReference,
    ) -> bool:
        """Check whether a PhysicalEntityNode satisfies the linguistic reference."""
        # Concept / Type check
        if ref.concept_name:
            cname = ref.concept_name.lower()
            etype = entity.entity_type.lower()
            if cname != etype and cname not in etype:
                return False

        # Property constraints (e.g., color="red")
        props = getattr(entity, "properties", None) or getattr(entity, "observed_properties", {}) or {}
        for prop_key, prop_val in ref.properties.items():
            ent_val = props.get(prop_key)
            if ent_val is None or str(ent_val).lower() != str(prop_val).lower():
                return False

        return True
