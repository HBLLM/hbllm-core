"""Grounding Resolver for A16.

Maps symbolic EntityReferences in SemanticFrames to physical entities (A13)
and grounded concepts (A15) stored in the HCIR CognitiveGraph.

Detects explicit epistemic error states:
- GROUNDING_FAILED: No entity in HCIR matches the reference criteria.
- AMBIGUOUS_REFERENCE: Multiple candidate entities match a definite reference.

Auto-creation policy (open-world assumption):
- ASSERTION frames: If a definite entity reference ("the ball") has no match,
  a new PhysicalEntityNode is created on-the-fly. Language is treated as a
  perception modality that can introduce new entities.
- COMMAND frames: Grounding is best-effort — ungrounded targets are passed
  through as raw concept names for the CommandDispatcher to interpret as tool
  parameters (e.g. "List the directory" → fs.list path="directory").
- QUERY frames: Strict grounding — missing entities → GROUNDING_FAILED.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    GroundedConceptNode,
    HCIRNodeType,
    PhysicalEntityNode,
)
from hbllm.hcir.language.core.reference import ReferenceResolver
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    GroundedSemanticFrame,
    LanguageErrorType,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.language.english.normalizer import damerau_levenshtein

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
        workspace: Any | None = None,
        lexicon: Any | None = None,
    ) -> None:
        self._graph = graph
        self._ref_resolver = reference_resolver or ReferenceResolver()
        self._workspace = workspace
        self._lexicon = lexicon

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

        new_mentions: list[tuple[str, str, dict[str, Any]]] = []

        for role, ref in sorted_args:
            if ref is None:
                continue

            # 1. Check for anaphora ("it", "that", "this", "they", "them")
            if ref.specifier == "anaphoric" or (
                ref.concept_name
                and ref.concept_name.lower() in {"it", "this", "that", "they", "them"}
            ):
                resolved_mention = self._ref_resolver.resolve_anaphor(ref)
                if resolved_mention and self._graph.get_node(resolved_mention.entity_id):
                    grounded.grounded_entities[role] = resolved_mention.entity_id
                    continue
                elif resolved_mention and resolved_mention.concept_name:
                    # Conceptual discourse topic (e.g., "photosynthesis" from prior turn)
                    frame.set_role(
                        role,
                        EntityReference(
                            concept_name=resolved_mention.concept_name,
                            properties=dict(ref.properties),
                            specifier="generic",
                            raw_text=ref.raw_text,
                            discourse_id=resolved_mention.entity_id,
                        ),
                    )
                    grounded.grounded_concepts[role] = resolved_mention.entity_id
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

            # 3. Match Physical Entities in A13 (with fuzzy fallback for untrusted typos)
            matching_entities: list[PhysicalEntityNode] = []
            for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
                if isinstance(node, PhysicalEntityNode):
                    if self._entity_matches_reference(node, ref):
                        matching_entities.append(node)

            if not matching_entities and ref.concept_name and len(ref.concept_name) >= 4:
                cname = ref.concept_name.lower()
                is_trusted = bool(
                    self._lexicon is not None
                    and hasattr(self._lexicon, "is_trusted_word")
                    and self._lexicon.is_trusted_word(cname)
                )
                if not is_trusted:
                    max_dist = 2 if len(cname) >= 6 else 1
                    best_etype: str | None = None
                    best_dist = max_dist + 1
                    fuzzy_candidates: list[PhysicalEntityNode] = []
                    for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
                        if not isinstance(node, PhysicalEntityNode):
                            continue
                        etype = node.entity_type.lower()
                        d = damerau_levenshtein(cname, etype, max_dist)
                        if d <= max_dist:
                            corrected_ref = EntityReference(
                                concept_name=node.entity_type,
                                properties=dict(ref.properties),
                                specifier=ref.specifier,
                                raw_text=ref.raw_text,
                                discourse_id=ref.discourse_id,
                            )
                            if self._entity_matches_reference(node, corrected_ref):
                                if d < best_dist:
                                    best_dist = d
                                    best_etype = node.entity_type
                                    fuzzy_candidates = [node]
                                elif d == best_dist and node.entity_type == best_etype:
                                    fuzzy_candidates.append(node)
                    if fuzzy_candidates and best_etype:
                        matching_entities = fuzzy_candidates
                        ref = EntityReference(
                            concept_name=best_etype,
                            properties=dict(ref.properties),
                            specifier=ref.specifier,
                            raw_text=ref.raw_text,
                            discourse_id=ref.discourse_id,
                        )
                        frame.set_role(role, ref)

            grounded.candidate_entities[role] = [e.id for e in matching_entities]

            is_conceptual_pred = frame.predicate in ("is_a", "causes", "requires", "enables")
            is_spatial_pred = frame.predicate in (
                "located_on",
                "located_in",
                "located_at",
                "located_inside",
                "located_within",
                "located_under",
                "located_beneath",
                "located_above",
                "located_near",
                "located_beside",
                "below",
                "above",
                "near",
            )

            # 4. Evaluate Definite vs Indefinite/Generic matching
            if ref.specifier == "definite":  # "the red ball"
                if len(matching_entities) == 0:
                    # ── Open-world auto-creation policy ──
                    # ASSERTION (physical/spatial/property): Create new entity on-the-fly
                    if (
                        frame.frame_type == FrameType.ASSERTION
                        and ref.concept_name
                        and not is_conceptual_pred
                    ):
                        new_entity = self._auto_create_entity(ref)
                        grounded.grounded_entities[role] = new_entity.id
                        new_mentions.append((new_entity.id, ref.concept_name, ref.properties))
                    # COMMAND: Relaxed grounding — pass raw concept through
                    elif frame.frame_type == FrameType.COMMAND:
                        pass
                    # CONCEPTUAL ASSERTION ("is_a", "causes", "requires"): keep conceptual
                    elif frame.frame_type == FrameType.ASSERTION and is_conceptual_pred:
                        pass
                    else:
                        # QUERY: Strict grounding
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
                    grounded.grounding_error_detail = f"Reference '{ref.raw_text}' is ambiguous; matches {len(matching_entities)} entities."
                    return grounded
                else:
                    chosen_entity = matching_entities[0]
                    grounded.grounded_entities[role] = chosen_entity.id
                    new_mentions.append(
                        (
                            chosen_entity.id,
                            ref.concept_name or chosen_entity.entity_type,
                            ref.properties,
                        )
                    )

            elif ref.specifier in (
                "indefinite",
                "generic",
            ):  # "a ball" / "balls" / bare telegraphic
                if matching_entities:
                    chosen_entity = matching_entities[0]
                    grounded.grounded_entities[role] = chosen_entity.id
                    new_mentions.append(
                        (
                            chosen_entity.id,
                            ref.concept_name or chosen_entity.entity_type,
                            ref.properties,
                        )
                    )
                elif (
                    frame.frame_type == FrameType.ASSERTION and is_spatial_pred and ref.concept_name
                ):
                    new_entity = self._auto_create_entity(ref)
                    grounded.grounded_entities[role] = new_entity.id
                    new_mentions.append((new_entity.id, ref.concept_name, ref.properties))

        # Register mentions in discourse only after all references in frame are resolved
        for ent_id, c_name, props in new_mentions:
            self._ref_resolver.register_mention(
                entity_id=ent_id,
                concept_name=c_name,
                properties=props,
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
        props = (
            getattr(entity, "properties", None) or getattr(entity, "observed_properties", {}) or {}
        )
        for prop_key, prop_val in ref.properties.items():
            ent_val = props.get(prop_key)
            if ent_val is None or str(ent_val).lower() != str(prop_val).lower():
                return False

        return True

    # ── Open-World Entity Auto-Creation ───────────────────────────────

    def _auto_create_entity(self, ref: EntityReference) -> PhysicalEntityNode:
        """Create a new PhysicalEntityNode from a linguistic reference.

        Called when an ASSERTION references a definite entity that doesn't exist
        in the graph. Language is treated as a perception modality that can
        introduce new entity hypotheses into the world model.
        """
        from hbllm.hcir.transactions import TransactionOp, TransactionOperation

        concept_name = ref.concept_name or "entity"
        entity = PhysicalEntityNode(
            entity_type=concept_name,
            entity_name=concept_name,
            name=concept_name,
            properties=dict(ref.properties) if ref.properties else {},
            tags=["language_introduced", concept_name],
        )

        # Commit through workspace if available, else direct graph add
        if self._workspace is not None and hasattr(self._workspace, "commit_operations"):
            self._workspace.commit_operations(
                [
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=entity.id,
                        node_data=entity.model_dump(),
                    )
                ],
                author="grounding_resolver",
            )
        else:
            self._graph.add_node(entity)

        logger.info(
            "GroundingResolver: auto-created PhysicalEntityNode '%s' (id=%s) from assertion",
            concept_name,
            entity.id,
        )
        return entity
