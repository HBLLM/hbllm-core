"""HCIR Gateway for A16.

Connects Language SemanticFrames to HCIR cognitive operations:
- Assertions -> Ingested as EvidenceNodes (Language is an evidence modality).
- Queries -> Evaluated against A13 World Model & A11 Epistemics, returning CognitiveEpistemicState.
- Commands -> Converted to GoalNode / Intent proposals for A12 Execution & Planning.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EvidenceNode,
    GoalNode,
    GroundedConceptNode,
    HCIRNodeType,
    PhysicalEntityNode,
)
from hbllm.hcir.language.core.epistemic_policy import CognitiveEpistemicState
from hbllm.hcir.language.core.semantic_frame import (
    GroundedSemanticFrame,
    ThematicRole,
)
from hbllm.hcir.nodes import HCIREdgeType

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

    def __init__(
        self,
        graph: CognitiveGraph,
        workspace: Any | None = None,
        ontology_store: Any | None = None,
        tiered_workspace: Any | None = None,
    ) -> None:
        self._graph = graph
        if workspace is None:
            from hbllm.hcir.workspace import HCIRWorkspaceState

            workspace = HCIRWorkspaceState(graph=graph)
        self._workspace = workspace
        self._ontology_store = ontology_store
        self._tiered_workspace = tiered_workspace

    def _commit_node(self, node: EvidenceNode | GoalNode) -> None:
        from hbllm.hcir.transactions import TransactionOp, TransactionOperation

        self._workspace.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node.id,
                    node_data=node.model_dump(),
                )
            ],
            author="hcir_language_gateway",
        )

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
        is_spatial_pred = (frame.predicate or "").startswith("located_") or frame.predicate in (
            "below",
            "above",
            "near",
            "is_property",
        )
        object_ref = (
            (frame.get_role(ThematicRole.LOCATION) or frame.get_role(ThematicRole.PATIENT))
            if is_spatial_pred
            else (frame.get_role(ThematicRole.PATIENT) or frame.get_role(ThematicRole.LOCATION))
        )

        subject_name: str = (
            subject_ref.concept_name if (subject_ref and subject_ref.concept_name) else "entity"
        )
        target_entity_id = grounded.grounded_entities.get(
            ThematicRole.THEME
        ) or grounded.grounded_entities.get(ThematicRole.AGENT)

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
        self._commit_node(evidence)

        # If entity exists, update observed property / spatial relation via transactional copy
        if target_entity_id:
            ent = self._graph.get_node(target_entity_id)
            if isinstance(ent, PhysicalEntityNode):
                from hbllm.hcir.transactions import TransactionOp, TransactionOperation

                ent_copy = ent.model_copy(deep=True)
                target_dict = (
                    ent_copy.properties
                    if hasattr(ent_copy, "properties") and isinstance(ent_copy.properties, dict)
                    else ent_copy.observed_properties
                )
                if subject_ref and subject_ref.properties:
                    target_dict.update(subject_ref.properties)
                if object_ref and object_ref.properties:
                    target_dict.update(object_ref.properties)
                self._workspace.commit_operations(
                    [
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE,
                            node_id=ent_copy.id,
                            node_data=ent_copy.model_dump(),
                        )
                    ],
                    author="hcir_language_gateway",
                )

        # Create spatial edge if both subject and object are grounded entities
        location_entity_id = grounded.grounded_entities.get(
            ThematicRole.LOCATION
        ) or grounded.grounded_entities.get(ThematicRole.PATIENT)
        if target_entity_id and location_entity_id and target_entity_id != location_entity_id:
            predicate = frame.predicate or "located_on"
            edge_type = self._predicate_to_edge_type(predicate)
            if edge_type is not None:
                from hbllm.hcir.graph import HCIREdge
                from hbllm.hcir.transactions import TransactionOp, TransactionOperation

                edge = HCIREdge(
                    edge_type=edge_type,
                    sources=[target_entity_id],
                    targets=[location_entity_id],
                    properties={"predicate": predicate, "speaker": speaker},
                )
                # Check for duplicate edges
                existing_edges = {e.id for e in self._graph.edges_from(target_entity_id)}
                if edge.id not in existing_edges:
                    self._workspace.commit_operations(
                        [
                            TransactionOperation(
                                op=TransactionOp.UPSERT_EDGE,
                                edge_id=edge.id,
                                edge_data=edge.model_dump(),
                            )
                        ],
                        author="hcir_language_gateway",
                    )
                    logger.debug(
                        "HCIRGateway: Created %s edge %s → %s",
                        edge_type.value,
                        target_entity_id,
                        location_entity_id,
                    )

        logger.debug(
            "HCIRGateway: Ingested language assertion as EvidenceNode %s (%s)",
            evidence.id,
            evidence.methodology,
        )

        # Also materialize conceptual, property, causal, dependency & SVO relations into persistent tier
        is_spatial_only = frame.predicate in (
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
        loc_or_inst_ref = (
            frame.get_role(ThematicRole.LOCATION)
            or frame.get_role(ThematicRole.DESTINATION)
            or frame.get_role(ThematicRole.INSTRUMENT)
        )
        effective_obj = obj_desc
        if not effective_obj and loc_or_inst_ref and loc_or_inst_ref.concept_name:
            effective_obj = loc_or_inst_ref.concept_name

        concept_rel = (
            "has_property" if frame.predicate == "is_property" else (frame.predicate or "")
        )
        if (
            concept_rel
            and not is_spatial_only
            and subject_name
            and effective_obj
            and self._tiered_workspace is not None
        ):
            try:
                p_ws = getattr(self._tiered_workspace, "persistent", None) or self._workspace
                if p_ws is not None and hasattr(p_ws, "commit_operations"):
                    grant = getattr(self._tiered_workspace, "_system_grant", None)
                    raw_sent = getattr(frame.metadata, "raw_text", "") or claim_str
                    is_negated = not getattr(frame.metadata, "polarity", True)
                    from hbllm.hcir.language.english.morphology import singularize_noun
                    from hbllm.hcir.language.reading_curriculum import _extract_np_modifiers

                    s_raw_clean = subject_name.lower().replace(" ", "_")
                    subj_raw_surface = (getattr(subject_ref, "raw_text", "") or "").strip()
                    is_proper_s_name = bool(
                        subj_raw_surface
                        and subj_raw_surface[0].isupper()
                        and not subj_raw_surface.lower().startswith(("a ", "an ", "the "))
                        and s_raw_clean.endswith(("as", "os"))
                    )
                    s_clean = s_raw_clean if is_proper_s_name else singularize_noun(s_raw_clean)
                    o_raw_clean = effective_obj.lower().replace(" ", "_")
                    o_clean = (
                        o_raw_clean
                        if concept_rel == "has_property"
                        else singularize_noun(o_raw_clean)
                    )

                    mod_pairs: list[tuple[str, str]] = []
                    if concept_rel == "is_a" and object_ref and object_ref.raw_text:
                        mod_pairs.extend(_extract_np_modifiers(object_ref.raw_text))
                    if concept_rel != "has_property":
                        for extra_ref in (
                            frame.get_role(ThematicRole.LOCATION),
                            frame.get_role(ThematicRole.DESTINATION),
                            frame.get_role(ThematicRole.INSTRUMENT),
                        ):
                            if extra_ref and extra_ref.concept_name:
                                prep = str(extra_ref.properties.get("prep", "")).strip().lower()
                                tgt_c = singularize_noun(
                                    extra_ref.concept_name.lower().replace(" ", "_")
                                )
                                if prep and tgt_c:
                                    mod_pairs.append((prep, tgt_c))

                    from hbllm.hcir.transactions import TransactionOp, TransactionOperation

                    if not is_negated:
                        # Retract any prior contradiction edges or negative beliefs for clean belief revision
                        p_graph = getattr(p_ws, "graph", None)
                        if p_graph is not None:
                            src_id = f"t4_concept_{s_clean}"
                            dst_id = f"t4_concept_{o_clean}"
                            retract_ops: list[TransactionOperation] = []
                            for edge in list(p_graph.edges_from(src_id)):
                                if any(t == dst_id for t in edge.targets):
                                    rel = str(edge.properties.get("relation", "")).lower()
                                    etype = getattr(
                                        edge.edge_type, "value", str(edge.edge_type)
                                    ).lower()
                                    if (
                                        etype == "contradicts"
                                        or rel == f"not_{concept_rel}"
                                        or rel == concept_rel
                                    ):
                                        retract_ops.append(
                                            TransactionOperation(
                                                op=TransactionOp.REMOVE_EDGE,
                                                edge_id=edge.id,
                                            )
                                        )
                            # Also check reverse edges for contradiction
                            for edge in list(p_graph.edges_from(dst_id)):
                                if any(t == src_id for t in edge.targets):
                                    etype = getattr(
                                        edge.edge_type, "value", str(edge.edge_type)
                                    ).lower()
                                    if etype == "contradicts":
                                        retract_ops.append(
                                            TransactionOperation(
                                                op=TransactionOp.REMOVE_EDGE,
                                                edge_id=edge.id,
                                            )
                                        )
                            for c_eid in (
                                f"t4_contradicts_{src_id}__{dst_id}",
                                f"t4_contradicts_{src_id}__{dst_id}__not_{concept_rel}",
                            ):
                                if (
                                    getattr(p_graph, "get_edge", None)
                                    and p_graph.get_edge(c_eid) is not None
                                ):
                                    retract_ops.append(
                                        TransactionOperation(
                                            op=TransactionOp.REMOVE_EDGE,
                                            edge_id=c_eid,
                                        )
                                    )
                            neg_b_id = f"t4_belief_{s_clean}_not_{concept_rel}_{o_clean}"
                            if p_graph.get_node(neg_b_id) is not None:
                                retract_ops.append(
                                    TransactionOperation(
                                        op=TransactionOp.REMOVE_NODE,
                                        node_id=neg_b_id,
                                    )
                                )
                            if retract_ops:
                                p_ws.commit_operations(
                                    retract_ops,
                                    author="hcir_language_gateway",
                                    grant=grant,
                                )

                        from hbllm.hcir.language.reading_curriculum import (
                            ExtractedProposition,
                            commit_propositions,
                        )

                        props_to_commit = [
                            ExtractedProposition(
                                subject=s_clean,
                                relation=concept_rel,
                                obj=o_clean,
                                source_file="dialogue_assertion",
                                sentence=raw_sent,
                                polarity=True,
                                modifiers=tuple(mod_pairs),
                            )
                        ]
                        inf_verb = (
                            str(object_ref.properties.get("infinitive_verb", "")).strip().lower()
                            if object_ref and object_ref.properties
                            else ""
                        )
                        if inf_verb and concept_rel == "causes":
                            props_to_commit.append(
                                ExtractedProposition(
                                    subject=s_clean,
                                    relation="causes",
                                    obj=f"{o_clean}_{inf_verb}",
                                    source_file="dialogue_assertion",
                                    sentence=raw_sent,
                                )
                            )
                            props_to_commit.append(
                                ExtractedProposition(
                                    subject=s_clean,
                                    relation="causes",
                                    obj=inf_verb,
                                    source_file="dialogue_assertion",
                                    sentence=raw_sent,
                                )
                            )

                        commit_propositions(
                            props_to_commit,
                            p_ws,
                            ontology_store=self._ontology_store,
                            grant=grant,
                            author="hcir_language_gateway",
                        )
                    else:
                        from hbllm.hcir.graph import BeliefNode, ConceptNode, HCIREdge
                        from hbllm.hcir.nodes import HCIREdgeType

                        src_id = f"t4_concept_{s_clean}"
                        dst_id = f"t4_concept_{o_clean}"
                        tx_ops: list[TransactionOperation] = []

                        # Ensure concept nodes exist (without erasing existing definitions)
                        p_graph = getattr(p_ws, "graph", None)
                        for term, cid in ((s_clean, src_id), (o_clean, dst_id)):
                            display = term.replace("_", " ")
                            existing_n = p_graph.get_node(cid) if p_graph is not None else None
                            existing_def = (
                                getattr(existing_n, "definition", None)
                                if existing_n is not None
                                else None
                            )
                            if existing_def:
                                continue
                            cnode = ConceptNode(
                                id=cid,
                                label=display,
                                domain="dialogue_negation",
                                tags=["t4_negation", display],
                            )
                            tx_ops.append(
                                TransactionOperation(
                                    op=TransactionOp.UPSERT_NODE,
                                    node_id=cnode.id,
                                    node_data=cnode.model_dump(),
                                )
                            )

                        # Retract existing positive edges between src_id and dst_id
                        if p_graph is not None:
                            for edge in list(p_graph.edges_from(src_id)):
                                if any(t == dst_id for t in edge.targets):
                                    rel = str(edge.properties.get("relation", "")).lower()
                                    etype = getattr(
                                        edge.edge_type, "value", str(edge.edge_type)
                                    ).lower()
                                    if rel == concept_rel or (
                                        concept_rel in ("is_a", "causes", "requires", "enables")
                                        and (
                                            rel in ("is_a", "causes", "requires", "enables")
                                            or etype
                                            in ("generalizes", "supports", "causes", "requires")
                                        )
                                    ):
                                        tx_ops.append(
                                            TransactionOperation(
                                                op=TransactionOp.REMOVE_EDGE,
                                                edge_id=edge.id,
                                            )
                                        )

                        # Retract existing positive belief node if any
                        belief_id = f"t4_belief_{s_clean}_{concept_rel}_{o_clean}"
                        if p_graph is not None and p_graph.get_node(belief_id) is not None:
                            tx_ops.append(
                                TransactionOperation(
                                    op=TransactionOp.REMOVE_NODE,
                                    node_id=belief_id,
                                )
                            )

                        # Commit contradiction edge
                        edge_id = f"t4_contradicts_{src_id}__{dst_id}"
                        contra_edge = HCIREdge(
                            id=edge_id,
                            edge_type=HCIREdgeType.CONTRADICTS,
                            sources=[src_id],
                            targets=[dst_id],
                            weight=1.0,
                            properties={
                                "relation": f"not_{concept_rel}",
                                "polarity": False,
                                "source_file": "dialogue_assertion",
                                "sentence": raw_sent,
                            },
                        )
                        tx_ops.append(
                            TransactionOperation(
                                op=TransactionOp.UPSERT_EDGE,
                                edge=contra_edge,
                            )
                        )

                        # Commit negative belief node
                        bnode = BeliefNode(
                            id=f"t4_belief_{s_clean}_not_{concept_rel}_{o_clean}",
                            statement=f"{s_clean.replace('_', ' ')} not_{concept_rel} {o_clean.replace('_', ' ')}",
                            confidence=0.98,
                            tags=["t4_negation", f"not_{concept_rel}"],
                            properties={"sentence": raw_sent},
                        )
                        tx_ops.append(
                            TransactionOperation(
                                op=TransactionOp.UPSERT_NODE,
                                node_id=bnode.id,
                                node_data=bnode.model_dump(),
                            )
                        )

                        p_ws.commit_operations(tx_ops, author="hcir_language_gateway", grant=grant)
            except Exception as exc:
                logger.debug("HCIRGateway: conceptual proposition commit skipped: %s", exc)

        return evidence

    # ── Query: Epistemic State Resolution ─────────────────────────────

    def process_query(self, grounded: GroundedSemanticFrame) -> CognitiveEpistemicState:
        """Query HCIR world model and epistemics for a grounded question."""
        frame = grounded.frame

        # 0. Specialized query routing
        if frame.query_target == "verification":
            return self._process_verification_query(grounded)
        elif frame.query_target == "cause":
            return self._process_cause_query(grounded)
        elif frame.query_target in ("procedure", "requirement") or frame.predicate == "requires":
            return self._process_requirement_query(grounded)
        elif frame.query_target in ("enables", "enable") or frame.predicate == "enables":
            return self._process_enable_query(grounded)
        elif frame.query_target == "identity":
            return self._process_identity_query(grounded)
        elif frame.query_target == "svo_object":
            return self._process_svo_object_query(grounded)
        elif frame.query_target == "svo_location":
            return self._process_svo_location_query(grounded)
        elif frame.query_target == "svo_subject":
            return self._process_svo_subject_query(grounded)

        target_role = (
            ThematicRole.THEME
            if frame.get_role(ThematicRole.THEME) is not None
            else (
                ThematicRole.AGENT
                if frame.get_role(ThematicRole.AGENT) is not None
                else (
                    ThematicRole.THEME
                    if ThematicRole.THEME in grounded.grounded_entities
                    else ThematicRole.AGENT
                )
            )
        )
        entity_id = grounded.grounded_entities.get(target_role)

        subject_ref = frame.get_role(target_role)
        subject_name: str = (
            subject_ref.concept_name if (subject_ref and subject_ref.concept_name) else "entity"
        )

        # If entity could not be grounded, try epistemic prior before giving up
        if not entity_id:
            prior = self._lookup_epistemic_prior(
                subject_name, query_target=frame.query_target or "definition"
            )
            if prior is not None:
                return prior
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
            ent_props = (
                ent_node.properties
                if hasattr(ent_node, "properties") and isinstance(ent_node.properties, dict)
                else ent_node.observed_properties
            )

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
                    if (
                        isinstance(concept_node, GroundedConceptNode)
                        and "location" in concept_node.feature_prototype
                    ):
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
                n
                for n in self._graph.nodes_by_type(HCIRNodeType.EVIDENCE)
                if isinstance(n, EvidenceNode)
                and ("language_evidence" in n.tags and subject_name in n.tags)
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

            # ── Epistemic Prior Fallback: "What is X?" definition queries ──
            # When prop_key is "property" (from "property_property" predicate, i.e.
            # "What is X?"), consult LexicalOntologyStore and persistent memory.
            if prop_key == "property":
                prior = self._lookup_epistemic_prior(subject_name)
                if prior is not None:
                    return prior

            return CognitiveEpistemicState(
                target_predicate=prop_key,
                target_subject=subject_name,
                is_known=False,
            )

        # 3. Verification / Yes-No Query ("Is the ball on the table?")
        elif frame.query_target == "verification":
            object_role = (
                ThematicRole.LOCATION
                if ThematicRole.LOCATION in grounded.grounded_entities
                else ThematicRole.PATIENT
            )
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
                    if (
                        isinstance(target_node, PhysicalEntityNode)
                        and target_node.entity_type == expected_object_name
                    ):
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

        # ── General fallback: consult epistemic priors before giving up ──
        prior = self._lookup_epistemic_prior(subject_name)
        if prior is not None:
            return prior

        return CognitiveEpistemicState(
            target_predicate=frame.predicate,
            target_subject=subject_name,
            is_known=False,
        )

    # ── Specialized Conceptual Query Processors ───────────────────────

    def _find_concept_ids(self, name: str) -> set[str]:
        """Find matching ConceptNode IDs across persistent, brain, and working graphs (O(1) first)."""
        from hbllm.hcir.language.english.morphology import singularize_noun

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        b_ws = getattr(self._tiered_workspace, "brain", None)
        b_graph = getattr(b_ws, "graph", None)
        graphs = [g for g in (p_graph, b_graph, self._graph) if g is not None]

        clean = name.lower().strip().replace(" ", "_")
        sing = singularize_noun(clean)
        forms = (clean, sing) if sing != clean else (clean,)
        ids: set[str] = set()
        cand_ids = {
            pfx for form in forms for pfx in (f"t4_concept_{form}", f"concept_{form}", form)
        }
        for g in graphs:
            for cid in cand_ids:
                if g.get_node(cid) is not None:
                    ids.add(cid)
        if ids:
            return ids

        # Fallback: label scan only when direct O(1) ID lookup had no hit
        clean_sp = clean.replace("_", " ")
        sing_sp = sing.replace("_", " ")
        for g in graphs:
            for node in g.nodes_by_type(HCIRNodeType.CONCEPT):
                lbl = (
                    (getattr(node, "label", "") or getattr(node, "concept_name", "") or "")
                    .lower()
                    .strip()
                )
                if lbl in (clean, clean_sp, sing, sing_sp):
                    ids.add(node.id)
        return ids

    def _collect_is_a_ancestors(
        self, seed_ids: set[str], graphs: list[Any], max_depth: int = 4
    ) -> list[str]:
        """Return ordered concept IDs starting with seed_ids followed by transitive is_a parents."""
        ordered = list(seed_ids)
        visited = set(seed_ids)
        queue = [(sid, 0) for sid in ordered]
        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for g in graphs:
                for edge in g.edges_from(curr):
                    if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                        continue
                    rel = str(edge.properties.get("relation", "")).lower()
                    etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                    if etype == "generalizes" or rel in ("is_a", "type_of"):
                        for tid in edge.targets:
                            if tid not in visited:
                                visited.add(tid)
                                ordered.append(tid)
                                queue.append((tid, depth + 1))
        return ordered

    def _process_verification_query(
        self, grounded: GroundedSemanticFrame
    ) -> CognitiveEpistemicState:
        """Process verification / yes-no queries for both physical entities and conceptual knowledge."""
        frame = grounded.frame
        target_role = (
            ThematicRole.THEME
            if frame.get_role(ThematicRole.THEME) is not None
            else (
                ThematicRole.AGENT
                if frame.get_role(ThematicRole.AGENT) is not None
                else ThematicRole.THEME
            )
        )
        object_role = (
            ThematicRole.PATIENT
            if frame.get_role(ThematicRole.PATIENT) is not None
            else (
                ThematicRole.LOCATION
                if frame.get_role(ThematicRole.LOCATION) is not None
                else (
                    ThematicRole.DESTINATION
                    if frame.get_role(ThematicRole.DESTINATION) is not None
                    else ThematicRole.PATIENT
                )
            )
        )
        entity_id = grounded.grounded_entities.get(target_role)
        object_id = grounded.grounded_entities.get(object_role)

        subject_ref = frame.get_role(target_role)
        subject_name = (
            subject_ref.concept_name if (subject_ref and subject_ref.concept_name) else "entity"
        )
        object_ref = frame.get_role(object_role)
        expected_object_name = (
            object_ref.concept_name if (object_ref and object_ref.concept_name) else ""
        )
        pred = frame.predicate or "is_a"

        # Case 1: Physical / Spatial / Property verification on working-graph PhysicalEntityNode
        if entity_id and (object_id or expected_object_name or pred.startswith("property_")):
            ent_node = self._graph.get_node(entity_id)
            ent_props: dict[str, Any] = (
                ent_node.properties
                if hasattr(ent_node, "properties") and isinstance(ent_node.properties, dict)
                else getattr(ent_node, "observed_properties", {})
            )
            if pred.startswith("property_"):
                prop_key = pred.replace("property_", "")
                prop_val = ent_props.get(prop_key)
                if prop_val is not None:
                    expected_vals = set()
                    if expected_object_name:
                        expected_vals.add(expected_object_name.lower())
                    if object_ref and object_ref.properties:
                        expected_vals.update(str(v).lower() for v in object_ref.properties.values())
                    matched = bool(expected_vals and str(prop_val).lower() in expected_vals)
                    return CognitiveEpistemicState(
                        target_predicate=pred,
                        target_subject=subject_name,
                        target_object=expected_object_name,
                        confidence=0.96 if matched else 0.92,
                        uncertainty=0.04 if matched else 0.08,
                        support_count=3 if matched else 2,
                        is_known=True,
                        raw_belief_value=matched,
                    )

            is_connected = False
            for edge in self._graph.edges_from(entity_id):
                if object_id and object_id in edge.targets:
                    is_connected = True
                    break
                for target_id in edge.targets:
                    target_node = self._graph.get_node(target_id)
                    if (
                        isinstance(target_node, PhysicalEntityNode)
                        and target_node.entity_type == expected_object_name
                    ):
                        is_connected = True
                        break
            if not is_connected and expected_object_name:
                if str(ent_props.get("location", "")).lower() == expected_object_name.lower():
                    is_connected = True

            return CognitiveEpistemicState(
                target_predicate=pred or "located_on",
                target_subject=subject_name,
                target_object=expected_object_name,
                confidence=0.96 if is_connected else 0.92,
                uncertainty=0.04 if is_connected else 0.08,
                support_count=3 if is_connected else 2,
                contradiction_count=0,
                is_known=True,
                raw_belief_value=is_connected,
            )

        # Case 2: Conceptual verification
        return self._verify_conceptual_relation(subject_name, expected_object_name, pred)

    def _verify_conceptual_relation(
        self, subj: str, obj: str, pred: str
    ) -> CognitiveEpistemicState:
        """Verify conceptual relations (taxonomy, causality, properties, SVO) over HCIR graphs."""
        from hbllm.hcir.language.english.morphology import singularize_noun

        s_clean = singularize_noun(subj.lower().strip().replace(" ", "_"))
        raw_o_clean = obj.lower().strip().replace(" ", "_")
        norm_pred = "has_property" if pred.startswith("property_") else pred
        o_clean = raw_o_clean if norm_pred == "has_property" else singularize_noun(raw_o_clean)

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        b_ws = getattr(self._tiered_workspace, "brain", None)
        b_graph = getattr(b_ws, "graph", None)
        graphs = [g for g in (p_graph, b_graph, self._graph) if g is not None]

        s_ids = self._find_concept_ids(s_clean)
        o_ids = self._find_concept_ids(o_clean) if o_clean else set()

        has_s_ont = bool(
            self._ontology_store is not None and self._ontology_store.has_word(s_clean)
        )
        has_o_ont = bool(
            o_clean and self._ontology_store is not None and self._ontology_store.has_word(o_clean)
        )

        # Unknown subject -> Abstain; unknown object when object required (except property modifiers) -> Abstain
        if not s_ids and not has_s_ont:
            return CognitiveEpistemicState(
                target_predicate=norm_pred,
                target_subject=subj,
                target_object=obj,
                is_known=False,
                confidence=0.05,
                uncertainty=0.95,
                support_count=0,
            )
        if o_clean and norm_pred in ("is_a", "type_of", "causes", "enables", "requires"):
            if not o_ids and not has_o_ont:
                return CognitiveEpistemicState(
                    target_predicate=norm_pred,
                    target_subject=subj,
                    target_object=obj,
                    is_known=False,
                    confidence=0.05,
                    uncertainty=0.95,
                    support_count=0,
                )

        # Check for explicit contradictions in graphs
        contradiction_found = False
        for g in graphs:
            for sid in s_ids:
                for edge in g.edges_from(sid):
                    etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                    rel = str(edge.properties.get("relation", "")).lower()
                    is_contra = etype == "contradicts" or rel in (
                        f"not_{norm_pred}",
                        f"not_{pred}",
                        "not_is_a",
                        "not_causes",
                        "not_requires",
                        "not_enables",
                        "not_type_of",
                        "not_a",
                        "not_has_property",
                    )
                    if is_contra and (not o_clean or any(t in o_ids for t in edge.targets)):
                        contradiction_found = True
                        break
                if contradiction_found:
                    break
            if contradiction_found:
                break

        if contradiction_found:
            return CognitiveEpistemicState(
                target_predicate=norm_pred,
                target_subject=subj,
                target_object=obj,
                is_known=True,
                raw_belief_value=False,
                confidence=0.98,
                uncertainty=0.02,
                support_count=3,
                contradiction_count=0,
            )

        if norm_pred in ("is_a", "type_of"):
            found = False
            visited = set(s_ids)
            queue = list(s_ids)
            while queue:
                curr = queue.pop(0)
                if curr in o_ids:
                    found = True
                    break
                for g in graphs:
                    for edge in g.edges_from(curr):
                        if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                            continue
                        rel = str(edge.properties.get("relation", "")).lower()
                        etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                        if etype == "generalizes" or rel in ("is_a", "type_of"):
                            for target_id in edge.targets:
                                if target_id not in visited:
                                    visited.add(target_id)
                                    queue.append(target_id)

            if not found and self._ontology_store is not None:
                try:
                    hyps = {
                        h.lower().replace("_", " ")
                        for h in self._ontology_store.get_hypernyms(s_clean)
                    }
                    if o_clean.replace("_", " ") in hyps:
                        found = True
                except Exception as exc:
                    logger.debug("HCIRGateway: ontology hypernym verification error: %s", exc)

            return CognitiveEpistemicState(
                target_predicate="is_a",
                target_subject=subj,
                target_object=obj,
                is_known=True,
                raw_belief_value=found,
                confidence=0.98 if found else 0.95,
                uncertainty=0.02 if found else 0.05,
                support_count=3 if found else 0,
                contradiction_count=0,
            )

        if norm_pred == "has_property":
            all_ids = self._collect_is_a_ancestors(s_ids, graphs)
            prop_found = False
            for g in graphs:
                for sid in all_ids:
                    for edge in g.edges_from(sid):
                        if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                            continue
                        if edge.properties.get("polarity") is False:
                            continue
                        rel = str(edge.properties.get("relation", "")).lower()
                        etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                        if (rel == "has_property" or etype == "has_property") and (
                            any(t in o_ids for t in edge.targets)
                            or any(o_clean in t.lower() for t in edge.targets)
                        ):
                            prop_found = True
                            break
                        mods = edge.properties.get("modifiers") or {}
                        adj_mods = mods.get("adj") if isinstance(mods, dict) else []
                        if adj_mods and o_clean in [str(a).lower() for a in adj_mods]:
                            prop_found = True
                            break
                    if prop_found:
                        break
                if prop_found:
                    break

            if prop_found:
                return CognitiveEpistemicState(
                    target_predicate="has_property",
                    target_subject=subj,
                    target_object=obj,
                    is_known=True,
                    raw_belief_value=True,
                    confidence=0.98,
                    uncertainty=0.02,
                    support_count=3,
                )
            return CognitiveEpistemicState(
                target_predicate="has_property",
                target_subject=subj,
                target_object=obj,
                is_known=False,
                confidence=0.05,
                uncertainty=0.95,
                support_count=0,
            )

        elif pred in ("causes", "enables"):
            direct_found = False
            for g in graphs:
                for sid in s_ids:
                    for edge in g.edges_from(sid):
                        rel = str(edge.properties.get("relation", "")).lower()
                        etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                        if etype in ("causes", "enables") or rel in ("causes", "enables"):
                            if any(t in o_ids for t in edge.targets):
                                direct_found = True
                                break
                    if direct_found:
                        break
                if direct_found:
                    break

            if direct_found:
                return CognitiveEpistemicState(
                    target_predicate=pred,
                    target_subject=subj,
                    target_object=obj,
                    is_known=True,
                    raw_belief_value=True,
                    confidence=0.98,
                    uncertainty=0.02,
                    support_count=3,
                    contradiction_count=0,
                )

            # Check for reverse causal edge (obj causes subj)
            reverse_found = False
            for g in graphs:
                for oid in o_ids:
                    for edge in g.edges_from(oid):
                        rel = str(edge.properties.get("relation", "")).lower()
                        etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                        if etype in ("causes", "enables") or rel in ("causes", "enables"):
                            if any(t in s_ids for t in edge.targets):
                                reverse_found = True
                                break
                    if reverse_found:
                        break
                if reverse_found:
                    break

            if reverse_found:
                return CognitiveEpistemicState(
                    target_predicate=pred,
                    target_subject=subj,
                    target_object=obj,
                    is_known=True,
                    raw_belief_value=False,
                    confidence=0.98,
                    uncertainty=0.02,
                    support_count=3,
                    contradiction_count=0,
                )

            # Neither direct nor reverse causal relationship recorded -> open-world epistemic abstention
            return CognitiveEpistemicState(
                target_predicate=pred,
                target_subject=subj,
                target_object=obj,
                is_known=False,
                confidence=0.08,
                uncertainty=0.92,
                support_count=0,
                contradiction_count=0,
            )

        elif pred == "requires":
            direct_found = False
            for g in graphs:
                for sid in s_ids:
                    for edge in g.edges_from(sid):
                        rel = str(edge.properties.get("relation", "")).lower()
                        etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                        if etype == "requires" or rel == "requires":
                            if any(t in o_ids for t in edge.targets):
                                direct_found = True
                                break
                    if direct_found:
                        break
                if direct_found:
                    break

            if direct_found:
                return CognitiveEpistemicState(
                    target_predicate="requires",
                    target_subject=subj,
                    target_object=obj,
                    is_known=True,
                    raw_belief_value=True,
                    confidence=0.98,
                    uncertainty=0.02,
                    support_count=3,
                    contradiction_count=0,
                )

            # No requirement recorded -> open-world epistemic abstention
            return CognitiveEpistemicState(
                target_predicate="requires",
                target_subject=subj,
                target_object=obj,
                is_known=False,
                confidence=0.08,
                uncertainty=0.92,
                support_count=0,
                contradiction_count=0,
            )

        # General SVO verb verification (e.g., eat, live, fly, walk, absorb, release)
        all_ids = self._collect_is_a_ancestors(s_ids, graphs)
        verb_found = False
        for g in graphs:
            for sid in all_ids:
                for edge in g.edges_from(sid):
                    if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                        continue
                    if edge.properties.get("polarity") is False:
                        continue
                    rel = str(edge.properties.get("relation", "")).lower()
                    if rel == norm_pred or rel.rstrip("s") == norm_pred.rstrip("s"):
                        if not o_clean or any(t in o_ids for t in edge.targets):
                            verb_found = True
                            break
                if verb_found:
                    break
            if verb_found:
                break

        if verb_found:
            return CognitiveEpistemicState(
                target_predicate=norm_pred,
                target_subject=subj,
                target_object=obj,
                is_known=True,
                raw_belief_value=True,
                confidence=0.96,
                uncertainty=0.04,
                support_count=3,
            )

        return CognitiveEpistemicState(
            target_predicate=norm_pred,
            target_subject=subj,
            target_object=obj,
            is_known=False,
            confidence=0.08,
            uncertainty=0.92,
            support_count=0,
        )

    @staticmethod
    def _label_for_node_id(g: Any, node_id: str) -> str:
        node = g.get_node(node_id)
        label = getattr(node, "label", None) or getattr(node, "concept_name", None)
        if label:
            return str(label)
        return node_id.replace("t4_concept_", "").replace("concept_", "").replace("_", " ")

    def _process_cause_query(self, grounded: GroundedSemanticFrame) -> CognitiveEpistemicState:
        """Resolve 'What causes X?' queries by traversing upstream causal edges."""
        frame = grounded.frame
        patient_ref = frame.get_role(ThematicRole.PATIENT) or frame.get_role(ThematicRole.THEME)
        patient_name = patient_ref.concept_name if patient_ref else ""
        if not patient_name:
            return CognitiveEpistemicState(
                target_predicate="causes",
                target_subject="",
                is_known=False,
                confidence=0.0,
            )

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        graphs = [g for g in (p_graph, self._graph) if g is not None]
        pat_ids = self._find_concept_ids(patient_name)

        cause_found: str | None = None
        for g in graphs:
            for pid in pat_ids:
                for edge in g.edges_to(pid):
                    if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                        continue
                    rel = str(edge.properties.get("relation", "")).lower()
                    etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                    if etype == "causes" or rel == "causes":
                        if edge.sources:
                            cause_found = self._label_for_node_id(g, edge.sources[0])
                            break
                if cause_found:
                    break
            if cause_found:
                break

        if cause_found:
            return CognitiveEpistemicState(
                target_predicate="causes",
                target_subject=cause_found,
                target_object=patient_name,
                is_known=True,
                raw_belief_value=cause_found,
                confidence=0.98,
                uncertainty=0.02,
                support_count=3,
            )

        prior = self._lookup_epistemic_prior(patient_name, query_target="cause")
        if prior is not None:
            return prior

        return CognitiveEpistemicState(
            target_predicate="causes",
            target_subject=patient_name,
            is_known=False,
            confidence=0.05,
            uncertainty=0.95,
        )

    def _process_svo_object_query(self, grounded: GroundedSemanticFrame) -> CognitiveEpistemicState:
        """Resolve 'What do X <verb>?' queries over SVO edges."""
        frame = grounded.frame
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        subj_name = theme_ref.concept_name if (theme_ref and theme_ref.concept_name) else ""
        pred = (frame.predicate or "").lower()
        if not subj_name or not pred:
            return CognitiveEpistemicState(
                target_predicate=pred,
                target_subject=subj_name,
                is_known=False,
                confidence=0.05,
                uncertainty=0.95,
            )

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        b_ws = getattr(self._tiered_workspace, "brain", None)
        b_graph = getattr(b_ws, "graph", None)
        graphs = [g for g in (p_graph, b_graph, self._graph) if g is not None]

        s_ids = self._find_concept_ids(subj_name)
        all_ids = self._collect_is_a_ancestors(s_ids, graphs)
        targets: list[str] = []
        for g in graphs:
            for sid in all_ids:
                for edge in g.edges_from(sid):
                    if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                        continue
                    if edge.properties.get("polarity") is False:
                        continue
                    rel = str(edge.properties.get("relation", "")).lower()
                    if rel == pred or rel.rstrip("s") == pred.rstrip("s"):
                        for tid in edge.targets:
                            lbl = self._label_for_node_id(g, tid)
                            if lbl and lbl not in targets:
                                targets.append(lbl)
                if targets and sid in s_ids:
                    break
            if targets:
                break

        if targets:
            ans = targets[0] if len(targets) == 1 else ", ".join(targets[:4])
            return CognitiveEpistemicState(
                target_predicate=pred,
                target_subject=subj_name,
                target_object=ans,
                is_known=True,
                raw_belief_value=ans,
                confidence=0.96,
                uncertainty=0.04,
                support_count=3,
            )

        return CognitiveEpistemicState(
            target_predicate=pred,
            target_subject=subj_name,
            is_known=False,
            confidence=0.05,
            uncertainty=0.95,
        )

    def _process_svo_location_query(
        self, grounded: GroundedSemanticFrame
    ) -> CognitiveEpistemicState:
        """Resolve 'Where do X <verb>?' queries over SVO/location edges."""
        frame = grounded.frame
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        subj_name = theme_ref.concept_name if (theme_ref and theme_ref.concept_name) else ""
        pred = (frame.predicate or "").lower()
        if not subj_name or not pred:
            return CognitiveEpistemicState(
                target_predicate=pred,
                target_subject=subj_name,
                is_known=False,
                confidence=0.05,
                uncertainty=0.95,
            )

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        b_ws = getattr(self._tiered_workspace, "brain", None)
        b_graph = getattr(b_ws, "graph", None)
        graphs = [g for g in (p_graph, b_graph, self._graph) if g is not None]

        s_ids = self._find_concept_ids(subj_name)
        all_ids = self._collect_is_a_ancestors(s_ids, graphs)
        found_loc: str | None = None
        found_prep = "in"
        for g in graphs:
            for sid in all_ids:
                for edge in g.edges_from(sid):
                    if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                        continue
                    if edge.properties.get("polarity") is False:
                        continue
                    rel = str(edge.properties.get("relation", "")).lower()
                    mods = edge.properties.get("modifiers") or {}
                    prep = mods.get("prep", "") if isinstance(mods, dict) else ""
                    if rel == pred or rel.rstrip("s") == pred.rstrip("s"):
                        if prep and edge.targets:
                            found_loc = self._label_for_node_id(g, edge.targets[0])
                            found_prep = str(prep)
                            break
                        elif edge.targets and found_loc is None:
                            found_loc = self._label_for_node_id(g, edge.targets[0])
                if found_loc:
                    break
            if found_loc:
                break

        if found_loc:
            return CognitiveEpistemicState(
                target_predicate=pred,
                target_subject=subj_name,
                target_object=found_loc,
                is_known=True,
                raw_belief_value=found_loc,
                confidence=0.96,
                uncertainty=0.04,
                support_count=3,
                provenance=f"prep:{found_prep}",
            )

        return CognitiveEpistemicState(
            target_predicate=pred,
            target_subject=subj_name,
            is_known=False,
            confidence=0.05,
            uncertainty=0.95,
        )

    def _process_svo_subject_query(
        self, grounded: GroundedSemanticFrame
    ) -> CognitiveEpistemicState:
        """Resolve 'Who <verb/was Adj>?' and 'What <verb> <obj>?' subject-seeking queries."""
        frame = grounded.frame
        pred = (frame.predicate or "").lower()
        target_ref = (
            frame.get_role(ThematicRole.PATIENT)
            or frame.get_role(ThematicRole.LOCATION)
            or frame.get_role(ThematicRole.DESTINATION)
        )
        obj_name = target_ref.concept_name if (target_ref and target_ref.concept_name) else ""

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        b_ws = getattr(self._tiered_workspace, "brain", None)
        b_graph = getattr(b_ws, "graph", None)
        graphs = [g for g in (p_graph, b_graph, self._graph) if g is not None]

        subjects: list[str] = []
        if obj_name:
            o_ids = self._find_concept_ids(obj_name)
            for g in graphs:
                for oid in o_ids:
                    for edge in g.edges_to(oid):
                        if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                            continue
                        if edge.properties.get("polarity") is False:
                            continue
                        rel = str(edge.properties.get("relation", "")).lower()
                        if rel == pred or rel.rstrip("s") == pred.rstrip("s"):
                            for sid in edge.sources:
                                lbl = self._label_for_node_id(g, sid)
                                if lbl and lbl not in subjects:
                                    subjects.append(lbl)
                    if subjects:
                        break
                if subjects:
                    break

            # Fallback for property modifiers attached to is_a edges or PhysicalEntityNodes
            if not subjects and pred == "has_property":
                obj_low = obj_name.lower()
                for g in graphs:
                    for node in g.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
                        props = getattr(node, "properties", None) or {}
                        if any(str(v).lower() == obj_low for v in props.values()):
                            subjects.append(getattr(node, "entity_type", "entity"))
                    if subjects:
                        break
                    for node in g.nodes_by_type(HCIRNodeType.CONCEPT):
                        for edge in g.edges_from(node.id):
                            mods = edge.properties.get("modifiers") or {}
                            adjs = mods.get("adj") if isinstance(mods, dict) else []
                            if adjs and obj_low in [str(a).lower() for a in adjs]:
                                lbl = self._label_for_node_id(g, node.id)
                                if lbl and lbl not in subjects:
                                    subjects.append(lbl)
                        if subjects:
                            break
                    if subjects:
                        break

        if subjects:
            ans = subjects[0] if len(subjects) == 1 else ", ".join(subjects[:4])
            return CognitiveEpistemicState(
                target_predicate=pred,
                target_subject=ans,
                target_object=obj_name,
                is_known=True,
                raw_belief_value=ans,
                confidence=0.96,
                uncertainty=0.04,
                support_count=3,
            )

        return CognitiveEpistemicState(
            target_predicate=pred,
            target_subject="",
            target_object=obj_name,
            is_known=False,
            confidence=0.05,
            uncertainty=0.95,
        )

    def _process_requirement_query(
        self, grounded: GroundedSemanticFrame
    ) -> CognitiveEpistemicState:
        """Resolve 'What does X require?' queries by traversing downstream requirement edges."""
        frame = grounded.frame
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        theme_name = theme_ref.concept_name if theme_ref else ""
        if not theme_name:
            return CognitiveEpistemicState(
                target_predicate="requires",
                target_subject="",
                is_known=False,
                confidence=0.0,
            )

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        graphs = [g for g in (p_graph, self._graph) if g is not None]
        th_ids = self._find_concept_ids(theme_name)

        req_found: str | None = None
        for g in graphs:
            for tid in th_ids:
                for edge in g.edges_from(tid):
                    rel = str(edge.properties.get("relation", "")).lower()
                    etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                    if etype == "requires" or rel == "requires":
                        if edge.targets:
                            tgt_node = g.get_node(edge.targets[0])
                            req_found = (
                                getattr(tgt_node, "label", None)
                                or getattr(tgt_node, "concept_name", None)
                                or edge.targets[0]
                                .replace("t4_concept_", "")
                                .replace("concept_", "")
                            )
                            break
                if req_found:
                    break
            if req_found:
                break

        if req_found:
            return CognitiveEpistemicState(
                target_predicate="requires",
                target_subject=theme_name,
                target_object=req_found,
                is_known=True,
                raw_belief_value=req_found,
                confidence=0.98,
                uncertainty=0.02,
                support_count=3,
            )

        prior = self._lookup_epistemic_prior(theme_name, query_target="procedure")
        if prior is not None:
            return prior

        return CognitiveEpistemicState(
            target_predicate="requires",
            target_subject=theme_name,
            is_known=False,
            confidence=0.05,
            uncertainty=0.95,
        )

    def _process_enable_query(self, grounded: GroundedSemanticFrame) -> CognitiveEpistemicState:
        """Resolve 'What does X enable?' or 'What enables Y?' queries."""
        frame = grounded.frame
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        patient_ref = frame.get_role(ThematicRole.PATIENT)

        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        graphs = [g for g in (p_graph, self._graph) if g is not None]

        # Case 1: Downstream query - "What does X enable?"
        if theme_ref and theme_ref.concept_name and not patient_ref:
            theme_name = theme_ref.concept_name
            th_ids = self._find_concept_ids(theme_name)
            target_found: str | None = None
            for g in graphs:
                for tid in th_ids:
                    for edge in g.edges_from(tid):
                        rel = str(edge.properties.get("relation", "")).lower()
                        etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                        if etype in ("causes", "enables") or rel in ("causes", "enables"):
                            if edge.targets:
                                tgt_node = g.get_node(edge.targets[0])
                                target_found = (
                                    getattr(tgt_node, "label", None)
                                    or getattr(tgt_node, "concept_name", None)
                                    or edge.targets[0]
                                    .replace("t4_concept_", "")
                                    .replace("concept_", "")
                                )
                                break
                    if target_found:
                        break
                if target_found:
                    break

            if target_found:
                return CognitiveEpistemicState(
                    target_predicate="enables",
                    target_subject=theme_name,
                    target_object=target_found,
                    is_known=True,
                    raw_belief_value=target_found,
                    confidence=0.98,
                    uncertainty=0.02,
                    support_count=3,
                )
            return CognitiveEpistemicState(
                target_predicate="enables",
                target_subject=theme_name,
                is_known=False,
                confidence=0.05,
                uncertainty=0.95,
            )

        # Case 2: Upstream query - "What enables Y?"
        if patient_ref and patient_ref.concept_name:
            patient_name = patient_ref.concept_name
            pat_ids = self._find_concept_ids(patient_name)
            enabler_found: str | None = None
            for g in graphs:
                for pid in pat_ids:
                    for edge in g.edges_to(pid):
                        if getattr(edge, "edge_type", None) == HCIREdgeType.CONTRADICTS:
                            continue
                        rel = str(edge.properties.get("relation", "")).lower()
                        etype = getattr(edge.edge_type, "value", str(edge.edge_type)).lower()
                        if etype in ("causes", "enables") or rel in ("causes", "enables"):
                            if edge.sources:
                                enabler_found = self._label_for_node_id(g, edge.sources[0])
                                break
                    if enabler_found:
                        break
                if enabler_found:
                    break

            if enabler_found:
                return CognitiveEpistemicState(
                    target_predicate="enables",
                    target_subject=enabler_found,
                    target_object=patient_name,
                    is_known=True,
                    raw_belief_value=enabler_found,
                    confidence=0.98,
                    uncertainty=0.02,
                    support_count=3,
                )
            return CognitiveEpistemicState(
                target_predicate="enables",
                target_subject=patient_name,
                is_known=False,
                confidence=0.05,
                uncertainty=0.95,
            )

        return CognitiveEpistemicState(
            target_predicate="enables",
            target_subject="",
            is_known=False,
            confidence=0.0,
        )

    def _process_identity_query(self, grounded: GroundedSemanticFrame) -> CognitiveEpistemicState:
        """Resolve 'Who is X?' identity queries against EvidenceNodes and ConceptNodes."""
        frame = grounded.frame
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        theme_name = theme_ref.concept_name if theme_ref else ""
        if not theme_name:
            return CognitiveEpistemicState(
                target_predicate="identity",
                target_subject="",
                is_known=False,
                confidence=0.0,
            )

        id_clean = theme_name.lower().strip().replace(" ", "_")
        clean_space = id_clean.replace("_", " ")
        p_ws = getattr(self._tiered_workspace, "persistent", None)
        p_graph = getattr(p_ws, "graph", None)
        graphs = [g for g in (p_graph, self._graph) if g is not None]

        identity_found: str | None = None
        for g in graphs:
            for node in g.all_nodes():
                lbl = (
                    getattr(node, "label", "") or getattr(node, "concept_name", "") or ""
                ).lower()
                nid = node.id.lower()
                if (
                    clean_space in lbl
                    or lbl in clean_space
                    or id_clean in lbl
                    or lbl in id_clean
                    or id_clean in nid
                ):
                    defn = getattr(node, "definition", "")
                    if defn:
                        identity_found = defn
                        break
            if identity_found:
                break

        if not identity_found:
            for node in self._graph.all_nodes():
                tags = getattr(node, "tags", [])
                methodology = getattr(node, "methodology", "")
                if any(
                    clean_space in str(t).lower()
                    or str(t).lower() in clean_space
                    or id_clean in str(t).lower()
                    for t in tags
                ):
                    if "is_a" in methodology:
                        identity_found = methodology.split("is_a")[-1].strip()
                        break

        if identity_found:
            clean_id = identity_found
            if "(" in clean_id and ")" in clean_id:
                clean_id = clean_id.split("(")[1].split(")")[0].strip()
            return CognitiveEpistemicState(
                target_predicate="identity",
                target_subject=theme_name,
                target_object=clean_id,
                is_known=True,
                raw_belief_value=clean_id,
                confidence=0.98,
                uncertainty=0.02,
                support_count=3,
            )

        return CognitiveEpistemicState(
            target_predicate="identity",
            target_subject=theme_name,
            is_known=False,
            confidence=0.05,
            uncertainty=0.95,
        )

    # ── Epistemic Prior Lookup ────────────────────────────────────────

    def _lookup_epistemic_prior(
        self, subject_name: str, query_target: str = "definition"
    ) -> CognitiveEpistemicState | None:
        """Consult ConceptExplainer, LexicalOntologyStore, and persistent memory.

        Returns a definition/explanation CognitiveEpistemicState if found, else None.
        Multi-hop traversal synthesizes definition, taxonomy, requirements, causes, and beliefs.
        """
        try:
            from hbllm.hcir.capabilities.concept_explainer import ConceptExplainer

            explainer = ConceptExplainer(
                tiered_workspace=self._tiered_workspace,
                ontology_store=self._ontology_store,
            )
            snapshot = explainer.gather_knowledge(subject_name)
            if snapshot.has_knowledge:
                if query_target == "cause":
                    discourse = explainer.explain_why(subject_name)
                    pred_name = "causes"
                elif query_target == "procedure":
                    discourse = explainer.explain_how(subject_name)
                    pred_name = "requires"
                else:
                    discourse = explainer.synthesize_discourse(snapshot)
                    pred_name = "definition"

                return CognitiveEpistemicState(
                    target_predicate=pred_name,
                    target_subject=subject_name,
                    target_object=discourse,
                    confidence=0.90 if snapshot.definition else 0.75,
                    uncertainty=0.10 if snapshot.definition else 0.25,
                    support_count=max(1, len(snapshot.hypernyms) + len(snapshot.causes)),
                    is_known=True,
                    raw_belief_value=discourse,
                    provenance=snapshot.provenance or "concept_explainer",
                )
        except Exception as exc:
            logger.debug("HCIRGateway: concept explainer error for %r: %s", subject_name, exc)

        return None

    # ── Command: Goal & Action Proposals ──────────────────────────────

    def process_command(self, grounded: GroundedSemanticFrame) -> GoalNode:
        """Convert a grounded command frame into a GoalNode in HCIR."""
        frame = grounded.frame
        action_verb = frame.predicate
        patient_ref = frame.get_role(ThematicRole.PATIENT) or frame.get_role(ThematicRole.THEME)
        destination_ref = frame.get_role(ThematicRole.DESTINATION) or frame.get_role(
            ThematicRole.LOCATION
        )

        patient_name = patient_ref.concept_name if patient_ref else "object"
        dest_name = destination_ref.concept_name if destination_ref else ""

        goal_spec = f"{action_verb} {patient_name} to {dest_name}".strip()
        goal = GoalNode(
            description=goal_spec,
            priority=0.8,
            tags=["language_command", frame.metadata.language, action_verb],
        )
        self._commit_node(goal)
        return goal

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _predicate_to_edge_type(predicate: str) -> HCIREdgeType | None:
        """Map a semantic predicate to an HCIREdgeType for spatial/relational edges."""
        _MAP: dict[str, HCIREdgeType] = {
            "located_on": HCIREdgeType.LOCATED_IN,
            "located_in": HCIREdgeType.LOCATED_IN,
            "located_at": HCIREdgeType.LOCATED_AT,
            "located_inside": HCIREdgeType.LOCATED_IN,
            "located_within": HCIREdgeType.LOCATED_IN,
            "located_under": HCIREdgeType.BELOW,
            "located_beneath": HCIREdgeType.BELOW,
            "below": HCIREdgeType.BELOW,
            "under": HCIREdgeType.BELOW,
            "located_above": HCIREdgeType.ABOVE,
            "above": HCIREdgeType.ABOVE,
            "located_beside": HCIREdgeType.NEAR,
            "located_near": HCIREdgeType.NEAR,
            "near": HCIREdgeType.NEAR,
            "supports": HCIREdgeType.SUPPORTS,
            "part_of": HCIREdgeType.PART_OF,
            "contains": HCIREdgeType.CONTAINS,
            "is_a": HCIREdgeType.GENERALIZES,
            "causes": HCIREdgeType.CAUSES,
            "requires": HCIREdgeType.REQUIRES,
            "enables": HCIREdgeType.CAUSES,
        }
        return _MAP.get(predicate)
