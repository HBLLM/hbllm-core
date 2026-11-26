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
from collections.abc import Mapping
from typing import Any

from hbllm.hcir.fuzzy import FuzzyProfileId, apply_hedge, get_fuzzy_profile
from hbllm.hcir.graph import (
    CognitiveGraph,
    GroundedConceptNode,
    HCIRNodeType,
    PhysicalEntityNode,
)
from hbllm.hcir.language.core.normalizer import damerau_levenshtein
from hbllm.hcir.language.core.reference import ReferenceResolver
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    GroundedSemanticFrame,
    LanguageErrorType,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.seed.loader import load_constructions_spec

logger = logging.getLogger(__name__)


def _merged_fuzzy_lexical_rules(language: str = "en") -> dict[str, Any]:
    """Load declarative hedges, gradable adjectives, and spatial terms from YAML seeds."""
    langs = [language] + [l for l in ("en", "si", "ta") if l != language]
    hedges: dict[str, str] = {}
    gradable: dict[str, dict[str, Any]] = {}
    spatial: dict[str, str] = {}
    articles: list[str] = []
    for lang in langs:
        rules = load_constructions_spec(lang).get("lexical_rules", {})
        for k, v in (rules.get("hedges") or {}).items():
            hedges.setdefault(str(k).lower(), str(v))
        for k, v in (rules.get("gradable_adjectives") or {}).items():
            if isinstance(v, Mapping):
                gradable.setdefault(str(k).lower(), dict(v))
        for k, v in (rules.get("spatial_relations") or {}).items():
            spatial.setdefault(str(k).lower(), str(v))
        for art in rules.get("articles") or ():
            s_art = str(art).lower()
            if s_art not in articles:
                articles.append(s_art)
    return {
        "hedges": hedges,
        "gradable": gradable,
        "spatial": spatial,
        "articles": articles,
    }


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
        *,
        fuzzy_profile: str | FuzzyProfileId = FuzzyProfileId.PRODUCT,
        ambiguity_tie_margin: float = 0.08,
        min_referent_degree: float = 0.25,
    ) -> None:
        self._graph = graph
        self._ref_resolver = reference_resolver or ReferenceResolver()
        if workspace is None:
            from hbllm.hcir.workspace import HCIRWorkspaceState

            workspace = HCIRWorkspaceState(graph=graph)
        self._workspace = workspace
        self._lexicon = lexicon
        self._fuzzy_profile = FuzzyProfileId(fuzzy_profile)
        self._ambiguity_tie_margin = float(ambiguity_tie_margin)
        self._min_referent_degree = float(min_referent_degree)

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

            lang = getattr(getattr(frame, "metadata", None), "language", "en") or "en"
            lex_rules = _merged_fuzzy_lexical_rules(str(lang))

            # 3. Match Physical Entities in A13 (with fuzzy fallback for untrusted typos)
            matching_entities: list[PhysicalEntityNode] = []
            for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
                if isinstance(node, PhysicalEntityNode):
                    if self._entity_matches_reference(node, ref, lex_rules=lex_rules):
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
                            if self._entity_matches_reference(
                                node, corrected_ref, lex_rules=lex_rules
                            ):
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

            ranked_entities, ranked_scores, has_graded_signal = self._score_candidate_referents(
                frame=frame,
                role=role,
                ref=ref,
                candidates=matching_entities,
                grounded=grounded,
                lex_rules=lex_rules,
            )
            matching_entities = ranked_entities
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
            is_conceptual_query = frame.frame_type == FrameType.QUERY and (
                frame.query_target in ("svo_object", "svo_location", "svo_subject", "identity")
                or is_conceptual_pred
                or (frame.query_target == "verification" and not is_spatial_pred)
            )
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
                    # CONCEPTUAL / SVO QUERY: allow persistent conceptual/SVO resolution
                    elif is_conceptual_query:
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
                    top_deg = ranked_scores[0] if ranked_scores else 0.0
                    second_deg = ranked_scores[1] if len(ranked_scores) > 1 else 0.0
                    if (
                        has_graded_signal
                        and top_deg >= self._min_referent_degree
                        and (top_deg - second_deg) >= self._ambiguity_tie_margin
                    ):
                        chosen_entity = matching_entities[0]
                        grounded.grounded_entities[role] = chosen_entity.id
                        new_mentions.append(
                            (
                                chosen_entity.id,
                                ref.concept_name or chosen_entity.entity_type,
                                ref.properties,
                            )
                        )
                    else:
                        # Ambiguous reference (equal or within tie margin)
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

    def _score_candidate_referents(
        self,
        *,
        frame: SemanticFrame,
        role: ThematicRole,
        ref: EntityReference,
        candidates: list[PhysicalEntityNode],
        grounded: GroundedSemanticFrame,
        lex_rules: dict[str, Any],
    ) -> tuple[list[PhysicalEntityNode], list[float], bool]:
        """Rank candidate referents via `FuzzyLogicProfile` t-norms over graded cues."""
        if len(candidates) <= 1:
            return candidates, [1.0] * len(candidates), False

        prof = get_fuzzy_profile(self._fuzzy_profile)
        hedges_map: dict[str, str] = lex_rules["hedges"]
        gradable_map: dict[str, dict[str, Any]] = lex_rules["gradable"]
        spatial_map: dict[str, str] = lex_rules["spatial"]
        articles: list[str] = lex_rules["articles"]

        raw_low = (ref.raw_text or "").lower()
        hedge_id = "identity"
        for h_tok, h_fn in sorted(hedges_map.items(), key=lambda kv: -len(kv[0])):
            if f" {h_tok} " in f" {raw_low} ":
                hedge_id = h_fn
                break

        # Detect gradable dimension constraints from properties and raw_text
        requested_gradables: list[tuple[str, str, str]] = []
        for pk, pv in ref.properties.items():
            pv_low = str(pv).lower().strip()
            for adj_tok, adj_spec in gradable_map.items():
                if adj_tok == pv_low or f" {adj_tok} " in f" {pv_low} ":
                    dim = str(adj_spec.get("dimension", pk))
                    pol = str(adj_spec.get("polarity", "high"))
                    adj_h = str(adj_spec.get("default_hedge", hedge_id))
                    requested_gradables.append((dim, pol, adj_h))
        for adj_tok, adj_spec in gradable_map.items():
            if f" {adj_tok} " in f" {raw_low} ":
                dim = str(adj_spec.get("dimension", "size"))
                pol = str(adj_spec.get("polarity", "high"))
                adj_h = str(adj_spec.get("default_hedge", hedge_id))
                item = (dim, pol, adj_h)
                if item not in requested_gradables:
                    requested_gradables.append(item)

        # Detect landmark entity constraint from pre-grounded frame roles or raw_text spatial clause
        landmark_ids: list[tuple[str, str]] = []
        if role not in (ThematicRole.LOCATION, ThematicRole.DESTINATION, ThematicRole.SOURCE):
            for lrole in (ThematicRole.LOCATION, ThematicRole.DESTINATION, ThematicRole.SOURCE):
                if l_eid := grounded.grounded_entities.get(lrole):
                    pred_rel = spatial_map.get(
                        str(frame.predicate or "").replace("located_", "").lower(),
                        str(frame.predicate or "").upper(),
                    )
                    landmark_ids.append((l_eid, pred_rel))
        for sp_phrase, sp_rel in sorted(spatial_map.items(), key=lambda kv: -len(kv[0])):
            if f" {sp_phrase} " in f" {raw_low} ":
                after = raw_low.split(f" {sp_phrase} ", 1)[1].strip()
                for art in articles:
                    if after.startswith(art):
                        after = after[len(art) :].strip()
                if after:
                    for gnode in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
                        if (
                            isinstance(gnode, PhysicalEntityNode)
                            and gnode.entity_type.lower() == after
                        ):
                            landmark_ids.append((gnode.id, sp_rel))

        # Precompute min-max range for numeric gradable dimensions across candidates
        dim_ranges: dict[str, tuple[float, float]] = {}
        for dim, _, _ in requested_gradables:
            vals: list[float] = []
            for ent in candidates:
                props = (
                    getattr(ent, "properties", None)
                    or getattr(ent, "observed_properties", {})
                    or {}
                )
                for key_c in (dim, "area", "extent", "size") if dim == "size" else (dim,):
                    v = props.get(key_c)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.append(float(v))
                        break
            if vals:
                dim_ranges[dim] = (min(vals), max(vals))

        has_graded_signal = bool(requested_gradables or landmark_ids)
        scored: list[tuple[float, PhysicalEntityNode]] = []

        for ent in candidates:
            props = (
                getattr(ent, "properties", None) or getattr(ent, "observed_properties", {}) or {}
            )
            f_map = props.get("fuzzy_memberships")
            if isinstance(f_map, Mapping) and f_map:
                has_graded_signal = True

            deg = 1.0
            if ref.concept_name and isinstance(f_map, Mapping):
                c_key = f"concept:{ref.concept_name.upper()}"
                if c_key in f_map:
                    c_entry = f_map[c_key]
                    c_val = (
                        float(c_entry["degree"])
                        if isinstance(c_entry, Mapping) and "degree" in c_entry
                        else float(c_entry)
                    )
                    deg = prof.t_norm(deg, max(0.0, min(1.0, c_val)))

            for dim, pol, adj_h in requested_gradables:
                g_deg: float | None = None
                if isinstance(f_map, Mapping):
                    for k_f in (f"property:{dim}", dim):
                        if k_f in f_map:
                            e_f = f_map[k_f]
                            g_deg = (
                                float(e_f["degree"])
                                if isinstance(e_f, Mapping) and "degree" in e_f
                                else float(e_f)
                            )
                            break
                if g_deg is None and dim in dim_ranges:
                    lo, hi = dim_ranges[dim]
                    for key_c in (dim, "area", "extent", "size") if dim == "size" else (dim,):
                        raw_num = props.get(key_c)
                        if isinstance(raw_num, (int, float)) and not isinstance(raw_num, bool):
                            span = hi - lo
                            if span > 1e-9:
                                ratio = (float(raw_num) - lo) / span
                                norm_r = ratio if pol == "high" else (1.0 - ratio)
                                g_deg = 0.05 + 0.90 * norm_r
                            else:
                                g_deg = 0.5
                            break
                if g_deg is not None:
                    g_deg = apply_hedge(max(0.0, min(1.0, g_deg)), adj_h)
                    deg = prof.t_norm(deg, g_deg)

            for l_eid, rel_hint in landmark_ids:
                best_rel_deg = 0.0
                for edge in list(self._graph.edges_from(ent.id)) + list(
                    self._graph.edges_from(l_eid)
                ):
                    if l_eid not in edge.targets and ent.id not in edge.targets:
                        continue
                    eprops = edge.properties if isinstance(edge.properties, Mapping) else {}
                    pred_u = str(
                        eprops.get("predicate", getattr(edge.edge_type, "value", edge.edge_type))
                    ).upper()
                    if rel_hint and rel_hint not in pred_u and pred_u not in rel_hint:
                        continue
                    edeg = (
                        float(eprops["semantic_degree"])
                        if isinstance(eprops.get("semantic_degree"), (int, float))
                        else float(edge.weight)
                    )
                    best_rel_deg = max(best_rel_deg, max(0.0, min(1.0, edeg)))
                if best_rel_deg > 0.0 or any(self._graph.edges_from(ent.id)):
                    deg = prof.t_norm(deg, best_rel_deg)

            scored.append((deg, ent))

        if not has_graded_signal:
            return candidates, [1.0] * len(candidates), False

        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [item[1] for item in scored], [item[0] for item in scored], True

    def _entity_matches_reference(
        self,
        entity: PhysicalEntityNode,
        ref: EntityReference,
        *,
        lex_rules: dict[str, Any] | None = None,
    ) -> bool:
        """Check whether a PhysicalEntityNode satisfies the linguistic reference."""
        # Concept / Type check
        if ref.concept_name:
            cname = ref.concept_name.lower()
            etype = entity.entity_type.lower()
            if cname != etype and cname not in etype:
                return False

        # Property constraints (e.g., color="red" stays crisp; gradables score downstream)
        rules = lex_rules or _merged_fuzzy_lexical_rules()
        gradable_map = rules.get("gradable", {})
        hedges_map = rules.get("hedges", {})
        props = (
            getattr(entity, "properties", None) or getattr(entity, "observed_properties", {}) or {}
        )
        for prop_key, prop_val in ref.properties.items():
            pv_low = str(prop_val).lower().strip()
            toks = [t for t in pv_low.split() if t not in hedges_map]
            is_gradable_val = prop_key in gradable_map or any(t in gradable_map for t in toks)
            if is_gradable_val:
                continue
            ent_val = props.get(prop_key)
            if ent_val is None or str(ent_val).lower() != pv_low:
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
        from hbllm.hcir.types import Scope

        tenant_id = getattr(self._workspace, "tenant_id", None) or "default"
        concept_name = ref.concept_name or "entity"
        entity = PhysicalEntityNode(
            entity_type=concept_name,
            entity_name=concept_name,
            name=concept_name,
            properties=dict(ref.properties) if ref.properties else {},
            scope=Scope(tenant_id=tenant_id),
            tags=["language_introduced", concept_name],
        )

        # Commit atomically through workspace TransactionManager
        self._workspace.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=entity.id,
                    node_data=entity.model_dump(),
                )
            ],
            author="grounding_resolver",
            tenant_id=tenant_id,
        )

        logger.info(
            "GroundingResolver: auto-created PhysicalEntityNode '%s' (id=%s) from assertion",
            concept_name,
            entity.id,
        )
        return entity
