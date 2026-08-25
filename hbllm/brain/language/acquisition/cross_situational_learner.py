"""Cross-Situational Learner & State Transition Grounding for A17.

Accumulates evidence across varied visual scenes, state transitions,
spatial configurations, and property contrasts to ground novel lexical tokens.
"""

from __future__ import annotations

from typing import Any

from hbllm.brain.concepts.grounded_concept_registry import GroundedConceptRegistry
from hbllm.brain.language.acquisition.lexical_hypothesis import (
    EvidenceSourceType,
    LexicalEvidence,
    LexicalHypothesisSet,
    LexicalTargetType,
)
from hbllm.brain.language.acquisition.scoring import apply_evidence_to_candidate
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode


class CrossSituationalLearner:
    """Discovers lexical groundings across scenes, transitions, and contexts."""

    def __init__(
        self,
        graph: CognitiveGraph,
        concept_registry: GroundedConceptRegistry | None = None,
    ) -> None:
        self._graph = graph
        self._concept_registry = concept_registry or GroundedConceptRegistry(graph)

    # ── Fast Mapping & Scene Observation ──────────────────────────────

    def observe_situated_token(
        self,
        token: str,
        hypothesis_set: LexicalHypothesisSet,
        visible_entity_ids: list[str],
        spatial_edges: list[tuple[str, str, str]] | None = None,  # (source_id, edge_type, target_id)
        state_delta: dict[str, Any] | None = None,
        language: str = "en",
        timestamp: float = 0.0,
    ) -> list[LexicalEvidence]:
        """Process a situated token utterance in the context of visible entities/events.

        Generates LexicalEvidence and updates the competing LexicalHypothesisSet.
        Does NOT presuppose the semantic category of the token.
        """
        generated_evidence: list[LexicalEvidence] = []

        # 1. Candidate Generation across Physical Entities (Concept, Individual, Property)
        for entity_id in visible_entity_ids:
            entity = self._graph.get_node(entity_id)
            if not isinstance(entity, PhysicalEntityNode):
                continue

            props = entity.properties if hasattr(entity, "properties") and isinstance(entity.properties, dict) else entity.observed_properties

            # Candidate: Individual Instance
            c_ind = hypothesis_set.add_or_get_candidate(
                target_type=LexicalTargetType.INDIVIDUAL,
                target_id=entity_id,
                target_value={"entity_type": entity.entity_type},
                timestamp=timestamp,
            )
            ev_ind = LexicalEvidence(
                source_type=EvidenceSourceType.CROSS_SITUATIONAL,
                token=token,
                language=language,
                target_type=LexicalTargetType.INDIVIDUAL,
                target_value=entity_id,
                is_positive=True,
                context_entities=visible_entity_ids,
                timestamp=timestamp,
            )
            apply_evidence_to_candidate(c_ind, ev_ind)
            generated_evidence.append(ev_ind)

            # Candidate: Concept / Category (Entity Type or shape prototype)
            cat_name = entity.entity_type
            c_cat = hypothesis_set.add_or_get_candidate(
                target_type=LexicalTargetType.CONCEPT,
                target_id=cat_name,
                target_value={"shape": props.get("shape", cat_name)},
                timestamp=timestamp,
            )
            ev_cat = LexicalEvidence(
                source_type=EvidenceSourceType.CROSS_SITUATIONAL,
                token=token,
                language=language,
                target_type=LexicalTargetType.CONCEPT,
                target_value=cat_name,
                is_positive=True,
                context_entities=visible_entity_ids,
                timestamp=timestamp,
            )
            apply_evidence_to_candidate(c_cat, ev_cat)
            generated_evidence.append(ev_cat)

            # Candidate: Properties (e.g. color, size, material)
            for prop_key, prop_val in props.items():
                if prop_key in ("freshness", "timestamp", "location", "shape") or not isinstance(prop_val, (str, int, float)):
                    continue
                prop_target_id = f"{prop_key}:{prop_val}"
                c_prop = hypothesis_set.add_or_get_candidate(
                    target_type=LexicalTargetType.PROPERTY,
                    target_id=prop_target_id,
                    target_value={prop_key: prop_val},
                    timestamp=timestamp,
                )
                ev_prop = LexicalEvidence(
                    source_type=EvidenceSourceType.CROSS_SITUATIONAL,
                    token=token,
                    language=language,
                    target_type=LexicalTargetType.PROPERTY,
                    target_value={prop_key: prop_val},
                    is_positive=True,
                    context_entities=visible_entity_ids,
                    timestamp=timestamp,
                )
                apply_evidence_to_candidate(c_prop, ev_prop)
                generated_evidence.append(ev_prop)

        # 2. Candidate Generation across Spatial Relations (Prepositions)
        if spatial_edges:
            for src_id, rel_type, tgt_id in spatial_edges:
                rel_target_id = f"relation:{rel_type.lower()}"
                c_rel = hypothesis_set.add_or_get_candidate(
                    target_type=LexicalTargetType.RELATION,
                    target_id=rel_target_id,
                    target_value={"relation": rel_type, "source": src_id, "target": tgt_id},
                    timestamp=timestamp,
                )
                ev_rel = LexicalEvidence(
                    source_type=EvidenceSourceType.SPATIAL_RELATIONAL,
                    token=token,
                    language=language,
                    target_type=LexicalTargetType.RELATION,
                    target_value=rel_type,
                    is_positive=True,
                    context_entities=visible_entity_ids,
                    timestamp=timestamp,
                )
                apply_evidence_to_candidate(c_rel, ev_rel)
                generated_evidence.append(ev_rel)

        # 3. Candidate Generation across State Transitions (Actions & Events)
        if state_delta:
            action_name = str(state_delta.get("action", state_delta.get("transition", "transition")))
            is_agentive = state_delta.get("agentive", True)
            target_type = LexicalTargetType.ACTION if is_agentive else LexicalTargetType.EVENT

            c_act = hypothesis_set.add_or_get_candidate(
                target_type=target_type,
                target_id=f"transition:{action_name.lower()}",
                target_value=state_delta,
                timestamp=timestamp,
            )
            ev_act = LexicalEvidence(
                source_type=EvidenceSourceType.ACTION_TRANSITION,
                token=token,
                language=language,
                target_type=target_type,
                target_value=state_delta,
                is_positive=True,
                state_delta=state_delta,
                context_entities=visible_entity_ids,
                timestamp=timestamp,
            )
            apply_evidence_to_candidate(c_act, ev_act)
            generated_evidence.append(ev_act)

        # 4. Cross-Situational Distractor Pruning:
        # Penalize candidates in the hypothesis set that were NOT present in this scene
        for candidate in hypothesis_set.candidates:
            if candidate.target_type == LexicalTargetType.INDIVIDUAL and candidate.target_id not in visible_entity_ids:
                # Individual was absent in this scene where token was used -> contradict
                ev_neg = LexicalEvidence(
                    source_type=EvidenceSourceType.CROSS_SITUATIONAL,
                    token=token,
                    language=language,
                    target_type=LexicalTargetType.INDIVIDUAL,
                    target_value=candidate.target_id,
                    is_positive=False,
                    context_entities=visible_entity_ids,
                    timestamp=timestamp,
                )
                apply_evidence_to_candidate(candidate, ev_neg)
                generated_evidence.append(ev_neg)

        return generated_evidence
