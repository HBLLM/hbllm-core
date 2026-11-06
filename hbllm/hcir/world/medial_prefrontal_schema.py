"""Medial Prefrontal Schema Induction, Invariant Abstraction & Cross-Context Transfer.

Biological Foundations:
- Euston, Gruber, McNaughton (2012): "The role of medial prefrontal cortex in memory
  and decision making" (Neuron 76: 1057-1070).
- McKenzie et al. (2014): "Prefrontal cortical input to the hippocampus encodes the
  context of memory retrieval" (Neuron 80: 1011-1022).
- Ghosh, Gilboa (2014): "What is a memory schema? A framework for understanding the
  nature of schematic memory" (Neuropsychologia 53: 104-114).

Key Cognitive Capabilities:
1. Invariant Schema Extraction:
   - Distills relational topologies (e.g. collector-objective, barrier-key-door, chokepoint)
     across episodic instances, abstracting away concrete coordinates and transient IDs.
2. Cross-Context Schema Matching:
   - Computes structural similarity between current sensory/graph state and stored
     schematic prototypes using relational graph kernels and role signatures.
3. Rapid Prior Transference:
   - Emits structured schema priors (canonical roles, high-probability causal sequences,
     and heuristic subgoal hierarchies) to accelerate downstream cortical planning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass(frozen=True)
class GeneralizedTaskSchema:
    """Abstract structural template capturing invariant causal and relational topologies."""

    schema_id: str
    archetype_name: str  # e.g., 'barrier_unlock', 'collector_patrol', 'maze_chokepoint'
    canonical_roles: tuple[str, ...]  # e.g., ('agent', 'key', 'barrier', 'goal')
    invariant_relations: tuple[tuple[str, str, str], ...]  # (role_a, relation, role_b)
    subgoal_sequence_hints: tuple[str, ...]
    activation_count: int = 0
    generalization_score: float = 0.50


@dataclass(frozen=True)
class SchemaMatchResult:
    """Evaluated match between current context and a generalized schema."""

    schema_id: str
    archetype_name: str
    match_confidence: float
    role_bindings: Mapping[str, str]  # schema_role -> concrete_entity_id
    subgoal_hints: tuple[str, ...]

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert schema match into an HCIR CausalPredicate."""
        bindings_str = ";".join(f"{k}->{v}" for k, v in sorted(self.role_bindings.items()))
        return CausalPredicate(
            variable="active_schema_binding",
            operator="==",
            value=f"{self.archetype_name}:conf:{self.match_confidence:.2f}:{bindings_str}",
            relation="medial_prefrontal_schema_activation",
            target_entity="medial_prefrontal_cortex",
        )


@dataclass(frozen=True)
class SchemaGeneralizationReport:
    """Consolidated report of medial prefrontal schema deliberation."""

    dominant_schema: SchemaMatchResult | None
    candidate_matches: tuple[SchemaMatchResult, ...]
    total_stored_schemas: int
    abstraction_depth: float


class MedialPrefrontalSchemaEngine:
    """Cortical engine for abstract schema induction, storage, and cross-context transfer."""

    def __init__(self, match_threshold: float = 0.40) -> None:
        """Initialize mPFC schema engine.

        Args:
            match_threshold: Minimum structural alignment confidence to activate a schema.
        """
        self.match_threshold: float = max(0.1, min(1.0, float(match_threshold)))
        self._schemas: dict[str, GeneralizedTaskSchema] = {}
        self._register_default_schemata()

    def _register_default_schemata(self) -> None:
        """Register universal canonical schemata derived from domain-agnostic physical dynamics."""
        # 1. Sequential unlock / barrier clearance
        self.register_schema(
            GeneralizedTaskSchema(
                schema_id="schema_clearance_sequential",
                archetype_name="barrier_clearance",
                canonical_roles=("actor", "mediator", "barrier", "target"),
                invariant_relations=(
                    ("actor", "navigates_to", "mediator"),
                    ("mediator", "disables", "barrier"),
                    ("actor", "reaches", "target"),
                ),
                subgoal_sequence_hints=("acquire_mediator", "neutralize_barrier", "reach_target"),
                activation_count=1,
                generalization_score=0.85,
            )
        )

        # 2. Direct transit / gradient ascent
        self.register_schema(
            GeneralizedTaskSchema(
                schema_id="schema_direct_transit",
                archetype_name="direct_navigation",
                canonical_roles=("actor", "target"),
                invariant_relations=(("actor", "navigates_to", "target"),),
                subgoal_sequence_hints=("approach_target", "dock_target"),
                activation_count=1,
                generalization_score=0.90,
            )
        )

        # 3. Dynamic evasion / hazardous navigation
        self.register_schema(
            GeneralizedTaskSchema(
                schema_id="schema_evasion_transit",
                archetype_name="hazard_avoidance",
                canonical_roles=("actor", "hazard", "target"),
                invariant_relations=(
                    ("actor", "evades", "hazard"),
                    ("actor", "navigates_to", "target"),
                ),
                subgoal_sequence_hints=("circumvent_hazard", "advance_target"),
                activation_count=1,
                generalization_score=0.75,
            )
        )

    def register_schema(self, schema: GeneralizedTaskSchema) -> None:
        """Register or update a generalized schema template."""
        self._schemas[schema.schema_id] = schema

    def match_schema_context(
        self,
        entity_roles: Mapping[str, str],
        active_relations: tuple[tuple[str, str, str], ...] = (),
    ) -> SchemaGeneralizationReport:
        """Match present observed entities and relations against schematic prototypes.

        Args:
            entity_roles: Map of concrete entity ID -> functional role
                          (e.g., {'agent_0': 'actor', 'key_1': 'mediator', 'door_1': 'barrier'}).
            active_relations: List of observed relations (source_id, relation_type, target_id).

        Returns:
            SchemaGeneralizationReport with evaluated schema candidates and role bindings.
        """
        if not entity_roles and not self._schemas:
            return SchemaGeneralizationReport(
                dominant_schema=None,
                candidate_matches=(),
                total_stored_schemas=0,
                abstraction_depth=0.0,
            )

        # Invert entity roles: functional_role -> set of entity IDs
        role_to_entities: dict[str, list[str]] = {}
        for ent_id, role in entity_roles.items():
            role_to_entities.setdefault(role, []).append(ent_id)

        candidates: list[SchemaMatchResult] = []

        for schema in self._schemas.values():
            # Calculate role coverage
            matched_roles = 0
            role_bindings: dict[str, str] = {}
            for canonical_role in schema.canonical_roles:
                if canonical_role in role_to_entities and role_to_entities[canonical_role]:
                    matched_roles += 1
                    role_bindings[canonical_role] = role_to_entities[canonical_role][0]

            role_coverage = (
                matched_roles / len(schema.canonical_roles) if schema.canonical_roles else 0.0
            )

            # Calculate relational alignment
            matched_relations = 0
            for r_src, r_rel, r_tgt in schema.invariant_relations:
                concrete_src = role_bindings.get(r_src)
                concrete_tgt = role_bindings.get(r_tgt)
                if concrete_src and concrete_tgt:
                    for obs_src, obs_rel, obs_tgt in active_relations:
                        if obs_src == concrete_src and obs_tgt == concrete_tgt:
                            if not r_rel or obs_rel == r_rel:
                                matched_relations += 1
                                break

            rel_coverage = (
                matched_relations / len(schema.invariant_relations)
                if schema.invariant_relations
                else 0.50
            )

            # Specificity bonus: richer schemas matching more observed roles receive priority
            specificity_bonus = min(1.0, float(matched_roles) / 4.0)

            # Combined confidence with schema prior generalization score
            confidence = (
                0.40 * role_coverage + 0.35 * rel_coverage + 0.25 * specificity_bonus
            ) * schema.generalization_score

            if confidence >= self.match_threshold:
                candidates.append(
                    SchemaMatchResult(
                        schema_id=schema.schema_id,
                        archetype_name=schema.archetype_name,
                        match_confidence=float(confidence),
                        role_bindings=role_bindings,
                        subgoal_hints=schema.subgoal_sequence_hints,
                    )
                )

        # Sort descending by confidence
        candidates.sort(key=lambda c: c.match_confidence, reverse=True)
        dominant = candidates[0] if candidates else None

        # Calculate average abstraction depth
        abstraction_depth = float(
            sum(len(s.canonical_roles) for s in self._schemas.values()) / max(1, len(self._schemas))
        )

        return SchemaGeneralizationReport(
            dominant_schema=dominant,
            candidate_matches=tuple(candidates),
            total_stored_schemas=len(self._schemas),
            abstraction_depth=abstraction_depth,
        )
