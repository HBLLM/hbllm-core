from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle
from hbllm.hcir.grounding.contracts.quality import GroundingQuality
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef


@dataclass
class GroundedConcept:
    """A multi-faceted, predictive, relational concept in the SGE Working Store.

    Represents an emergent or consolidated concept through exemplary prototypes,
    running feature distributions, structural invariants, behavioral affordances,
    and late-bound linguistic aliases.
    """

    concept_ref: GroundedConceptRef
    concept_version: int = 1
    lifecycle_state: ConceptLifecycle = ConceptLifecycle.EMERGING

    # Structural & Feature Representations
    prototypes: list[dict[str, Any]] = field(default_factory=list)
    feature_distributions: dict[str, tuple[float, float]] = field(default_factory=dict)
    feature_counts: dict[str, int] = field(default_factory=dict)
    feature_m2: dict[str, float] = field(default_factory=dict)  # For Welford variance
    structural_invariants: set[str] = field(default_factory=set)

    # Compositional & Relational Bindings (G6)
    parts: dict[str, str] = field(default_factory=dict)  # part_name -> concept_id
    role_bindings: dict[str, str] = field(default_factory=dict)  # role_name -> entity_or_concept_id
    higher_order_relations: list[dict[str, Any]] = field(
        default_factory=list
    )  # relations between concepts/relations
    relational_graph: Any | None = None  # Relational scene graph representation

    # Behavioral & Functional Affordances
    relational_patterns: list[str] = field(default_factory=list)
    affordances: set[str] = field(default_factory=set)
    predictive_models: dict[str, str] = field(default_factory=dict)

    # Empirical Grounding Anchors
    exemplar_entity_ids: set[str] = field(default_factory=set)
    counterexamples: set[str] = field(default_factory=set)
    predictive_accuracy: float = 0.5
    grounding_quality: GroundingQuality = field(default_factory=GroundingQuality.zero)

    # Late-Bound Linguistic Grounding
    linguistic_aliases: dict[str, float] = field(default_factory=dict)

    # Timestamps & History
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    lifecycle_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.concept_ref, GroundedConceptRef):
            raise TypeError(f"concept_ref must be GroundedConceptRef, got {type(self.concept_ref)}")
        if not self.lifecycle_history:
            self.lifecycle_history.append(
                {
                    "state": self.lifecycle_state.value,
                    "version": self.concept_version,
                    "timestamp": self.created_at,
                    "reason": "Initial concept creation",
                }
            )

    def update_feature_observation(self, feature_name: str, value: float) -> None:
        """Online update of feature running mean and variance using Welford's algorithm."""
        n = self.feature_counts.get(feature_name, 0) + 1
        self.feature_counts[feature_name] = n

        mean, _ = self.feature_distributions.get(feature_name, (0.0, 0.0))
        delta = value - mean
        new_mean = mean + (delta / n)

        m2 = self.feature_m2.get(feature_name, 0.0) + delta * (value - new_mean)
        self.feature_m2[feature_name] = m2

        var = (m2 / (n - 1)) if n > 1 else 0.0
        self.feature_distributions[feature_name] = (new_mean, var)
        self.last_updated_at = time.time()

    def add_prototype(self, prototype_features: dict[str, Any], max_prototypes: int = 8) -> None:
        """Adds an exemplary feature prototype, keeping the prototype set bounded."""
        self.prototypes.append(dict(prototype_features))
        if len(self.prototypes) > max_prototypes:
            self.prototypes.pop(0)

        # Update running distributions for numeric features
        for k, v in prototype_features.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.update_feature_observation(k, float(v))

        self.last_updated_at = time.time()

    def add_exemplar(self, entity_id: str) -> None:
        """Anchors a positive exemplar entity to this concept."""
        self.exemplar_entity_ids.add(entity_id)
        self.last_updated_at = time.time()

    def add_counterexample(self, entity_id: str) -> None:
        """Anchors a negative boundary counterexample (e.g. helmet vs bowl)."""
        self.counterexamples.add(entity_id)
        self.last_updated_at = time.time()

    def add_affordance(self, affordance: str) -> None:
        """Attaches an action affordance (e.g. 'GRASP', 'OPEN', 'CONTAIN')."""
        self.affordances.add(affordance.upper())
        self.last_updated_at = time.time()

    def add_linguistic_alias(self, label: str, confidence: float = 1.0) -> None:
        """Binds a late linguistic label to the grounded concept."""
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
        self.linguistic_aliases[label.lower()] = confidence
        self.last_updated_at = time.time()

    def update_grounding_quality(self, quality: GroundingQuality) -> None:
        """Updates the 7-dimensional grounding quality assessment."""
        if not isinstance(quality, GroundingQuality):
            raise TypeError(f"quality must be GroundingQuality, got {type(quality)}")
        self.grounding_quality = quality
        self.predictive_accuracy = quality.predictive_utility
        self.last_updated_at = time.time()

    def transition_lifecycle(
        self,
        new_state: ConceptLifecycle,
        reason: str = "",
    ) -> None:
        """Transitions the concept to a new lifecycle state and increments its version."""
        if new_state == self.lifecycle_state:
            return

        self.concept_version += 1
        self.lifecycle_state = new_state
        self.last_updated_at = time.time()
        self.lifecycle_history.append(
            {
                "state": new_state.value,
                "version": self.concept_version,
                "timestamp": self.last_updated_at,
                "reason": reason or f"Transitioned to {new_state.value}",
            }
        )

    transition_to = transition_lifecycle

    @property
    def is_composite(self) -> bool:
        """Indicates whether this concept is a composite constructed from parts or relations."""
        return bool(self.parts or self.relational_graph is not None or self.higher_order_relations)

    def add_part(self, part_name: str, concept_id: str) -> None:
        """Bind a constituent concept as a structural part (part-whole composition)."""
        self.parts[part_name] = concept_id
        self.last_updated_at = time.time()

    def bind_role(self, role_name: str, filler: str) -> None:
        """Bind a semantic/functional role to a filler entity or concept."""
        self.role_bindings[role_name] = filler
        self.last_updated_at = time.time()

    def add_higher_order_relation(
        self,
        relation: str,
        source: str,
        target: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a higher-order relational predicate between concepts or relations (e.g. enables, prevents)."""
        entry: dict[str, Any] = {
            "relation": relation,
            "source": source,
            "target": target,
            "confidence": float(confidence),
        }
        if metadata:
            entry["metadata"] = dict(metadata)
        self.higher_order_relations.append(entry)
        self.last_updated_at = time.time()

    def set_relational_graph(self, graph: Any) -> None:
        """Attach a relational scene graph representing the structural topology of this concept."""
        self.relational_graph = graph
        self.last_updated_at = time.time()

    def to_causal_predicates(self) -> list[Any]:
        """Export compositional, part-whole, and higher-order relations as formal CausalPredicates."""
        from hbllm.hcir.world.causal_discovery import CausalPredicate

        preds: list[CausalPredicate] = []
        cid = self.concept_ref.id

        # Part-whole predicates
        for part_name, part_cid in self.parts.items():
            preds.append(
                CausalPredicate(
                    variable=f"composition:part:{cid}->{part_cid}",
                    operator="==",
                    value=part_name,
                    relation="part_of",
                    target_entity=part_cid,
                )
            )

        # Role-filler predicates
        for role_name, filler in self.role_bindings.items():
            preds.append(
                CausalPredicate(
                    variable=f"composition:role:{cid}->{role_name}",
                    operator="==",
                    value=filler,
                    relation="role_filler",
                    target_entity=filler,
                )
            )

        # Higher-order relation predicates
        for hor in self.higher_order_relations:
            rel = hor["relation"]
            src = hor["source"]
            tgt = hor["target"]
            conf = hor.get("confidence", 1.0)
            preds.append(
                CausalPredicate(
                    variable=f"higher_order:{src}->{tgt}",
                    operator="==",
                    value=f"conf:{conf:.2f}",
                    relation=rel,
                    target_entity=tgt,
                )
            )

        return preds

    def to_transaction_operations(self) -> list[Any]:
        """Propose atomic HCIR transactions persisting compositional structure and edges."""
        from hbllm.hcir.transactions import TransactionOp, TransactionOperation

        ops: list[TransactionOperation] = []
        cid = self.concept_ref.id

        ops.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"concept_{cid}",
                node_data={
                    "concept_id": cid,
                    "lifecycle": self.lifecycle_state.value,
                    "is_composite": self.is_composite,
                    "parts": dict(self.parts),
                    "role_bindings": dict(self.role_bindings),
                    "higher_order_relations": list(self.higher_order_relations),
                },
            )
        )

        for part_name, part_cid in self.parts.items():
            ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_id=f"part_of_{cid}_{part_cid}",
                    edge_data={
                        "source": f"concept_{cid}",
                        "target": f"concept_{part_cid}",
                        "relation": "PART_OF",
                        "part_name": part_name,
                    },
                )
            )

        for role_name, filler in self.role_bindings.items():
            ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_id=f"role_{cid}_{role_name}",
                    edge_data={
                        "source": f"concept_{cid}",
                        "target": str(filler),
                        "relation": "ROLE_FILLER",
                        "role": role_name,
                    },
                )
            )

        for idx, hor in enumerate(self.higher_order_relations):
            rel = hor["relation"]
            src = hor["source"]
            tgt = hor["target"]
            ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_id=f"higher_order_{cid}_{idx}",
                    edge_data={
                        "source": str(src),
                        "target": str(tgt),
                        "relation": str(rel).upper(),
                        "confidence": hor.get("confidence", 1.0),
                    },
                )
            )

        return ops
