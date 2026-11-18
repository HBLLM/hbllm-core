"""Dorsolateral Prefrontal Cortex (dlPFC) Hierarchical Rule Abstraction & Schema Transfer.

Biologically grounded in Badre & D'Esposito (2007, 2009) rostro-caudal prefrontal hierarchy:
1. First-Order Relational Rule Extraction:
   - Generalizes concrete state transitions into abstract causal schemas (P => Q):
     strips specific grid coordinates and retains relational predicates
     (e.g., adjacent_to(agent, switch) & locked(door) => toggle(door)).
2. Zero-Shot Cross-Task Rule Transfer:
   - When encountering novel, structurally isomorphic problem instances, dlPFC pattern-matches
     active state conditions against stored abstract schemas, generating high-confidence
     action recommendations without trial-and-error relearning.
3. Confidence Calibration & Schema Pruning:
   - Tracks empirical support and counter-examples for each abstract rule schema,
     decaying confidence when transferred rules fail in novel environments.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class AbstractRuleSchema:
    """An abstract first-order relational production rule schema."""

    schema_id: str
    antecedent_conditions: dict[
        str, Any
    ]  # Abstract relational invariants (e.g. {"has_key": True, "near_door": True})
    consequent_action: str  # Action type (e.g. "interact", "navigate_through", "push")
    observed_support: int = 1
    counter_examples: int = 0
    confidence: float = 0.80

    def matches_state(self, current_features: dict[str, Any]) -> bool:
        """Check whether current state features satisfy antecedent conditions."""
        for k, v in self.antecedent_conditions.items():
            if current_features.get(k) != v:
                return False
        return True


@dataclass
class RuleTransferResult:
    """Outcome of evaluating abstract rule schemas against an active task state."""

    applicable_schemas: list[AbstractRuleSchema]
    recommended_action: str | None
    match_confidence: float
    is_transfer_viable: bool

    def to_causal_predicate(self) -> CausalPredicate:
        """Export rule transfer result as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="abstract_rule_transfer",
            operator="==",
            value=f"action:{self.recommended_action or 'none'}:conf:{self.match_confidence:.2f}:matches:{len(self.applicable_schemas)}",
            relation="dlpfc_rule_abstraction",
            target_entity="dlpfc_prefrontal",
        )


class DorsolateralRuleAbstractionEngine:
    """dlPFC rule abstraction engine inducing relational schemas and performing zero-shot transfer."""

    def __init__(
        self,
        min_support_for_transfer: int = 2,
        confidence_threshold: float = 0.65,
        max_schemas: int = 500,
    ) -> None:
        """Initialize dlPFC rule abstraction engine.

        Args:
            min_support_for_transfer: Minimum observations required before schema can be applied.
            confidence_threshold: Confidence required to recommend action transfer.
            max_schemas: Maximum number of abstract schemas to retain.
        """
        self.min_support = max(1, int(min_support_for_transfer))
        self.confidence_thresh = float(confidence_threshold)
        self.max_schemas = max(10, int(max_schemas))

        self.schemas: dict[str, AbstractRuleSchema] = {}
        self.total_evaluations: int = 0

    def induce_rule_schema(
        self,
        abstract_antecedents: dict[str, Any],
        successful_action: str,
    ) -> AbstractRuleSchema:
        """Induce or reinforce an abstract relational rule schema from observed success.

        Args:
            abstract_antecedents: Normalized relational state features (e.g., {"adjacent_switch": True}).
            successful_action: Chosen motor action that achieved positive progress.

        Returns:
            AbstractRuleSchema updated or newly created.
        """
        schema_key = self._generate_schema_key(abstract_antecedents, successful_action)

        if schema_key in self.schemas:
            s = self.schemas[schema_key]
            s.observed_support += 1
            # Laplace / Bayesian confidence smoothing
            s.confidence = float(
                (s.observed_support + 1) / (s.observed_support + s.counter_examples + 2)
            )
            return s

        if len(self.schemas) >= self.max_schemas:
            least_key = min(self.schemas.keys(), key=lambda k: self.schemas[k].observed_support)
            self.schemas.pop(least_key, None)

        schema = AbstractRuleSchema(
            schema_id=f"rule_{uuid.uuid4().hex[:8]}",
            antecedent_conditions=abstract_antecedents.copy(),
            consequent_action=successful_action,
            observed_support=1,
            counter_examples=0,
            confidence=0.67,  # (1+1)/(1+0+2)
        )
        self.schemas[schema_key] = schema
        return schema

    def record_counter_example(
        self,
        abstract_antecedents: dict[str, Any],
        failed_action: str,
    ) -> None:
        """Record a failure of an abstract rule, penalizing its confidence."""
        schema_key = self._generate_schema_key(abstract_antecedents, failed_action)
        if schema_key in self.schemas:
            s = self.schemas[schema_key]
            s.counter_examples += 1
            s.confidence = float(
                (s.observed_support + 1) / (s.observed_support + s.counter_examples + 2)
            )

    def evaluate_rule_transfer(
        self,
        current_state_features: dict[str, Any],
    ) -> RuleTransferResult:
        """Pattern-match current state against stored abstract schemas for zero-shot action recommendation.

        Args:
            current_state_features: Abstracted relational features of the current scene.

        Returns:
            RuleTransferResult indicating applicable schemas and recommended action.
        """
        self.total_evaluations += 1
        applicable: list[AbstractRuleSchema] = []

        for schema in self.schemas.values():
            if (
                schema.observed_support >= self.min_support
                and schema.confidence >= self.confidence_thresh
            ):
                if schema.matches_state(current_state_features):
                    applicable.append(schema)

        if not applicable:
            return RuleTransferResult(
                applicable_schemas=[],
                recommended_action=None,
                match_confidence=0.0,
                is_transfer_viable=False,
            )

        # Sort by confidence and support
        applicable.sort(key=lambda s: (s.confidence, s.observed_support), reverse=True)
        best_schema = applicable[0]

        return RuleTransferResult(
            applicable_schemas=applicable,
            recommended_action=best_schema.consequent_action,
            match_confidence=best_schema.confidence,
            is_transfer_viable=True,
        )

    def _generate_schema_key(self, antecedents: dict[str, Any], action: str) -> str:
        """Generate a deterministic string signature for an antecedent-action pair."""
        ant_items = sorted([(k, str(v)) for k, v in antecedents.items()])
        ant_str = ";".join(f"{k}={v}" for k, v in ant_items)
        return f"{ant_str}==>{action}"

    def reset(self) -> None:
        """Clear all stored abstract schemas."""
        self.schemas.clear()
        self.total_evaluations = 0
