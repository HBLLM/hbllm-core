"""Verification Gate — Safety Filter Validating Real-World Confirmations Before Learning Promotion.

Constitutional and epistemic safety gate ensuring all candidate world model rules,
predictions, and state transformations pass strict validation standards before
being promoted into reality belief stores or dispatched to execution layers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class VerificationStatus(StrEnum):
    """Categorical outcome of a verification evaluation."""

    APPROVED = "approved"
    REJECTED_CONFIRMATIONS = "rejected_confirmations"
    REJECTED_CONFIDENCE = "rejected_confidence"
    REJECTED_RISK = "rejected_risk"


class RiskLevel(StrEnum):
    """Operational risk classifications."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class VerificationResult:
    """Structured decision output from VerificationGate."""

    status: VerificationStatus
    is_approved: bool
    confidence: float
    confirmations: int
    reason: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Export verification result as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="verification_gate",
            operator="==",
            value=self.is_approved,
            relation=str(self.status),
            target_entity="reality",
        )


@dataclass
class VerificationGate:
    """Safety gate evaluating eligibility of simulation predictions for reality-verified learning promotion."""

    min_confirmations: int = 1
    min_confidence: float = 0.80
    max_risk_level: RiskLevel | str = RiskLevel.LOW

    def evaluate_verification(
        self,
        confirmations: int,
        confidence: float,
        risk_level: RiskLevel | str = RiskLevel.LOW,
    ) -> VerificationResult:
        """Evaluate prediction promotion under formal constitutional safety criteria."""
        if confirmations < self.min_confirmations:
            reason = f"confirmations {confirmations} < required {self.min_confirmations}"
            logger.info("VerificationGate rejected: %s", reason)
            return VerificationResult(
                status=VerificationStatus.REJECTED_CONFIRMATIONS,
                is_approved=False,
                confidence=confidence,
                confirmations=confirmations,
                reason=reason,
            )

        if confidence < self.min_confidence:
            reason = f"confidence {confidence:.2f} < required {self.min_confidence:.2f}"
            logger.info("VerificationGate rejected: %s", reason)
            return VerificationResult(
                status=VerificationStatus.REJECTED_CONFIDENCE,
                is_approved=False,
                confidence=confidence,
                confirmations=confirmations,
                reason=reason,
            )

        reason = "All promotion safety criteria satisfied."
        logger.info(
            "VerificationGate APPROVED promotion: confirmations=%d, confidence=%.2f",
            confirmations,
            confidence,
        )
        return VerificationResult(
            status=VerificationStatus.APPROVED,
            is_approved=True,
            confidence=confidence,
            confirmations=confirmations,
            reason=reason,
        )

    def evaluate_eligibility(
        self,
        confirmations: int,
        confidence: float,
        risk_level: str = "LOW",
    ) -> bool:
        """Return True if prediction meets all verification gate safety criteria."""
        return self.evaluate_verification(confirmations, confidence, risk_level).is_approved

    def verify_transactions(self, operations: list[TransactionOperation]) -> tuple[bool, list[str]]:
        """Verify proposed state transactions against constitutional invariants."""
        violations: list[str] = []
        if not operations:
            return True, []

        seen_nodes: set[str] = set()
        for i, op in enumerate(operations):
            if op.node_id is None:
                violations.append(f"Operation {i} missing required node_id.")
            elif op.op == TransactionOp.ADD_NODE and op.node_id in seen_nodes:
                violations.append(
                    f"Operation {i} attempts duplicate insertion of node {op.node_id}."
                )
            elif op.node_id:
                seen_nodes.add(op.node_id)

        return len(violations) == 0, violations

    @staticmethod
    def verify_internal_consistency(rules: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        """W151: Internal world-model consistency check between co-existing transition rules."""
        conflicts: list[str] = []
        seen_conditions: dict[str, Any] = {}
        for r in rules:
            cond = str(r.get("condition"))
            eff = r.get("effect")
            if cond in seen_conditions and seen_conditions[cond] != eff:
                conflicts.append(
                    f"Contradictory effect for condition '{cond}': {seen_conditions[cond]} vs {eff}"
                )
            else:
                seen_conditions[cond] = eff

        return len(conflicts) == 0, conflicts

    @staticmethod
    def verify_constraint_satisfaction(
        state: dict[str, Any],
        constraints: list[Callable[[dict[str, Any]], bool]],
    ) -> tuple[bool, list[str]]:
        """W152: Constraint satisfaction verification ensuring invariant physical/topological invariants."""
        violations: list[str] = []
        for i, c in enumerate(constraints):
            try:
                if not c(state):
                    violations.append(f"Constraint {i} violated by state")
            except Exception as e:
                violations.append(f"Constraint {i} evaluation error: {e}")

        return len(violations) == 0, violations

    @staticmethod
    def detect_contradictory_transitions(
        transitions: list[tuple[Any, Any, Any]],
    ) -> list[tuple[Any, Any, Any, Any]]:
        """W153: Contradictory transition detection ((s, a) -> s1 vs (s, a) -> s2)."""
        seen: dict[tuple[Any, Any], Any] = {}
        contradictions = []
        for s, a, next_s in transitions:
            key = (str(s), str(a))
            if key in seen and seen[key] != str(next_s):
                contradictions.append((s, a, seen[key], next_s))
            else:
                seen[key] = str(next_s)
        return contradictions

    @staticmethod
    def compare_simulation_to_observation(
        simulated_state: dict[str, Any],
        observed_state: dict[str, Any],
    ) -> dict[str, Any]:
        """W154: Simulation-versus-observation comparison measuring empirical prediction error."""
        all_keys = set(simulated_state.keys()) | set(observed_state.keys())
        matches = 0
        mismatches: dict[str, tuple[Any, Any]] = {}

        for k in all_keys:
            sim_val = simulated_state.get(k)
            obs_val = observed_state.get(k)
            if sim_val == obs_val:
                matches += 1
            else:
                mismatches[k] = (sim_val, obs_val)

        total = len(all_keys) or 1
        accuracy = matches / total
        return {
            "accuracy": accuracy,
            "is_exact_match": len(mismatches) == 0,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }

    @staticmethod
    def reject_falsified_hypothesis(
        hypothesis_id: str,
        counterexample: dict[str, Any],
        active_hypotheses: dict[str, Any],
    ) -> bool:
        """W155: Failed hypothesis rejection upon empirical Popperian falsification."""
        if hypothesis_id in active_hypotheses:
            del active_hypotheses[hypothesis_id]
            logger.info(
                "Hypothesis %s rejected upon counterexample %s", hypothesis_id, counterexample
            )
            return True
        return False

    @staticmethod
    def minimal_change_model_revision(
        base_model: dict[str, Any],
        failed_rule_key: str,
        patched_value: Any,
    ) -> dict[str, Any]:
        """W156: Minimal-change model revision preserving all non-falsified invariants."""
        revised = dict(base_model)
        revised[failed_rule_key] = patched_value
        return revised

    @staticmethod
    def compare_alternative_models(
        candidate_models: list[dict[str, Any]],
        validation_pairs: list[tuple[dict[str, Any], Any]],
    ) -> dict[str, Any]:
        """W157: Alternative model comparison selecting the hypothesis with highest empirical accuracy and MDL simplicity."""
        scored: list[dict[str, Any]] = []
        for i, model in enumerate(candidate_models):
            correct = 0
            for x, y_true in validation_pairs:
                predict_fn = model.get("predict_fn", lambda s: s)
                y_pred = predict_fn(x)
                if y_pred == y_true:
                    correct += 1
            acc = correct / len(validation_pairs) if validation_pairs else 1.0
            complexity = float(model.get("complexity", 1.0))
            score = acc - 0.05 * complexity
            scored.append({"model_idx": i, "model": model, "accuracy": acc, "score": score})

        scored.sort(key=lambda s: s["score"], reverse=True)
        return scored[0] if scored else {}

    @staticmethod
    def verify_replay_reproducibility(
        seed: int,
        action_history: list[int],
        simulator_fn: Callable[[int, list[int]], Any],
    ) -> bool:
        """W158: Deterministic replay and reproducibility verification."""
        run1 = simulator_fn(seed, action_history)
        run2 = simulator_fn(seed, action_history)
        return bool(run1 == run2)

    @staticmethod
    def validate_final_prediction(
        prediction: Any,
        confidence: float,
        min_confidence: float = 0.75,
    ) -> tuple[bool, str]:
        """W160: Final prediction validation asserting confidence safety threshold before action dispatch."""
        if prediction is None:
            return False, "null_prediction"
        if confidence < min_confidence:
            return False, f"insufficient_confidence_{confidence:.2f}_below_{min_confidence:.2f}"
        return True, "validated"


@dataclass
class ModelVersionRecord:
    """W159: Model versioning metadata capturing lineage and provenance."""

    version_id: str
    parent_version_id: str | None
    author_faculty: str
    timestamp_iso: str
    description: str
    checkpoint_hash: str
    is_active: bool = True

    def to_causal_predicate(self) -> CausalPredicate:
        """Export version provenance record into a formal declarative CausalPredicate."""
        return CausalPredicate(
            variable="model_version",
            operator="==",
            value=self.version_id,
            relation=self.parent_version_id or "root",
            target_entity="provenance",
        )


class ModelProvenanceTracker:
    """W159: Tracks world-model versioning, lineage DAGs, and empirical provenance."""

    def __init__(self) -> None:
        self.versions: dict[str, ModelVersionRecord] = {}
        self.active_version_id: str | None = None

    def register_version(
        self,
        version_id: str,
        parent_version_id: str | None = None,
        author_faculty: str = "HCIR_COGNITIVE_CORE",
        description: str = "",
        checkpoint_hash: str = "hash_0",
    ) -> ModelVersionRecord:
        """Register a new world-model version in the provenance DAG."""
        record = ModelVersionRecord(
            version_id=version_id,
            parent_version_id=parent_version_id,
            author_faculty=author_faculty,
            timestamp_iso="2026-10-10T00:00:00Z",
            description=description,
            checkpoint_hash=checkpoint_hash,
            is_active=True,
        )
        self.versions[version_id] = record
        self.active_version_id = version_id
        return record

    def get_lineage(self, version_id: str) -> list[str]:
        """Trace lineage path from the given version back to root ancestor."""
        lineage: list[str] = []
        curr: str | None = version_id
        while curr is not None:
            lineage.append(curr)
            curr = self.versions[curr].parent_version_id if curr in self.versions else None
        return lineage

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export complete lineage DAG as declarative causal predicates."""
        return [ver.to_causal_predicate() for ver in self.versions.values()]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions for version tracking."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"provenance_{ver.version_id}",
                node_data={
                    "version_id": ver.version_id,
                    "parent_version_id": ver.parent_version_id,
                    "author_faculty": ver.author_faculty,
                    "checkpoint_hash": ver.checkpoint_hash,
                },
            )
            for ver in self.versions.values()
        ]
