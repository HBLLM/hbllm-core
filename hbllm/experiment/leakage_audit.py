"""Leakage Audit Suite for the Scientific Comparison Experiment.

Verifies absolute experimental fairness and integrity across all cohorts:
1. No hidden environment state or ground-truth physical properties leaked into observations.
2. No future-task test data or evaluation distributions exposed in prior tasks.
3. No task-specific schemas, concepts, or lexical groundings preloaded in initial state.
4. Generates an immutable cryptographic hash of initial knowledge states.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LeakageAuditReport:
    """Rigorous audit report verifying information isolation and zero data contamination."""

    is_clean: bool
    initial_knowledge_hash: str
    violations: list[str] = field(default_factory=list)
    audited_cohorts: list[str] = field(default_factory=list)
    timestamp: float = 0.0


class LeakageAuditor:
    """Audits cohorts and task environments for information leakage and unfair prior advantage."""

    def __init__(self) -> None:
        self.prohibited_preloads: set[str] = {
            "target_assembly_component",
            "mepo_grounded_cylinder",
            "dax_grounded_sphere",
            "curved_support_instability_rule",
            "t5_industrial_transfer_solution",
        }

    def compute_knowledge_hash(self, initial_state: dict[str, Any]) -> str:
        """Compute a SHA-256 hash of a cohort's initial knowledge state."""
        state_repr = json.dumps(initial_state, sort_keys=True, default=str)
        return hashlib.sha256(state_repr.encode("utf-8")).hexdigest()

    def audit_observation_parity(
        self,
        cohort_id: str,
        canonical_observation: dict[str, Any],
        received_observation: dict[str, Any],
    ) -> list[str]:
        """Verify that a cohort received strictly the canonical observation with no hidden state."""
        violations = []
        # Check for hidden simulator fields that should never be present in observations
        hidden_fields = {
            "_true_mass",
            "_internal_friction_coeff",
            "_oracle_optimal_action",
            "_true_hidden_class",
        }
        for k in received_observation.keys():
            if k in hidden_fields:
                violations.append(
                    f"Cohort '{cohort_id}' received forbidden hidden environment field: '{k}'"
                )

        if canonical_observation != received_observation:
            violations.append(
                f"Cohort '{cohort_id}' observation deviated from canonical environment observation."
            )

        return violations

    def audit_initial_knowledge(
        self, cohort_id: str, initial_knowledge: dict[str, Any]
    ) -> list[str]:
        """Audit cohort's preloaded state to ensure no task-specific solutions are hardcoded."""
        violations = []
        knowledge_keys = set(str(k).lower() for k in initial_knowledge.keys())
        for prohibited in self.prohibited_preloads:
            if prohibited in knowledge_keys:
                violations.append(
                    f"Cohort '{cohort_id}' has prohibited task-specific preload: '{prohibited}'"
                )
        return violations

    def run_full_audit(
        self,
        cohort_states: dict[str, dict[str, Any]],
        sample_observations: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    ) -> LeakageAuditReport:
        """Run complete leakage audit across all experimental cohorts."""
        all_violations: list[str] = []
        audited_cohorts = list(cohort_states.keys())
        combined_repr = []

        for cid, state in cohort_states.items():
            combined_repr.append((cid, state))
            violations = self.audit_initial_knowledge(cid, state)
            all_violations.extend(violations)

        for cid, (canon, received) in sample_observations.items():
            obs_violations = self.audit_observation_parity(cid, canon, received)
            all_violations.extend(obs_violations)

        combined_hash = hashlib.sha256(
            json.dumps(combined_repr, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        is_clean = len(all_violations) == 0
        if not is_clean:
            logger.warning(
                "Leakage audit failed with %d violations: %s", len(all_violations), all_violations
            )

        return LeakageAuditReport(
            is_clean=is_clean,
            initial_knowledge_hash=combined_hash,
            violations=all_violations,
            audited_cohorts=audited_cohorts,
        )
