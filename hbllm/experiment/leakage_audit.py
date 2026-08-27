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
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Canonical task-specific tokens that must NEVER appear in initial cohort memory/knowledge
PROHIBITED_TASK_PRELOADS: frozenset[str] = frozenset(
    {
        # E1 Concept tokens
        "concept_item",
        "is_concept",
        # E2 Lexical tokens
        "mepo",
        "dax",
        # E3 Simulation entities
        "support_flat",
        "support_curved",
        "support_soft",
        "support_stable",
        "support_dome",
        # E4 Calibration tokens
        "irregular_bevel",
        "micro_groove",
        "non_euclidean_mesh",
        "quantum_foam_base",
        # E5 Active Discovery scenarios
        "hollow_cube",
        "magnetic_cylinder",
        "fragile_glass",
        "uneven_weight_block",
        "solid_granite",
        # E6 Transfer tokens
        "nucleus_electron",
        "star_comet",
        "yellow_blue_balls",
        "yellow_lamp_globe",
        # E7 Curriculum tokens
        "t1_spatialstacking",
        "t2_containerpacking",
        "t3_balancebeam",
        "t4_obstaclenav",
        "t5_toolaffordance",
        # Explicit cheat keys
        "target_assembly_component",
        "mepo_grounded_cylinder",
        "dax_grounded_sphere",
        "curved_support_instability_rule",
        "t5_industrial_transfer_solution",
    }
)

FORBIDDEN_HIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "_true_mass",
        "_internal_friction_coeff",
        "_oracle_optimal_action",
        "_true_hidden_class",
        "_ground_truth_utility",
        "_is_oracle_active",
        "_oracle_optimal_probe",
    }
)


@dataclass
class LeakageAuditReport:
    """Rigorous audit report verifying information isolation and zero data contamination."""

    is_clean: bool
    initial_knowledge_hash: str
    violations: list[str] = field(default_factory=list)
    audited_cohorts: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class LeakageAuditor:
    """Audits cohorts and task environments for information leakage and unfair prior advantage."""

    def __init__(self, prohibited_preloads: set[str] | None = None) -> None:
        self.prohibited_preloads: set[str] = (
            set(prohibited_preloads)
            if prohibited_preloads is not None
            else set(PROHIBITED_TASK_PRELOADS)
        )

    def compute_knowledge_hash(self, initial_state: Any) -> str:
        """Compute a deterministic SHA-256 hash of a cohort's initial knowledge state."""
        state_repr = json.dumps(
            self._extract_state_dict(initial_state), sort_keys=True, default=str
        )
        return hashlib.sha256(state_repr.encode("utf-8")).hexdigest()

    def _extract_state_dict(self, target: Any) -> dict[str, Any]:
        """Extract a serializable representation from a cohort or dictionary."""
        if isinstance(target, dict):
            return target

        extracted: dict[str, Any] = {}
        cohort_id = getattr(target, "cohort_id", type(target).__name__)
        extracted["cohort_id"] = str(cohort_id)

        # Inspect public attributes
        for attr in dir(target):
            if attr.startswith("__"):
                continue
            val = getattr(target, attr, None)
            if callable(val):
                continue
            if isinstance(val, (str, int, float, bool, list, dict, set, tuple)):
                extracted[attr] = val
            elif hasattr(val, "__dict__"):
                extracted[attr] = {
                    k: v
                    for k, v in val.__dict__.items()
                    if not k.startswith("_") and not callable(v)
                }
        return extracted

    def audit_cohort_instance(self, cohort: Any) -> list[str]:
        """Deep audit of a live cohort instance for hardcoded preloads or oracle references."""
        violations: list[str] = []
        cohort_id = getattr(cohort, "cohort_id", type(cohort).__name__)

        # 1. Check for forbidden references to environment or oracle
        for attr in ("oracle", "_oracle", "env", "_env", "canonical_env"):
            if hasattr(cohort, attr):
                val = getattr(cohort, attr)
                if val is not None:
                    violations.append(
                        f"Cohort '{cohort_id}' directly references simulator oracle via attribute '{attr}'."
                    )

        # 2. Extract state representation and scan for prohibited task tokens
        state_dict = self._extract_state_dict(cohort)
        state_str = json.dumps(state_dict, default=str).lower()

        for prohibited in self.prohibited_preloads:
            if prohibited.lower() in state_str:
                violations.append(
                    f"Cohort '{cohort_id}' contains prohibited task-specific preload: '{prohibited}'"
                )

        return violations

    def audit_initial_knowledge(
        self, cohort_id: str, initial_knowledge: dict[str, Any]
    ) -> list[str]:
        """Audit cohort's preloaded state to ensure no task-specific solutions are hardcoded."""
        violations: list[str] = []
        state_str = json.dumps(initial_knowledge, default=str).lower()
        for prohibited in self.prohibited_preloads:
            if prohibited.lower() in state_str:
                violations.append(
                    f"Cohort '{cohort_id}' has prohibited task-specific preload: '{prohibited}'"
                )
        return violations

    def audit_observation_parity(
        self,
        cohort_id: str,
        canonical_observation: dict[str, Any] | Any,
        received_observation: dict[str, Any] | Any,
    ) -> list[str]:
        """Verify that a cohort received strictly the canonical observation with no hidden state."""
        violations: list[str] = []

        canon_dict = (
            canonical_observation
            if isinstance(canonical_observation, dict)
            else (
                canonical_observation.__dict__ if hasattr(canonical_observation, "__dict__") else {}
            )
        )
        rec_dict = (
            received_observation
            if isinstance(received_observation, dict)
            else (
                received_observation.__dict__ if hasattr(received_observation, "__dict__") else {}
            )
        )

        # Recursively check for hidden fields in received observation
        self._check_hidden_fields(cohort_id, rec_dict, violations)

        if canon_dict != rec_dict:
            violations.append(
                f"Cohort '{cohort_id}' observation deviated from canonical environment observation."
            )

        return violations

    def _check_hidden_fields(self, cohort_id: str, data: Any, violations: list[str]) -> None:
        """Recursively scan data structures for forbidden hidden fields."""
        if isinstance(data, dict):
            for k, v in data.items():
                if k in FORBIDDEN_HIDDEN_FIELDS:
                    violations.append(
                        f"Cohort '{cohort_id}' received forbidden hidden environment field: '{k}'"
                    )
                self._check_hidden_fields(cohort_id, v, violations)
        elif isinstance(data, list):
            for item in data:
                self._check_hidden_fields(cohort_id, item, violations)

    def run_full_audit(
        self,
        cohorts: list[Any] | dict[str, dict[str, Any]],
        sample_observations: dict[str, tuple[dict[str, Any], dict[str, Any]]]
        | list[Any]
        | None = None,
    ) -> LeakageAuditReport:
        """Run complete leakage audit across all experimental cohorts and observations."""
        all_violations: list[str] = []
        audited_cohorts: list[str] = []
        combined_repr = []

        # Audit cohorts
        if isinstance(cohorts, dict):
            for cid, state in cohorts.items():
                audited_cohorts.append(cid)
                combined_repr.append((cid, state))
                all_violations.extend(self.audit_initial_knowledge(cid, state))
        elif isinstance(cohorts, list):
            for c in cohorts:
                cid = getattr(c, "cohort_id", type(c).__name__)
                audited_cohorts.append(cid)
                state = self._extract_state_dict(c)
                combined_repr.append((cid, state))
                all_violations.extend(self.audit_cohort_instance(c))

        # Audit observations if provided
        if isinstance(sample_observations, dict):
            for cid, (canon, received) in sample_observations.items():
                all_violations.extend(self.audit_observation_parity(cid, canon, received))
        elif isinstance(sample_observations, list):
            for task in sample_observations:
                if hasattr(task, "__dict__"):
                    self._check_hidden_fields(type(task).__name__, task.__dict__, all_violations)

        combined_hash = hashlib.sha256(
            json.dumps(combined_repr, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        is_clean = len(all_violations) == 0
        if not is_clean:
            logger.warning(
                "Leakage audit failed with %d violations: %s",
                len(all_violations),
                all_violations,
            )

        return LeakageAuditReport(
            is_clean=is_clean,
            initial_knowledge_hash=combined_hash,
            violations=all_violations,
            audited_cohorts=audited_cohorts,
        )
