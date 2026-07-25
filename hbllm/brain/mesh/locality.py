"""Cognitive Locality Engine.

Determines what minimum context is required for a delegated task,
preventing over-sharing of cognition and maintaining local sovereignty.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from hbllm.brain.mesh.capsule import CognitiveOwnership, TaskCapsule
from hbllm.brain.mesh.registry import TaskPriorityClass

logger = logging.getLogger(__name__)


class DataSensitivity(str, Enum):
    """Data sensitivity levels for classification."""

    PUBLIC = "public"  # Safe to share anywhere
    INTERNAL = "internal"  # Safe within trusted cluster
    CONFIDENTIAL = "confidential"  # Only within local node
    RESTRICTED = "restricted"  # Never share externally


class NodeTypeClass(str, Enum):
    """Node type classifications for locality rules."""

    LOCAL = "local"  # Same physical device
    TRUSTED_CLUSTER = "trusted_cluster"  # Same trusted network
    CLOUD = "cloud"  # External cloud service
    UNTRUSTED = "untrusted"  # Untrusted external node


@dataclass
class DataRule:
    """A data minimization rule."""

    pattern: str  # Regex pattern to match keys
    sensitivity: DataSensitivity
    allowed_targets: list[NodeTypeClass]
    redaction_strategy: str = "remove"  # remove, redact, hash


class CognitiveLocalityEngine:
    """Calculates the minimal required subgraph for delegation."""

    def __init__(self, local_node_id: str) -> None:
        self.local_node_id = local_node_id
        self._rules = self._init_default_rules()
        self._audit_log: list[dict[str, Any]] = []

    def _init_default_rules(self) -> list[DataRule]:
        """Initialize default data minimization rules."""
        return [
            # Biometric data - highly restricted
            DataRule(
                pattern=r"biometric|fingerprint|face_id|iris|voiceprint",
                sensitivity=DataSensitivity.RESTRICTED,
                allowed_targets=[NodeTypeClass.LOCAL],
                redaction_strategy="remove",
            ),
            # Personal identifiers - confidential
            DataRule(
                pattern=r"ssn|social_security|tax_id|passport|driver_license",
                sensitivity=DataSensitivity.CONFIDENTIAL,
                allowed_targets=[NodeTypeClass.LOCAL, NodeTypeClass.TRUSTED_CLUSTER],
                redaction_strategy="redact",
            ),
            # Health data - confidential
            DataRule(
                pattern=r"health|medical|diagnosis|treatment|prescription",
                sensitivity=DataSensitivity.CONFIDENTIAL,
                allowed_targets=[NodeTypeClass.LOCAL, NodeTypeClass.TRUSTED_CLUSTER],
                redaction_strategy="redact",
            ),
            # Financial data - confidential
            DataRule(
                pattern=r"credit_card|bank_account|financial|transaction|payment",
                sensitivity=DataSensitivity.CONFIDENTIAL,
                allowed_targets=[NodeTypeClass.LOCAL, NodeTypeClass.TRUSTED_CLUSTER],
                redaction_strategy="redact",
            ),
            # Location data - internal
            DataRule(
                pattern=r"gps|location|coordinates|address|geolocation",
                sensitivity=DataSensitivity.INTERNAL,
                allowed_targets=[NodeTypeClass.LOCAL, NodeTypeClass.TRUSTED_CLUSTER],
                redaction_strategy="redact",
            ),
            # Communication data - internal
            DataRule(
                pattern=r"email|phone|message|chat|conversation",
                sensitivity=DataSensitivity.INTERNAL,
                allowed_targets=[NodeTypeClass.LOCAL, NodeTypeClass.TRUSTED_CLUSTER],
                redaction_strategy="redact",
            ),
            # User preferences - internal
            DataRule(
                pattern=r"preference|setting|config|profile",
                sensitivity=DataSensitivity.INTERNAL,
                allowed_targets=[NodeTypeClass.LOCAL, NodeTypeClass.TRUSTED_CLUSTER],
                redaction_strategy="hash",
            ),
        ]

    def _classify_node_type(self, target_node: str) -> NodeTypeClass:
        """Classify the target node type based on its ID."""
        target_lower = target_node.lower()

        if "cloud" in target_lower or "external" in target_lower:
            return NodeTypeClass.CLOUD
        elif "untrusted" in target_lower or "public" in target_lower:
            return NodeTypeClass.UNTRUSTED
        elif "cluster" in target_lower or "peer" in target_lower:
            return NodeTypeClass.TRUSTED_CLUSTER
        else:
            return NodeTypeClass.LOCAL

    def _apply_locality_filters(self, state: dict[str, Any], target_node: str) -> dict[str, Any]:
        """Strip highly sensitive local data before sending it out of the personal cluster."""
        filtered = {}
        target_type = self._classify_node_type(target_node)

        for key, value in state.items():
            should_include = True
            filtered_value = value
            matched_rule = None

            # Check against all rules
            for rule in self._rules:
                if re.search(rule.pattern, key, re.IGNORECASE):
                    matched_rule = rule
                    # Check if target is allowed
                    if target_type not in rule.allowed_targets:
                        should_include = False
                        # Apply redaction strategy
                        if rule.redaction_strategy == "redact":
                            filtered_value = self._redact_value(value)
                            should_include = True  # Include redacted version
                        elif rule.redaction_strategy == "hash":
                            filtered_value = self._hash_value(value)
                            should_include = True  # Include hashed version
                        break

            if should_include:
                filtered[key] = filtered_value

            # Log data sharing for audit
            if matched_rule and not should_include:
                self._audit_log.append(
                    {
                        "key": key,
                        "rule": rule.pattern,
                        "sensitivity": rule.sensitivity.value,
                        "target": target_node,
                        "target_type": target_type.value,
                        "action": "removed",
                    }
                )
                logger.debug(
                    "Removed sensitive data: %s (sensitivity=%s) from %s",
                    key,
                    rule.sensitivity.value,
                    target_node,
                )
            elif matched_rule and filtered_value != value:
                self._audit_log.append(
                    {
                        "key": key,
                        "rule": rule.pattern,
                        "sensitivity": rule.sensitivity.value,
                        "target": target_node,
                        "target_type": target_type.value,
                        "action": matched_rule.redaction_strategy,
                    }
                )
                logger.debug(
                    "Redacted data: %s (sensitivity=%s) to %s using %s",
                    key,
                    rule.sensitivity.value,
                    target_node,
                    matched_rule.redaction_strategy,
                )

        return filtered

    def _redact_value(self, value: Any) -> Any:
        """Redact sensitive value while preserving structure."""
        if isinstance(value, str):
            if len(value) <= 4:
                return "***"
            return value[:2] + "*" * (len(value) - 4) + value[-2:]
        elif isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._redact_value(v) for v in value]
        else:
            return "***"

    def _hash_value(self, value: Any) -> str:
        """Hash sensitive value for identification without exposure."""
        import hashlib

        value_str = str(value)
        return hashlib.sha256(value_str.encode()).hexdigest()[:16]

    def create_task_capsule(
        self,
        goal_id: str,
        target_node_id: str,
        authority_node: str,
        priority: TaskPriorityClass,
        world_state_subgraph: dict[str, Any],
        causal_edges: list[dict[str, Any]],
        utility_constraints: dict[str, float],
        permissions_scope: list[str],
    ) -> TaskCapsule:
        """Packages only the necessary context for the target node."""

        # Determine strict context boundary
        # For privacy, if the target is a CLOUD_SERVER, we might anonymize or drop PII
        filtered_state = self._apply_locality_filters(world_state_subgraph, target_node_id)

        capsule = TaskCapsule(
            goal_id=goal_id,
            ownership=CognitiveOwnership(
                origin_node=self.local_node_id,
                authority_node=authority_node,
                execution_node=target_node_id,
                verification_node=self.local_node_id,
            ),
            priority=priority,
            required_entities=filtered_state,
            causal_dependencies=causal_edges,
            utility_constraints=utility_constraints,
            permissions_scope=permissions_scope,
        )

        return capsule

    def add_rule(self, rule: DataRule) -> None:
        """Add a custom data minimization rule."""
        self._rules.append(rule)
        logger.info(
            "Added data minimization rule: %s (sensitivity=%s)",
            rule.pattern,
            rule.sensitivity.value,
        )

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log of data sharing decisions."""
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()
