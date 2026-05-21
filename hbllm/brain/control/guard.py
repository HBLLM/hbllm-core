"""Explanation-First Mode and Intent Integrity Engine.

Intercepts SENSITIVE actions, creates Immutable Approval Snapshots,
and halts execution until human approval is granted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.control.permissions import ActionClass, PermissionRegistry
from hbllm.brain.observability.tracer import DecisionTraceLedger

logger = logging.getLogger(__name__)


@dataclass
class IntentEnvelope:
    """A bounded approval session requested from the user."""

    envelope_id: str = field(default_factory=lambda: f"env_{uuid.uuid4().hex[:8]}")
    goal_description: str = ""
    planned_actions: list[dict[str, Any]] = field(default_factory=list)
    risk_level: ActionClass = ActionClass.SAFE
    execution_window_s: float = 900.0  # 15 minutes
    allowed_scopes: list[str] = field(default_factory=list)
    explanation: str = ""

    # Intent Integrity Hash
    immutable_hash: str = ""

    def compute_hash(self) -> str:
        """Hash the critical execution components to prevent drift after approval."""
        # We hash the stringified actions and scopes
        payload = json.dumps(
            {
                "actions": self.planned_actions,
                "scopes": self.allowed_scopes,
                "window": self.execution_window_s,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IntentIntegrityEngine:
    """Ensures that approved plans are not mutated before execution."""

    def __init__(self) -> None:
        self.approved_envelopes: dict[str, IntentEnvelope] = {}

    def record_approval(self, envelope: IntentEnvelope) -> None:
        """Save the approved envelope with its immutable hash."""
        envelope.immutable_hash = envelope.compute_hash()
        self.approved_envelopes[envelope.envelope_id] = envelope

    def verify_integrity(
        self,
        envelope_id: str,
        current_actions: list[dict[str, Any]],
        current_scopes: list[str],
        current_window: float,
    ) -> bool:
        """Verify that the execution parameters still match the approved hash."""
        envelope = self.approved_envelopes.get(envelope_id)
        if not envelope:
            return False

        current_payload = json.dumps(
            {"actions": current_actions, "scopes": current_scopes, "window": current_window},
            sort_keys=True,
        )

        current_hash = hashlib.sha256(current_payload.encode("utf-8")).hexdigest()

        if current_hash != envelope.immutable_hash:
            logger.critical("Intent Integrity Violation! Execution graph mutated after approval.")
            return False

        return True


class SecurityGuard:
    """Intercepts tasks, manages explanation-first mode, and enforces integrity."""

    def __init__(self, permissions: PermissionRegistry, tracer: DecisionTraceLedger) -> None:
        self.permissions = permissions
        self.tracer = tracer
        self.integrity = IntentIntegrityEngine()

    def intercept_plan(
        self, goal_id: str, goal_desc: str, actions: list[dict[str, Any]], trace_id: str
    ) -> IntentEnvelope | None:
        """Scan a proposed plan. If SENSITIVE, generate an IntentEnvelope for human approval."""
        highest_risk = ActionClass.SAFE
        required_scopes = []

        for action in actions:
            tool = action.get("tool_name", "")
            scope = action.get("scope", "")

            action_class = self.permissions.get_tool_class(tool)

            # Upgrade risk level
            if action_class == ActionClass.CRITICAL:
                highest_risk = ActionClass.CRITICAL
            elif action_class == ActionClass.SENSITIVE and highest_risk != ActionClass.CRITICAL:
                highest_risk = ActionClass.SENSITIVE
            elif action_class == ActionClass.USER_AWARE and highest_risk == ActionClass.SAFE:
                highest_risk = ActionClass.USER_AWARE

            if scope:
                required_scopes.append(f"{tool}:{scope}")

            # If there is already a valid TrustGrant for this sensitive action, we might skip intercept
            if action_class in (ActionClass.SENSITIVE, ActionClass.CRITICAL):
                if self.permissions.check_permission(tool, scope):
                    logger.info("SecurityGuard: Action %s is covered by active TrustGrant.", tool)
                    # Downgrade localized risk for this evaluation since it's pre-approved
                    continue

        # If highest unresolved risk requires approval
        if highest_risk in (ActionClass.SENSITIVE, ActionClass.CRITICAL):
            explanation = self.tracer.explain_decision(trace_id)
            envelope = IntentEnvelope(
                goal_description=goal_desc,
                planned_actions=actions,
                risk_level=highest_risk,
                allowed_scopes=required_scopes,
                explanation=explanation,
            )
            return envelope

        return None
