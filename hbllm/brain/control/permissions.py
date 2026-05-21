"""Trust Boundaries and Scoped Trust Tokens.

Defines the ActionClass for tools and the TrustGrant mechanism
for scoped, contextual, expiring, and revocable trust.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class ActionClass(StrEnum):
    """Classification of action risk levels."""

    SAFE = "safe"  # Background cognition, non-mutating OS reads
    USER_AWARE = "user_aware"  # Mutating local state, drafting emails
    SENSITIVE = "sensitive"  # Sending emails, deleting data, external DB modifications
    CRITICAL = "critical"  # System-level destructive actions, payments


@dataclass
class TrustGrant:
    """A scoped, contextual trust token."""

    grant_id: str = field(default_factory=lambda: f"tg_{uuid.uuid4().hex[:8]}")
    tool_name: str = ""
    allowed_scope: str = ""  # e.g., "recipient:alice@company.com"
    granted_at: float = field(default_factory=time.time)
    expires_in_s: float = 86400 * 7  # Default 7 days
    max_actions: int = 10
    actions_used: int = 0
    requires_same_template: bool = False
    revoked: bool = False

    @property
    def is_valid(self) -> bool:
        """Check if the trust grant is still valid."""
        if self.revoked:
            return False
        if self.actions_used >= self.max_actions:
            return False
        if time.time() > self.granted_at + self.expires_in_s:
            return False
        return True

    def consume(self) -> bool:
        """Attempt to consume one action from this grant."""
        if not self.is_valid:
            return False
        self.actions_used += 1
        return True


class PermissionRegistry:
    """Manages tool classifications and active trust grants."""

    def __init__(self) -> None:
        self._tool_classes: dict[str, ActionClass] = {}
        self._active_grants: dict[str, list[TrustGrant]] = {}

    def register_tool(self, tool_name: str, action_class: ActionClass) -> None:
        """Register a tool's baseline risk classification."""
        self._tool_classes[tool_name] = action_class

    def get_tool_class(self, tool_name: str) -> ActionClass:
        """Get the baseline classification for a tool. Defaults to SENSITIVE for safety."""
        return self._tool_classes.get(tool_name, ActionClass.SENSITIVE)

    def issue_grant(self, grant: TrustGrant) -> None:
        """Issue a new scoped trust grant."""
        if grant.tool_name not in self._active_grants:
            self._active_grants[grant.tool_name] = []
        self._active_grants[grant.tool_name].append(grant)

    def check_permission(self, tool_name: str, requested_scope: str) -> bool:
        """Check if an action is allowed by baseline class or a valid TrustGrant."""
        action_class = self.get_tool_class(tool_name)

        # SAFE actions are always allowed
        if action_class == ActionClass.SAFE:
            return True

        # USER_AWARE might require a short cancellation window, handled elsewhere, but conceptually allowed
        if action_class == ActionClass.USER_AWARE:
            return True

        # SENSITIVE and CRITICAL require a valid TrustGrant matching the scope
        grants = self._active_grants.get(tool_name, [])
        for grant in grants:
            # Very simplistic scope matching for now
            if grant.is_valid and requested_scope.startswith(grant.allowed_scope):
                return True

        return False

    def consume_grant(self, tool_name: str, requested_scope: str) -> bool:
        """Consume a grant after the action is executed."""
        grants = self._active_grants.get(tool_name, [])
        for grant in grants:
            if grant.is_valid and requested_scope.startswith(grant.allowed_scope):
                return grant.consume()
        return False
