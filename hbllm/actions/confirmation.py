"""Action Confirmation Gate — safety layer for dangerous operations.

Intercepts high-risk actions before execution, requiring explicit
user confirmation. "Sir, shall I really unlock the front door?"

Risk Tiers:
    Tier 0 (Safe):       Read operations, queries → auto-approve
    Tier 1 (Low risk):   Light control, volume → auto-approve + log
    Tier 2 (Medium):     Thermostat, blinds → confirm if unusual context
    Tier 3 (High risk):  Lock/unlock, file delete, shell → always confirm

Architecture:
    1. ActionRiskClassifier assigns a risk tier to each action
    2. ConfirmationGate intercepts Tier 2+ actions
    3. Publishes `action.confirm.request` and waits for response
    4. Auto-denies after timeout (default 60s)
    5. All decisions logged to audit trail
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


# ── Risk Classification ──────────────────────────────────────────────────────


@dataclass
class RiskAssessment:
    """Risk assessment for a proposed action."""

    action: str
    tier: int  # 0-3
    reason: str
    requires_confirmation: bool
    auto_approve: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionRiskClassifier:
    """Classifies actions into risk tiers.

    Tier 0 (Safe):       Read-only operations
    Tier 1 (Low risk):   Non-destructive control (lights, volume)
    Tier 2 (Medium):     State-changing control (thermostat, blinds)
    Tier 3 (High risk):  Security-critical (locks, file deletion, shell)
    """

    # Action patterns mapped to risk tiers
    TIER_MAP: dict[str, int] = {
        # Tier 0 — Safe reads
        "read": 0,
        "query": 0,
        "search": 0,
        "list": 0,
        "get": 0,
        "status": 0,
        "check": 0,
        # Tier 1 — Low risk
        "light.on": 1,
        "light.off": 1,
        "light.brightness": 1,
        "light.color": 1,
        "light.toggle": 1,
        "volume.set": 1,
        "volume.mute": 1,
        "speaker.play": 1,
        "speaker.stop": 1,
        "notification.send": 1,
        # Tier 2 — Medium risk
        "thermostat.set_temp": 2,
        "thermostat.mode": 2,
        "blinds.open": 2,
        "blinds.close": 2,
        "blinds.position": 2,
        "fan.on": 2,
        "fan.off": 2,
        "fan.speed": 2,
        "plug.on": 2,
        "plug.off": 2,
        "camera.snapshot": 2,
        "file.create": 2,
        "app.open": 2,
        # Tier 3 — High risk
        "lock.unlock": 3,
        "lock.lock": 3,
        "file.delete": 3,
        "shell.execute": 3,
        "camera.stream": 3,
        "system.shutdown": 3,
        "system.restart": 3,
        "network.disconnect": 3,
    }

    def __init__(
        self,
        custom_overrides: dict[str, int] | None = None,
        auto_approve_tier: int = 1,
    ) -> None:
        """
        Args:
            custom_overrides: Additional action → tier mappings.
            auto_approve_tier: Actions at or below this tier are auto-approved.
        """
        self.tier_map = dict(self.TIER_MAP)
        if custom_overrides:
            self.tier_map.update(custom_overrides)
        self.auto_approve_tier = auto_approve_tier

    def classify(self, action: str, context: dict[str, Any] | None = None) -> RiskAssessment:
        """Classify an action's risk level.

        Args:
            action: The action identifier (e.g., "lock.unlock", "file.delete").
            context: Optional context (time of day, user location, etc.).

        Returns:
            RiskAssessment with tier, reason, and confirmation requirement.
        """
        context = context or {}

        # Look for exact match first, then prefix match
        tier = self.tier_map.get(action)
        if tier is None:
            # Try prefix matching (e.g., "light.on" matches "light")
            for pattern, t in sorted(self.tier_map.items(), key=lambda x: -len(x[0])):
                if action.startswith(pattern):
                    tier = t
                    break

        if tier is None:
            # Unknown actions default to Tier 2 (medium risk)
            tier = 2

        # Context-based tier escalation
        import datetime

        hour = datetime.datetime.now().hour
        if tier >= 2 and (hour >= 22 or hour < 7):
            # Escalate medium-risk actions during night hours
            tier = min(3, tier + 1)

        requires_confirmation = tier > self.auto_approve_tier
        auto_approve = tier <= self.auto_approve_tier

        reason = self._explain_tier(action, tier)

        return RiskAssessment(
            action=action,
            tier=tier,
            reason=reason,
            requires_confirmation=requires_confirmation,
            auto_approve=auto_approve,
            metadata={"context": context, "hour": hour},
        )

    def _explain_tier(self, action: str, tier: int) -> str:
        tier_names = {
            0: "Safe (read-only operation)",
            1: "Low risk (non-destructive control)",
            2: "Medium risk (state-changing operation)",
            3: "High risk (security-critical or destructive)",
        }
        return f"Action '{action}' classified as Tier {tier}: {tier_names.get(tier, 'Unknown')}"


# ── Confirmation Gate ────────────────────────────────────────────────────────


@dataclass
class ConfirmationRequest:
    """A pending confirmation request."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action: str = ""
    risk_assessment: RiskAssessment | None = None
    tenant_id: str = "default"
    created_at: float = field(default_factory=time.time)
    timeout_s: float = 60.0
    # Resolved state
    approved: bool | None = None  # None = pending
    resolved_at: float | None = None
    resolved_by: str | None = None  # "user", "timeout", "auto"

    @property
    def is_pending(self) -> bool:
        return self.approved is None

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout_s


class ConfirmationGate:
    """Intercepts high-risk actions and requires user confirmation.

    Usage::

        gate = ConfirmationGate(bus=message_bus)
        await gate.start()

        # Before executing a dangerous action:
        approved = await gate.request_confirmation(
            action="lock.unlock",
            tenant_id="user1",
            context={"device": "front_door"},
        )
        if approved:
            # Execute the action
            ...
    """

    def __init__(
        self,
        bus: Any | None = None,
        classifier: ActionRiskClassifier | None = None,
        default_timeout_s: float = 60.0,
    ) -> None:
        self.bus = bus
        self.classifier = classifier or ActionRiskClassifier()
        self.default_timeout_s = default_timeout_s

        self._pending: dict[str, ConfirmationRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._history: list[ConfirmationRequest] = []
        self._max_history = 200

        # Telemetry
        self._total_requests = 0
        self._auto_approved = 0
        self._user_approved = 0
        self._user_denied = 0
        self._timed_out = 0

    async def start(self) -> None:
        """Subscribe to confirmation response events."""
        if self.bus:
            await self.bus.subscribe("action.confirm.response", self._on_response)
            logger.info("ConfirmationGate started")

    async def request_confirmation(
        self,
        action: str,
        tenant_id: str = "default",
        context: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Request confirmation for an action.

        Returns True if approved, False if denied or timed out.
        Blocks until the user responds or the timeout expires.
        """
        self._total_requests += 1

        # Classify risk
        assessment = self.classifier.classify(action, context)

        # Auto-approve safe actions
        if assessment.auto_approve:
            self._auto_approved += 1
            logger.debug("Auto-approved action '%s' (Tier %d)", action, assessment.tier)
            return True

        # Create confirmation request
        timeout = timeout_s or self.default_timeout_s
        request = ConfirmationRequest(
            action=action,
            risk_assessment=assessment,
            tenant_id=tenant_id,
            timeout_s=timeout,
        )

        self._pending[request.request_id] = request
        self._events[request.request_id] = asyncio.Event()

        # Publish confirmation request to user
        if self.bus:
            await self.bus.publish(
                "action.confirm.request",
                Message(
                    type=MessageType.EVENT,
                    source_node_id="confirmation_gate",
                    topic="action.confirm.request",
                    tenant_id=tenant_id,
                    payload={
                        "request_id": request.request_id,
                        "action": action,
                        "tier": assessment.tier,
                        "reason": assessment.reason,
                        "context": context or {},
                        "timeout_s": timeout,
                    },
                ),
            )

            # Also push as a notification for user visibility
            await self.bus.publish(
                "proactive.push",
                Message(
                    type=MessageType.EVENT,
                    source_node_id="confirmation_gate",
                    topic="proactive.push",
                    tenant_id=tenant_id,
                    payload={
                        "title": f"⚠️ Confirm: {action}",
                        "body": assessment.reason,
                        "priority": "critical" if assessment.tier >= 3 else "high",
                        "category": "confirmation",
                        "metadata": {
                            "request_id": request.request_id,
                            "action": action,
                            "requires_response": True,
                        },
                    },
                ),
            )

        logger.info(
            "Confirmation requested for '%s' (Tier %d, timeout=%ds, id=%s)",
            action,
            assessment.tier,
            timeout,
            request.request_id,
        )

        # Wait for response
        try:
            await asyncio.wait_for(
                self._events[request.request_id].wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Timed out — deny by default
            self._timed_out += 1
            request.approved = False
            request.resolved_at = time.time()
            request.resolved_by = "timeout"
            logger.warning(
                "Confirmation TIMED OUT for '%s' (id=%s) — denying action",
                action,
                request.request_id,
            )

        # Clean up
        self._events.pop(request.request_id, None)
        self._pending.pop(request.request_id, None)
        self._history.append(request)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        return request.approved is True

    async def _on_response(self, msg: Message) -> None:
        """Handle confirmation response from user."""
        request_id = msg.payload.get("request_id", "")
        approved = msg.payload.get("approved", False)

        request = self._pending.get(request_id)
        if request is None:
            logger.warning("Confirmation response for unknown request: %s", request_id)
            return

        request.approved = approved
        request.resolved_at = time.time()
        request.resolved_by = "user"

        if approved:
            self._user_approved += 1
            logger.info("Action '%s' APPROVED by user (id=%s)", request.action, request_id)
        else:
            self._user_denied += 1
            logger.info("Action '%s' DENIED by user (id=%s)", request.action, request_id)

        # Wake up the waiter
        event = self._events.get(request_id)
        if event:
            event.set()

    def stats(self) -> dict[str, Any]:
        """Confirmation gate statistics."""
        return {
            "total_requests": self._total_requests,
            "auto_approved": self._auto_approved,
            "user_approved": self._user_approved,
            "user_denied": self._user_denied,
            "timed_out": self._timed_out,
            "pending": len(self._pending),
        }
