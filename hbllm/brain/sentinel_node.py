"""
Sentinel Node — proactive governance monitor.

Continuously watches runtime context (time, sensors, world state) and
evaluates owner rules against that context. When a rule triggers due to
a state change, the Sentinel takes corrective action or escalates to the
owner for approval.

Bus topics:
  Subscribes to:
    context.update — state changes from sensors, clock, etc.
  Publishes to:
    sentinel.action    — corrective commands (e.g., lock door)
    sentinel.alert     — escalations requiring owner approval
    sensory.output     — direct notifications to the user/owner
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.policy_engine import PolicyEngine
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class SentinelAlert:
    """A record of a triggered rule and the action taken."""

    rule_name: str
    violation: str
    action_taken: str  # "corrective", "alert", "blocked"
    context_snapshot: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class SentinelNode(Node):
    """
    Proactive governance monitor that watches world state.

    Unlike DecisionNode (which gates responses reactively), the Sentinel
    actively monitors context changes and triggers corrective actions
    when owner rules are violated by the current world state.

    Example flow:
      1. Sensor publishes: context.update {time_hour: 21, door_state: "open"}
      2. Sentinel evaluates rules against new context
      3. Rule "no open door after 9pm" triggers
      4. Sentinel publishes: sentinel.action {action: "lock_door"}

    Also supports:
    - Periodic context polling (fallback when no bus updates arrive)
    - Alert escalation for ambiguous situations
    - Action history for audit trail
    """

    def __init__(
        self,
        node_id: str,
        policy_engine: PolicyEngine | None = None,
        poll_interval: float = 60.0,
        max_history: int = 100,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["proactive_monitoring", "governance_sentinel"],
        )
        self.policy_engine = policy_engine
        self.poll_interval = poll_interval
        self.max_history = max_history

        self._current_context: dict[str, Any] = {}
        self._alert_history: list[SentinelAlert] = []
        self._poll_task: asyncio.Task | None = None
        self._triggered_rules: set[str] = set()  # Prevent repeated alerts

    @property
    def alert_history(self) -> list[SentinelAlert]:
        return list(self._alert_history)

    @property
    def current_context(self) -> dict[str, Any]:
        return dict(self._current_context)

    async def on_start(self) -> None:
        logger.info("Starting SentinelNode (poll_interval=%.1fs)", self.poll_interval)
        await self.bus.subscribe("context.update", self._on_context_update)
        self.bus.add_interceptor(self._interceptor)
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def _interceptor(self, message: Message) -> Message | None:
        """Proactively evaluate messages against the policy engine before delivery."""
        if not self.policy_engine or message.is_security_cleared:
            return message

        # Extract text to evaluate
        text_to_eval = ""
        if "text" in message.payload:
            text_to_eval = message.payload["text"]
        elif "code" in message.payload:
            text_to_eval = message.payload["code"]
        elif "command" in message.payload:
            text_to_eval = str(message.payload["command"])

        if text_to_eval:
            result = self.policy_engine.evaluate(
                text=text_to_eval,
                tenant_id=message.tenant_id,
                context=self._current_context,
            )
            if not result.passed:
                violation = result.violations[0] if result.violations else "Policy violation"
                logger.warning(
                    "[Sentinel] Interceptor BLOCKED message %s: %s", message.id, violation
                )

                # Send an error back if it was a query
                if message.type == MessageType.QUERY:
                    error_msg = message.create_error(
                        f"BLOCKED BY SENTINEL: {violation}", code="POLICY_DENY"
                    )
                    asyncio.create_task(self.bus.publish(error_msg.topic, error_msg))

                # Drop the original message
                return None

            # Message is safe
            message.is_security_cleared = True

        return message

    async def on_stop(self) -> None:
        logger.info("Stopping SentinelNode")
        if self._poll_task:
            self._poll_task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        """Handle direct queries (e.g., 'list alerts', 'get context')."""
        if message.type != MessageType.QUERY:
            return None

        action = message.payload.get("action", "status")

        if action == "status":
            return message.create_response(
                {
                    "context": self._current_context,
                    "alert_count": len(self._alert_history),
                    "triggered_rules": list(self._triggered_rules),
                }
            )
        elif action == "alerts":
            limit = message.payload.get("limit", 10)
            return message.create_response(
                {
                    "alerts": [
                        {
                            "rule": a.rule_name,
                            "violation": a.violation,
                            "action_taken": a.action_taken,
                            "timestamp": a.timestamp,
                        }
                        for a in self._alert_history[-limit:]
                    ],
                }
            )
        elif action == "clear_alerts":
            self._alert_history.clear()
            self._triggered_rules.clear()
            return message.create_response({"status": "cleared"})

        return None

    async def _on_context_update(self, message: Message) -> Message | None:
        """
        Handle a context change from sensors/system.

        Expected payload:
            {"time_hour": 21, "door_state": "open", "baby_state": "sleeping", ...}
        """
        payload = message.payload
        if not payload:
            return None

        # Merge new context values into current state
        self._current_context.update(payload)

        logger.debug("[Sentinel] Context updated: %s", payload)

        # Evaluate all policies against the new context
        await self._evaluate_context(message.tenant_id or "default")

        return None

    async def _poll_loop(self) -> None:
        """Periodically re-evaluate rules even without explicit context updates."""
        while self._running:
            await asyncio.sleep(self.poll_interval)
            if self._current_context and self.policy_engine:
                await self._evaluate_context("default")

    async def _evaluate_context(self, tenant_id: str) -> None:
        """
        Evaluate all policies against the current context.

        For proactive monitoring, we generate a synthetic "state description"
        from the context dict and run it through the PolicyEngine. If a rule
        triggers, we take corrective action.
        """
        if not self.policy_engine:
            return

        # Build a text description of the current state for pattern matching
        state_text = self._context_to_text(self._current_context)

        result = self.policy_engine.evaluate(
            text=state_text,
            tenant_id=tenant_id,
            context=self._current_context,
        )

        if not result.passed:
            for violation in result.violations:
                await self._handle_violation(
                    violation=violation,
                    applied_policies=result.applied_policies,
                    tenant_id=tenant_id,
                )
        elif result.warnings:
            for warning in result.warnings:
                await self._handle_warning(warning, tenant_id)

    def _context_to_text(self, context: dict[str, Any]) -> str:
        """
        Convert context dict to a natural language description
        for pattern matching against DENY/REQUIRE policies.
        """
        parts = []
        if "door_state" in context:
            parts.append(f"The door is {context['door_state']}")
        if "time_hour" in context:
            h = context["time_hour"]
            ampm = "am" if h < 12 else "pm"
            h12 = h if h <= 12 else h - 12
            parts.append(f"The time is {h12}{ampm}")
        if "baby_state" in context:
            parts.append(f"The baby is {context['baby_state']}")
        if "person_type" in context:
            parts.append(f"A {context['person_type']} is present")
        if "location" in context:
            parts.append(f"Location is {context['location']}")

        # Include any other keys as generic state descriptions
        known_keys = {
            "door_state",
            "time_hour",
            "time_minute",
            "baby_state",
            "person_type",
            "location",
            "day_of_week",
            "is_weekend",
        }
        for key, value in context.items():
            if key not in known_keys:
                parts.append(f"{key.replace('_', ' ')} is {value}")

        return ". ".join(parts) + "." if parts else "No state information available."

    async def _handle_violation(
        self,
        violation: str,
        applied_policies: list[str],
        tenant_id: str,
    ) -> None:
        """Take corrective action for a policy violation."""
        # Extract rule name from the violation string
        rule_name = applied_policies[0] if applied_policies else "unknown"

        # Avoid repeating the same alert
        alert_key = f"{rule_name}:{hash(str(self._current_context)) % 10000}"
        if alert_key in self._triggered_rules:
            return
        self._triggered_rules.add(alert_key)

        # Determine corrective action based on context
        corrective_action = self._determine_corrective_action(rule_name)

        # Record alert
        alert = SentinelAlert(
            rule_name=rule_name,
            violation=violation,
            action_taken=corrective_action["type"],
            context_snapshot=dict(self._current_context),
        )
        self._alert_history.append(alert)
        if len(self._alert_history) > self.max_history:
            self._alert_history = self._alert_history[-self.max_history :]

        logger.warning(
            "[Sentinel] VIOLATION: %s → action: %s",
            violation,
            corrective_action["type"],
        )

        if corrective_action["type"] == "corrective":
            # Publish a corrective command
            await self.bus.publish(
                "sentinel.action",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    topic="sentinel.action",
                    payload={
                        "action": corrective_action["command"],
                        "reason": violation,
                        "rule": rule_name,
                    },
                ),
            )

        # Always notify the owner
        await self.bus.publish(
            "sensory.output",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=tenant_id,
                topic="sensory.output",
                payload={
                    "text": f"⚠️ Rule triggered: {violation}. "
                    f"Action: {corrective_action.get('description', corrective_action['type'])}.",
                    "source": "sentinel",
                    "severity": "high",
                },
            ),
        )

    async def _handle_warning(self, warning: str, tenant_id: str) -> None:
        """Log and optionally surface a policy warning."""
        logger.info("[Sentinel] Warning: %s", warning)
        await self.bus.publish(
            "sentinel.alert",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=tenant_id,
                topic="sentinel.alert",
                payload={
                    "warning": warning,
                    "context": self._current_context,
                    "severity": "low",
                },
            ),
        )

    def _determine_corrective_action(self, rule_name: str) -> dict[str, str]:
        """
        Determine the corrective action for a rule violation.

        Maps known rule patterns to corrective commands. Falls back to
        'alert' (notify owner) for unknown rules.
        """
        rule_lower = rule_name.lower()

        # Door-related rules
        if "door" in rule_lower:
            return {
                "type": "corrective",
                "command": "lock_door",
                "description": "Locking the door automatically",
            }

        # Noise/volume-related rules
        if any(w in rule_lower for w in ("noise", "volume", "loud", "soft", "quiet")):
            return {
                "type": "corrective",
                "command": "reduce_volume",
                "description": "Reducing volume level",
            }

        # Light-related rules
        if "light" in rule_lower:
            return {
                "type": "corrective",
                "command": "adjust_lights",
                "description": "Adjusting lighting",
            }

        # Default: escalate to owner
        return {
            "type": "alert",
            "command": "notify_owner",
            "description": "Alerting the owner for manual action",
        }

    def stats(self) -> dict[str, Any]:
        """Return sentinel statistics."""
        return {
            "context_keys": list(self._current_context.keys()),
            "alert_count": len(self._alert_history),
            "triggered_rules": len(self._triggered_rules),
            "is_monitoring": self._poll_task is not None and not self._poll_task.done(),
        }
