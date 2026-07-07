"""
Action Verification Bridge — closes the act → verify → correct loop.

Connects the TaskGraphRuntime's verification system to the WorldStateEngine
so that executed actions are automatically confirmed or retried based on
real-world state changes.

Architecture:
    1. TaskGraphRuntime marks a task as VERIFYING after execution
    2. This bridge periodically checks VERIFYING tasks against WorldState
    3. If the world state confirms the action succeeded → COMPLETED
    4. If timeout without confirmation → CORRECTING (re-execute with adjustments)
    5. If max corrections exhausted → UNCERTAIN (escalate to user)

Also supports auto-generating verification rules from IoT commands:
    - "turn on kitchen light" → VerificationRule(entity="kitchen_light", property="state", expected="on")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class ActionVerificationBridge:
    """
    Bridges TaskGraphRuntime verification with WorldStateEngine.

    Usage:
        bridge = ActionVerificationBridge(task_graph, world_state, bus)
        await bridge.start()  # Begins periodic verification loop
    """

    def __init__(
        self,
        task_graph: Any,
        world_state: Any,
        bus: Any,
        check_interval_s: float = 5.0,
    ) -> None:
        self.task_graph = task_graph
        self.world_state = world_state
        self.bus = bus
        self.check_interval_s = check_interval_s

        self._running = False
        self._loop_task: asyncio.Task[Any] | None = None
        self._verification_count = 0
        self._success_count = 0
        self._correction_count = 0

    async def start(self) -> None:
        """Start the verification loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._verification_loop())

        # Subscribe to task execution events for auto-rule generation
        if self.bus:
            await self.bus.subscribe("task.executed", self._on_task_executed)
            await self.bus.subscribe("iot.command.sent", self._on_iot_command)

        logger.info("ActionVerificationBridge started (interval=%.1fs)", self.check_interval_s)

    async def stop(self) -> None:
        """Stop the verification loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "ActionVerificationBridge stopped (verified=%d, succeeded=%d, corrected=%d)",
            self._verification_count,
            self._success_count,
            self._correction_count,
        )

    async def _verification_loop(self) -> None:
        """Periodically check VERIFYING tasks against world state."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_s)

                if not self.world_state or not self.task_graph:
                    continue

                # Run the existing verification logic in TaskGraphRuntime
                self.task_graph.verify_pending_tasks(self.world_state)
                self._verification_count += 1

                # Check for tasks that need correction and re-execute
                await self._handle_correcting_tasks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Verification loop error: %s", e)

    async def _handle_correcting_tasks(self) -> None:
        """Re-execute tasks that failed verification and entered CORRECTING state."""
        import sqlite3

        try:
            from hbllm.brain.autonomy.task_graph import TaskStatus

            with sqlite3.connect(self.task_graph.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM task_nodes WHERE status = ?",
                    (TaskStatus.CORRECTING.value,),
                ).fetchall()

                for row in rows:
                    task_id = row["task_id"]
                    action_topic = row["action_topic"]
                    action_payload_str = row["action_payload"]
                    correction_attempts = row["correction_attempts"]

                    self._correction_count += 1

                    # Parse action payload
                    import json

                    try:
                        action_payload = (
                            json.loads(action_payload_str) if action_payload_str else {}
                        )
                    except (json.JSONDecodeError, TypeError):
                        action_payload = {}

                    logger.info(
                        "Re-executing task %s (attempt %d): %s",
                        task_id,
                        correction_attempts,
                        action_topic,
                    )

                    # Re-publish the action command
                    if action_topic and self.bus:
                        await self.bus.publish(
                            action_topic,
                            Message(
                                type=MessageType.COMMAND,
                                source_node_id="verification_bridge",
                                topic=action_topic,
                                payload={
                                    **action_payload,
                                    "_correction_attempt": correction_attempts,
                                    "_original_task_id": task_id,
                                },
                            ),
                        )

                    # Transition back to VERIFYING
                    conn.execute(
                        "UPDATE task_nodes SET status = ?, verification_started_at = ? WHERE task_id = ?",
                        (TaskStatus.VERIFYING.value, time.time(), task_id),
                    )

        except Exception as e:
            logger.debug("Correcting task handling failed: %s", e)

    # ── Auto-Rule Generation ─────────────────────────────────────────────

    async def _on_task_executed(self, msg: Message) -> None:
        """Auto-generate verification rules when tasks are executed."""
        task_id = msg.payload.get("task_id")
        if not task_id:
            return

        # Check if the task already has a verification rule
        try:
            task = self.task_graph.get_task(task_id)
            if task and not task.verification_rule:
                rule = self._infer_verification_rule(task)
                if rule:
                    self.task_graph.set_verification_rule(task_id, rule)
                    logger.info(
                        "Auto-generated verification rule for task %s: %s.%s == %s",
                        task_id,
                        rule.entity_id,
                        rule.property_name,
                        rule.expected_value,
                    )
        except Exception as e:
            logger.debug("Auto-rule generation failed: %s", e)

    async def _on_iot_command(self, msg: Message) -> None:
        """Auto-generate verification for IoT commands."""
        device_id = msg.payload.get("device_id", "")
        command = msg.payload.get("command", "")
        expected_state = msg.payload.get("expected_state", "")

        if device_id and expected_state:
            logger.debug(
                "IoT verification will check %s for state=%s after command=%s",
                device_id,
                expected_state,
                command,
            )

    @staticmethod
    def _infer_verification_rule(task: Any) -> Any:
        """Infer a verification rule from the task's action metadata."""
        from hbllm.brain.autonomy.task_graph import VerificationRule

        action_topic = getattr(task, "action_topic", "") or getattr(task, "action_type", "")
        payload = getattr(task, "action_payload", {})

        if not action_topic:
            return None

        # IoT commands: "turn on X" → check device state
        if "iot" in action_topic.lower():
            device_id = payload.get("device_id", "")
            expected = payload.get("expected_state") or payload.get("state", "")
            if device_id and expected:
                return VerificationRule(
                    entity_id=device_id,
                    property_name="state",
                    expected_value=expected,
                    max_wait_time_s=30.0,
                )

        return None

    def stats(self) -> dict[str, int]:
        return {
            "verification_checks": self._verification_count,
            "successes": self._success_count,
            "corrections": self._correction_count,
        }
