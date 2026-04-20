import time
import uuid
from typing import Any

from hbllm.network.messages import Message, MessageType


class ScheduleEventTool:
    """Tool for the LLM to schedule a one-off future event."""

    name = "schedule_event"
    description = (
        "Schedule a specialized action to occur at a future time. "
        "Will publish a payload to the provided route_topic when time has elapsed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "delay_seconds": {
                "type": "number",
                "description": "How many seconds from now to trigger the event.",
            },
            "route_topic": {
                "type": "string",
                "description": "The destination MessageBus topic (e.g. 'robot.move' or 'api.fetch').",
            },
            "payload": {
                "type": "object",
                "description": "JSON payload to send at the target time.",
            },
            "retry_policy": {
                "type": "string",
                "enum": ["fire_and_forget", "retry"],
                "description": "Whether to retry if the initial execution fails.",
                "default": "fire_and_forget",
            },
        },
        "required": ["delay_seconds", "route_topic", "payload"],
    }

    async def execute(self, env: Any, **kwargs: Any) -> Any:
        try:
            delay = float(kwargs["delay_seconds"])
            trigger_time = time.time() + delay
            task_id = f"sch_{uuid.uuid4().hex[:8]}"

            # Send EVENT to SchedulerNode
            schedule_msg = Message(
                type=MessageType.EVENT,
                source_node_id="tool_router",
                topic="system.scheduler.schedule",
                payload={
                    "task_id": task_id,
                    "tenant_id": getattr(env, "tenant_id", "default"),
                    "trigger_time": trigger_time,
                    "route_topic": kwargs["route_topic"],
                    "payload": kwargs["payload"],
                    "retry_policy": kwargs.get("retry_policy", "fire_and_forget"),
                },
            )

            # The 'env' injected into tools is generally an abstraction that might wrap bus
            if hasattr(env, "bus"):
                await env.bus.publish("system.scheduler.schedule", schedule_msg)
                return {"status": "success", "task_id": task_id, "trigger_time": trigger_time}
            else:
                return {"status": "error", "error": "Execution environment lacks MessageBus"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class ScheduleRecurringTool:
    """Tool for the LLM to schedule cron-based recurring events."""

    name = "schedule_recurring"
    description = (
        "Schedule a recurring background action. Must be tracked by cron syntax. "
        "Useful for polling an API every 10 minutes or sending daily reports."
    )
    parameters = {
        "type": "object",
        "properties": {
            "cron_expression": {
                "type": "string",
                "description": "Cron string e.g. '*/10 * * * *' (every 10 minutes).",
            },
            "route_topic": {
                "type": "string",
                "description": "The destination MessageBus topic.",
            },
            "payload": {
                "type": "object",
                "description": "JSON payload to send on interval.",
            },
        },
        "required": ["cron_expression", "route_topic", "payload"],
    }

    async def execute(self, env: Any, **kwargs: Any) -> Any:
        try:
            task_id = f"cron_{uuid.uuid4().hex[:8]}"
            schedule_msg = Message(
                type=MessageType.EVENT,
                source_node_id="tool_router",
                topic="system.scheduler.schedule",
                payload={
                    "task_id": task_id,
                    "tenant_id": getattr(env, "tenant_id", "default"),
                    "trigger_time": time.time(),  # immediate evaluation context
                    "cron_expression": kwargs["cron_expression"],
                    "route_topic": kwargs["route_topic"],
                    "payload": kwargs["payload"],
                },
            )

            if hasattr(env, "bus"):
                await env.bus.publish("system.scheduler.schedule", schedule_msg)
                return {"status": "success", "task_id": task_id}
            else:
                return {"status": "error", "error": "Execution environment lacks MessageBus"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class CancelTaskTool:
    """Tool for the LLM to cancel scheduled tasks."""

    name = "cancel_task"
    description = "Cancels a previously scheduled task or recurring event using its task_id."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the scheduled task to cancel.",
            },
        },
        "required": ["task_id"],
    }

    async def execute(self, env: Any, **kwargs: Any) -> Any:
        try:
            task_id = kwargs["task_id"]
            cancel_msg = Message(
                type=MessageType.EVENT,
                source_node_id="tool_router",
                topic="system.scheduler.cancel",
                payload={"task_id": task_id},
            )

            if hasattr(env, "bus"):
                await env.bus.publish("system.scheduler.cancel", cancel_msg)
                return {"status": "cancel_request_dispatched"}
            else:
                return {"status": "error", "error": "Execution environment lacks MessageBus"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
