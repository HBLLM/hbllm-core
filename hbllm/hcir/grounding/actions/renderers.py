from __future__ import annotations

import math
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from hbllm.hcir.grounding.contracts.actions import (
    RenderedActionPayload,
    SymbolicActionIntent,
)


@runtime_checkable
class ActionRenderer(Protocol):
    """Protocol for pure functional transformations from SymbolicActionIntent to RenderedActionPayload."""

    @property
    def renderer_name(self) -> str: ...

    def can_render(self, intent: SymbolicActionIntent) -> bool: ...

    def render(
        self, intent: SymbolicActionIntent, context: dict[str, Any] | None = None
    ) -> RenderedActionPayload: ...


class KinematicRenderer:
    """Renders high-level spatial intents into physical motion waypoints and trajectories."""

    @property
    def renderer_name(self) -> str:
        return "KinematicRenderer"

    def can_render(self, intent: SymbolicActionIntent) -> bool:
        return intent.action_type.upper() in (
            "NAVIGATE_TO",
            "MOVE",
            "GRASP",
            "PUSH",
            "PLACE",
            "MANIPULATE",
        )

    def render(
        self, intent: SymbolicActionIntent, context: dict[str, Any] | None = None
    ) -> RenderedActionPayload:
        ctx = context or {}
        curr_pos = ctx.get("current_position", (0.0, 0.0))
        target_pos = ctx.get("target_position", (1.0, 1.0))
        speed = float(intent.parameters.get("speed", 1.0))

        dx = target_pos[0] - curr_pos[0]
        dy = target_pos[1] - curr_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        duration = dist / max(0.1, speed)

        # Generate linear interpolation waypoints
        num_waypoints = max(2, int(dist * 2))
        waypoints = [
            (
                curr_pos[0] + (dx * i) / (num_waypoints - 1),
                curr_pos[1] + (dy * i) / (num_waypoints - 1),
            )
            for i in range(num_waypoints)
        ]

        payload_data = {
            "trajectory_waypoints": waypoints,
            "target_coordinates": target_pos,
            "command": intent.action_type.lower(),
            "target_entity_id": intent.target_ref.id,
            "speed": speed,
            "gripper_state": intent.parameters.get(
                "gripper_state", "closed" if intent.action_type.upper() == "GRASP" else "open"
            ),
        }

        return RenderedActionPayload(
            payload_id=f"payload_kin_{uuid.uuid4().hex[:8]}",
            intent_id=intent.intent_id,
            renderer_name=self.renderer_name,
            execution_channel="kinematic_actuator",
            payload_data=payload_data,
            estimated_duration_s=duration,
            created_at=time.time(),
        )


class GuiRenderer:
    """Renders virtual and user interface actions (click, type, select)."""

    @property
    def renderer_name(self) -> str:
        return "GuiRenderer"

    def can_render(self, intent: SymbolicActionIntent) -> bool:
        return intent.action_type.upper() in (
            "CLICK",
            "TYPE",
            "SELECT",
            "SCROLL",
            "DRAG",
        )

    def render(
        self, intent: SymbolicActionIntent, context: dict[str, Any] | None = None
    ) -> RenderedActionPayload:
        ctx = context or {}
        element_bbox = ctx.get("element_bbox", (100.0, 100.0, 50.0, 20.0))
        target_x = element_bbox[0] + element_bbox[2] / 2.0
        target_y = element_bbox[1] + element_bbox[3] / 2.0

        payload_data = {
            "ui_action": intent.action_type.lower(),
            "coordinates": (target_x, target_y),
            "element_id": intent.target_ref.id,
            "text": intent.parameters.get("text", ""),
            "button": intent.parameters.get("button", "left"),
        }

        return RenderedActionPayload(
            payload_id=f"payload_gui_{uuid.uuid4().hex[:8]}",
            intent_id=intent.intent_id,
            renderer_name=self.renderer_name,
            execution_channel="gui_event_stream",
            payload_data=payload_data,
            estimated_duration_s=0.1,
            created_at=time.time(),
        )


class DialogueRenderer:
    """Renders communicative, speech, and textual interaction actions."""

    @property
    def renderer_name(self) -> str:
        return "DialogueRenderer"

    def can_render(self, intent: SymbolicActionIntent) -> bool:
        return intent.action_type.upper() in (
            "INFORM",
            "ASK",
            "WARN",
            "CONFIRM",
            "EXPLAIN",
        )

    def render(
        self, intent: SymbolicActionIntent, context: dict[str, Any] | None = None
    ) -> RenderedActionPayload:
        ctx = context or {}
        template = intent.parameters.get("template", "")
        message = intent.parameters.get("message", "")

        if not message and template:
            try:
                message = template.format(**ctx)
            except Exception:
                message = template
        elif not message:
            message = f"{intent.action_type}: {intent.target_ref.id}"

        payload_data = {
            "dialogue_act": intent.action_type.lower(),
            "text": message,
            "addressed_to": intent.target_ref.id,
            "tone": intent.parameters.get("tone", "neutral"),
        }

        return RenderedActionPayload(
            payload_id=f"payload_dlg_{uuid.uuid4().hex[:8]}",
            intent_id=intent.intent_id,
            renderer_name=self.renderer_name,
            execution_channel="speech_tts_channel",
            payload_data=payload_data,
            estimated_duration_s=max(0.5, len(message) * 0.05),
            created_at=time.time(),
        )
