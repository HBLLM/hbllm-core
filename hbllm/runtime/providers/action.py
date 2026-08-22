"""Action Provider Protocol — HBLLM Cognitive Runtime.

Action providers execute intents in the physical or digital world.
The interface uses structured ``ActionIntent`` objects with explicit
safety constraints, preconditions, and authorization — because
``"say hello"`` and ``"move robotic arm to position X"`` cannot
have the same risk model.

Architecture::

    ActionIntent
        ├── TTS provider
        ├── OS command provider
        ├── ROS2 provider
        ├── CAN bus provider
        └── notification provider

The same cognitive runtime produces intents.  Which action provider
executes them is a deployment detail.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from hbllm.hcir.types import Provenance
from hbllm.runtime.providers.capability import ProviderCapability

# ═══════════════════════════════════════════════════════════════════════════
# Request & Response Types
# ═══════════════════════════════════════════════════════════════════════════


class ActionIntent(BaseModel):
    """Structured action request with safety model.

    Carries everything an action provider needs to execute safely:
    what to do, the expected effect, safety constraints, and
    authorization level.

    Attributes:
        action_type: What kind of action (``"speak"``, ``"move_arm"``,
            ``"send_notification"``, ``"run_command"``).
        target: What to act on (``"speaker_0"``, ``"arm_left"``,
            ``"user_123"``).
        parameters: Action-specific parameters.
        preconditions: Conditions that must be true before execution.
        expected_effect: What should change in the world state.
        safety_constraints: Limits on execution (force limits,
            speed limits, content filters).
        authorization: Permission level required (``"user"``,
            ``"admin"``, ``"safety_override"``).
        provenance: Why this action was requested (audit trail).
        max_duration_ms: Maximum time allowed for execution.
    """

    action_type: str = ""
    target: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    expected_effect: str = ""
    safety_constraints: list[str] = Field(default_factory=list)
    authorization: str = ""
    provenance: Provenance = Field(default_factory=Provenance)
    max_duration_ms: int = 5000


class ExecutionResult(BaseModel):
    """Result from any action provider.

    Captures success/failure, actual effect, timing, and error
    details — regardless of whether the provider was TTS, OS,
    or robot motor control.

    Attributes:
        success: Whether the action completed successfully.
        action_type: Echo of the requested action type.
        actual_effect: What actually changed (may differ from expected).
        error: Error description if failed.
        duration_ms: Wall-clock duration in milliseconds.
        provider_id: Which provider executed this action.
    """

    success: bool = False
    action_type: str = ""
    actual_effect: str = ""
    error: str = ""
    duration_ms: float = 0.0
    provider_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ActionProvider(Protocol):
    """Action execution — TTS, OS commands, robot motor control.

    ``"say hello"`` and ``"move robotic arm to position X"``
    cannot have the same risk model.  The ``ActionIntent``
    carries explicit safety constraints and authorization.

    The same cognitive runtime produces intents.  Which action
    provider executes them is a deployment detail:

    - Desktop: TTS + OS commands
    - Mobile: TTS + notifications
    - Robot: TTS + ROS2/CAN motor commands
    """

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for this provider."""
        ...

    async def execute(self, intent: ActionIntent) -> ExecutionResult:
        """Execute an action intent.

        Args:
            intent: Structured action request with safety model.

        Returns:
            ExecutionResult with success/failure and actual effect.
        """
        ...

    async def initialize(self) -> None:
        """Initialize provider resources."""
        ...

    async def shutdown(self) -> None:
        """Release provider resources."""
        ...
