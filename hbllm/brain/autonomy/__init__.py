"""Autonomy subsystem — the executive brain layer of HBLLM.

Provides continuous cognition, attentional control, and adaptive state
management that shifts HBLLM from reactive request→response to a
persistent, proactive cognitive organism.

Components
──────────
- ``CognitiveStateMachine`` — hierarchical state control (brainstem)
- ``AttentionSystem`` — multi-factor event prioritization
- ``AutonomyCore`` — hybrid event+tick cognitive heartbeat
- ``TaskGraphRuntime`` — persistent DAG-based goal execution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "AttentionEvent",
    "AttentionSystem",
    "AutonomyCore",
    "CognitiveState",
    "CognitiveStateCategory",
    "CognitiveStateMachine",
    "Goal",
    "IncrementalContextWindow",
    "InternalThought",
    "ScoredEvent",
    "TaskGraphRuntime",
    "TaskNode",
    "TickProfile",
]


def __getattr__(name: str):  # noqa: ANN202
    """Lazy imports to avoid pulling heavy dependencies at package load."""
    if name in (
        "CognitiveState",
        "CognitiveStateCategory",
        "CognitiveStateMachine",
        "TickProfile",
    ):
        from hbllm.brain.autonomy.state_machine import (
            CognitiveState,
            CognitiveStateCategory,
            CognitiveStateMachine,
            TickProfile,
        )

        return locals()[name]

    if name in (
        "AttentionEvent",
        "AttentionSystem",
        "IncrementalContextWindow",
        "ScoredEvent",
    ):
        from hbllm.brain.autonomy.attention import (
            AttentionEvent,
            AttentionSystem,
            IncrementalContextWindow,
            ScoredEvent,
        )

        return locals()[name]

    if name in ("AutonomyCore", "InternalThought"):
        from hbllm.brain.autonomy.loop import AutonomyCore, InternalThought

        return locals()[name]

    if name in ("Goal", "TaskGraphRuntime", "TaskNode"):
        from hbllm.brain.autonomy.task_graph import Goal, TaskGraphRuntime, TaskNode

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
