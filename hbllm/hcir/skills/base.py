"""Base Protocol and Abstractions for Standardized Hierarchical Skills.

Every HCIR skill represents a composable, reusable cognitive building block.
Each skill implements:
1. Invariant Recognition (`can_handle`): Evaluates whether visual and affordance signatures match.
2. Hierarchical Plan Synthesis (`plan`): Composes lower-level primitives into an action sequence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialActionIntent

if TYPE_CHECKING:
    from hbllm.drivers.base import DriverAction


class BaseHierarchicalSkill(ABC):
    """Abstract base class for all standardized HCIR hierarchical skills."""

    skill_name: str = "base_skill"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE

    @abstractmethod
    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate visual and action invariants to determine if this skill applies.

        Args:
            grid: 2D numpy array visual observation.
            available_actions: List of discrete action integers currently valid in the environment.
            metadata: Optional dictionary with auxiliary context (e.g. current level, score, tags).

        Returns:
            True if the skill can solve or advance the current state, False otherwise.
        """
        ...

    @abstractmethod
    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]] | list[int]:
        """Synthesize a sequence of discrete actions by composing lower-level primitives.

        Args:
            grid: 2D numpy array visual observation.
            current_level: 0-indexed environment level.
            metadata: Optional dictionary with auxiliary context.

        Returns:
            List of action tuples (action_id, parameters_dict) or raw action integers.
        """
        ...

    def plan_as_driver_actions(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[DriverAction]:
        """Synthesize a plan and normalize all outputs to standard DriverAction objects.

        Returns:
            List of DriverAction objects ready for the CognitiveBlackbox execution queue.
        """
        from hbllm.drivers.base import DriverAction

        raw_plan = self.plan(grid, current_level=current_level, metadata=metadata)
        driver_actions: list[DriverAction] = []
        for item in raw_plan:
            if isinstance(item, tuple):
                act_id, params = item
                intent = (
                    SpatialActionIntent.INTERACT
                    if act_id in (5, 6, 7) or bool(params)
                    else self.semantic_intent
                )
                driver_actions.append(
                    DriverAction(
                        action_id=int(act_id),
                        semantic_intent=intent,
                        parameters=dict(params) if params is not None else {},
                    )
                )
            elif isinstance(item, (int, np.integer)):
                act_id = int(item)
                intent = (
                    SpatialActionIntent.INTERACT if act_id in (5, 6, 7) else self.semantic_intent
                )
                driver_actions.append(
                    DriverAction(
                        action_id=act_id,
                        semantic_intent=intent,
                    )
                )
            elif isinstance(item, DriverAction):
                driver_actions.append(item)
        return driver_actions
