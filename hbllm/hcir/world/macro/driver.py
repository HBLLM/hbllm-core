"""Macro Action Driver Manager & Pluggable Registry."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.macro.models import MacroAction

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.macro.planner import MacroActionPlan

logger = logging.getLogger(__name__)


class MacroActionDriverManager:
    """Central registry and dispatch manager for pluggable domain-specific macro-actions and drivers.

    Decouples core neuro-symbolic reasoning from any specific domain, game, benchmark,
    or peripheral (e.g. ARC-AGI, AI2-THOR, Crafter, MineDojo, ROS2).
    Domain drivers or runner scripts register their custom macro-action operators
    and synthesis hooks here.
    """

    _registry: dict[str, type[MacroAction]] = {}
    _synthesis_hooks: list[
        Callable[
            [AutonomousEpistemicEngine, np.ndarray, Sequence[Any], Sequence[Any] | None],
            MacroActionPlan | None,
        ]
    ] = []

    @classmethod
    def register_macro_action(cls, name: str, op_class: type[MacroAction]) -> None:
        """Register a domain-specific macro-action operator class."""
        cls._registry[name] = op_class
        logger.info("MacroActionDriverManager: Registered operator '%s'", name)

    @classmethod
    def register_synthesis_hook(
        cls,
        hook: Callable[
            [AutonomousEpistemicEngine, np.ndarray, Sequence[Any], Sequence[Any] | None],
            MacroActionPlan | None,
        ],
    ) -> None:
        if hook not in cls._synthesis_hooks:
            cls._synthesis_hooks.append(hook)

    @classmethod
    def get_registered_actions(cls) -> dict[str, type[MacroAction]]:
        return dict(cls._registry)

    @classmethod
    def get_registered_action(cls, name: str) -> type[MacroAction] | None:
        return cls._registry.get(name)

    @classmethod
    def run_synthesis_hooks(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        objects: Sequence[Any] | None = None,
    ) -> MacroActionPlan | None:
        """Query registered domain drivers for specialized plan synthesis."""
        for hook in cls._synthesis_hooks:
            try:
                plan = hook(engine, curr_grid, available_actions, objects)
                if plan is not None and not plan.is_empty():
                    return plan
            except Exception as e:
                logger.debug("MacroActionDriverManager: Hook error: %s", e)
        return None

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._synthesis_hooks.clear()
