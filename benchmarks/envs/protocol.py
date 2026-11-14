"""Closed-Loop Environment Protocol.

Defines the formal environment protocol for closed-loop evaluation:
1. reset(seed) -> EmbodiedObservation (raw sensors, zero oracle labels).
2. step(action) -> (EmbodiedObservation, reward, done, info).
3. get_oracle_action() -> Action (optimal upper-bound baseline).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hbllm.hcir.world.percept import EmbodiedObservation

# Canonical discrete motor action indices
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_TOGGLE = 6
ACTION_INTERACT = 7


@runtime_checkable
class BenchmarkEnvironment(Protocol):
    """Protocol for seedable, closed-loop benchmark environments."""

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        """Reset environment to initial state using deterministic seed."""
        ...

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        """Advance environment state by executing agent's proposed action.

        Returns:
            (next_observation, reward, done, info_dict)
        """
        ...

    def get_oracle_action(self) -> Any | None:
        """Compute the optimal ground-truth action for upper-bound oracle baseline."""
        ...
