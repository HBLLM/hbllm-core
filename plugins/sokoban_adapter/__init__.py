"""
Sokoban Adapter Plugin for HBLLM Core.

Provides deadlock detection, forward push-space search, and combinatorial puzzle
solving adapters connecting Sokoban environments to typed CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import SokobanActionAdapter
from .benchmark import PureHCIRSokobanAgent, run_sokoban_benchmark, run_sokoban_tier
from .environment import StandaloneSokobanEnv, make_sokoban_env
from .perception import SokobanPerceptionAdapter
from .types import (
    SokobanAction,
    SokobanDeadlockType,
    SokobanObservation,
    SokobanTier,
    SokobanTile,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "sokoban-adapter"
PLUGIN_VERSION = "1.0.0"


from .predicates import register_sokoban_predicates

# Register domain predicates with EmbodiedCausalOperator
register_sokoban_predicates()


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    register_sokoban_predicates()
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    register_sokoban_predicates()
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRSokobanAgent",
    "SokobanAction",
    "SokobanActionAdapter",
    "SokobanDeadlockType",
    "SokobanObservation",
    "SokobanPerceptionAdapter",
    "SokobanTier",
    "SokobanTile",
    "StandaloneSokobanEnv",
    "make_sokoban_env",
    "register",
    "run_sokoban_benchmark",
    "run_sokoban_tier",
    "setup",
]
