"""
MineDojo Adapter Plugin for HBLLM Core.

Provides perception, 3D voxel modeling, and causal crafting DAG adapters
connecting MineDojo / Minecraft open-ended embodied domains to typed CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import MineDojoActionAdapter
from .benchmark import PureHCIRMineDojoAgent, run_minedojo_benchmark
from .environment import (
    NativeMineDojoWrapper,
    StandaloneMineDojoEnv,
    make_minedojo_env,
)
from .perception import MineDojoPerceptionAdapter
from .types import (
    MineDojoAction,
    MineDojoGoal,
    MineDojoInventory,
    MineDojoObservation,
    MineDojoVoxel,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "minedojo-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "MineDojoAction",
    "MineDojoActionAdapter",
    "MineDojoGoal",
    "MineDojoInventory",
    "MineDojoObservation",
    "MineDojoPerceptionAdapter",
    "MineDojoVoxel",
    "NativeMineDojoWrapper",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRMineDojoAgent",
    "StandaloneMineDojoEnv",
    "make_minedojo_env",
    "register",
    "run_minedojo_benchmark",
    "setup",
]
