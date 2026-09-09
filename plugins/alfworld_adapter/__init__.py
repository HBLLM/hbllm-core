"""
ALFWorld Adapter Plugin for HBLLM Core.

Provides perception, causal affordance planning, and multi-task evaluation
adapters connecting ALFWorld / ALFRED household domains to typed CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import ALFWorldActionAdapter
from .benchmark import PureHCIRALFWorldAgent, run_alfworld_benchmark
from .environment import (
    NativeALFWorldWrapper,
    StandaloneALFWorldEnv,
    make_alfworld_env,
)
from .perception import ALFWorldPerceptionAdapter
from .types import (
    ALFWorldAffordance,
    ALFWorldGoal,
    ALFWorldObject,
    ALFWorldObservation,
    ALFWorldReceptacle,
    ALFWorldTaskType,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "alfworld-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "ALFWorldActionAdapter",
    "ALFWorldAffordance",
    "ALFWorldGoal",
    "ALFWorldObject",
    "ALFWorldObservation",
    "ALFWorldPerceptionAdapter",
    "ALFWorldReceptacle",
    "ALFWorldTaskType",
    "NativeALFWorldWrapper",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRALFWorldAgent",
    "StandaloneALFWorldEnv",
    "make_alfworld_env",
    "register",
    "run_alfworld_benchmark",
    "setup",
]
