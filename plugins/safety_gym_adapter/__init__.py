"""
Safety-Gymnasium Adapter Plugin for HBLLM Core.

Provides perception, constrained safety planning, and benchmark adapters connecting
Safety-Gymnasium navigation and hazard avoidance domains to typed CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import SafetyGymActionAdapter
from .benchmark import PureHCIRSafetyAgent, run_safety_gym_benchmark
from .environment import StandaloneSafetyGymEnv, make_safety_gym_env
from .perception import SafetyGymPerceptionAdapter
from .types import (
    SafetyEntity,
    SafetyEntityType,
    SafetyGoal,
    SafetyGymAction,
    SafetyObservation,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "safety-gym-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRSafetyAgent",
    "SafetyEntity",
    "SafetyEntityType",
    "SafetyGoal",
    "SafetyGymAction",
    "SafetyGymActionAdapter",
    "SafetyGymPerceptionAdapter",
    "SafetyObservation",
    "StandaloneSafetyGymEnv",
    "make_safety_gym_env",
    "register",
    "run_safety_gym_benchmark",
    "setup",
]
