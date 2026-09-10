"""
Crafter Adapter Plugin for HBLLM Core.

Provides perception, causal action planning, and benchmark adapters bridging
Crafter survival and technology-tree environments to typed HCIR CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import CrafterActionAdapter
from .benchmark import PureHCIRCrafterAgent, run_crafter_benchmark
from .environment import NativeCrafterWrapper, StandaloneCrafterEnv, make_crafter_env
from .perception import CrafterPerceptionAdapter
from .types import (
    ACTION_NAMES,
    NAME_TO_ACTION,
    CrafterAchievement,
    CrafterAction,
    CrafterGoal,
    CrafterInventory,
    CrafterObject,
    CrafterObservation,
    CrafterVitals,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "crafter-adapter"
PLUGIN_VERSION = "1.0.0"


from .predicates import register_crafter_predicates

# Register domain predicates with EmbodiedCausalOperator
register_crafter_predicates()


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    register_crafter_predicates()
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    register_crafter_predicates()
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "ACTION_NAMES",
    "NAME_TO_ACTION",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "CrafterAchievement",
    "CrafterAction",
    "CrafterActionAdapter",
    "CrafterGoal",
    "CrafterInventory",
    "CrafterObject",
    "CrafterObservation",
    "CrafterPerceptionAdapter",
    "CrafterVitals",
    "NativeCrafterWrapper",
    "PureHCIRCrafterAgent",
    "StandaloneCrafterEnv",
    "make_crafter_env",
    "register",
    "run_crafter_benchmark",
    "setup",
]
