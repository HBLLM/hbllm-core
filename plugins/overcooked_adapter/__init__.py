"""
Overcooked-AI Adapter Plugin for HBLLM Core.

Provides multi-agent cooperative kitchen dynamics, culinary recipe DAG execution,
and counter exchange adapters connecting Overcooked simulations to typed CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import OvercookedActionAdapter
from .benchmark import (
    PureHCIROvercookedAgent,
    run_overcooked_benchmark,
    run_overcooked_tier,
)
from .environment import NativeOvercookedWrapper, StandaloneOvercookedEnv, make_overcooked_env
from .perception import OvercookedPerceptionAdapter
from .types import (
    AgentState,
    CulinaryItem,
    KitchenTile,
    OvercookedAction,
    OvercookedObservation,
    OvercookedTier,
    PotState,
    PotStatus,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "overcooked-adapter"
PLUGIN_VERSION = "1.0.0"


from .predicates import register_overcooked_predicates

# Register domain predicates with EmbodiedCausalOperator
register_overcooked_predicates()


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    register_overcooked_predicates()
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    register_overcooked_predicates()
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "AgentState",
    "CulinaryItem",
    "KitchenTile",
    "NativeOvercookedWrapper",
    "OvercookedAction",
    "OvercookedActionAdapter",
    "OvercookedObservation",
    "OvercookedPerceptionAdapter",
    "OvercookedTier",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PotState",
    "PotStatus",
    "PureHCIROvercookedAgent",
    "StandaloneOvercookedEnv",
    "make_overcooked_env",
    "register",
    "run_overcooked_benchmark",
    "run_overcooked_tier",
    "setup",
]
