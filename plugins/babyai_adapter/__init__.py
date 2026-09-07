"""
BabyAI / MiniGrid Adapter Plugin for HBLLM Core.

Provides perception and action adapters bridging MiniGrid/BabyAI environments
to typed HCIR CognitiveGraphs and planning engines.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import BabyAIActionAdapter
from .environment import BabyAIEnvironment, create_babyai_level
from .mission import BabyAIMissionParser
from .perception import BabyAIPerceptionAdapter
from .types import (
    DIR_TO_VEC,
    IDX_TO_ACTION,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    IDX_TO_STATE,
    BabyAIGoal,
    MiniGridAction,
    MiniGridColor,
    MiniGridDirection,
    MiniGridObjectType,
    MiniGridObservation,
    MiniGridState,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "babyai-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "BabyAIActionAdapter",
    "BabyAIEnvironment",
    "BabyAIGoal",
    "BabyAIMissionParser",
    "BabyAIPerceptionAdapter",
    "DIR_TO_VEC",
    "IDX_TO_ACTION",
    "IDX_TO_COLOR",
    "IDX_TO_OBJECT",
    "IDX_TO_STATE",
    "MiniGridAction",
    "MiniGridColor",
    "MiniGridDirection",
    "MiniGridObjectType",
    "MiniGridObservation",
    "MiniGridState",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "create_babyai_level",
    "register",
    "setup",
]
