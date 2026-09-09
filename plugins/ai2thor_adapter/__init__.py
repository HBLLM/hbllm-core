"""
AI2-THOR Adapter Plugin for HBLLM Core.

Provides perception, 3D scene-graph modeling, and causal embodied manipulation
adapters connecting AI2-THOR photorealistic simulation to typed CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import AI2ThorActionAdapter
from .benchmark import PureHCIRAI2ThorAgent, run_ai2thor_benchmark
from .environment import (
    NativeAI2ThorWrapper,
    StandaloneAI2ThorEnv,
    make_ai2thor_env,
)
from .perception import AI2ThorPerceptionAdapter
from .types import (
    AI2ThorActionType,
    AI2ThorAgentPose,
    AI2ThorGoal,
    AI2ThorObjectMetadata,
    AI2ThorObservation,
    AI2ThorVector3,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "ai2thor-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "AI2ThorActionAdapter",
    "AI2ThorActionType",
    "AI2ThorAgentPose",
    "AI2ThorGoal",
    "AI2ThorObjectMetadata",
    "AI2ThorObservation",
    "AI2ThorPerceptionAdapter",
    "AI2ThorVector3",
    "NativeAI2ThorWrapper",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRAI2ThorAgent",
    "StandaloneAI2ThorEnv",
    "make_ai2thor_env",
    "register",
    "run_ai2thor_benchmark",
    "setup",
]
