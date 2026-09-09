"""
NetHack Adapter Plugin for HBLLM Core.

Provides perception, tactical causal dungeon crawling, and benchmark adapters
connecting NetHack / MiniHack procedural roguelike domains to typed CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import NetHackActionAdapter
from .benchmark import PureHCIRNetHackAgent, run_nethack_benchmark
from .environment import (
    NativeNetHackWrapper,
    StandaloneNetHackEnv,
    make_nethack_env,
)
from .perception import NetHackPerceptionAdapter
from .types import (
    ACTION_VECTORS,
    GLYPH_CHARS,
    NetHackAction,
    NetHackGlyph,
    NetHackGoal,
    NetHackObservation,
    NetHackStats,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "nethack-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "ACTION_VECTORS",
    "GLYPH_CHARS",
    "NetHackAction",
    "NetHackActionAdapter",
    "NetHackGlyph",
    "NetHackGoal",
    "NetHackObservation",
    "NetHackPerceptionAdapter",
    "NetHackStats",
    "NativeNetHackWrapper",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRNetHackAgent",
    "StandaloneNetHackEnv",
    "make_nethack_env",
    "register",
    "run_nethack_benchmark",
    "setup",
]
