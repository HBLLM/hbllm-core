"""
Digital Agent Adapter Plugin for HBLLM Core.

Provides POSIX shell tool interaction, in-memory virtual filesystem sandboxing,
DOM automation, and proactive safety policy filters for digital embodiment.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import DigitalActionAdapter
from .benchmark import (
    PureHCIRDigitalAgent,
    run_digital_benchmark,
    run_digital_tier,
)
from .environment import StandaloneDigitalEnv, make_digital_env
from .perception import DigitalPerceptionAdapter
from .types import (
    DigitalAction,
    DigitalActionType,
    DigitalObservation,
    DigitalTier,
    DOMNode,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "digital-agent-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "DOMNode",
    "DigitalAction",
    "DigitalActionAdapter",
    "DigitalActionType",
    "DigitalObservation",
    "DigitalPerceptionAdapter",
    "DigitalTier",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRDigitalAgent",
    "StandaloneDigitalEnv",
    "make_digital_env",
    "register",
    "run_digital_benchmark",
    "run_digital_tier",
    "setup",
]
