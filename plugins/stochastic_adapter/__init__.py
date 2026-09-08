"""
Stochastic Robustness Adapter Plugin for HBLLM Core.

Provides sensorimotor noise simulation, epistemic belief filtering, and surprise-resilient
closed-loop planning for embodied navigation under uncertainty.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import StochasticActionAdapter
from .benchmark import (
    PureHCIRStochasticAgent,
    run_stochastic_benchmark,
    run_stochastic_tier,
)
from .environment import StandaloneStochasticEnv, make_stochastic_env
from .perception import StochasticPerceptionAdapter
from .types import (
    EpistemicEntityBelief,
    StochasticAction,
    StochasticObservation,
    StochasticTier,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "stochastic-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "EpistemicEntityBelief",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRStochasticAgent",
    "StandaloneStochasticEnv",
    "StochasticAction",
    "StochasticActionAdapter",
    "StochasticObservation",
    "StochasticPerceptionAdapter",
    "StochasticTier",
    "make_stochastic_env",
    "register",
    "run_stochastic_benchmark",
    "run_stochastic_tier",
    "setup",
]
