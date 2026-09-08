"""
Semantic Ambiguity Adapter Plugin for HBLLM Core.

Provides natural language intent grounding, linguistic ambiguity classification,
and hybrid Guided HCIR causal execution for underspecified directives.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import (
    GuidedHCIRSemanticPlanner,
    PureHCIRSemanticPlanner,
    SemanticActionAdapter,
)
from .benchmark import (
    SemanticAgentCohort,
    run_semantic_benchmark,
    run_semantic_tier,
)
from .environment import StandaloneSemanticEnv, make_semantic_env
from .perception import SemanticPerceptionAdapter
from .types import (
    CausalSubgoal,
    SemanticObservation,
    SemanticTier,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "semantic-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return []


__all__ = [
    "CausalSubgoal",
    "GuidedHCIRSemanticPlanner",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PureHCIRSemanticPlanner",
    "SemanticActionAdapter",
    "SemanticAgentCohort",
    "SemanticObservation",
    "SemanticPerceptionAdapter",
    "SemanticTier",
    "StandaloneSemanticEnv",
    "make_semantic_env",
    "register",
    "run_semantic_benchmark",
    "run_semantic_tier",
    "setup",
]
