"""Epistemic Integration — wiring helper for AutonomyCore.

Provides a single-function entry point to connect the full epistemic
runtime to the rest of the brain.

Usage::

    from hbllm.brain.epistemics.integration import wire_epistemics

    loop = wire_epistemics(
        autonomy_core=core,
        graph=graph,
        data_dir="/path/to/data",
        llm=llm,
    )

This creates and wires:
    - EpistemicMemory (persistent reasoning history)
    - EpistemicCalibrationEngine (meta-epistemic self-calibration)
    - CounterfactualReasoner (what-if epistemic analysis)
    - DiscoveryWorkspace (research program lifecycle)
    - EpistemicLoop (orchestrator, registered as proactive handler)
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.epistemics.calibration import EpistemicCalibrationEngine
from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner
from hbllm.brain.epistemics.epistemic_loop import EpistemicLoop
from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.interfaces import InvestigationBudget
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


def wire_epistemics(
    autonomy_core: Any,
    graph: CognitiveGraph,
    data_dir: str,
    llm: Any | None = None,
    reputation_tracker: Any | None = None,
    budget: InvestigationBudget | None = None,
    calibration_interval: int = 5,
    max_investigations_per_cycle: int = 3,
    max_ideas_per_investigation: int = 15,
) -> EpistemicLoop:
    """Wire the full epistemic runtime into AutonomyCore.

    Creates and connects all epistemic components, then registers
    the ``EpistemicLoop`` as a proactive handler on the AutonomyCore.

    After calling this function, the epistemic loop will run on
    every cognitive tick (slow path), performing autonomous
    investigation, hypothesis generation, and belief revision.

    Args:
        autonomy_core: The AutonomyCore to register with.
        graph: The shared HCIR cognitive graph.
        data_dir: Directory for persistent data (workspace + memory).
        llm: Optional LLM instance for creative reasoning.
        reputation_tracker: Optional SourceReputationTracker.
        budget: Default investigation budget per cycle.
        calibration_interval: Run calibration every N cycles.
        max_investigations_per_cycle: Max targets per cycle.
        max_ideas_per_investigation: Max ideas per target.

    Returns:
        The fully-wired EpistemicLoop instance.
    """
    # Create persistent stores
    workspace = DiscoveryWorkspace(data_dir=data_dir, graph=graph)
    memory = EpistemicMemory(data_dir=data_dir)

    # Create meta-cognition engines
    calibrator = EpistemicCalibrationEngine(memory=memory)
    counterfactual = CounterfactualReasoner(graph=graph)

    # Create the orchestrator loop with all engines wired
    loop = EpistemicLoop(
        graph=graph,
        workspace=workspace,
        llm=llm,
        reputation_tracker=reputation_tracker,
        budget=budget,
        max_investigations_per_cycle=max_investigations_per_cycle,
        max_ideas_per_investigation=max_ideas_per_investigation,
        memory=memory,
        calibration=calibrator,
        counterfactual=counterfactual,
        calibration_interval=calibration_interval,
    )

    # Register as proactive handler
    autonomy_core.add_proactive_handler("epistemic", loop.run_cycle)

    logger.info(
        "Epistemic runtime wired: loop + memory + calibration + counterfactual "
        "(calibration every %d cycles, max %d investigations/cycle)",
        calibration_interval,
        max_investigations_per_cycle,
    )

    return loop
