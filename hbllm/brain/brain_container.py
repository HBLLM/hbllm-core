"""
Brain Container — Bootstrap factory for the Cognitive Operating System.

Constructs all cognitive services, wires their dependencies, and
produces a fully initialized ``BrainContext``.

This is the single entry point for system construction. Individual
cognitive nodes never construct their own dependencies.

Architecture::

    BrainContainer.build(config)
        │
        ├─→ construct OscillationManager (BrainClock)
        ├─→ construct CapabilityRegistry
        ├─→ construct TraceCollector
        ├─→ construct SimulationEngine
        ├─→ construct GoalMemory
        ├─→ construct NeuromodulationEngine
        ├─→ construct CognitiveState
        │
        ├─→ assemble BrainServices
        ├─→ assemble BrainState
        │
        ├─→ register capabilities
        │
        └─→ return BrainContext

Usage::

    from hbllm.brain.brain_container import BrainContainer

    context = BrainContainer.build()
    # or with config:
    context = BrainContainer.build(BrainConfig(trace_retention=5000))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.brain.brain_context import BrainContext, BrainServices, BrainState
from hbllm.brain.capability_registry import CapabilityRegistry
from hbllm.brain.cognitive_state import CognitiveState
from hbllm.brain.neuromodulation import NeuromodulationEngine
from hbllm.brain.prediction import CognitivePredictors
from hbllm.brain.simulation_engine import SimulationEngine
from hbllm.brain.snn.oscillations import OscillationManager
from hbllm.brain.trace import TraceCollector
from hbllm.memory.belief_graph import BeliefGraph
from hbllm.memory.goal_memory import GoalMemory

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BrainConfig:
    """Configuration for brain bootstrap.

    Attributes:
        trace_retention: Max completed traces to retain.
        simulation_threshold: Default approval threshold for simulation.
        enable_simulation: Whether to enable the simulation engine.
        enable_prediction: Whether to enable cognitive predictors.
    """

    trace_retention: int = 1000
    simulation_threshold: float = 0.5
    enable_simulation: bool = True
    enable_prediction: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# BrainContainer — Bootstrap factory
# ═══════════════════════════════════════════════════════════════════════════


class BrainContainer:
    """Factory that constructs and wires the cognitive system.

    All dependency injection happens here. Cognitive nodes receive
    their dependencies through ``BrainContext`` — they never
    construct services themselves.
    """

    @staticmethod
    def build(
        config: BrainConfig | None = None,
        bus: Any = None,
    ) -> BrainContext:
        """Construct and wire the complete cognitive system.

        Args:
            config: Configuration overrides.
            bus: External message bus instance (optional).

        Returns:
            Fully initialized BrainContext.
        """
        cfg = config or BrainConfig()

        logger.info("Bootstrapping cognitive system...")

        # ── Construct infrastructure services ────────────────────────
        clock = OscillationManager()
        registry = CapabilityRegistry()
        traces = TraceCollector(max_retained=cfg.trace_retention)

        # ── Construct cognitive state ────────────────────────────────
        cognitive_state = CognitiveState()
        goals = GoalMemory()
        neuromod = NeuromodulationEngine()
        belief_graph = BeliefGraph()

        # ── Construct predictors ─────────────────────────────────────
        predictors = CognitivePredictors() if cfg.enable_prediction else None

        # ── Construct simulation engine ──────────────────────────────
        simulation = None
        if cfg.enable_simulation:
            simulation = SimulationEngine(
                goal_provider=goals,
                belief_graph=belief_graph,
                predictors=predictors,
                approval_threshold=cfg.simulation_threshold,
            )

        # ── Assemble services layer ──────────────────────────────────
        services = BrainServices(
            clock=clock,
            bus=bus,
            capability_registry=registry,
            traces=traces,
            simulation=simulation,
        )

        # ── Assemble state layer ─────────────────────────────────────
        state = BrainState(
            cognitive_state=cognitive_state,
            goals=goals,
            neuromodulation=neuromod,
        )

        # ── Register capabilities ────────────────────────────────────
        if simulation:
            registry.register(
                "simulation_engine",
                simulation,
                ["simulation", "deliberation", "candidate_evaluation"],
            )

        registry.register(
            "brain_clock",
            clock,
            ["timing", "oscillation", "phase_gating"],
        )

        registry.register(
            "goal_memory",
            goals,
            ["goal_tracking", "goal_hierarchy", "goal_provider"],
        )

        registry.register(
            "belief_graph",
            belief_graph,
            ["belief_tracking", "provenance", "evidence"],
        )

        if predictors:
            registry.register(
                "cognitive_predictors",
                predictors,
                ["prediction", "anticipation", "markov"],
            )

        registry.register(
            "neuromodulation",
            neuromod,
            ["neuromodulation", "transmitter_control"],
        )

        registry.register(
            "trace_collector",
            traces,
            ["observability", "tracing", "debugging"],
        )

        # ── Assemble context ─────────────────────────────────────────
        context = BrainContext(services=services, state=state)

        logger.info(
            "Cognitive system bootstrapped: %d services, %d capabilities",
            len(registry.all_services),
            len(registry.all_capabilities),
        )

        return context
