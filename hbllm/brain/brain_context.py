"""
Brain Context — Separated service lifetime from runtime state.

Provides three distinct layers for the Cognitive Operating System:

    ``BrainServices``
        Service lifetime — constructed once at bootstrap, injected everywhere.
        Contains stateless infrastructure: simulation engine, clock, bus,
        capability registry, trace collector.

    ``BrainState``
        Runtime state — serializable, checkpointable, mutable.
        Contains the system's current cognitive condition: goals,
        neuromodulator levels, attention state, working memory.

    ``BrainContext``
        Coordination layer — holds services + state together.
        Never a service locator; constructed by ``BrainContainer``.

Design rationale:
    Separating services from state makes serialization, checkpointing,
    and distributed execution cleaner. ``BrainState`` can be snapshotted
    for debugging or restored after crashes. ``BrainServices`` remain
    stable across state transitions.

Architecture::

    BrainContainer.build(config)
            │
            ▼
    ┌─────────────────┐
    │  BrainContext    │
    │                  │
    │  ┌────────────┐  │
    │  │BrainServices│  │
    │  │ simulation  │  │
    │  │ clock       │  │
    │  │ bus         │  │
    │  │ registry    │  │
    │  │ traces      │  │
    │  └────────────┘  │
    │                  │
    │  ┌────────────┐  │
    │  │ BrainState  │  │
    │  │ cognitive   │  │
    │  │ goals       │  │
    │  │ neuromod    │  │
    │  │ attention   │  │
    │  └────────────┘  │
    └─────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.capability_registry import CapabilityRegistry
from hbllm.brain.cognitive_state import CognitiveState
from hbllm.brain.neuromodulation import NeuromodulationEngine
from hbllm.brain.snn.oscillations import OscillationManager
from hbllm.brain.trace import TraceCollector
from hbllm.memory.goal_memory import GoalMemory

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BrainServices — service lifetime (constructed once, injected everywhere)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BrainServices:
    """Service lifetime layer — stateless infrastructure.

    These are constructed once at bootstrap and injected into
    cognitive subsystems. They do not change during runtime.

    Attributes:
        clock: Global timing service (oscillations, timers, circadian).
        bus: Message bus for inter-node communication.
        capability_registry: Dynamic service discovery.
        traces: Cognitive trace collector for observability.
        simulation: Simulation engine for mental rehearsal (optional).
    """

    clock: OscillationManager = field(default_factory=OscillationManager)
    bus: Any = None  # MessageBus — typed as Any to avoid circular imports
    capability_registry: CapabilityRegistry = field(default_factory=CapabilityRegistry)
    traces: TraceCollector = field(default_factory=TraceCollector)
    simulation: Any = None  # SimulationEngine — resolved at bootstrap


# ═══════════════════════════════════════════════════════════════════════════
# BrainState — runtime state (serializable, checkpointable)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BrainState:
    """Runtime state layer — mutable, serializable.

    Can be snapshotted for debugging, persisted for recovery,
    or diffed across cognitive cycles.

    Attributes:
        cognitive_state: Current cognitive condition (urgency, curiosity, etc.).
        goals: Active goal hierarchy.
        neuromodulation: Neurotransmitter levels.
        working_memory: Contents of the workspace (references).
        attention_focus: Current attention target.
        cycle_count: Number of cognitive cycles completed.
    """

    cognitive_state: CognitiveState = field(default_factory=CognitiveState)
    goals: GoalMemory = field(default_factory=GoalMemory)
    neuromodulation: NeuromodulationEngine = field(default_factory=NeuromodulationEngine)
    working_memory: list[Any] = field(default_factory=list)
    attention_focus: str = ""
    cycle_count: int = 0

    def snapshot(self) -> dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "cognitive_state": self.cognitive_state.to_dict()
            if hasattr(self.cognitive_state, "to_dict")
            else str(self.cognitive_state),
            "goal_count": len(self.goals._goals) if hasattr(self.goals, "_goals") else 0,
            "neuromodulation": self.neuromodulation.state.to_dict()
            if hasattr(self.neuromodulation, "state")
            and hasattr(self.neuromodulation.state, "to_dict")
            else {},
            "working_memory_size": len(self.working_memory),
            "attention_focus": self.attention_focus,
            "cycle_count": self.cycle_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# BrainContext — coordination layer
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BrainContext:
    """Coordination layer — holds services and state together.

    This is the single object passed to cognitive subsystems that
    need access to shared infrastructure. It is NOT a service locator;
    all dependencies are wired at construction time by ``BrainContainer``.

    Usage::

        # In a cognitive node:
        def __init__(self, context: BrainContext):
            self._clock = context.services.clock
            self._goals = context.state.goals
            self._traces = context.services.traces
    """

    services: BrainServices = field(default_factory=BrainServices)
    state: BrainState = field(default_factory=BrainState)

    def start_trace(self, source: str = "", tenant_id: str = "default"):
        """Convenience: start a cognitive trace."""
        return self.services.traces.start_trace(source=source, tenant_id=tenant_id)

    def stats(self) -> dict[str, Any]:
        """Combined system statistics."""
        return {
            "services": {
                "capabilities": self.services.capability_registry.stats(),
                "traces": self.services.traces.stats(),
                "clock": self.services.clock.stats(),
            },
            "state": self.state.snapshot(),
        }
