"""Discovery — autonomous scientific cognition subsystem.

The Discovery package implements the epistemic operating system layer
of HBLLM.  It evolves existing cognitive capabilities (beliefs, evidence,
hypotheses, experiments) into a unified discovery lifecycle.

Architecture::

    Existing Cognition
    ├── BeliefStore
    ├── ContradictionDetector
    ├── ExperimentEngine
    ├── CausalGraph
    └── SimulationEngine
            │
            ▼
    Discovery Layer (this package)
    ├── DiscoveryBeliefManager  → wraps BeliefStore
    ├── DiscoveryWorkspace      → runtime for ResearchProgram
    ├── SourceReputation        → epistemic trust tracking
    └── interfaces              → protocols for all components

Design principles:
    1. Evolve existing cognition, don't duplicate it.
    2. Discovery is a cognitive mode, not a separate subsystem.
    3. One brain, not two.
"""

from hbllm.brain.discovery.belief_manager import DiscoveryBeliefManager
from hbllm.brain.discovery.interfaces import (
    IBeliefReviser,
    IContradictionSeeker,
    IExperimentDesigner,
    IHypothesisGenerator,
    IPredictionTracker,
    ISourceReputationTracker,
)
from hbllm.brain.discovery.reputation import SourceReputation, SourceReputationTracker
from hbllm.brain.discovery.workspace import DiscoveryWorkspace, ResearchProgram

__all__ = [
    # Concrete implementations
    "DiscoveryBeliefManager",
    "DiscoveryWorkspace",
    "ResearchProgram",
    "SourceReputation",
    "SourceReputationTracker",
    # Protocols
    "IBeliefReviser",
    "IContradictionSeeker",
    "IExperimentDesigner",
    "IHypothesisGenerator",
    "IPredictionTracker",
    "ISourceReputationTracker",
]
