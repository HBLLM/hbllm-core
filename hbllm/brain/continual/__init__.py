"""A22 Lifelong Continual Learning Substrate Package.

Provides Three-Layer Memory Store (Fast, Slow, Immutable), Tri-Modal Sleep Replay,
Provenance-Preserving Adaptive Compaction, Dependency-Analyzed Stability Gating,
and Continual Learning Loop without Catastrophic Forgetting.
"""

from hbllm.brain.continual.compaction import (
    CompactionReport,
    ProvenancePreservingCompactor,
)
from hbllm.brain.continual.consolidation_engine import (
    LifelongLearningLoop,
    LifelongRetentionAudit,
    SleepConsolidationEngine,
    SleepCycleSummary,
)
from hbllm.brain.continual.replay import (
    ContrastivePair,
    ReplayCandidate,
    ReplayKind,
    SleepReplayEngine,
)
from hbllm.brain.continual.stability_gate import (
    CandidateUpdate,
    GateVerdict,
    PlasticityStabilityEngine,
    StabilityGateReport,
)
from hbllm.brain.continual.store import (
    DualStoreMemory,
    EpisodicTrace,
    ImmutableEvent,
    MemoryLayer,
    VersionedKnowledgeRecord,
)

__all__ = [
    "CandidateUpdate",
    "CompactionReport",
    "ContrastivePair",
    "DualStoreMemory",
    "EpisodicTrace",
    "GateVerdict",
    "ImmutableEvent",
    "LifelongLearningLoop",
    "LifelongRetentionAudit",
    "MemoryLayer",
    "PlasticityStabilityEngine",
    "ProvenancePreservingCompactor",
    "ReplayCandidate",
    "ReplayKind",
    "SleepConsolidationEngine",
    "SleepCycleSummary",
    "SleepReplayEngine",
    "StabilityGateReport",
    "VersionedKnowledgeRecord",
]
