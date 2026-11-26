"""A22 Lifelong Continual Learning Substrate Package.

Provides Three-Layer Memory Store (Fast, Slow, Immutable), Tri-Modal Sleep Replay,
Provenance-Preserving Adaptive Compaction, Dependency-Analyzed Stability Gating,
and Continual Learning Loop without Catastrophic Forgetting.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "CompactionReport": ("hbllm.brain.continual.compaction", "CompactionReport"),
    "ProvenancePreservingCompactor": (
        "hbllm.brain.continual.compaction",
        "ProvenancePreservingCompactor",
    ),
    "LifelongLearningLoop": ("hbllm.brain.continual.consolidation_engine", "LifelongLearningLoop"),
    "LifelongRetentionAudit": (
        "hbllm.brain.continual.consolidation_engine",
        "LifelongRetentionAudit",
    ),
    "SleepConsolidationEngine": (
        "hbllm.brain.continual.consolidation_engine",
        "SleepConsolidationEngine",
    ),
    "SleepCycleSummary": ("hbllm.brain.continual.consolidation_engine", "SleepCycleSummary"),
    "ContrastivePair": ("hbllm.brain.continual.replay", "ContrastivePair"),
    "ReplayCandidate": ("hbllm.brain.continual.replay", "ReplayCandidate"),
    "ReplayKind": ("hbllm.brain.continual.replay", "ReplayKind"),
    "SleepReplayEngine": ("hbllm.brain.continual.replay", "SleepReplayEngine"),
    "CandidateUpdate": ("hbllm.brain.continual.stability_gate", "CandidateUpdate"),
    "GateVerdict": ("hbllm.brain.continual.stability_gate", "GateVerdict"),
    "PlasticityStabilityEngine": (
        "hbllm.brain.continual.stability_gate",
        "PlasticityStabilityEngine",
    ),
    "StabilityGateReport": ("hbllm.brain.continual.stability_gate", "StabilityGateReport"),
    "DualStoreMemory": ("hbllm.brain.continual.store", "DualStoreMemory"),
    "EpisodicTrace": ("hbllm.brain.continual.store", "EpisodicTrace"),
    "ImmutableEvent": ("hbllm.brain.continual.store", "ImmutableEvent"),
    "MemoryLayer": ("hbllm.brain.continual.store", "MemoryLayer"),
    "VersionedKnowledgeRecord": ("hbllm.brain.continual.store", "VersionedKnowledgeRecord"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
