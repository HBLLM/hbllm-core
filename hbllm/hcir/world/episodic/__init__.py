"""Modular Hippocampal & Habenular Episodic Memory Faculty Package."""

from __future__ import annotations

from hbllm.hcir.world.episodic.attractor import (
    CA3RecurrentPatternCompleter,
    DentateGyrusPatternSeparator,
    HippocampalAttractorNetwork,
    PatternCompletionResult,
    PatternSeparationResult,
    _project_vector_to_dim,
)
from hbllm.hcir.world.episodic.cortex import HippocampalEpisodicCortex
from hbllm.hcir.world.episodic.models import (
    EpisodicTransition,
    ProceduralMacro,
    ReplayCreditAssignment,
    TDReplaySummary,
    compress_action_trajectory,
)
from hbllm.hcir.world.episodic.replay import (
    ConsolidatedKnowledgeItem,
    DistilledSchema,
    DualStoreConsolidationEngine,
    HippocampalReplayConsolidator,
)

__all__ = [
    "CA3RecurrentPatternCompleter",
    "ConsolidatedKnowledgeItem",
    "DentateGyrusPatternSeparator",
    "DistilledSchema",
    "DualStoreConsolidationEngine",
    "EpisodicTransition",
    "HippocampalAttractorNetwork",
    "HippocampalEpisodicCortex",
    "HippocampalReplayConsolidator",
    "PatternCompletionResult",
    "PatternSeparationResult",
    "ProceduralMacro",
    "ReplayCreditAssignment",
    "TDReplaySummary",
    "_project_vector_to_dim",
    "compress_action_trajectory",
]
