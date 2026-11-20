"""Hippocampal & Lateral Habenular Episodic Memory Faculty.

Biologically modeled on mammalian hippocampal sharp-wave ripples (SWR) and
habenular anti-reward signaling:
1. Hippocampal Episodic Trajectory Buffer
2. Sharp-Wave Ripple (SWR) Backward Replay
3. Spatial Hazard & Line-of-Sight Deduction
4. Episodic Trajectory Recall (Forward Preplay Retrieval)

This module is a lightweight backward-compatible facade delegating to .
"""

from __future__ import annotations

from typing import Any

from hbllm.hcir.transactions import TransactionOperation
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.engine_state import CognitiveContext
from hbllm.hcir.world.episodic import (
    CA3RecurrentPatternCompleter,
    ConsolidatedKnowledgeItem,
    DentateGyrusPatternSeparator,
    DistilledSchema,
    DualStoreConsolidationEngine,
    EpisodicTransition,
    HippocampalAttractorNetwork,
    HippocampalEpisodicCortex,
    HippocampalReplayConsolidator,
    PatternCompletionResult,
    PatternSeparationResult,
    ProceduralMacro,
    ReplayCreditAssignment,
    TDReplaySummary,
    _project_vector_to_dim,
    compress_action_trajectory,
)

__all__ = [
    "CA3RecurrentPatternCompleter",
    "ConsolidatedKnowledgeItem",
    "DentateGyrusPatternSeparator",
    "DistilledSchema",
    "DualStoreConsolidationEngine",
    "EpisodicStage",
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


class EpisodicStage:
    """Pure CorticalStage for Stage 2 / Episodic Memory Consolidation.

    In accordance with Invariant 1 and WP1, this stage observes an immutable CognitiveContext
    and emits a pure StageResult containing consolidated episodic schemas and state mutations.
    """

    name: str = "episodic_stage"

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        ops: list[TransactionOperation] = []
        signals: dict[str, Any] = {}
        terminal = ctx.signals.get("terminal")
        if terminal:
            signals["episode_boundary"] = True
            signals["consolidated_transitions"] = len(ctx.failed_transitions)

        return StageResult(
            author=self.name,
            operations=ops,
            signals=signals,
            proposed_action=None,
        )
