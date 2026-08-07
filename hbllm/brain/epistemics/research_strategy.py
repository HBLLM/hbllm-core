"""Research Strategy — pluggable strategy patterns for epistemic investigation.

Strategies determine *how* the epistemic loop allocates attention
and resources across its capabilities.

Available strategies::

    EXPLORATION      — maximize coverage of unknown territory
    VERIFICATION     — deepen evidence for existing hypotheses
    REPLICATION      — reproduce existing findings
    COUNTEREXAMPLE   — actively seek falsifying evidence
    SYNTHESIS        — integrate and consolidate findings
    ABDUCTIVE        — generate best explanations from observations
    SYSTEMATIC       — methodical coverage of hypothesis space

Each strategy configures the weights and priorities used by the
EpistemicLoop when dispatching tasks.

Usage::

    manager = ResearchStrategyManager(graph=graph)
    config = manager.get_strategy_config(ResearchStrategyType.EXPLORATION)
    # config.idea_weight, config.verification_weight, ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.hcir.graph import (
    CognitiveGraph,
    HypothesisNode,
    UnknownNode,
)
from hbllm.hcir.types import ResearchStrategyType

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration weights for a research strategy.

    These weights control how the EpistemicLoop distributes attention
    across different epistemic activities.

    All weights are [0.0, 1.0].  Higher = more emphasis.
    """

    strategy: ResearchStrategyType = ResearchStrategyType.EXPLORATION
    idea_generation_weight: float = 0.5
    hypothesis_validation_weight: float = 0.5
    prediction_tracking_weight: float = 0.5
    experiment_design_weight: float = 0.5
    contradiction_scanning_weight: float = 0.5
    evidence_evaluation_weight: float = 0.5
    explanation_building_weight: float = 0.3
    max_ideas_per_round: int = 15
    max_experiments_per_round: int = 3
    falsification_emphasis: float = 0.5
    novelty_emphasis: float = 0.5
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Definitions
# ═══════════════════════════════════════════════════════════════════════════

_STRATEGY_CONFIGS: dict[ResearchStrategyType, StrategyConfig] = {
    ResearchStrategyType.EXPLORATION: StrategyConfig(
        strategy=ResearchStrategyType.EXPLORATION,
        idea_generation_weight=0.9,
        hypothesis_validation_weight=0.4,
        prediction_tracking_weight=0.3,
        experiment_design_weight=0.4,
        contradiction_scanning_weight=0.6,
        evidence_evaluation_weight=0.3,
        explanation_building_weight=0.2,
        max_ideas_per_round=20,
        max_experiments_per_round=2,
        falsification_emphasis=0.3,
        novelty_emphasis=0.9,
        description="Maximize coverage of unknown territory. Generate many ideas, scan for contradictions.",
    ),
    ResearchStrategyType.VERIFICATION: StrategyConfig(
        strategy=ResearchStrategyType.VERIFICATION,
        idea_generation_weight=0.2,
        hypothesis_validation_weight=0.8,
        prediction_tracking_weight=0.8,
        experiment_design_weight=0.7,
        contradiction_scanning_weight=0.3,
        evidence_evaluation_weight=0.9,
        explanation_building_weight=0.6,
        max_ideas_per_round=5,
        max_experiments_per_round=5,
        falsification_emphasis=0.7,
        novelty_emphasis=0.2,
        description="Deepen evidence for existing hypotheses. Focus on experiments and evaluation.",
    ),
    ResearchStrategyType.REPLICATION: StrategyConfig(
        strategy=ResearchStrategyType.REPLICATION,
        idea_generation_weight=0.1,
        hypothesis_validation_weight=0.3,
        prediction_tracking_weight=0.9,
        experiment_design_weight=0.8,
        contradiction_scanning_weight=0.2,
        evidence_evaluation_weight=0.9,
        explanation_building_weight=0.4,
        max_ideas_per_round=3,
        max_experiments_per_round=5,
        falsification_emphasis=0.5,
        novelty_emphasis=0.1,
        description="Reproduce existing findings. Heavy prediction tracking and experiment design.",
    ),
    ResearchStrategyType.COUNTEREXAMPLE_SEARCH: StrategyConfig(
        strategy=ResearchStrategyType.COUNTEREXAMPLE_SEARCH,
        idea_generation_weight=0.3,
        hypothesis_validation_weight=0.5,
        prediction_tracking_weight=0.6,
        experiment_design_weight=0.7,
        contradiction_scanning_weight=0.9,
        evidence_evaluation_weight=0.7,
        explanation_building_weight=0.3,
        max_ideas_per_round=10,
        max_experiments_per_round=3,
        falsification_emphasis=1.0,
        novelty_emphasis=0.5,
        description="Actively seek falsifying evidence. Maximum contradiction scanning.",
    ),
    ResearchStrategyType.SYNTHESIS: StrategyConfig(
        strategy=ResearchStrategyType.SYNTHESIS,
        idea_generation_weight=0.2,
        hypothesis_validation_weight=0.3,
        prediction_tracking_weight=0.3,
        experiment_design_weight=0.2,
        contradiction_scanning_weight=0.4,
        evidence_evaluation_weight=0.6,
        explanation_building_weight=1.0,
        max_ideas_per_round=5,
        max_experiments_per_round=1,
        falsification_emphasis=0.3,
        novelty_emphasis=0.3,
        description="Integrate and consolidate findings. Maximum explanation building.",
    ),
    ResearchStrategyType.ABDUCTIVE: StrategyConfig(
        strategy=ResearchStrategyType.ABDUCTIVE,
        idea_generation_weight=0.7,
        hypothesis_validation_weight=0.7,
        prediction_tracking_weight=0.5,
        experiment_design_weight=0.5,
        contradiction_scanning_weight=0.5,
        evidence_evaluation_weight=0.6,
        explanation_building_weight=0.5,
        max_ideas_per_round=15,
        max_experiments_per_round=3,
        falsification_emphasis=0.5,
        novelty_emphasis=0.7,
        description="Generate best explanations from observations. Balanced idea generation and validation.",
    ),
    ResearchStrategyType.SYSTEMATIC: StrategyConfig(
        strategy=ResearchStrategyType.SYSTEMATIC,
        idea_generation_weight=0.5,
        hypothesis_validation_weight=0.6,
        prediction_tracking_weight=0.6,
        experiment_design_weight=0.6,
        contradiction_scanning_weight=0.5,
        evidence_evaluation_weight=0.6,
        explanation_building_weight=0.5,
        max_ideas_per_round=10,
        max_experiments_per_round=3,
        falsification_emphasis=0.6,
        novelty_emphasis=0.5,
        description="Methodical coverage of hypothesis space. Balanced across all activities.",
    ),
}


class ResearchStrategyManager:
    """Manages and recommends research strategies.

    Provides strategy configurations for the EpistemicLoop and can
    recommend strategy changes based on program state.
    """

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph

    def get_strategy_config(
        self,
        strategy: ResearchStrategyType,
    ) -> StrategyConfig:
        """Get the configuration for a specific strategy."""
        return _STRATEGY_CONFIGS.get(strategy, _STRATEGY_CONFIGS[ResearchStrategyType.EXPLORATION])

    def recommend_strategy(
        self,
        program_id: str,
        current_strategy: ResearchStrategyType,
    ) -> ResearchStrategyType:
        """Recommend a strategy based on program state.

        Strategy transitions::

            New program → EXPLORATION
            Many hypotheses, few tests → VERIFICATION
            Strong hypotheses → COUNTEREXAMPLE_SEARCH
            Many findings → SYNTHESIS
            Stale program → EXPLORATION (reset)
        """
        # Count program state
        unknowns = 0
        hypotheses = 0
        tested = 0
        strong = 0

        for _node in self._graph.all_nodes():
            node_id = _node.id
            node = self._graph.get_node(node_id)
            if isinstance(node, UnknownNode):
                if getattr(node, "research_program_id", "") == program_id:
                    unknowns += 1
            elif isinstance(node, HypothesisNode):
                if getattr(node, "research_program_id", "") == program_id:
                    hypotheses += 1
                    if node.linked_experiments:
                        tested += 1
                    if node.uncertainty.confidence >= 0.7:
                        strong += 1

        # Decision logic
        if hypotheses == 0:
            return ResearchStrategyType.EXPLORATION

        test_ratio = tested / max(1, hypotheses)
        if test_ratio < 0.3:
            return ResearchStrategyType.VERIFICATION

        if strong >= 2:
            return ResearchStrategyType.COUNTEREXAMPLE_SEARCH

        if test_ratio > 0.7 and hypotheses >= 3:
            return ResearchStrategyType.SYNTHESIS

        return current_strategy
