"""Tests for ResearchStrategyManager — strategy recommendation + switching."""

from __future__ import annotations

from hbllm.brain.epistemics.research_strategy import (
    ResearchStrategyManager,
    ResearchStrategyType,
)
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph, HypothesisNode


class TestActiveStrategy:
    """Test active strategy property and switching."""

    def test_default_strategy(self, graph: CognitiveGraph) -> None:
        mgr = ResearchStrategyManager(graph=graph)
        assert mgr.active_strategy == ResearchStrategyType.EXPLORATION

    def test_set_active_strategy_enum(self, graph: CognitiveGraph) -> None:
        mgr = ResearchStrategyManager(graph=graph)
        mgr.set_active_strategy(ResearchStrategyType.VERIFICATION)
        assert mgr.active_strategy == ResearchStrategyType.VERIFICATION

    def test_set_active_strategy_string(self, graph: CognitiveGraph) -> None:
        mgr = ResearchStrategyManager(graph=graph)
        mgr.set_active_strategy("synthesis")
        assert mgr.active_strategy == ResearchStrategyType.SYNTHESIS

    def test_set_active_strategy_name(self, graph: CognitiveGraph) -> None:
        mgr = ResearchStrategyManager(graph=graph)
        mgr.set_active_strategy("COUNTEREXAMPLE_SEARCH")
        assert mgr.active_strategy == ResearchStrategyType.COUNTEREXAMPLE_SEARCH

    def test_set_unknown_strategy_ignored(self, graph: CognitiveGraph) -> None:
        mgr = ResearchStrategyManager(graph=graph)
        mgr.set_active_strategy("nonexistent_strategy")
        # Should remain at default
        assert mgr.active_strategy == ResearchStrategyType.EXPLORATION


class TestActiveConfig:
    """Test active_config property."""

    def test_active_config_matches_strategy(
        self,
        graph: CognitiveGraph,
    ) -> None:
        mgr = ResearchStrategyManager(graph=graph)
        config = mgr.active_config
        assert config.strategy == ResearchStrategyType.EXPLORATION

        mgr.set_active_strategy(ResearchStrategyType.VERIFICATION)
        config = mgr.active_config
        assert config.strategy == ResearchStrategyType.VERIFICATION


class TestGetStrategyConfig:
    """Test config retrieval."""

    def test_all_strategies_have_configs(self, graph: CognitiveGraph) -> None:
        mgr = ResearchStrategyManager(graph=graph)
        for strategy in ResearchStrategyType:
            config = mgr.get_strategy_config(strategy)
            assert config is not None
            assert config.max_ideas_per_round > 0


class TestRecommendStrategy:
    """Test strategy recommendation based on program state."""

    def test_empty_program_recommends_exploration(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Empty", "Q")
        mgr = ResearchStrategyManager(graph=graph)

        rec = mgr.recommend_strategy(
            prog.program_id,
            ResearchStrategyType.EXPLORATION,
        )
        assert rec == ResearchStrategyType.EXPLORATION

    def test_many_untested_hypotheses_recommends_verification(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Unverified", "Q")
        mgr = ResearchStrategyManager(graph=graph)

        # Add hypotheses without experiments
        for i in range(5):
            h = HypothesisNode(claim=f"Hypothesis {i}")
            h.research_program_id = prog.program_id
            graph.upsert_node(h)

        rec = mgr.recommend_strategy(
            prog.program_id,
            ResearchStrategyType.EXPLORATION,
        )
        # With 5 hypotheses and 0 tested → ratio < 0.3 → VERIFICATION
        assert rec == ResearchStrategyType.VERIFICATION
