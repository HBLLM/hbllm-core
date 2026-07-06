"""
Milestone 4: Deliberative Brain — Unit Tests.

Validates the SimulationEngine, HeuristicCritic, and candidate
comparison pipeline.
"""

from __future__ import annotations

import time

import pytest

from hbllm.brain.simulation_engine import (
    HeuristicCritic,
    SimulationEngine,
    SimulationResult,
)
from hbllm.memory.belief_graph import BeliefGraph, BeliefRecord
from hbllm.memory.goal_memory import GoalCube, GoalMemory

# ═══════════════════════════════════════════════════════════════════════════
# HeuristicCritic Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHeuristicCritic:
    """Validate weighted heuristic scoring."""

    @pytest.mark.asyncio
    async def test_high_alignment_high_score(self) -> None:
        critic = HeuristicCritic()
        score = await critic.score(
            candidate="Help user debug auth",
            consequences=[],
            goal_alignment=0.9,
            belief_consistency=0.9,
            user_reaction=0.8,
        )
        assert score > 0.7

    @pytest.mark.asyncio
    async def test_low_alignment_low_score(self) -> None:
        critic = HeuristicCritic()
        score = await critic.score(
            candidate="Irrelevant response",
            consequences=["confusion", "frustration", "wasted time"],
            goal_alignment=0.1,
            belief_consistency=0.3,
            user_reaction=-0.5,
        )
        assert score < 0.5

    @pytest.mark.asyncio
    async def test_score_bounded_zero_one(self) -> None:
        critic = HeuristicCritic()
        high = await critic.score("x", [], 1.0, 1.0, 1.0)
        low = await critic.score("x", ["a"] * 20, 0.0, 0.0, -1.0)
        assert 0.0 <= high <= 1.0
        assert 0.0 <= low <= 1.0

    @pytest.mark.asyncio
    async def test_risk_penalty_for_many_consequences(self) -> None:
        critic = HeuristicCritic()
        no_risk = await critic.score("safe", [], 0.7, 0.7, 0.5)
        high_risk = await critic.score("risky", ["c"] * 10, 0.7, 0.7, 0.5)
        assert no_risk > high_risk


# ═══════════════════════════════════════════════════════════════════════════
# SimulationEngine Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulationEngine:
    """Validate mental rehearsal and candidate evaluation."""

    @pytest.mark.asyncio
    async def test_simulate_returns_result(self) -> None:
        engine = SimulationEngine()
        result = await engine.simulate("Help user write tests")
        assert isinstance(result, SimulationResult)
        assert result.candidate == "Help user write tests"
        assert result.simulation_time >= 0

    @pytest.mark.asyncio
    async def test_goal_aligned_candidate_approved(self) -> None:
        """Candidate aligned with active goals should be approved."""
        goals = GoalMemory()
        await goals.add_goal(
            GoalCube(
                id="g1",
                description="Debug authentication module",
                priority=0.9,
            )
        )

        engine = SimulationEngine(
            goal_provider=goals,
            approval_threshold=0.3,
        )
        result = await engine.simulate(
            "I'll help you debug the authentication module",
            tenant_id="default",
        )
        assert result.goal_alignment > 0.3
        assert result.approved

    @pytest.mark.asyncio
    async def test_goal_misaligned_candidate_rejected(self) -> None:
        """Candidate that violates goals should be rejected with reason."""
        goals = GoalMemory()
        await goals.add_goal(
            GoalCube(
                id="g1",
                description="Fix critical production outage",
                priority=1.0,
            )
        )

        engine = SimulationEngine(
            goal_provider=goals,
            approval_threshold=0.8,
        )
        result = await engine.simulate(
            "Let me tell you about unrelated trivia xyz",
            tenant_id="default",
        )
        # Unrelated candidate should have low goal alignment
        assert result.goal_alignment < 0.8
        assert not result.approved
        assert result.rejection_reason is not None

    @pytest.mark.asyncio
    async def test_compare_candidates_ranked(self) -> None:
        """compare_candidates should return results sorted by score."""
        goals = GoalMemory()
        await goals.add_goal(
            GoalCube(
                id="g1",
                description="Write unit tests for payment module",
            )
        )

        engine = SimulationEngine(goal_provider=goals, approval_threshold=0.3)

        results = await engine.compare_candidates(
            candidates=[
                "Random unrelated topic about weather",
                "I will write payment module unit tests now",
                "Let me review the payment code first",
            ],
            tenant_id="default",
        )

        assert len(results) == 3
        # Results should be sorted by critic_score descending
        assert results[0].critic_score >= results[1].critic_score
        assert results[1].critic_score >= results[2].critic_score

    @pytest.mark.asyncio
    async def test_belief_inconsistency_lowers_score(self) -> None:
        """Contested beliefs should lower belief consistency score."""
        belief_graph = BeliefGraph()
        await belief_graph.record_belief(
            BeliefRecord(
                id="b1",
                memory_id="m1",
                created_by="test",
                created_at=time.time(),
                reason="test",
                trigger="test",
            )
        )
        await belief_graph.add_contradiction("m1", "m2", strength=0.8)

        engine = SimulationEngine(belief_graph=belief_graph)
        result = await engine.simulate("Some action")
        assert result.belief_consistency < 1.0

    @pytest.mark.asyncio
    async def test_simulation_stats_tracked(self) -> None:
        """Statistics should be tracked across simulations."""
        engine = SimulationEngine(approval_threshold=0.3)
        await engine.simulate("Action 1")
        await engine.simulate("Action 2")

        stats = engine.stats()
        assert stats["simulations_run"] == 2
        assert stats["approvals"] + stats["rejections"] == 2

    @pytest.mark.asyncio
    async def test_no_providers_neutral_scores(self) -> None:
        """Without providers, scores should be neutral (not extreme)."""
        engine = SimulationEngine()
        result = await engine.simulate("Any action")
        # Without goal or belief provider, should get neutral values
        assert result.goal_alignment == 0.5
        assert result.belief_consistency == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# SimulationResult Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulationResult:
    """Validate result serialization."""

    def test_to_dict(self) -> None:
        result = SimulationResult(
            candidate="Test action",
            predicted_consequences=["effect_1"],
            goal_alignment=0.85,
            critic_score=0.72,
            approved=True,
        )
        d = result.to_dict()
        assert d["approved"] is True
        assert d["goal_alignment"] == 0.85
        assert d["critic_score"] == 0.72
