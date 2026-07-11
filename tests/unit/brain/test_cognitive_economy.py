"""Tests for the Cognitive Economy (Opportunity Market, Budget, Reflection, Episode Lifecycle)."""

from __future__ import annotations

import time

from hbllm.brain.autonomy.cognitive_budget import CognitiveBudget
from hbllm.brain.autonomy.opportunity import Opportunity
from hbllm.brain.autonomy.opportunity_market import OpportunityMarket
from hbllm.brain.autonomy.opportunity_source import ReflectionSource
from hbllm.brain.autonomy.presence_state import PresenceState
from hbllm.brain.planning.workspace_node import WorkspaceEpisode


def test_cognitive_budget_can_afford():
    budget = CognitiveBudget(available_cpu=0.5, available_gpu_time=0.5, interruption_capacity=0.5)

    # Affordable opportunity
    opp_easy = Opportunity(
        id="opp_easy",
        source="test",
        category="test",
        priority=0.8,
        urgency=0.8,
        confidence=0.9,
        created_at=time.time(),
        resource_cost=0.2,
        interruption_cost=0.1,
    )
    assert budget.can_afford(opp_easy) is True

    # Too expensive resource-wise
    opp_expensive = Opportunity(
        id="opp_expensive",
        source="test",
        category="test",
        priority=0.8,
        urgency=0.8,
        confidence=0.9,
        created_at=time.time(),
        resource_cost=0.8,
        interruption_cost=0.1,
    )
    assert budget.can_afford(opp_expensive) is False

    # Too intrusive interruption-wise
    opp_intrusive = Opportunity(
        id="opp_intrusive",
        source="test",
        category="test",
        priority=0.8,
        urgency=0.8,
        confidence=0.9,
        created_at=time.time(),
        resource_cost=0.2,
        interruption_cost=0.6,
    )
    assert budget.can_afford(opp_intrusive) is False


def test_opportunity_market_selection():
    budget = CognitiveBudget(available_cpu=0.8, available_gpu_time=0.8, interruption_capacity=0.8)
    market = OpportunityMarket(budget=budget)

    # Opportunity 1: high utility
    opp1 = Opportunity(
        id="opp1",
        source="test",
        category="test",
        priority=0.9,
        urgency=0.9,
        expected_value=0.9,
        confidence=0.95,
        created_at=time.time(),
        resource_cost=0.1,
        interruption_cost=0.1,
    )

    # Opportunity 2: low utility
    opp2 = Opportunity(
        id="opp2",
        source="test",
        category="test",
        priority=0.3,
        urgency=0.3,
        expected_value=0.3,
        confidence=0.5,
        created_at=time.time(),
        resource_cost=0.5,
        interruption_cost=0.5,
    )

    selected = market.select_opportunities([opp1, opp2])
    assert len(selected) == 2
    assert selected[0].id == "opp1"
    assert selected[1].id == "opp2"


def test_opportunity_dependencies_and_conflicts():
    market = OpportunityMarket()
    market.completed_opportunities.add("dep1")
    market.running_opportunities.add("run1")

    opp_with_dep_met = Opportunity(
        id="opp_dep_met",
        source="test",
        category="test",
        priority=0.8,
        urgency=0.8,
        created_at=time.time(),
        confidence=0.9,
        requires=["dep1"],
    )
    opp_with_dep_unmet = Opportunity(
        id="opp_dep_unmet",
        source="test",
        category="test",
        priority=0.8,
        urgency=0.8,
        created_at=time.time(),
        confidence=0.9,
        requires=["dep2"],
    )
    opp_conflict = Opportunity(
        id="opp_conflict",
        source="test",
        category="test",
        priority=0.8,
        urgency=0.8,
        created_at=time.time(),
        confidence=0.9,
        conflicts=["run1"],
    )

    selected = market.select_opportunities([opp_with_dep_met, opp_with_dep_unmet, opp_conflict])
    assert len(selected) == 1
    assert selected[0].id == "opp_dep_met"


def test_reflection_source_triggers():
    source = ReflectionSource()
    source.add_trigger("user_idle", {"level": "deep_idle"})
    source.add_trigger("goal_failed", {"goal_id": "g1"})

    opps = source.detect(PresenceState(), None)
    assert len(opps) == 2

    # Check details of user_idle trigger reflection opportunity
    idle_opp = next(o for o in opps if "user_idle" in o.id)
    assert idle_opp.category == "reflection"
    assert idle_opp.expected_value == 0.8
    assert idle_opp.resource_cost == 0.7
    assert "deep_memory_consolidation" in idle_opp.reason


def test_workspace_episode_lifecycle():
    episode = WorkspaceEpisode(
        corr_id="corr_123",
        tenant_id="tenant_1",
        session_id="session_1",
        original_query={"text": "hello"},
    )
    assert episode.status == "Created"
    assert episode.resolved is False

    episode.status = "Reasoning"
    assert episode.status == "Reasoning"

    episode.resolved = True
    assert episode.resolved is True
