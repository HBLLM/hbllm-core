"""Tests for DreamCoder Wake/Sleep Library Learning & Macro Skill Abstraction Engine.

Verifies:
1. Subgoal pattern extraction (k-grams), recurrence tracking, and DreamCoder utility scoring.
2. Cross-context reuse filtering and ranking.
3. Learned heuristic search proposal distribution conditioned on context features.
4. Two-phase wake/sleep skill abstraction cycle.
5. Persistent tier storage and hydration of macro skills.
6. Integration with AutonomousEpistemicEngine and ExecutiveRuntime offline consolidation.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_episodic import EpisodicTransition
from hbllm.hcir.world.library_learning import (
    LearnedProposalDistribution,
    LibraryLearner,
    SubgoalPatternTracker,
)


def test_subgoal_pattern_tracker_kgrams_and_metrics() -> None:
    """Validate extraction of k-grams, invocation tracking, and utility scoring."""
    tracker = SubgoalPatternTracker(min_pattern_length=2, max_pattern_length=4)
    trace = [
        SpatialActionIntent.NAVIGATE_TO,
        SpatialActionIntent.INTERACT,
        SpatialActionIntent.NAVIGATE_TO,
        SpatialActionIntent.ACTUATE,
    ]

    updated = tracker.record_subgoal_trace(trace, context_id="room_alpha", success=True)
    assert len(updated) > 0

    pat = ("NAVIGATE", "INTERACT")
    rec = tracker.patterns.get(pat)
    assert rec is not None
    assert rec.invocation_count == 1
    assert rec.success_count == 1
    assert rec.failure_count == 0
    assert "room_alpha" in rec.contexts_seen
    assert rec.success_rate == 1.0
    assert rec.context_diversity == 1
    assert rec.compression_gain == 1.0

    # Repeat in a second context
    tracker.record_subgoal_trace(trace, context_id="room_beta", success=True)
    assert rec.invocation_count == 2
    assert rec.context_diversity == 2
    assert rec.utility_score > 0.0


def test_qualifying_patterns_filtering_and_ranking() -> None:
    """Validate that only patterns meeting cross-context reuse and success thresholds qualify."""
    tracker = SubgoalPatternTracker(min_pattern_length=2, max_pattern_length=3)

    # Pattern A: seen in 2 contexts, 100% success
    trace_a = ["NAVIGATE", "INTERACT", "DROP"]
    tracker.record_subgoal_trace(trace_a, context_id="ctx_1", success=True)
    tracker.record_subgoal_trace(trace_a, context_id="ctx_2", success=True)

    # Pattern B: seen in 1 context only
    trace_b = ["ACTIVATE", "PICKUP"]
    tracker.record_subgoal_trace(trace_b, context_id="ctx_1", success=True)
    tracker.record_subgoal_trace(trace_b, context_id="ctx_1", success=True)

    # Pattern C: seen in 2 contexts, but 100% failure
    trace_c = ["ALIGN", "MANIPULATE"]
    tracker.record_subgoal_trace(trace_c, context_id="ctx_1", success=False)
    tracker.record_subgoal_trace(trace_c, context_id="ctx_2", success=False)

    qualifying = tracker.get_qualifying_patterns(
        min_invocations=2,
        min_contexts=2,
        min_success_rate=0.5,
        min_utility=0.5,
    )

    qualifying_pats = [q.pattern for q in qualifying]
    assert ("NAVIGATE", "INTERACT") in qualifying_pats
    assert ("ACTIVATE", "PICKUP") not in qualifying_pats  # only 1 context
    assert ("ALIGN", "MANIPULATE") not in qualifying_pats  # failed


def test_learned_proposal_distribution_updates_and_sampling() -> None:
    """Validate empirical Dirichlet-multinomial proposal distribution conditioned on context features."""
    dist = LearnedProposalDistribution(prior_pseudocount=0.1)

    # When context has keys and doors, NAVIGATE and INTERACT succeed
    ctx_key_door = {"has_keys": True, "has_doors": True, "num_barriers": 3}
    for _ in range(5):
        dist.update(ctx_key_door, ["NAVIGATE", "INTERACT"], success=True)

    # When context has movable boxes, PUSH and ALIGN succeed
    ctx_boxes = {"has_cargo": True, "num_boxes": 2}
    for _ in range(5):
        dist.update(ctx_boxes, ["ALIGN", "MANIPULATE"], success=True)

    # Query proposals for key_door context
    proposals_door = dist.sample_candidates(ctx_key_door, top_k=2)
    top_actions_door = [p[0] for p in proposals_door]
    assert "NAVIGATE" in top_actions_door or "INTERACT" in top_actions_door

    # Query proposals for boxes context
    proposals_box = dist.sample_candidates(ctx_boxes, top_k=2)
    top_actions_box = [p[0] for p in proposals_box]
    assert "ALIGN" in top_actions_box or "MANIPULATE" in top_actions_box


def test_wake_sleep_consolidation_cycle() -> None:
    """Validate full two-phase wake/sleep skill abstraction cycle."""
    learner = LibraryLearner()

    trace = ["NAVIGATE", "INTERACT", "ACTIVATE"]
    # Wake phase across context 1
    learner.wake_record_episode(
        intents_or_steps=trace,
        context_id="env_dungeon_1",
        context_features={"has_keys": True},
        success=True,
    )
    # Wake phase across context 2
    learner.wake_record_episode(
        intents_or_steps=trace,
        context_id="env_dungeon_2",
        context_features={"has_keys": True},
        success=True,
    )

    # Before sleep, no macros abstracted yet
    assert len(learner.macro_skills) == 0

    # Sleep phase consolidation
    abstracted = learner.sleep_consolidation(min_invocations=2, min_contexts=2)
    assert len(abstracted) > 0
    macro_name = abstracted[0]
    assert macro_name in learner.macro_skills

    skill = learner.macro_skills[macro_name]
    assert isinstance(skill, DeclarativeNeuroSymbolicSkill)
    assert skill.program is not None
    assert len(skill.program.subgoals) >= 2

    # Instructions can be compiled cleanly
    stream = skill.program.compile_to_stream(skill_name=macro_name)
    assert len(stream.instructions) > 0


def test_persistent_workspace_roundtrip_macro_skills() -> None:
    """Validate persisting abstracted macro skills to TieredWorkspace.persistent and re-hydrating."""
    tiered = TieredWorkspace()
    learner = LibraryLearner()

    trace = ["PICKUP", "NAVIGATE", "DROP"]
    learner.wake_record_episode(trace, context_id="ctx_warehouse_a", success=True)
    learner.wake_record_episode(trace, context_id="ctx_warehouse_b", success=True)

    abstracted = learner.sleep_consolidation(
        tiered_workspace=tiered,
        tenant_id="test_tenant",
        min_invocations=2,
        min_contexts=2,
    )
    assert len(abstracted) > 0
    macro_name = abstracted[0]

    # Verify SkillNode exists in persistent workspace
    persistent_nodes = list(tiered.persistent.graph.nodes.values())
    skill_nodes = [n for n in persistent_nodes if getattr(n, "properties", {}).get("is_macro")]
    assert len(skill_nodes) > 0
    assert any(n.name == macro_name for n in skill_nodes)

    # Re-hydrate in a fresh LibraryLearner
    fresh_learner = LibraryLearner()
    loaded_count = fresh_learner.load_from_workspace(tiered, tenant_id="test_tenant")
    assert loaded_count > 0
    assert macro_name in fresh_learner.macro_skills
    restored_skill = fresh_learner.macro_skills[macro_name]
    assert isinstance(restored_skill, DeclarativeNeuroSymbolicSkill)


def test_autonomous_epistemic_engine_integration_wake_and_proposal() -> None:
    """Validate AutonomousEpistemicEngine wake ingestion and transferable skill querying."""
    ws = HCIRWorkspaceState()
    services = KernelServices.create_default(ws)
    engine = AutonomousEpistemicEngine(services=services)
    engine.current_context_id = "test_chamber_1"

    # Simulate an episodic sequence in episodic_cortex
    engine.episodic_cortex.current_episode = [
        EpisodicTransition(
            step=0,
            avatar_pos=(1, 1),
            action_id=1,  # displacement action
            reward=0.0,
            grid_snapshot=np.zeros((5, 5), dtype=np.int32),
        ),
        EpisodicTransition(
            step=1,
            avatar_pos=(1, 2),
            action_id=1,
            reward=1.0,
            grid_snapshot=np.zeros((5, 5), dtype=np.int32),
        ),
    ]

    # Consolidate episode with positive reward
    res = engine.consolidate_episodic_memory(terminal_reward=1.0)
    assert "schemas" in res

    # Verify wake phase recorded trace in library_learner
    assert len(engine.library_learner.pattern_tracker.patterns) > 0

    # Query proposals
    proposals = engine.library_learner.sample_proposals({"has_barriers": False}, top_k=3)
    assert len(proposals) > 0


@pytest.mark.asyncio
async def test_executive_runtime_offline_sleep_consolidation() -> None:
    """Validate ExecutiveRuntime triggering sleep consolidation and abstracting macro skills."""
    ws = HCIRWorkspaceState()
    services = KernelServices.create_default(ws, tiered_workspace=TieredWorkspace())
    runtime = ExecutiveRuntime(services)
    learner = LibraryLearner()
    runtime.attach_library_learner(learner)

    # Seed traces across 2 contexts
    trace = ["NAVIGATE", "INTERACT"]
    learner.wake_record_episode(trace, context_id="domain_1", success=True)
    learner.wake_record_episode(trace, context_id="domain_1", success=True)
    learner.wake_record_episode(trace, context_id="domain_2", success=True)

    # Run offline consolidation
    await runtime.run_offline_consolidation()

    # Macro skill should have been abstracted and persisted
    assert len(learner.macro_skills) > 0
    macro_name = list(learner.macro_skills.keys())[0]
    assert "macro_" in macro_name

    # Check node in persistent workspace
    persistent_nodes = list(services.tiered_workspace.persistent.graph.nodes.values())
    macro_nodes = [n for n in persistent_nodes if getattr(n, "name", "") == macro_name]
    assert len(macro_nodes) == 1


def test_library_learner_skill_compiles_to_macro_plan_and_guides_scheduler() -> None:
    """Consolidated LibraryLearner skills compile to MacroActionPlan and guide goal ranking."""
    import numpy as np

    from hbllm.hcir.skills.declarative import (
        DeclarativeNeuroSymbolicSkill,
        SpatialActionIntent,
        SubgoalSequence,
        SymbolicSubgoal,
    )
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.causal_discovery import CausalPredicate
    from hbllm.hcir.world.cortex_planner import EpistemicGoalScheduler
    from hbllm.hcir.world.macro_actions import InteractWith, MacroActionPlan, NavigateTo

    class LearnedSwitchGoalSkill(DeclarativeNeuroSymbolicSkill):
        skill_name = "learned_switch_then_goal"
        program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.INTERACT, target_query={"role": "actuator"}),
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE_TO, target_query={"role": "goal"}),
        )

    engine = AutonomousEpistemicEngine(enable_logging=False)
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1  # avatar
    grid[2, 3] = 5  # actuator
    grid[0, 4] = 9  # goal
    engine.avatar_pos = (2, 2)
    engine.avatar_feature = 1
    engine.avatar_features.add(1)
    engine.symbolic_theory.induce_actuator(5, evidence="test")
    engine.symbolic_theory.induce_goal(9, evidence="test")
    engine.record_learned_goal((0, 4))

    plan = MacroActionPlan.from_skill(
        LearnedSwitchGoalSkill(), engine, grid, available_actions=[1, 2, 3, 4, 5]
    )
    assert plan is not None and not plan.is_empty()
    assert isinstance(plan.actions[0], InteractWith)
    assert isinstance(plan.actions[1], NavigateTo)

    # Proposal distribution prefers INTERACT when has_switches is active
    engine2 = AutonomousEpistemicEngine(enable_logging=False)
    engine2.library_learner.wake_record_episode(
        ["INTERACT", "INTERACT", "NAVIGATE"],
        context_id="switch_domain",
        context_features={"has_switches": True},
        success=True,
    )
    engine2.library_learner.wake_record_episode(
        ["INTERACT", "NAVIGATE"],
        context_id="switch_domain_2",
        context_features={"has_switches": True},
        success=True,
    )
    grid2 = np.zeros((5, 5), dtype=int)
    grid2[0, 0] = 1  # avatar at (0, 0)
    grid2[0, 4] = 9  # closer candidate at distance 4
    grid2[4, 4] = 5  # distant walkable switch at distance 8
    engine2.avatar_pos = (0, 0)
    engine2.avatar_feature = 1
    engine2.avatar_features.add(1)
    engine2.symbolic_theory.induce_actuator(5, evidence="test")
    engine2.symbolic_theory.induce_walkable(5, evidence="test_walkable_switch")
    engine2.symbolic_theory.induce_conditional_affordance(
        feature=99,
        condition=CausalPredicate(variable="trigger", operator="==", value=5),
        evidence="test_switch_context",
        prerequisite_feature=5,
    )
    engine2.symbolic_theory.induce_candidate_goal(9, evidence="test")
    engine2.symbolic_theory.induce_candidate_goal(5, evidence="test")
    entities = engine2.extract_entities(grid2, 0)
    prop = EpistemicGoalScheduler.schedule_goals(
        engine=engine2,
        curr_grid=grid2,
        entities=entities,
        bg=0,
        av_feats={1},
        available_actions=[1, 2, 3, 4, 5],
    )
    assert prop.goals
    assert prop.goals[0] == (4, 4)
