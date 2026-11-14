"""Unit tests for Theory of Mind Agency Detection and Bayesian Inverse Planning.

Validates intentional agency classification, inverse planning goal inferences,
and intent-conditioned forward trajectory projections.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.transactions import TransactionOp
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.social_cognition import (
    AgencyStatus,
    InferredIntent,
    SocialCognitionEngine,
)


def test_passive_ballistic_entity_classification() -> None:
    """Entities moving inertially or bouncing off walls remain classified as physical objects."""
    engine = SocialCognitionEngine(agency_turn_threshold=2)
    grid = np.zeros((10, 10), dtype=int)

    # Ballistic linear movement: (1, 1) -> (2, 1) -> (3, 1) -> (4, 1)
    engine.track_entity_position("rock_1", feature_id=5, position=(1, 1), grid=grid)
    engine.track_entity_position("rock_1", feature_id=5, position=(2, 1), grid=grid)
    model = engine.track_entity_position("rock_1", feature_id=5, position=(3, 1), grid=grid)

    assert model.agency_status == AgencyStatus.PHYSICAL
    assert model.non_inertial_turns == 0

    # Project trajectory linearly
    projected = engine.project_intent_trajectory("rock_1", horizon=3, grid=grid)
    assert projected == [(4, 1), (5, 1), (6, 1)]


def test_non_inertial_turns_trigger_intentional_agency() -> None:
    """Entities taking spontaneous non-inertial turns in open space become intentional agents."""
    engine = SocialCognitionEngine(agency_turn_threshold=2)
    grid = np.zeros((10, 10), dtype=int)

    # Move down: (2, 2) -> (3, 2)
    engine.track_entity_position("patroller_1", feature_id=8, position=(2, 2), grid=grid)
    engine.track_entity_position("patroller_1", feature_id=8, position=(3, 2), grid=grid)

    # Turn right in empty space: (3, 3) (Turn 1)
    engine.track_entity_position("patroller_1", feature_id=8, position=(3, 3), grid=grid)
    model = engine.tracked_agents["patroller_1"]
    assert model.non_inertial_turns == 1
    assert model.agency_status == AgencyStatus.PHYSICAL

    # Turn up in empty space: (2, 3) (Turn 2 -> Threshold reached)
    model = engine.track_entity_position("patroller_1", feature_id=8, position=(2, 3), grid=grid)
    assert model.non_inertial_turns == 2
    assert model.agency_status == AgencyStatus.INTENTIONAL_AGENT


def test_bayesian_inverse_planning_pursuit() -> None:
    """Agent moving toward avatar is inferred as pursuing avatar."""
    engine = SocialCognitionEngine(rationality_beta=1.5)

    # Avatar is at (8, 8), resource at (1, 9)
    avatar_pos = (8, 8)
    candidate_goals = [(1, 9)]

    # Entity starts at (2, 2) and steps toward (8, 8): (3, 3) -> (4, 4) -> (5, 5)
    engine.track_entity_position("chaser", feature_id=3, position=(2, 2))
    engine.track_entity_position("chaser", feature_id=3, position=(3, 3))
    engine.track_entity_position("chaser", feature_id=3, position=(4, 4))
    engine.track_entity_position("chaser", feature_id=3, position=(5, 5))

    inferred = engine.invert_planner_for_intent(
        "chaser",
        avatar_pos=avatar_pos,
        candidate_goals=candidate_goals,
    )

    assert inferred.inferred_intent == InferredIntent.PURSUING_AVATAR
    assert inferred.most_likely_goal == (8, 8)
    assert inferred.confidence > 0.8
    assert inferred.goal_posterior[(8, 8)] > inferred.goal_posterior[(1, 9)]


def test_bayesian_inverse_planning_resource_seeking() -> None:
    """Agent moving toward environmental resource is inferred as seeking resource."""
    engine = SocialCognitionEngine(rationality_beta=1.5)

    avatar_pos = (9, 1)
    resource_pos = (2, 9)

    # Entity steps toward resource: (2, 2) -> (2, 4) -> (2, 6)
    engine.track_entity_position("forager", feature_id=4, position=(2, 2))
    engine.track_entity_position("forager", feature_id=4, position=(2, 4))
    engine.track_entity_position("forager", feature_id=4, position=(2, 6))

    inferred = engine.invert_planner_for_intent(
        "forager",
        avatar_pos=avatar_pos,
        candidate_goals=[resource_pos],
    )

    assert inferred.inferred_intent == InferredIntent.SEEKING_RESOURCE
    assert inferred.most_likely_goal == resource_pos
    assert inferred.confidence > 0.8


def test_intent_conditioned_trajectory_projection() -> None:
    """Projected trajectory moves towards inferred goal rather than continuing straight."""
    engine = SocialCognitionEngine()
    engine.track_entity_position("predator", feature_id=7, position=(3, 3))
    engine.track_entity_position("predator", feature_id=7, position=(4, 3))
    engine.track_entity_position("predator", feature_id=7, position=(4, 4))
    engine.tracked_agents["predator"].agency_status = AgencyStatus.INTENTIONAL_AGENT

    # Avatar is at (7, 4)
    engine.invert_planner_for_intent("predator", avatar_pos=(7, 4), candidate_goals=[])

    proj = engine.project_intent_trajectory("predator", horizon=3)
    assert len(proj) == 3
    # Trajectory moves downward toward row 7
    assert proj[0][0] > 4
    assert proj[-1][0] >= 6


def test_causal_predicate_and_transaction_export() -> None:
    """Inferred intentions export to formal declarative CausalPredicates and transactions."""
    engine = SocialCognitionEngine()
    engine.track_entity_position("npc_1", feature_id=6, position=(0, 0))
    engine.track_entity_position("npc_1", feature_id=6, position=(1, 1))
    agent = engine.invert_planner_for_intent("npc_1", avatar_pos=(5, 5))

    predicate = agent.to_causal_predicate()
    assert predicate.variable == "social_agent:npc_1"
    assert predicate.relation == "inferred_intent"
    assert "PURSUING_AVATAR" in predicate.value

    tx_ops = agent.to_transaction_operations()
    assert len(tx_ops) == 1
    assert tx_ops[0].op == TransactionOp.UPSERT_NODE
    assert tx_ops[0].node_id == "intentional_agent_npc_1"
    assert tx_ops[0].node_data["inferred_intent"] == "PURSUING_AVATAR"


def test_epistemic_engine_social_tracking_integration() -> None:
    """AutonomousEpistemicEngine tracks and infers agent intents cleanly."""
    epistemic_engine = AutonomousEpistemicEngine()
    epistemic_engine.avatar_pos = (9, 9)
    epistemic_engine.goals = [(0, 9)]

    model = epistemic_engine.track_and_infer_agent_intents(
        entity_id="ghost_1",
        feature_id=2,
        position=(1, 1),
    )
    assert model.entity_id == "ghost_1"

    # Step towards avatar
    model = epistemic_engine.track_and_infer_agent_intents(
        entity_id="ghost_1",
        feature_id=2,
        position=(3, 3),
    )
    assert model.most_likely_goal == (9, 9)
    assert model.inferred_intent == InferredIntent.PURSUING_AVATAR
