"""Unit tests for Round 24 Frontier AGI Cognitive Architectures:
- Hierarchical Active Inference with Free Energy Precision Control (Bogacz 2017, Friston 2020)
- Relational Graph Neural-Symbolic Policy Distillation (Garnelo & Shanahan 2019, Gentner 1997)
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.cortex_episodic import (
    EpisodicTransition,
    HippocampalEpisodicCortex,
)
from hbllm.hcir.world.hierarchical_precision_control import (
    HierarchicalPrecisionController,
)
from hbllm.hcir.world.relational_policy_distillation import (
    RelationalPolicyDistiller,
    SpatialRelationType,
)


def test_hierarchical_active_inference_precision_control() -> None:
    """Validate Hierarchical Active Inference with Free Energy Precision Control:
    In stable dynamics, sensory precision is high and volatility is low.
    In volatile/chaotic shocks, sensory precision is downweighted and epistemic exploration gain rises.
    """
    controller = HierarchicalPrecisionController(
        base_learning_rate=0.5,
        volatility_half_life=8,
        prior_rigidity=1.0,
    )

    # 1. Feed 10 stable steps with negligible prediction errors
    grid_a = np.zeros((10, 10), dtype=int)
    grid_b = np.zeros((10, 10), dtype=int)

    for s in range(10):
        state = controller.record_observation(
            step=s,
            predicted_grid=grid_a,
            observed_grid=grid_b,
            is_reset=False,
        )

    assert state.sensory_precision >= 2.0
    assert state.environmental_volatility <= 0.15
    assert not state.is_volatile_regime
    assert state.adaptive_learning_rate >= 0.40

    # 2. Feed sudden chaotic environmental shock / reset
    shock_grid = np.random.randint(1, 8, size=(10, 10))
    for s in range(10, 15):
        shock_state = controller.record_observation(
            step=s,
            predicted_grid=grid_a,
            observed_grid=shock_grid,
            is_reset=True,
        )

    # High volatility triggers volatile regime and increases epistemic exploration gain
    assert shock_state.environmental_volatility > 0.30
    assert shock_state.sensory_precision < state.sensory_precision
    assert shock_state.epistemic_exploration_gain > 1.0


def test_relational_graph_policy_distillation() -> None:
    """Validate Relational Graph Neural-Symbolic Policy Distillation:
    Distills recurring relational state-action configurations into invariant production rules,
    and compiles them into executable DeclarativeNeuroSymbolicSkills.
    """
    entities = [
        SpatialEntity(id="ent_key", feature_id=3, cells=[(2, 2)], role=EntityRole.RESOURCE),
        SpatialEntity(id="ent_door", feature_id=1, cells=[(4, 4)], role=EntityRole.OBSTACLE),
        SpatialEntity(id="ent_goal", feature_id=2, cells=[(8, 8)], role=EntityRole.GOAL),
    ]

    # Trajectory:
    # Step 0: Agent at (2, 1) [Adjacent to key at (2, 2)] -> Executes Action 3 (Move Right)
    # Step 1: Agent at (2, 2) [Coincident with key] -> Executes Action 5 (Interact)
    # Step 2: Agent at (4, 3) [Adjacent to door at (4, 4)] -> Executes Action 3 (Move Right)
    trajectory: list[EpisodicTransition] = [
        EpisodicTransition(step=0, avatar_pos=(2, 1), action_id=3),
        EpisodicTransition(step=1, avatar_pos=(2, 2), action_id=5),
        EpisodicTransition(step=2, avatar_pos=(4, 3), action_id=3),
    ]

    rules = RelationalPolicyDistiller.distill_trajectory(
        trajectory=trajectory,
        entities=entities,
        min_support=1,
        min_confidence=0.5,
    )

    assert len(rules) >= 2

    # Check that a rule for interacting/coinciding with the key was discovered
    coincident_rules = [
        r
        for r in rules
        if r.premise.relation_type == SpatialRelationType.COINCIDENT
        and r.premise.target_role == EntityRole.RESOURCE
    ]
    assert len(coincident_rules) == 1
    assert coincident_rules[0].recommended_action == 5
    assert not coincident_rules[0].is_inhibition

    # Compile rules into a DeclarativeNeuroSymbolicSkill
    skill = RelationalPolicyDistiller.compile_to_declarative_skill(
        rules=rules,
        skill_name="KeyDoorRelationalSkill",
    )
    assert skill.skill_name == "KeyDoorRelationalSkill"
    assert skill.program is not None
    assert len(skill.program.subgoals) >= 2


def test_hippocampal_relational_distillation_integration() -> None:
    """Validate HippocampalEpisodicCortex relational distillation integration:
    Allows episodic memory to distill relational policies during sleep consolidation.
    """
    hippocampus = HippocampalEpisodicCortex()

    entities = [
        SpatialEntity(id="ent_gem", feature_id=6, cells=[(5, 5)], role=EntityRole.RESOURCE),
    ]

    hippocampus.record_transition(
        step=0,
        avatar_pos=(5, 4),
        action=2,  # Move right
    )
    hippocampus.record_transition(
        step=1,
        avatar_pos=(5, 5),
        action=4,  # Collect
        is_win=True,
    )

    rules = hippocampus.distill_relational_policy(entities=entities)
    assert len(rules) >= 1
    assert any(r.premise.target_role == EntityRole.RESOURCE for r in rules)
    assert len(hippocampus.distilled_relational_rules) >= 1
