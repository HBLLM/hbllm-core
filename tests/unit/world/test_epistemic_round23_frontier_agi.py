from __future__ import annotations

import numpy as np

from hbllm.hcir.skills.declarative import SpatialActionIntent, SymbolicSubgoal
from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.world.cortex_episodic import (
    EpisodicTransition,
    FalsifiableAnalogyEngine,
    HippocampalEpisodicCortex,
)
from hbllm.hcir.world.cortex_perception import PerceptionEngine, SpatialEntity
from hbllm.hcir.world.foveal_saccade_attention import FovealSaccadeAttention
from hbllm.hcir.world.shapley_credit_attribution import (
    CounterfactualShapleyCreditAttributor,
)
from hbllm.hcir.world.structure_mapping_functor import (
    CategoryTheoreticStructureMappingEngine,
)


def test_category_theoretic_structure_mapping_functor() -> None:
    """Validate Category-Theoretic Structure Mapping Functor (Gentner's SME):
    Induces functor morphism between source domain and target domain, aligns relational
    topological structures, and pushes declarative subgoals forward into target entities.
    """
    # Source domain: Key at (2, 2) [feat 3], Barrier at (4, 4) [feat 1], Goal at (8, 8) [feat 2]
    src_entities = [
        SpatialEntity(id="src_key", feature_id=3, cells=[(2, 2)], role=EntityRole.RESOURCE),
        SpatialEntity(id="src_door", feature_id=1, cells=[(4, 4)], role=EntityRole.OBSTACLE),
        SpatialEntity(id="src_goal", feature_id=2, cells=[(8, 8)], role=EntityRole.GOAL),
    ]

    # Target domain: Switch at (3, 3) [feat 7], Portal at (5, 5) [feat 9], Target at (12, 12) [feat 6]
    tgt_entities = [
        SpatialEntity(id="tgt_switch", feature_id=7, cells=[(3, 3)], role=EntityRole.ACTUATOR),
        SpatialEntity(id="tgt_portal", feature_id=9, cells=[(5, 5)], role=EntityRole.PORTAL),
        SpatialEntity(id="tgt_target", feature_id=6, cells=[(12, 12)], role=EntityRole.GOAL),
    ]

    src_subgoals = [
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE,
            target_query={"target_role": "key", "feature": 3},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.INTERACT,
            target_query={"target_role": "goal", "feature": 2},
        ),
    ]

    # Induce Functor F: C_source -> C_target
    result = CategoryTheoreticStructureMappingEngine.induce_functor(
        source_entities=src_entities,
        target_entities=tgt_entities,
        source_domain="game_A",
        target_domain="game_B",
        source_subgoals=src_subgoals,
        grid_shape=(20, 20),
    )

    assert result.alignment_score >= 0.60
    assert result.feature_mapping[3] == 7  # Key -> Switch
    assert result.feature_mapping[2] == 6  # Goal -> Target
    assert len(result.pushed_subgoals) == 2
    # First pushed subgoal now references feature 7 (the switch)
    assert result.pushed_subgoals[0].target_query["feature"] == 7

    # Integration with FalsifiableAnalogyEngine
    analogy_engine = FalsifiableAnalogyEngine()
    morphism = analogy_engine.induce_structure_mapping_functor(
        source_entities=src_entities,
        target_entities=tgt_entities,
        source_domain="game_A",
        target_domain="game_B",
        source_subgoals=src_subgoals,
    )
    assert morphism is not None
    assert morphism.confidence >= 0.60
    assert morphism.feature_mapping[3] == 7


def test_counterfactual_shapley_credit_attribution() -> None:
    """Validate Counterfactual Shapley Credit Attribution (Deep Credit Assignment):
    When an episode is lost, identifies the root causal step (e.g. corner deadlock at step 2)
    rather than uniformly blaming the last steps of harmless wandering.
    """
    grid = np.zeros((10, 10), dtype=int)
    barriers = {(0, 1), (1, 0)}  # Forms corner deadlock at (0, 0)
    goals = {(8, 8)}

    # Trajectory:
    # Step 0: (2, 2)
    # Step 1: (1, 1)
    # Step 2: (0, 0) -> Irreversible corner deadlock created!
    # Steps 3..9: Wandering trapped in/around corner
    # Step 10: Lost terminal
    trajectory: list[EpisodicTransition] = []
    trajectory.append(
        EpisodicTransition(step=0, avatar_pos=(2, 2), action_id=1, grid_snapshot=grid)
    )
    trajectory.append(
        EpisodicTransition(step=1, avatar_pos=(1, 1), action_id=1, grid_snapshot=grid)
    )
    trajectory.append(
        EpisodicTransition(step=2, avatar_pos=(0, 0), action_id=3, grid_snapshot=grid)
    )  # Deadlock
    for s in range(3, 10):
        trajectory.append(
            EpisodicTransition(step=s, avatar_pos=(0, 0), action_id=2, grid_snapshot=grid)
        )
    trajectory.append(
        EpisodicTransition(
            step=10, avatar_pos=(0, 0), action_id=2, is_failure=True, grid_snapshot=grid
        )
    )

    credit = CounterfactualShapleyCreditAttributor.attribute_blame(
        trajectory=trajectory,
        learned_goals=goals,
        learned_barriers=barriers,
    )

    # Root causal step is correctly identified as step 2 (where the deadlock occurred)
    assert credit.root_causal_step == 2
    assert credit.root_fatal_pos == (0, 0)
    assert credit.root_fatal_action == 3
    assert (
        "corner_deadlock" in credit.causal_explanation
        or "reachability_collapse" in credit.causal_explanation
    )

    # Integration test with HippocampalEpisodicCortex
    hippocampus = HippocampalEpisodicCortex()
    hippocampus.current_episode = list(trajectory)
    setattr(hippocampus, "learned_goals", goals)
    setattr(hippocampus, "learned_barriers", barriers)
    assignment = hippocampus.trigger_sharp_wave_ripple_replay(
        is_failure=True,
        is_success=False,
        prev_grid=grid,
    )
    assert assignment is not None
    assert assignment.root_causal_step == 2
    assert assignment.root_fatal_action == 3
    assert ((0, 0), 3) in assignment.inhibited_transitions


def test_foveal_saliency_and_active_saccading() -> None:
    """Validate Foveal Saliency & Active Visual Saccading (Parietal FEF/SC):
    Detects sensory difference masks and bounds visual attention to dynamic foveal windows.
    """
    prev_grid = np.zeros((30, 30), dtype=int)
    curr_grid = np.zeros((30, 30), dtype=int)

    # Localized 2x2 movement at rows 5..6, cols 7..8
    curr_grid[5:7, 7:9] = 4

    attention = FovealSaccadeAttention(fovea_padding=2)
    saccade = attention.compute_saccade(curr_grid=curr_grid, prev_grid=prev_grid)

    assert saccade.is_foveated_update
    assert saccade.changed_pixel_count == 4
    # Gaze centroid centered on the dynamic patch (5.5, 7.5)
    assert 5.0 <= saccade.gaze_centroid[0] <= 6.0
    assert 7.0 <= saccade.gaze_centroid[1] <= 8.0

    # PerceptionEngine integration
    saccade_p = PerceptionEngine.compute_foveal_saccade(curr_grid, prev_grid)
    assert saccade_p.is_foveated_update
    assert saccade_p.changed_pixel_count == 4
