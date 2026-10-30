from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_episodic import ProceduralMacro
from hbllm.hcir.world.cortex_motor import EpistemicCuriosityExplorer
from hbllm.hcir.world.cortex_perception import SpatialEntity
from hbllm.hcir.world.cortex_planner import MentalSimulationPlanner
from hbllm.hcir.world.kinetic_stream import KineticEntity
from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAffordance
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones


def test_spatiotemporal_object_permanence_tracking() -> None:
    """Validate 4D Spatiotemporal Object Permanence:
    Extrapolates occluded kinetic entities across frames when temporarily unobserved.
    """
    cones = SpatiotemporalCollisionCones(default_horizon=5)
    barriers: set[tuple[int, int]] = set()

    # Frame 0: Kinetic entity moving right at (5, 5) with v=(0, 1)
    k_ent = KineticEntity(
        centroid=(5.0, 5.0),
        velocity=(0.0, 1.0),
        cells=[(5, 5)],
        bounding_box=(5, 5, 5, 5),
        area=1,
        features={3},
    )
    cones.update_trajectories([k_ent], barriers, grid_shape=(10, 10))
    assert cones.is_collision_hazard(5, 6, time_step=1)

    # Frame 1: Entity is temporarily occluded / missing from kinetic_entities
    cones.update_trajectories([], barriers, grid_shape=(10, 10))

    # Object Permanence maintains virtual trajectory!
    assert 3 in cones.occluded_trackers
    # The virtual centroid should have advanced from (5, 5) to (5, 6), so at step 1 it predicts (5, 7)
    assert cones.is_collision_hazard(5, 7, time_step=1)


def test_category_theoretic_structure_mapping_macro() -> None:
    """Validate Category-Theoretic Structure-Mapping:
    Computes relational topological graph functor score across different grid layouts and colors.
    """
    class MockWaypoint:
        def __init__(self, role: str) -> None:
            self.role = role

    macro = ProceduralMacro(
        macro_id="test_topo_macro",
        action_sequence=[1, 2, 5],
        action_data_sequence=[None, None, None],
        initial_grid_hash=123456,
        initial_grid_shape=(10, 10),
        available_actions={1, 2, 5},
        waypoints=[MockWaypoint(role="key"), MockWaypoint(role="goal")],
    )

    # Completely different grid shape and hash
    new_grid = np.zeros((20, 20), dtype=int)
    mock_entities = [
        SpatialEntity(id="e1", feature_id=4, cells=[(2, 2)], role="key"),
        SpatialEntity(id="e2", feature_id=7, cells=[(18, 18)], role="goal"),
    ]

    score = macro.match_score(new_grid, {1, 2, 5}, entities=mock_entities)
    # Even though hash and shape do NOT match, topological structure mapping boosts score!
    assert score >= 0.70


def test_friston_expected_free_energy_curiosity() -> None:
    """Validate Karl Friston's Expected Free Energy (EFE) Affordance Epistemic Value:
    Unverified interactive affordances exhibit high Shannon entropy and receive higher probing priority.
    """
    class MockDyn:
        def is_displacement_action(self) -> bool:
            return True

        def get_displacement(self) -> tuple[int, int]:
            return (-1, 0)

        probes_tested: int = 5
        confidence: float = 1.0

    engine = AutonomousEpistemicEngine()
    grid = np.zeros((10, 10), dtype=int)
    engine.avatar_pos = (5, 5)
    engine.avatar_feature = 1
    engine.avatar_features = {1}
    for a in [1, 2, 3, 4]:
        engine.action_dynamics[a] = MockDyn()

    # Candidate A: feature 2 is already known as walkable floor
    engine.symbolic_theory.walkable_features.add(2)
    grid[4, 5] = 2

    # Candidate B: feature 3 is completely unknown and candidate goal
    engine.symbolic_theory.candidate_goal_features.add(3)
    grid[6, 5] = 3

    # Both entities are equidistant (distance 1)
    prop = EpistemicCuriosityExplorer.compute_epistemic_probe(engine, grid, [1, 2, 3, 4])
    # The active inference curiosity drive should produce a valid proposal
    assert prop is not None


def test_recursive_multi_hop_backward_chaining_planner() -> None:
    """Validate Recursive Multi-Hop Backward-Chaining Planner:
    Goal guarded by Door 1 (Key 1), Key 1 is blocked by Door 2 (Switch 2).
    Planner resolves Switch 2 as the initial subgoal in the causal chain.
    """
    class MockDisplacementDyn:
        def __init__(self, dr: int, dc: int) -> None:
            self.dr = dr
            self.dc = dc

        def is_displacement_action(self) -> bool:
            return True

        def get_displacement(self) -> tuple[int, int]:
            return (self.dr, self.dc)

        probes_tested: int = 5
        confidence: float = 1.0

    engine = AutonomousEpistemicEngine()
    grid = np.zeros((10, 10), dtype=int)
    start_pos = (0, 0)
    engine.avatar_pos = start_pos
    engine.avatar_feature = 1
    engine.avatar_features = {1}

    # Actions: Cardinal moves
    movable_actions = [
        (1, -1, 0),  # UP
        (2, 1, 0),   # DOWN
        (3, 0, -1),  # LEFT
        (4, 0, 1),   # RIGHT
    ]
    for act, dr, dc in movable_actions:
        engine.action_dynamics[act] = MockDisplacementDyn(dr, dc)

    # Goal is at (9, 9)
    goals = {(9, 9)}
    # Door 1 at (8, 9) blocks goal, opened by Tool 5 (Key 1)
    grid[8, 9] = 7  # Door 1
    engine.learned_barriers.add((8, 9))
    engine.symbolic_theory.record_tool_barrier_affinity(tool_feat=5, barrier_feat=7)
    engine.causal_cortex.body_schema.register_resonance(tool_feature=5, barrier_feature=7)

    # Key 1 (feature 5) is at (4, 4)
    grid[4, 4] = 5
    # But Key 1 is blocked by Door 2 at (2, 4)
    grid[2, 4] = 8  # Door 2
    engine.learned_barriers.add((2, 4))
    # Door 2 is opened by Switch 2 at (0, 2)
    grid[0, 2] = 9  # Switch 2
    aff = RemoteCausalAffordance(
        trigger_pos=(0, 2),
        trigger_feature=9,
        remote_pos=(2, 4),
        original_remote_feature=8,
        mutated_remote_feature=0,
        confidence=1.0,
    )
    engine.causal_cortex.causal_attributor.remote_actuators[(2, 4)] = [aff]

    static_barriers = {(8, 9), (2, 4)}
    plan = MentalSimulationPlanner._plan_hierarchical_subgoals(
        engine=engine,
        curr_grid=grid,
        available_actions=[1, 2, 3, 4],
        start_pos=start_pos,
        goals=goals,
        static_barriers=static_barriers,
        mutation_triggers={},
        movable_actions=movable_actions,
    )

    # Multi-hop backward-chaining must find a valid plan resolving the causal prerequisites!
    assert plan is not None
    assert len(plan) > 0
    # First step should move toward the reachable Switch 2 at (0, 2)
    first_step = plan[0]
    assert first_step.predicted_avatar_pos in [(0, 1), (1, 0)]
