"""Unit tests for WholeGridPredictor (W083 Milestone).

Verifies:
1. Canvas dimension inference across SAME, FIXED, SCALED, and CROP modes.
2. Background color estimation.
3. Raster execution and clipping.
4. Exact match and pixel divergence calculation.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.predictors.whole_grid import (
    DimensionMode,
    WholeGridPredictor,
)
from hbllm.hcir.world.rule_induction import AffineOp, AffineRule


def test_estimate_background_color() -> None:
    """Verify background color detection from perimeter."""
    # 5x5 grid with black (0) perimeter and red (2) center
    grid = np.zeros((5, 5), dtype=int)
    grid[1:4, 1:4] = 2
    assert WholeGridPredictor.estimate_background_color(grid) == 0

    # 4x4 grid with teal (8) perimeter
    grid2 = np.full((4, 4), 8, dtype=int)
    grid2[1:3, 1:3] = 1
    assert WholeGridPredictor.estimate_background_color(grid2) == 8


def test_infer_dimension_rule_same() -> None:
    """Verify SAME dimension rule detection."""
    p1 = (np.zeros((4, 4), dtype=int), np.ones((4, 4), dtype=int))
    p2 = (np.zeros((6, 8), dtype=int), np.ones((6, 8), dtype=int))
    rule = WholeGridPredictor.infer_dimension_rule([p1, p2])
    assert rule.mode == DimensionMode.SAME
    assert rule.compute_output_shape((5, 5)) == (5, 5)


def test_infer_dimension_rule_fixed() -> None:
    """Verify FIXED dimension rule detection."""
    p1 = (np.zeros((4, 4), dtype=int), np.ones((3, 3), dtype=int))
    p2 = (np.zeros((6, 8), dtype=int), np.ones((3, 3), dtype=int))
    rule = WholeGridPredictor.infer_dimension_rule([p1, p2])
    assert rule.mode == DimensionMode.FIXED
    assert rule.compute_output_shape((10, 10)) == (3, 3)


def test_infer_dimension_rule_scaled() -> None:
    """Verify SCALED dimension rule detection."""
    p1 = (np.zeros((2, 3), dtype=int), np.ones((4, 6), dtype=int))
    p2 = (np.zeros((4, 5), dtype=int), np.ones((8, 10), dtype=int))
    rule = WholeGridPredictor.infer_dimension_rule([p1, p2])
    assert rule.mode == DimensionMode.SCALED
    assert rule.compute_output_shape((3, 4)) == (6, 8)


def test_render_prediction_with_padding() -> None:
    """Verify raster prediction with dimension adaptation and padding."""
    rule = AffineRule(AffineOp.ROT_90)
    inp = np.array([[1, 2], [3, 4]], dtype=int)

    # Output of 90 deg clockwise is 2x2: [[3, 1], [4, 2]]
    # Requesting target_shape 3x3 with background 0
    canvas = WholeGridPredictor.render_prediction(rule, inp, target_shape=(3, 3), default_bg=0)
    assert canvas.shape == (3, 3)
    assert canvas[0, 0] == 3
    assert canvas[0, 1] == 1
    assert canvas[1, 0] == 4
    assert canvas[1, 1] == 2
    assert canvas[2, 2] == 0


def test_compute_grid_metrics() -> None:
    """Verify exact match and pixel accuracy calculation."""
    g1 = np.array([[1, 2], [3, 4]], dtype=int)
    g2 = np.array([[1, 2], [3, 4]], dtype=int)
    g3 = np.array([[1, 2], [3, 9]], dtype=int)
    g_diff_shape = np.ones((3, 3), dtype=int)

    exact, acc, mismatches = WholeGridPredictor.compute_grid_metrics(g1, g2)
    assert exact is True
    assert acc == 1.0
    assert mismatches == 0

    exact, acc, mismatches = WholeGridPredictor.compute_grid_metrics(g1, g3)
    assert exact is False
    assert acc == 0.75
    assert mismatches == 1

    exact, acc, _ = WholeGridPredictor.compute_grid_metrics(g1, g_diff_shape)
    assert exact is False
    assert acc == 0.0


def test_whole_grid_predict_state_and_reality_model_ensemble() -> None:
    """Verify WholeGridPredictor integrates into PredictiveRealityModel."""
    from hbllm.hcir.world.predictive_reality import PredictiveRealityModel
    from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

    # 1. Direct predict_state call
    predictor = WholeGridPredictor()
    grid = np.array([[1, 2], [3, 4]], dtype=int)
    snap = WorldStateSnapshot(
        world_id="grid_world",
        variables={"grid": grid, "score": 10},
    )
    res_state, conf = predictor.predict_state(snap, "step_forward")
    assert "grid" in res_state
    assert np.array_equal(res_state["grid"], grid)
    assert conf == 0.90

    # 2. PredictiveRealityModel ensemble with grid variable
    model = PredictiveRealityModel()
    ensemble = model.predict(snap, "step_forward")
    assert "whole_grid" in ensemble.component_predictions
    assert "physics" in ensemble.component_predictions
    assert "snn" in ensemble.component_predictions
    assert ensemble.calibrated_confidence > 0.0

    # 3. Non-grid snapshot maintains standard ensemble without whole_grid
    non_grid_snap = WorldStateSnapshot(world_id="cont_world", variables={"temp": 50})
    non_grid_ensemble = model.predict(non_grid_snap, "cool")
    assert "whole_grid" not in non_grid_ensemble.component_predictions


def test_world_model_wiring_across_core_faculties() -> None:
    """Verify all newly implemented world modeling faculties are cleanly wired into core engines."""
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.cortex_episodic import FalsifiableAnalogyEngine, HippocampalEpisodicCortex
    from hbllm.hcir.world.object_state_graph import ObjectStateGraphPlanner, RelationalGraphMatcher
    from hbllm.hcir.world.spatiotemporal_tracker import MorphologicalDeformationTracker
    from hbllm.hcir.world.world_model_registry import WorldModelRegistry

    # 1. AutonomousEpistemicEngine instance attributes
    engine = AutonomousEpistemicEngine(exploration_budget=50)
    assert isinstance(engine.deformation_tracker, MorphologicalDeformationTracker)
    assert isinstance(engine.relational_matcher, RelationalGraphMatcher)
    assert isinstance(engine.analogy_engine, FalsifiableAnalogyEngine)

    # 2. ObjectStateGraphPlanner instance attribute
    planner = ObjectStateGraphPlanner()
    assert isinstance(planner.relational_matcher, RelationalGraphMatcher)

    # 3. HippocampalEpisodicCortex instance attribute
    hippocampus = HippocampalEpisodicCortex()
    assert isinstance(hippocampus.analogy_engine, FalsifiableAnalogyEngine)

    # 4. WorldModelRegistry default registration
    registry = WorldModelRegistry.create_default()
    grid_models = registry.list_active_models_for_domain("discrete_2d_grid")
    assert len(grid_models) == 1
    assert grid_models[0].model_id == "whole_grid_v1"
