"""Unit tests for HCIR Declarative Neuro-Symbolic Skill Grammar and Compilation Engine."""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.bytecode import Opcode
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    AnyOf,
    ColorDistributionPredicate,
    CutSetPredicate,
    DeclarativeNeuroSymbolicSkill,
    EntityBoundingBoxPredicate,
    EntityCountPredicate,
    GridDimensionPredicate,
    Not,
    PanelConstraint,
    SkillEvaluationContext,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent


@pytest.fixture
def sample_grid() -> np.ndarray:
    """Create a 64x64 grid with background 0, an agent at (10, 10), and a button console."""
    grid = np.zeros((64, 64), dtype=int)
    # Agent at (10, 10)
    grid[10:12, 10:12] = 1
    # Console panel on right (x in [42, 58], y in [10, 16] and [30, 36])
    grid[10:16, 42:58] = 9  # button 1: h=6, w=16
    grid[30:36, 42:58] = 6  # button 2: h=6, w=16
    return grid


def test_action_affordance_predicate(sample_grid: np.ndarray) -> None:
    ctx = SkillEvaluationContext(grid=sample_grid, available_actions=[1, 2, 3, 4, 6])

    pred_ok = ActionAffordancePredicate(required={1, 6}, forbidden={5})
    assert pred_ok.evaluate(ctx) is True

    pred_fail_req = ActionAffordancePredicate(required={7})
    assert pred_fail_req.evaluate(ctx) is False

    pred_fail_forbid = ActionAffordancePredicate(forbidden={6})
    assert pred_fail_forbid.evaluate(ctx) is False

    pred_exact = ActionAffordancePredicate(exact={1, 2, 3, 4, 6})
    assert pred_exact.evaluate(ctx) is True


def test_grid_dimension_predicate(sample_grid: np.ndarray) -> None:
    ctx = SkillEvaluationContext(grid=sample_grid, available_actions=[1])

    assert GridDimensionPredicate(exact_shape=(64, 64)).evaluate(ctx) is True
    assert GridDimensionPredicate(exact_shape=(32, 32)).evaluate(ctx) is False
    assert GridDimensionPredicate(min_height=50, max_height=70).evaluate(ctx) is True
    assert GridDimensionPredicate(min_width=100).evaluate(ctx) is False


def test_entity_predicates(sample_grid: np.ndarray) -> None:
    ctx = SkillEvaluationContext(grid=sample_grid, available_actions=[1])

    # 3 entities: agent(1), button1(9), button2(6)
    count_pred = EntityCountPredicate(min_count=3, max_count=3)
    assert count_pred.evaluate(ctx) is True

    # Check button bounding box (w: 10..20, h: 4..10)
    bbox_pred = EntityBoundingBoxPredicate(
        min_width=10, max_width=20, min_height=4, max_height=10, min_count=2
    )
    assert bbox_pred.evaluate(ctx) is True


def test_panel_constraint(sample_grid: np.ndarray) -> None:
    ctx = SkillEvaluationContext(grid=sample_grid, available_actions=[1])

    # Right side panel (col >= 0.6) has buttons
    panel_pred = PanelConstraint(
        min_col_ratio=0.6,
        contains_entities=EntityBoundingBoxPredicate(min_width=12, min_count=2),
    )
    assert panel_pred.evaluate(ctx) is True

    # Left side panel (col < 0.4) does NOT have 2 large buttons
    left_panel = PanelConstraint(
        max_col_ratio=0.4,
        contains_entities=EntityBoundingBoxPredicate(min_width=12, min_count=2),
    )
    assert left_panel.evaluate(ctx) is False


def test_boolean_combinators(sample_grid: np.ndarray) -> None:
    ctx = SkillEvaluationContext(grid=sample_grid, available_actions=[1, 2, 3, 4, 6])

    p1 = ActionAffordancePredicate(required={6})
    p2 = GridDimensionPredicate(exact_shape=(64, 64))
    p3 = EntityCountPredicate(min_count=100)  # False

    # Operator syntax
    assert (p1 & p2).evaluate(ctx) is True
    assert (p1 & p3).evaluate(ctx) is False
    assert (p1 | p3).evaluate(ctx) is True
    assert (~p3).evaluate(ctx) is True
    assert isinstance(AllOf(p1, p2), AllOf)
    assert isinstance(AnyOf(p1, p3), AnyOf)
    assert isinstance(Not(p3), Not)


def test_color_distribution_predicate(sample_grid: np.ndarray) -> None:
    ctx = SkillEvaluationContext(grid=sample_grid, available_actions=[1])

    color_pred = ColorDistributionPredicate(present_colors={0, 1, 6, 9}, absent_colors={5, 7})
    assert color_pred.evaluate(ctx) is True

    fail_pred = ColorDistributionPredicate(present_colors={2})
    assert fail_pred.evaluate(ctx) is False


def test_cut_set_predicate() -> None:
    # Create two rooms separated by a barrier wall
    grid = np.zeros((20, 20), dtype=int)
    grid[2:18, 2:8] = 1  # Room 1
    grid[2:18, 12:18] = 1  # Room 2
    # Columns 8..11 are background 0 (barrier)

    ctx = SkillEvaluationContext(grid=grid, available_actions=[1])
    cut_set_pred = CutSetPredicate(min_partitions=2)
    assert cut_set_pred.evaluate(ctx) is True


def test_subgoal_compilation_to_bytecode() -> None:
    subgoal = SymbolicSubgoal(
        intent=SpatialActionIntent.ACTUATE,
        target_query={"color": 9, "panel": "right"},
        timeout_steps=20,
    )
    instructions = subgoal.compile_to_instructions(author="test_skill")

    assert len(instructions) == 3
    assert instructions[0].opcode == Opcode.QUERY
    assert instructions[0].params["query"] == {"color": 9, "panel": "right"}
    assert instructions[1].opcode == Opcode.ASSERT
    assert instructions[1].params["intent"] == "ACTIVATE"
    assert instructions[2].opcode == Opcode.EXECUTE
    assert instructions[2].params["capability"] == "spatial.activate"

    # Sequence compilation
    seq = SubgoalSequence(
        subgoal,
        SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"role": "goal"}),
    )
    stream = seq.compile_to_stream(skill_name="test_sequence")
    assert stream.length == 6
    assert stream.author == "test_sequence"


def test_declarative_neuro_symbolic_skill(sample_grid: np.ndarray) -> None:
    class DummyConsoleSkill(DeclarativeNeuroSymbolicSkill):
        skill_name = "dummy_console_skill"
        semantic_intent = SpatialActionIntent.ACTUATE

        signature = AllOf(
            ActionAffordancePredicate(required={1, 6}),
            GridDimensionPredicate(exact_shape=(64, 64)),
            PanelConstraint(
                min_col_ratio=0.6,
                contains_entities=EntityBoundingBoxPredicate(min_width=10, min_count=2),
            ),
        )

        program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.ACTUATE, target_query={"color": 9}),
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"color": 6}),
        )

        def solve(
            self, ctx: SkillEvaluationContext, current_level: int = 0
        ) -> list[tuple[int, dict[str, int] | None]]:
            return [(6, {"x": 50, "y": 12}), (1, None)]

    skill = DummyConsoleSkill()

    # Invariant recognition
    assert skill.can_handle(sample_grid, [1, 2, 3, 4, 6]) is True
    assert skill.can_handle(sample_grid, [1, 2, 3, 4]) is False  # Missing action 6

    # Bytecode stream compilation
    stream = skill.compile_to_instruction_stream(sample_grid, [1, 6])
    assert stream.length == 6
    assert stream.author == "dummy_console_skill"

    # Plan as driver actions
    driver_actions = skill.plan_as_driver_actions(sample_grid)
    assert len(driver_actions) == 2
    assert driver_actions[0].action_id == 6
    assert driver_actions[0].parameters == {"x": 50, "y": 12}
    assert driver_actions[1].action_id == 1
