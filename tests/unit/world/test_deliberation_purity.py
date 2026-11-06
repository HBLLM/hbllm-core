from __future__ import annotations

from hbllm.hcir.world.cortex_deliberation import (
    DeliberationStage,
)
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import EngineState


def test_deliberation_stage_protocol_and_purity() -> None:
    stage = DeliberationStage()
    assert isinstance(stage, CorticalStage)
    assert stage.name == "deliberation_stage"

    state = EngineState(
        avatar_pos=(2, 2),
        avatar_feature=1,
        step_counter=10,
        action_dynamics={
            1: type("Dyn", (), {"delta_r": -1, "delta_c": 0})(),
            2: type("Dyn", (), {"delta_r": 1, "delta_c": 0})(),
        },
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert isinstance(res1, StageResult)
    assert isinstance(res2, StageResult)
    assert res1.author == "deliberation_stage"
    assert res1 == res2
    assert ctx == ctx_before


def test_deliberation_stage_arbitration_signal_purity() -> None:
    stage = DeliberationStage()
    state = EngineState(
        avatar_pos=(1, 1),
        step_counter=4,
        learned_goal_positions={(1, 2)},
    )
    ctx = state.freeze().with_signals({"mental_plan_action": 2})
    ctx_before = state.freeze().with_signals({"mental_plan_action": 2})

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert res1.proposed_action == 2
    assert res1 == res2
    assert ctx == ctx_before
