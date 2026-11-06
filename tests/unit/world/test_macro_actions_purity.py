from __future__ import annotations

import numpy as np

from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import EngineState
from hbllm.hcir.world.macro_actions import (
    MacroActionStage,
    MacroPlanSynthesizer,
)


def test_macro_plan_synthesizer_purity() -> None:
    state = EngineState(
        avatar_pos=(2, 2),
        avatar_feature=1,
        learned_barriers={(0, 0), (0, 1)},
        learned_goal_positions={(5, 5)},
        step_counter=3,
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    curr_grid = np.zeros((10, 10), dtype=np.int32)
    curr_grid[2, 2] = 1
    curr_grid[5, 5] = 2
    available_actions = [1, 2, 3, 4]

    # Synthesize twice on frozen ctx
    res1 = MacroPlanSynthesizer.synthesize_stage_result(ctx, curr_grid, available_actions)
    res2 = MacroPlanSynthesizer.synthesize_stage_result(ctx, curr_grid, available_actions)

    assert isinstance(res1, StageResult)
    assert isinstance(res2, StageResult)
    assert res1.author == res2.author
    assert res1.signals == res2.signals
    assert len(res1.operations) == len(res2.operations)

    # Immutable context must remain strictly identical
    assert ctx == ctx_before


def test_macro_action_stage_purity() -> None:
    stage = MacroActionStage()
    assert isinstance(stage, CorticalStage)
    assert stage.name == "macro_action_planner"

    state = EngineState(
        avatar_pos=(3, 3),
        avatar_feature=1,
        step_counter=10,
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert res1.author == "macro_action_planner"
    assert res1 == res2
    assert ctx == ctx_before
