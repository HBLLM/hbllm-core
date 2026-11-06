from __future__ import annotations

from hbllm.hcir.world.cortex_episodic import EpisodicStage
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import EngineState


def test_episodic_stage_protocol_and_purity() -> None:
    stage = EpisodicStage()
    assert isinstance(stage, CorticalStage)
    assert stage.name == "episodic_stage"

    state = EngineState(
        avatar_pos=(1, 1),
        avatar_feature=2,
        step_counter=8,
        learned_barriers={(0, 0), (2, 2)},
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert isinstance(res1, StageResult)
    assert isinstance(res2, StageResult)
    assert res1.author == "episodic_stage"
    assert res1 == res2
    assert ctx == ctx_before


def test_episodic_stage_with_terminal_signal_purity() -> None:
    stage = EpisodicStage()
    state = EngineState(step_counter=15)
    ctx = state.freeze().with_signals({"terminal": "SUCCESS"})
    ctx_before = state.freeze().with_signals({"terminal": "SUCCESS"})

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert res1.signals["episode_boundary"] is True
    assert res1 == res2
    assert ctx == ctx_before
