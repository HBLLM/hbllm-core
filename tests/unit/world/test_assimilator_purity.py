from __future__ import annotations

from hbllm.hcir.world.cortex_assimilator import (
    AssimilationStage,
    EpistemicPhase,
)
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import EngineState


def test_assimilator_stage_protocol_and_purity() -> None:
    stage = AssimilationStage()
    assert isinstance(stage, CorticalStage)
    assert stage.name == "assimilation_stage"

    state = EngineState(
        avatar_pos=(3, 3),
        avatar_feature=2,
        step_counter=5,
        phase=EpistemicPhase.EPISTEMIC_EXPLORATION,
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert isinstance(res1, StageResult)
    assert isinstance(res2, StageResult)
    assert res1.author == "assimilation_stage"
    assert res1.proposed_action is None
    assert res1 == res2
    assert ctx == ctx_before


def test_assimilator_stage_signal_purity() -> None:
    stage = AssimilationStage()
    state = EngineState(
        avatar_pos=(1, 1),
        step_counter=12,
        phase=EpistemicPhase.MENTAL_SIMULATION,
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert res1.signals.get("assimilated_phase") == "mental_simulation"
    assert res1 == res2
    assert ctx == ctx_before
