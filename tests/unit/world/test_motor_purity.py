from __future__ import annotations

from hbllm.hcir.world.cortex_motor import (
    EpistemicCuriosityExplorer,
    EpistemicProbeProposal,
    MotorStage,
)
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import EngineState


def test_motor_stage_protocol_and_purity() -> None:
    stage = MotorStage()
    assert isinstance(stage, CorticalStage)
    assert stage.name == "motor_stage"

    state = EngineState(
        avatar_pos=(2, 2),
        avatar_feature=1,
        step_counter=5,
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    res1 = stage.observe(ctx)
    res2 = stage.observe(ctx)

    assert isinstance(res1, StageResult)
    assert isinstance(res2, StageResult)
    assert res1.author == "motor_stage"
    assert res1 == res2
    assert ctx == ctx_before


def test_epistemic_probe_proposal_to_stage_result_purity() -> None:
    proposal = EpistemicProbeProposal(
        action=1,
        new_probe_target=(4, 5),
        new_probe_id="probe_test",
        tested_action=1,
        tested_action_position=(1, (2, 2)),
        probed_entity_ids_to_add=("entity_9",),
    )
    res = EpistemicCuriosityExplorer.proposal_to_stage_result(proposal)

    assert isinstance(res, StageResult)
    assert res.author == "cortex_motor"
    assert res.proposed_action == 1
    assert len(res.operations) >= 1
