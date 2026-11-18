from __future__ import annotations

import numpy as np

from hbllm.hcir.world.cortical_pipeline import CorticalPipeline, PipelineResult
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import CognitiveContext, EngineState


def test_cortical_pipeline_execution_and_purity() -> None:
    pipeline = CorticalPipeline()
    assert len(pipeline.stages) == 5

    state = EngineState(
        avatar_pos=(2, 2),
        avatar_feature=1,
        step_counter=1,
        curr_grid=np.zeros((5, 5), dtype=int),
        action_dynamics={
            1: type("Dyn", (), {"delta_r": -1, "delta_c": 0})(),
            2: type("Dyn", (), {"delta_r": 1, "delta_c": 0})(),
        },
    )
    ctx = state.freeze()
    ctx_before = state.freeze()

    res = pipeline.run(ctx)
    assert isinstance(res, PipelineResult)
    assert len(res.stage_results) == 5
    assert len(res.operations) >= 4
    final_act = res.final_action[0] if isinstance(res.final_action, tuple) else res.final_action
    assert final_act in (1, 2)
    assert ctx == ctx_before


def test_cortical_pipeline_with_mock_stages() -> None:
    class DummyStage1(CorticalStage):
        name: str = "stage1"

        def observe(self, ctx: CognitiveContext) -> StageResult:
            return StageResult(
                author=self.name,
                operations=[],
                signals={"stage1_done": True},
                proposed_action=None,
            )

    class DummyStage2(CorticalStage):
        name: str = "stage2"

        def observe(self, ctx: CognitiveContext) -> StageResult:
            assert ctx.signals.get("stage1_done") is True
            return StageResult(
                author=self.name,
                operations=[],
                signals={"stage2_done": True},
                proposed_action=42,
            )

    pipeline = CorticalPipeline(stages=[DummyStage1(), DummyStage2()])
    state = EngineState()
    ctx = state.freeze()

    res = pipeline.run(ctx)
    assert res.final_action == 42
    assert res.signals.get("stage1_done") is True
    assert res.signals.get("stage2_done") is True
