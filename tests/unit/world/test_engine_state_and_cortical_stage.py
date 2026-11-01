from __future__ import annotations

import pytest

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import CognitiveContext, EngineState


def test_engine_state_defaults() -> None:
    state = EngineState()
    assert state.phase == EpistemicPhase.MOTOR_GROUNDING
    assert state.step_counter == 0
    assert state.avatar_pos is None
    assert len(state.learned_barriers) == 0
    assert len(state.mental_plan) == 0


def test_engine_state_freeze_immutability() -> None:
    state = EngineState(
        avatar_pos=(2, 3),
        avatar_feature=5,
        learned_barriers={(1, 1), (1, 2)},
        step_counter=10,
    )
    ctx = state.freeze()
    assert isinstance(ctx, CognitiveContext)
    assert ctx.avatar_pos == (2, 3)
    assert ctx.avatar_feature == 5
    assert ctx.step_counter == 10
    assert (1, 1) in ctx.learned_barriers

    # Context is frozen / immutable
    with pytest.raises(Exception):
        ctx.avatar_pos = (4, 4)  # type: ignore[misc]


def test_cognitive_context_with_signals() -> None:
    state = EngineState(step_counter=5)
    ctx = state.freeze()
    ctx2 = ctx.with_signals({"perception_salience": 0.8, "detected_threat": True})

    assert ctx2.step_counter == 5
    assert ctx2.signals["perception_salience"] == 0.8
    assert ctx2.signals["detected_threat"] is True
    # Original ctx unchanged
    assert "perception_salience" not in ctx.signals


def test_stage_result_and_protocol() -> None:
    class DummyPerceptionStage:
        name = "dummy_perception"

        def observe(self, ctx: CognitiveContext) -> StageResult:
            ops = [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node={
                        "id": "obs_step_1",
                        "type": "ObservationNode",
                        "properties": {"step": ctx.step_counter},
                    },
                )
            ]
            return StageResult(
                author=self.name,
                operations=ops,
                signals={"target_located": True},
                proposed_action=1,
            )

    stage: CorticalStage = DummyPerceptionStage()
    state = EngineState(step_counter=1)
    ctx = state.freeze()

    res = stage.observe(ctx)
    assert res.author == "dummy_perception"
    assert len(res.operations) == 1
    assert res.signals["target_located"] is True
    assert res.proposed_action == 1


def test_engine_state_to_transaction_operations() -> None:
    state = EngineState(avatar_pos=(3, 4), step_counter=7)
    ops = state.to_transaction_operations(tenant_id="tenant_1")
    assert len(ops) >= 1
    assert any(op.op == TransactionOp.UPSERT_NODE for op in ops)


def test_autonomous_epistemic_engine_freeze() -> None:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

    engine = AutonomousEpistemicEngine(enable_logging=False, world_id="test_world")
    engine.step_counter = 12
    engine.avatar_pos = (5, 6)
    engine.learned_barriers.add((1, 2))

    ctx = engine.freeze()
    assert isinstance(ctx, CognitiveContext)
    assert ctx.step_counter == 12
    assert ctx.avatar_pos == (5, 6)
    assert (1, 2) in ctx.learned_barriers
    assert ctx.world_id == "test_world"
