"""Regression tests for per-instance scorer isolation (M-3) and demote() compensation check (M-8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from hbllm.hcir.graph import GoalNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.workspace_tiers import TieredWorkspace, WorkspaceTier
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.motor.effector import EffectorTargetScorer, MotorCortexEffector


@dataclass
class _DummyScorer(EffectorTargetScorer):
    tag: str

    def score_targets(
        self,
        engine: Any,
        curr_grid: Any,
        entities: Any,
        bg: int,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        return [(1, 1, 1.0)]


def test_target_scorer_registration_is_per_engine_instance() -> None:
    MotorCortexEffector.clear_target_scorers()
    eng_a = AutonomousEpistemicEngine(world_id="tenant_a")
    eng_b = AutonomousEpistemicEngine(world_id="tenant_b")

    scorer_a = _DummyScorer(tag="a")
    eng_a.register_target_scorer(scorer_a)

    assert scorer_a in eng_a.target_scorers
    assert scorer_a not in eng_b.target_scorers
    assert scorer_a not in MotorCortexEffector._registered_target_scorers

    eng_a.unregister_target_scorer(scorer_a)
    assert scorer_a not in eng_a.target_scorers


def test_tiered_workspace_demote_escalates_compensation_failure() -> None:
    tw = TieredWorkspace()
    node = GoalNode(id="g_demote_fail", description="demote compensation test")

    res = tw.persistent.commit_operations(
        [TransactionOperation(op=TransactionOp.UPSERT_NODE, node=node)],
        author="test_setup",
    )
    assert res.is_committed

    orig_brain_commit = tw.brain.commit_operations

    def _fail_source_remove(ops: Any, **kwargs: Any) -> Any:
        res_mock = MagicMock()
        res_mock.is_committed = False
        res_mock.status = "forced_remove_failure"
        return res_mock

    def _fail_dest_comp(ops: Any, **kwargs: Any) -> Any:
        if kwargs.get("author", "").endswith(":compensate"):
            comp_mock = MagicMock()
            comp_mock.is_committed = False
            comp_mock.status = "forced_compensation_failure"
            return comp_mock
        return orig_brain_commit(ops, **kwargs)

    tw.persistent.commit_operations = _fail_source_remove  # type: ignore[method-assign]
    tw.brain.commit_operations = _fail_dest_comp  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="compensation failed"):
        tw.demote("g_demote_fail", WorkspaceTier.PERSISTENT, WorkspaceTier.BRAIN)
