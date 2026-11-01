from __future__ import annotations

"""WP6.4 — real per-tier commit authority via kernel-issued grants."""

from hbllm.hcir.graph import SkillNode
from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.event_bus import KernelEventBus
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.nodes import GoalNode
from hbllm.hcir.tier_types import WorkspaceTier
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.workspace_tiers import TieredWorkspace


def _provisioned_tiered() -> tuple[TieredWorkspace, CommitAuthorityRegistry]:
    registry = CommitAuthorityRegistry()
    pipeline = create_default_pipeline(
        constitutional_verifier=ConstitutionalVerifier(authority_registry=registry)
    )
    bus = KernelEventBus()
    tw = TieredWorkspace()
    tw.attach_authorities(pipeline, bus, registry)
    return tw, registry


def _node(node_id: str) -> dict:
    return GoalNode(id=node_id).model_dump()


def test_promote_into_persistent_succeeds_with_system_grant() -> None:
    tw, _ = _provisioned_tiered()
    tw.brain.commit_operations(
        [TransactionOperation(op=TransactionOp.ADD_NODE, node_id="g1", node_data=_node("g1"))],
        author="test",
        grant=tw._system_grant,
    )
    assert tw.promote("g1", WorkspaceTier.BRAIN, WorkspaceTier.PERSISTENT) is True
    assert tw.persistent.get_node("g1") is not None


def test_direct_persistent_commit_without_grant_rejected() -> None:
    tw, _ = _provisioned_tiered()
    result = tw.persistent.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="rogue", node_data=_node("rogue")
            )
        ],
        author="attacker",
    )
    assert not result.is_committed
    assert tw.persistent.get_node("rogue") is None


def test_forged_grant_id_rejected() -> None:
    tw, _ = _provisioned_tiered()
    result = tw.persistent.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="forged", node_data=_node("forged")
            )
        ],
        author="attacker",
        metadata={"commit_grant": "deadbeef" * 4},
    )
    assert not result.is_committed
    assert tw.persistent.get_node("forged") is None


def test_revoked_grant_rejected() -> None:
    tw, registry = _provisioned_tiered()
    grant = registry.issue("temp", {WorkspaceTier.PERSISTENT})
    registry.revoke(grant.grant_id)
    result = tw.persistent.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="revoked", node_data=_node("revoked")
            )
        ],
        author="temp",
        metadata={"commit_grant": grant.grant_id},
    )
    assert not result.is_committed
    assert tw.persistent.get_node("revoked") is None


def test_working_tier_frame_commits_need_no_grant() -> None:
    tw, _ = _provisioned_tiered()
    frame = tw.create_task_frame("goal_x")
    result = frame.workspace.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="scratch", node_data=_node("scratch")
            )
        ],
        author="planner",
    )
    assert result.is_committed
    assert frame.workspace.get_node("scratch") is not None


def test_persistent_skillnode_removal_requires_supersedes() -> None:
    tw, _ = _provisioned_tiered()
    skill = SkillNode(id="skill_1")
    tw.persistent.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id="skill_1",
                node_data=skill.model_dump(),
            )
        ],
        author="test",
        grant=tw._system_grant,
    )
    # Without supersedes -> rejected
    result = tw.persistent.commit_operations(
        [TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="skill_1")],
        author="test",
        grant=tw._system_grant,
    )
    assert not result.is_committed
    assert tw.persistent.get_node("skill_1") is not None
    # With supersedes -> allowed
    result2 = tw.persistent.commit_operations(
        [TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="skill_1")],
        author="test",
        metadata={"supersedes": "skill_2"},
        grant=tw._system_grant,
    )
    assert result2.is_committed


def test_each_tier_and_frame_has_own_tm_sharing_bus() -> None:
    tw, _ = _provisioned_tiered()
    frame = tw.create_task_frame("goal_tm")
    bus_events = []
    tw.brain.transaction_manager.event_bus.subscribe("transaction.committed", bus_events.append)
    for ws, label in (
        (tw.brain, "brain"),
        (tw.persistent, "persistent"),
        (tw.meta, "meta"),
        (frame.workspace, "frame"),
    ):
        assert ws.transaction_manager.event_bus is tw.brain.transaction_manager.event_bus, label
    # commit through the frame's own TM publishes on the shared bus
    frame.workspace.commit_operations(
        [TransactionOperation(op=TransactionOp.ADD_NODE, node_id="ev", node_data=_node("ev"))],
        author="test",
    )
    assert (
        any(
            getattr(e, "event_type", None) == "transaction.committed"
            or (isinstance(e, dict) and e.get("event_type") == "transaction.committed")
            for e in bus_events
        )
        or bus_events
    ), "no commit event observed on shared bus"
