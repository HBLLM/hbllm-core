"""HCIR ISA v2 VM & Hypothesis Property Tests (register unification, CallStack, CoW FORK/MERGE)."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import ConceptNode, GoalNode, HCIRNodeType, SkillNode
from hbllm.hcir.interpreter import HCIRInterpreter, SyscallAllowlist
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.replay_debugger import ReplayDebugger
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace


@pytest.mark.asyncio
async def test_isa_v2_query_unify_assert_call_return_pipeline() -> None:
    """Registers propagate across QUERY -> UNIFY -> ASSERT -> CALL -> RETURN inside TieredWorkspace."""
    tiered = TieredWorkspace(tenant_id="tenant_v2")
    services = KernelServices.create_default(
        tiered.working_state,
        tiered_workspace=tiered,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_v2")),
    )
    target_node = ConceptNode(
        id="concept_v2_target",
        label="solar_panel",
        domain="energy",
        scope=Scope(tenant_id="tenant_v2"),
    )
    child_stream = InstructionStream(
        author="skill_child",
        instructions=[
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id="goal_bound_$r0",
                        description="Bind ?label in child frame",
                        scope=Scope(tenant_id="tenant_v2"),
                    ).model_dump()
                },
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
        ],
    )
    callee = SkillNode(
        id="skill_v2_bind",
        skill_name="bind_label",
        parameters=[{"name": "target_label", "required": True}],
        postcondition_pattern={"intent": "bind_concept", "label": "?target_label"},
        bytecode_stream=child_stream.model_dump(),
        scope=Scope(tenant_id="tenant_v2"),
    )
    receipt = tiered.brain.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=callee.id,
                node_data=callee.model_dump(),
            )
        ],
        author="test_v2",
        grant=tiered._system_grant,
        tenant_id="tenant_v2",
    )
    assert receipt.is_committed

    stream = InstructionStream(
        author="test_v2",
        instructions=[
            Instruction(opcode=Opcode.ASSERT, params={"node_data": target_node.model_dump()}),
            Instruction(opcode=Opcode.QUERY, params={"node_type": "concept", "bind": "$r0"}),
            Instruction(
                opcode=Opcode.UNIFY,
                params={
                    "source": "$r0",
                    "pattern": {"domain": "energy", "label": "?lbl"},
                    "bind": "$r1",
                },
            ),
            Instruction(
                opcode=Opcode.CALL,
                params={
                    "goal_pattern": {"intent": "bind_concept", "label": "?lbl"},
                    "args": {"target_label": "?lbl"},
                    "bind": "$r2",
                },
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r1", "$r2"]}),
        ],
    )
    interp = HCIRInterpreter(tiered.working_state, services=services)
    res = await interp.execute(stream)
    assert res.success is True
    assert tiered.working_state.get_node("concept_v2_target") is not None
    ret = res.events[0]["return_value"]
    assert isinstance(ret, list) and len(ret) == 2


@pytest.mark.asyncio
async def test_isa_v2_recursive_call_terminates_cleanly_within_depth_and_fuel() -> None:
    """Recursive self-CALL skills hit CallStack cycle/depth/fuel guards cleanly without infinite recursion."""
    tiered = TieredWorkspace(tenant_id="tenant_rec")
    services = KernelServices.create_default(
        tiered.working_state,
        tiered_workspace=tiered,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_rec")),
    )
    rec_stream = InstructionStream(
        author="test_rec",
        instructions=[
            Instruction(
                opcode=Opcode.CALL,
                params={"skill_id": "skill_self_loop", "args": {"step": "?step"}},
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
        ],
    )
    rec_skill = SkillNode(
        id="skill_self_loop",
        skill_name="self_loop",
        parameters=[{"name": "step", "required": False, "default": 0}],
        bytecode_stream=rec_stream.model_dump(),
        scope=Scope(tenant_id="tenant_rec"),
    )
    rec_receipt = tiered.brain.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=rec_skill.id,
                node_data=rec_skill.model_dump(),
            )
        ],
        author="test_rec",
        grant=tiered._system_grant,
        tenant_id="tenant_rec",
    )
    assert rec_receipt.is_committed

    interp = HCIRInterpreter(tiered.working_state, services=services, max_call_depth=6, max_fuel=40)
    res = await interp.execute(
        InstructionStream(
            author="caller",
            instructions=[Instruction(opcode=Opcode.CALL, params={"skill_id": "skill_self_loop"})],
        )
    )
    assert res.success is False
    assert res.error is not None


@pytest.mark.asyncio
async def test_isa_v2_strict_syscall_allowlist_and_fork_merge_rollback() -> None:
    """Strict allowlist blocks escaped capabilities while CoW FORK -> CALL -> ROLLBACK/MERGE stay isolated."""
    ws = HCIRWorkspaceState(tenant_id="tenant_iso")
    services = KernelServices.create_default(
        ws,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_iso")),
    )
    interp = HCIRInterpreter(ws, services=services)
    assert interp._allowlist.strict is True
    assert SyscallAllowlist().strict is True

    denied = await interp.execute(
        InstructionStream(
            author="test_iso",
            instructions=[
                Instruction(opcode=Opcode.EXECUTE, params={"capability": "planning.decompose"})
            ],
        )
    )
    assert denied.events[0]["results"][0]["blocked_by_allowlist"] is True

    # FORK -> ASSERT on branch -> ROLLBACK leaves main workspace clean
    rb_stream = InstructionStream(
        author="test_iso",
        instructions=[
            Instruction(opcode=Opcode.FORK, params={"branch_name": "sim_rb"}),
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "branch_name": "sim_rb",
                    "node_data": GoalNode(
                        id="goal_rolled_back",
                        description="Should not persist",
                        scope=Scope(tenant_id="tenant_iso"),
                    ).model_dump(),
                },
            ),
            Instruction(opcode=Opcode.ROLLBACK, params={"branch_name": "sim_rb"}),
        ],
    )
    rb_res = await interp.execute(rb_stream)
    assert rb_res.success is True
    assert ws.get_node("goal_rolled_back") is None

    # FORK -> ASSERT on branch -> MERGE promotes node into main workspace
    mg_stream = InstructionStream(
        author="test_iso",
        instructions=[
            Instruction(opcode=Opcode.FORK, params={"branch_name": "sim_mg"}),
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "branch_name": "sim_mg",
                    "node_data": GoalNode(
                        id="goal_merged",
                        description="Should merge",
                        scope=Scope(tenant_id="tenant_iso"),
                    ).model_dump(),
                },
            ),
            Instruction(opcode=Opcode.MERGE, params={"branch_name": "sim_mg"}),
        ],
    )
    mg_res = await interp.execute(mg_stream)
    assert mg_res.success is True
    assert ws.get_node("goal_merged") is not None


@given(
    target_val=st.integers(min_value=1, max_value=500),
    label_txt=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=3, max_size=12),
)
@settings(max_examples=12, deadline=None)
@pytest.mark.asyncio
async def test_hypothesis_replay_debugger_determinism(target_val: int, label_txt: str) -> None:
    """Hypothesis property: identical ISA v2 streams replay identically via ReplayDebugger."""
    ws = HCIRWorkspaceState(tenant_id="tenant_hyp")
    services = KernelServices.create_default(
        ws,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_hyp")),
    )
    stream = InstructionStream(
        author="prop_test",
        instructions=[
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": ConceptNode(
                        id=f"c_{label_txt}_{target_val}",
                        label=label_txt,
                        properties={"code": target_val},
                        scope=Scope(tenant_id="tenant_hyp"),
                    ).model_dump()
                },
            ),
            Instruction(
                opcode=Opcode.UNIFY,
                params={
                    "pattern": {"type": HCIRNodeType.CONCEPT.value, "label": "?l"},
                    "bind": "$r0",
                },
            ),
            Instruction(
                opcode=Opcode.BRANCH,
                params={"condition": "if_empty", "register": "$r0", "target_label": "done"},
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            Instruction(opcode=Opcode.WAIT, label="done", params={"timeout_ms": 1}),
        ],
    )
    debugger = ReplayDebugger(services)
    run_a = await debugger.replay_stream(stream, branch_name="rep_a")
    run_b = await debugger.replay_stream(stream, branch_name="rep_b")
    assert len(run_a) == len(run_b) and len(run_a) >= 3
    snap_a: list[tuple[str, int, Any]] = [
        (s.instruction.opcode.value, s.node_count, s.registers) for s in run_a
    ]
    snap_b: list[tuple[str, int, Any]] = [
        (s.instruction.opcode.value, s.node_count, s.registers) for s in run_b
    ]
    assert snap_a == snap_b
