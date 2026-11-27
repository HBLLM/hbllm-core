"""Execution-equivalence unit and Hypothesis property tests for `.hcir` round-trips."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.fuzzy import FuzzyExpr
from hbllm.hcir.graph import HCIREdge, HCIREdgeType, NodeLifecycle, PhysicalEntityNode, SkillNode
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
    ICapabilityExecutor,
)
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.sexpr import exact_digest, format_hcir, parse_hcir, semantic_digest
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.unification import EdgePattern, GraphPattern, NodePattern
from hbllm.hcir.workspace import HCIRWorkspaceState


class _MathFuzzyEvalExecutor(ICapabilityExecutor):
    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        fe = FuzzyExpr.model_validate(params["expr"])
        deg = fe.evaluate(params.get("bindings", {}), profile=params.get("profile_id", "product"))
        return {"degree": deg}


def _make_twin_systems(
    *,
    initial_nodes: list[Any],
    initial_edges: list[HCIREdge],
    skill: SkillNode | None = None,
    seed_mode: Literal["canonical", "exact"] = "canonical",
) -> tuple[
    tuple[HCIRWorkspaceState, HCIRInterpreter],
    tuple[HCIRWorkspaceState, HCIRInterpreter],
]:
    systems = []
    for roundtrip in (False, True):
        ws = HCIRWorkspaceState()
        ops: list[TransactionOperation] = []
        for n in initial_nodes:
            node_obj = (
                parse_hcir(format_hcir(n, mode=seed_mode)) if roundtrip else n.model_copy(deep=True)
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_obj.id,
                    node_data=node_obj.model_dump(),
                )
            )
        if skill is not None:
            s_obj = (
                parse_hcir(format_hcir(skill, mode=seed_mode))
                if roundtrip
                else skill.model_copy(deep=True)
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=s_obj.id,
                    node_data=s_obj.model_dump(),
                )
            )
        for e in initial_edges:
            edge_obj = (
                parse_hcir(format_hcir(e, mode=seed_mode)) if roundtrip else e.model_copy(deep=True)
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_id=edge_obj.id,
                    edge_data=edge_obj.model_dump(),
                )
            )
        if ops:
            res = ws.commit_operations(ops, author="test_setup")
            assert res.is_committed
        mgr = TransactionManager(ws)
        resolver = CapabilityResolver()
        resolver.register(
            CapabilityImplementation(
                capability_name="math.fuzzy.eval",
                implementation_id="math_fuzzy_eval_test",
                executor=_MathFuzzyEvalExecutor(),
                estimated_cost=1,
            )
        )
        scheduler = KernelInstructionScheduler()
        services = KernelServices(
            workspace=ws,
            transaction_manager=mgr,
            capability_resolver=resolver,
            scheduler=scheduler,
        )
        interp = HCIRInterpreter(ws, services, enable_fuzzy_reasoning=True)
        systems.append((ws, interp))
    return systems[0], systems[1]


def _normalize_exec_event(res: Any) -> dict[str, Any]:
    evt = dict(res.events[0]) if res.events else {}
    norm_results = []
    for item in evt.get("results", []):
        if isinstance(item, dict):
            cleaned = {k: v for k, v in item.items() if k != "tx_id"}
            norm_results.append(cleaned)
        else:
            norm_results.append(item)
    return {
        "success": res.success,
        "error": res.error,
        "return_value": evt.get("return_value"),
        "registers": evt.get("registers"),
        "capabilities_used": evt.get("capabilities_used"),
        "results": norm_results,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fit_degree,expect_merged", [(0.88, True), (0.25, False)])
async def test_nested_fork_fuzzy_unify_alpha_cut_call_merge_rollback_equivalence(
    fit_degree: float, expect_merged: bool
) -> None:
    """Execute FORK -> UNIFY(fuzzy) -> BRANCH(ALPHA_CUT) -> CALL -> MERGE/ROLLBACK across round-trip."""
    agent = PhysicalEntityNode(
        id="agent_1",
        entity_name="agent",
        entity_type="agent",
        properties={"x": 1, "near_goal": fit_degree},
        lifecycle=NodeLifecycle.ACTIVE,
        provenance=Provenance(created_by="test"),
        scope=Scope(tenant_id="default"),
    )
    door = PhysicalEntityNode(
        id="door_1",
        entity_name="door",
        entity_type="portal",
        properties={"locked": True},
        lifecycle=NodeLifecycle.ACTIVE,
        provenance=Provenance(created_by="test"),
        scope=Scope(tenant_id="default"),
    )
    edge = HCIREdge(
        id="edge_near_1",
        edge_type=HCIREdgeType.NEAR,
        sources=["agent_1"],
        targets=["door_1"],
        weight=fit_degree,
    )

    callee_skill = SkillNode(
        id="skill_mark_reached",
        skill_name="mark_reached",
        parameters=[{"name": "target", "required": True}],
        postcondition_pattern={"intent": "mark_reached", "target": "?target"},
        bytecode_stream=InstructionStream(
            author="test",
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={
                        "node_data": {
                            "id": "goal_reached_1",
                            "node_type": "goal",
                            "description": "Reached target",
                            "resolved": True,
                        },
                        "author": "skill_mark_reached",
                        "bind": "$r1",
                    },
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r1"]}),
            ],
        ).model_dump(),
        alpha_success=5.0,
        beta_failure=1.0,
    )

    fuzzy_pat = GraphPattern(
        nodes=[
            NodePattern(
                var="?a",
                node_type="physical_entity",
                fuzzy_memberships={"near_goal": 0.1},
            ),
            NodePattern(var="?d", node_type="physical_entity"),
        ],
        edges=[EdgePattern(source="?a", edge_type="near", target="?d", min_semantic_degree=0.1)],
    ).model_dump()

    program = InstructionStream(
        author="test_planner",
        instructions=[
            Instruction(opcode=Opcode.FORK, params={"branch_name": "sim_cf"}),
            Instruction(
                opcode=Opcode.UNIFY,
                params={
                    "pattern": fuzzy_pat,
                    "fuzzy": True,
                    "profile_id": "product",
                    "min_degree": 0.0,
                    "branch_name": "sim_cf",
                    "bind": {"__semantic_degree__": "$r0"},
                },
            ),
            Instruction(
                opcode=Opcode.BRANCH,
                params={
                    "condition": "if_degree_lt",
                    "register": "$r0",
                    "threshold": 0.5,
                    "target_label": "reject_branch",
                },
            ),
            Instruction(
                opcode=Opcode.CALL,
                params={
                    "goal_pattern": {"intent": "mark_reached", "target": "?d"},
                    "branch_name": "sim_cf",
                    "bind": "$r2",
                },
            ),
            Instruction(opcode=Opcode.MERGE, params={"branch_name": "sim_cf", "author": "planner"}),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r2"]}),
            Instruction(
                opcode=Opcode.ROLLBACK,
                params={"branch_name": "sim_cf", "reason": "below_alpha"},
                label="reject_branch",
            ),
            Instruction(opcode=Opcode.RETURN, params={"value": "rejected"}),
        ],
    )

    (ws_orig, interp_orig), (ws_rt, interp_rt) = _make_twin_systems(
        initial_nodes=[agent, door],
        initial_edges=[edge],
        skill=callee_skill,
        seed_mode="exact",
    )
    assert dict(ws_orig.graph.nodes) == dict(ws_rt.graph.nodes)
    assert dict(ws_orig.graph.edges) == dict(ws_rt.graph.edges)
    assert exact_digest(ws_orig.graph) == exact_digest(ws_rt.graph)
    rt_program = parse_hcir(format_hcir(program, mode="canonical"))

    res_orig = await interp_orig.execute(program)
    res_rt = await interp_rt.execute(rt_program)

    assert _normalize_exec_event(res_orig) == _normalize_exec_event(res_rt)
    assert semantic_digest(ws_orig.graph) == semantic_digest(ws_rt.graph)
    assert (ws_orig.get_node("goal_reached_1") is not None) is expect_merged
    if not expect_merged:
        assert dict(ws_orig.graph.nodes) == dict(ws_rt.graph.nodes)
        assert exact_digest(ws_orig.graph) == exact_digest(ws_rt.graph)

    # Also verify equivalence when workspace skill + entities are round-tripped in canonical mode
    (ws_c1, interp_c1), (ws_c2, interp_c2) = _make_twin_systems(
        initial_nodes=[agent, door],
        initial_edges=[edge],
        skill=callee_skill,
        seed_mode="canonical",
    )
    res_c1 = await interp_c1.execute(program)
    res_c2 = await interp_c2.execute(rt_program)
    assert _normalize_exec_event(res_c1) == _normalize_exec_event(res_c2)
    assert semantic_digest(ws_c1.graph) == semantic_digest(ws_c2.graph)


@pytest.mark.asyncio
@settings(max_examples=50, deadline=None)
@given(
    fit_deg=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    alpha=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    use_fuzzy_eval=st.booleans(),
    do_call=st.booleans(),
)
async def test_property_execution_equivalence(
    fit_deg: float, alpha: float, use_fuzzy_eval: bool, do_call: bool
) -> None:
    """Randomized multi-opcode programs execute identically before and after `.hcir` round-trip."""
    agent = PhysicalEntityNode(
        id="ent_a",
        entity_name="ent_a",
        entity_type="agent",
        properties={"near_goal": fit_deg},
        provenance=Provenance(created_by="prop"),
    )
    callee = SkillNode(
        id="prop_callee",
        skill_name="prop_callee",
        postcondition_pattern={"intent": "prop_act", "target": "?target"},
        bytecode_stream=InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={
                        "node_data": {
                            "id": "prop_asserted",
                            "node_type": "goal",
                            "description": "prop goal",
                        },
                        "bind": "$r1",
                    },
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r1"]}),
            ]
        ).model_dump(),
    )

    ins: list[Instruction] = [
        Instruction(opcode=Opcode.FORK, params={"branch_name": "cf"}),
        Instruction(
            opcode=Opcode.UNIFY,
            params={
                "pattern": GraphPattern(
                    nodes=[
                        NodePattern(
                            var="?e",
                            node_type="physical_entity",
                            fuzzy_memberships={"near_goal": 0.4},
                        )
                    ]
                ).model_dump(),
                "fuzzy": True,
                "profile_id": "product",
                "min_degree": 0.0,
                "branch_name": "cf",
                "bind": {"__semantic_degree__": "$r0"},
            },
        ),
    ]
    if use_fuzzy_eval:
        ins.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "math.fuzzy.eval",
                    "params": {
                        "expr": FuzzyExpr(
                            op="hedge",
                            args={"hedge": "very"},
                            children=[FuzzyExpr(op="const", args={"value": fit_deg})],
                        ).model_dump(),
                        "profile_id": "product",
                    },
                    "bind": "$r3",
                },
            )
        )
    ins.append(
        Instruction(
            opcode=Opcode.BRANCH,
            params={
                "condition": "if_degree_lt",
                "register": "$r0",
                "threshold": alpha,
                "target_label": "abort_cf",
            },
        )
    )
    if do_call:
        ins.append(
            Instruction(
                opcode=Opcode.CALL,
                params={
                    "goal_pattern": {"intent": "prop_act", "target": "?e"},
                    "branch_name": "cf",
                    "bind": "$r2",
                },
            )
        )
    ins.extend(
        [
            Instruction(opcode=Opcode.MERGE, params={"branch_name": "cf"}),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            Instruction(opcode=Opcode.ROLLBACK, params={"branch_name": "cf"}, label="abort_cf"),
            Instruction(opcode=Opcode.RETURN, params={"value": "aborted"}),
        ]
    )
    stream = InstructionStream(author="prop", instructions=ins)

    (ws_orig, interp_orig), (ws_rt, interp_rt) = _make_twin_systems(
        initial_nodes=[agent],
        initial_edges=[],
        skill=callee,
        seed_mode="canonical",
    )
    rt_stream = parse_hcir(format_hcir(stream, mode="canonical"))

    res_orig = await interp_orig.execute(stream)
    res_rt = await interp_rt.execute(rt_stream)

    assert _normalize_exec_event(res_orig) == _normalize_exec_event(res_rt)
    assert semantic_digest(ws_orig.graph) == semantic_digest(ws_rt.graph)
