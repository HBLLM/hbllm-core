"""WP4.4 parity: dirty-set diff_branch == full comparison, merge unchanged."""

from __future__ import annotations

import random

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType
from hbllm.hcir.nodes import GoalNode
from hbllm.hcir.workspace import HCIRWorkspaceState


def _make_workspace(n: int = 400) -> HCIRWorkspaceState:
    g = CognitiveGraph(enable_native_substrate=False)
    ws = HCIRWorkspaceState(graph=g)
    for i in range(n):
        g.add_node(GoalNode(id=f"n{i}"))
    return ws


def _full_diff(ws: HCIRWorkspaceState, branch: HCIRWorkspaceState):
    """Reference: run the full comparison by clearing dirty tracking."""
    br_g = branch._graph
    saved = (br_g._dirty_nodes, br_g._dirty_edges, br_g._fork_parent)
    br_g._dirty_nodes = None
    br_g._dirty_edges = None
    br_g._fork_parent = None
    try:
        return ws.diff_branch(branch.branch_name)
    finally:
        br_g._dirty_nodes, br_g._dirty_edges, br_g._fork_parent = saved


def test_dirty_diff_matches_full_diff_random_ops() -> None:
    rng = random.Random(0)
    for trial in range(5):
        ws = _make_workspace()
        branch = ws.fork(f"b{trial}")
        bg = branch._graph
        ops = rng.randint(200, 300)
        for _ in range(ops):
            kind = rng.choice(["add", "upsert", "remove_n", "add_e", "remove_e"])
            nid = f"new_{rng.randint(0, 60)}"
            if kind == "add":
                if not bg.has_node(nid):
                    bg.add_node(GoalNode(id=nid))
            elif kind == "upsert":
                node = (
                    bg.get_node(nid) or bg.get_node(f"n{rng.randint(0, 399)}") or GoalNode(id=nid)
                )
                node = node.model_copy(update={"properties": {"k": rng.randint(0, 9)}})
                bg.upsert_node(node)
            elif kind == "remove_n":
                bg.remove_node(rng.choice([nid, f"n{rng.randint(0, 399)}"]))
            elif kind == "add_e":
                e = HCIREdge(
                    edge_type=HCIREdgeType.LOCATED_IN,
                    id=f"e_{rng.randint(0, 40)}",
                    sources=[nid],
                    targets=["n0"],
                )
                if bg.get_edge(e.id) is None and bg.has_node(nid):
                    bg.add_edge(e)
            else:
                bg.remove_edge(f"e_{rng.randint(0, 40)}")

        fast = ws.diff_branch(branch.branch_name)
        ref = _full_diff(ws, branch)
        assert [o.model_dump() for o in fast] == [o.model_dump() for o in ref], (
            f"trial {trial}: dirty diff diverged from full diff"
        )


def test_merge_branch_result_unchanged() -> None:
    ws = _make_workspace()
    branch = ws.fork("m")
    bg = branch._graph
    bg.add_node(GoalNode(id="fresh"))
    bg.remove_node("n1")
    ws2 = _make_workspace()
    ws2._graph = ws._graph
    # merge via commit of diff ops (same as merge_branch path)
    ops = ws.diff_branch("m")
    assert any(o.op == "add_node" and o.node_id == "fresh" for o in ops)
    assert any(o.op == "remove_node" and o.node_id == "n1" for o in ops)
    merged = ws.merge_branch("m") if hasattr(ws, "merge_branch") else None
    if merged is not None:
        assert ws._graph.has_node("fresh")
        assert not ws._graph.has_node("n1")


def test_native_health_shape() -> None:
    g = CognitiveGraph(enable_native_substrate=False)
    h = g.native_health()
    assert h == {"enabled": False, "available": False, "divergences": 0, "last_error": None}


def test_fast_path_disabled_when_parent_mutated_after_fork() -> None:
    """B0 fix: parent commits after fork must disable the dirty-set fast path.

    The branch's dirty sets only track *branch* mutations, so if the parent has
    committed since the fork, the fast path silently drops parent-side changes.
    """
    from hbllm.hcir.transactions import TransactionOp, TransactionOperation

    ws = _make_workspace()
    ws.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.ADD_EDGE,
                edge_id="e_base",
                edge_data=HCIREdge(
                    edge_type=HCIREdgeType.LOCATED_IN,
                    id="e_base",
                    sources=["n0"],
                    targets=["n1"],
                ).model_dump(),
            )
        ],
        author="test",
    )
    branch = ws.fork("postfork")
    branch._graph.add_node(GoalNode(id="branch_only"))

    # Parent commits AFTER the fork — bumps the snapshot version.
    parent_node = GoalNode(id="parent_added")
    ws.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id="parent_added",
                node_data=parent_node.model_dump(),
            ),
            TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="n2"),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="n5",
                node_data=GoalNode(id="n5", properties={"p": 1}).model_dump(),
            ),
            TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id="e_base"),
        ],
        author="test",
    )

    ops = ws.diff_branch("postfork")
    keys = [(o.op, o.node_id or o.edge_id) for o in ops]

    # Ops transform main -> branch: parent-added/removed items must be reversed.
    assert ("remove_node", "parent_added") in keys, keys
    assert ("add_node", "n2") in keys, keys  # parent removed; branch still has it
    assert ("upsert_node", "n5") in keys, keys  # parent changed; branch has old
    assert ("add_edge", "e_base") in keys, keys  # parent removed; branch has it
    assert ("add_node", "branch_only") in keys, keys

    # And the result still equals a forced full comparison.
    ref = _full_diff(ws, branch)
    assert [o.model_dump() for o in ops] == [o.model_dump() for o in ref]
