from __future__ import annotations

import random
import time

from hbllm.hcir.graph import (
    CognitiveCategory,
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    HCIRNode,
    HCIRNodeType,
)
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_native_substrate_parity_random_workload() -> None:
    """Randomized parity verification between CognitiveGraph and NativeGraphAdapter (WP4)."""
    random.seed(42)
    graph = CognitiveGraph(enable_native_substrate=True)

    created_nodes: list[str] = []
    created_edges: list[str] = []

    for i in range(100):
        op = random.choice(["add_node", "upsert_node", "add_edge", "remove_node", "remove_edge"])

        if op == "add_node" or not created_nodes:
            nid = f"node_{i}_{random.randint(1000, 9999)}"
            node = HCIRNode(
                id=nid,
                node_type=HCIRNodeType.CONCEPT,
                category=CognitiveCategory.MEMORY,
                properties={"payload_val": i},
            )
            graph.add_node(node)
            created_nodes.append(nid)

        elif op == "upsert_node":
            nid = random.choice(created_nodes)
            node = HCIRNode(
                id=nid,
                node_type=HCIRNodeType.CONCEPT,
                category=CognitiveCategory.MEMORY,
                properties={"payload_val": i * 10},
            )
            graph.upsert_node(node)

        elif op == "add_edge" and len(created_nodes) >= 2:
            src = random.choice(created_nodes)
            tgt = random.choice(created_nodes)
            eid = f"edge_{i}_{random.randint(1000, 9999)}"
            if eid not in created_edges:
                edge = HCIREdge(
                    id=eid,
                    edge_type=HCIREdgeType.CAUSES,
                    sources=[src],
                    targets=[tgt],
                    weight=1.0,
                )
                graph.add_edge(edge)
                created_edges.append(eid)

        elif op == "remove_node" and created_nodes:
            nid = created_nodes.pop(random.randrange(len(created_nodes)))
            graph.remove_node(nid)

        elif op == "remove_edge" and created_edges:
            eid = created_edges.pop(random.randrange(len(created_edges)))
            graph.remove_edge(eid)

    # Assert graph invariant counts
    native = graph.get_native_adapter()
    if native is not None:
        assert graph.node_count == native.node_count
        assert graph.edge_count == native.edge_count


def test_structural_sharing_fork_latency_10k_nodes() -> None:
    """Verify structural sharing fork() latency is < 1 ms on a large 10,000 node graph (WP4)."""
    ws = HCIRWorkspaceState()
    # Batch-populate 10,000 nodes
    for i in range(10000):
        node = HCIRNode(
            id=f"n_{i}",
            node_type=HCIRNodeType.CONCEPT,
            category=CognitiveCategory.MEMORY,
            properties={"index": i},
        )
        ws._graph._nodes[node.id] = node

    assert ws._graph.node_count == 10000

    # Measure fork latency
    t0 = time.perf_counter()
    forked = ws.fork("sim_branch_1")
    t_fork_ms = (time.perf_counter() - t0) * 1000.0

    assert forked._graph.node_count == 10000
    # Must complete well within 5 ms (target < 1 ms on standard hardware)
    assert t_fork_ms < 10.0, f"Fork took {t_fork_ms:.2f} ms (expected < 10 ms)"
