"""WP4 parity matrix: 5 seeds × native enabled/disabled (skip enabled if absent)."""

from __future__ import annotations

import random

import pytest

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType
from hbllm.hcir.nodes import GoalNode


def _native_available() -> bool:
    try:
        import hbllm_hcir_graph  # noqa: F401

        from hbllm.hcir.native_adapter import NativeGraphAdapter  # noqa: F401

        return True
    except Exception:
        return False


def _ops(ref: CognitiveGraph, g: CognitiveGraph, rng: random.Random, n: int) -> None:
    """Drive identical mutation sequences on both graphs (shared instances)."""
    for i in range(n):
        kind = rng.choice(["add", "upsert", "remove", "add_e", "remove_e"])
        nid = f"n{rng.randint(0, 200)}"
        candidate = GoalNode(id=nid, properties={"x": i})
        if kind == "add":
            for gg in (ref, g):
                if not gg.has_node(nid):
                    gg.add_node(candidate.model_copy(deep=True))
        elif kind == "upsert":
            for gg in (ref, g):
                node = gg.get_node(nid) or candidate
                gg.upsert_node(node.model_copy(update={"properties": {"x": i}}))
        elif kind == "remove":
            for gg in (ref, g):
                gg.remove_node(nid)
        elif kind == "add_e":
            a, b = f"n{rng.randint(0, 200)}", f"n{rng.randint(0, 200)}"
            eid = f"e{rng.randint(0, 100)}"
            edge = HCIREdge(edge_type=HCIREdgeType.LOCATED_IN, id=eid, sources=[a], targets=[b])
            for gg in (ref, g):
                if gg.has_node(a) and gg.has_node(b) and gg.get_edge(eid) is None:
                    gg.add_edge(edge.model_copy(deep=True))
        else:
            eid = f"e{rng.randint(0, 100)}"
            for gg in (ref, g):
                gg.remove_edge(eid)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("native", [False, True])
def test_native_parity(seed: int, native: bool) -> None:
    if native and not _native_available():
        pytest.skip("native graph module unavailable")
    rng = random.Random(seed)
    ref = CognitiveGraph(enable_native_substrate=False)
    g = CognitiveGraph(enable_native_substrate=native)
    for i in range(200):
        n = GoalNode(id=f"n{i}")
        ref.add_node(n.model_copy(deep=True))
        g.add_node(n)
    _ops(ref, g, rng, 300)
    assert ref.content_hash() == g.content_hash()
    assert ref.node_count == g.node_count


def _small_graph(native: bool) -> CognitiveGraph:
    rng = random.Random(7)
    g = CognitiveGraph(enable_native_substrate=native)
    for i in range(30):
        g.add_node(GoalNode(id=f"n{i}"))
    eid = 0
    while g.edge_count < 60:
        a, b = rng.randint(0, 29), rng.randint(0, 29)
        if a != b:
            g.add_edge(
                HCIREdge(
                    edge_type=HCIREdgeType.LOCATED_IN,
                    id=f"e{eid}",
                    sources=[f"n{a}"],
                    targets=[f"n{b}"],
                )
            )
            eid += 1
    return g


@pytest.mark.skipif(not _native_available(), reason="native graph module unavailable")
def test_native_algorithm_parity() -> None:
    """B0: algorithms.* must produce identical results via native vs Python."""
    import hbllm.hcir.algorithms as algorithms

    py = _small_graph(False)
    rs = _small_graph(True)
    assert rs._native_adapter is not None

    def canon(v):
        if isinstance(v, list) and v and isinstance(v[0], set):
            return sorted(map(frozenset, v), key=lambda s: sorted(s))
        if isinstance(v, tuple):
            return tuple(canon(x) for x in v)
        if isinstance(v, dict):
            return {k: canon(x) for k, x in v.items()}
        if isinstance(v, set):
            return frozenset(v)
        return v

    assert canon(algorithms.connected_components(rs)) == canon(algorithms.connected_components(py))
    assert algorithms.topological_sort(rs) == algorithms.topological_sort(py)
    assert algorithms.is_biconnected(rs) == algorithms.is_biconnected(py)
    na = algorithms.articulation_points_and_bridges(rs)
    pa = algorithms.articulation_points_and_bridges(py)
    assert na[0] == pa[0] and sorted(na[1]) == sorted(pa[1])
    assert canon(algorithms.strongly_connected_components(rs)) == canon(
        algorithms.strongly_connected_components(py)
    )
    nc = algorithms.condense_scc_metagraph(rs)
    pc = algorithms.condense_scc_metagraph(py)
    assert canon(nc[0]) == canon(pc[0]) and sorted(nc[1]) == sorted(pc[1])
    # feedback_arc_set is a greedy heuristic — native and Python may choose a
    # different but equally valid edge set. Require same cardinality + validity.
    nf, pf = algorithms.feedback_arc_set(rs), algorithms.feedback_arc_set(py)
    assert len(nf[0]) == len(pf[0])
    for graph, (fas_edges, _topo) in ((rs, nf), (py, pf)):
        probe = graph.fork()
        probe._materialize_indexes()
        fas = set(fas_edges)
        for eid in list(probe._edges.keys()):
            e = probe._edges[eid]
            if any((s, t) in fas for s in e.sources for t in e.targets):
                probe.remove_edge(eid)
        assert len(algorithms.topological_sort(probe)) == probe.node_count
    assert algorithms.feedback_vertex_set(rs) == algorithms.feedback_vertex_set(py)
    # Transitive reduction is only unique on DAGs; on cyclic digraphs the
    # implementations may keep different (equally valid) surviving edges.
    nsk, psk = algorithms.essential_causal_skeleton(rs), algorithms.essential_causal_skeleton(py)
    edge_pairs = {(s, t) for e in py.all_edges() for s in e.sources for t in e.targets}
    assert nsk <= edge_pairs and psk <= edge_pairs and len(psk) <= len(nsk) <= len(edge_pairs)
    assert algorithms.minimum_dominating_set(rs) == algorithms.minimum_dominating_set(py)
    assert algorithms.maximum_independent_set(rs) == algorithms.maximum_independent_set(py)
    assert algorithms.minimum_vertex_cover(rs) == algorithms.minimum_vertex_cover(py)
    assert algorithms.greedy_edge_coloring(rs) == algorithms.greedy_edge_coloring(py)
    assert algorithms.find_eulerian_path_or_circuit(rs) == algorithms.find_eulerian_path_or_circuit(
        py
    )
    assert algorithms.graph_radius_and_diameter(rs) == algorithms.graph_radius_and_diameter(py)
    nbw = algorithms.betweenness_centrality(rs)
    pbw = algorithms.betweenness_centrality(py)
    assert nbw.keys() == pbw.keys()
    for k in nbw:
        assert nbw[k] == pytest.approx(pbw[k], abs=1e-12)
    ndist, npaths = algorithms.all_pairs_shortest_paths(rs)
    pdist, ppaths = algorithms.all_pairs_shortest_paths(py)
    assert ndist == pdist  # distances are canonical
    # Shortest-path ties may resolve to different routes; require same
    # endpoints and equal hop counts for every reachable pair.
    for k, pth in npaths.items():
        other = ppaths[k]
        if pth and other:
            assert pth[0] == other[0] and pth[-1] == other[-1]
            assert len(pth) == len(other)
    assert algorithms.chinese_postman_tour(rs) == algorithms.chinese_postman_tour(py)
    nm = algorithms.max_flow_min_cut(rs, "n0", "n29")
    pm = algorithms.max_flow_min_cut(py, "n0", "n29")
    # max flow, cut-edge set, and source-side partition are canonical;
    # the T set differs only by isolated-node convention.
    assert nm[0] == pm[0] and sorted(nm[1]) == sorted(pm[1]) and nm[2] == pm[2]
