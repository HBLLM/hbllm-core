from __future__ import annotations

"""Verification tests for native Rust compute and graph ABI bindings.

Ensures that compiled native libraries (hbllm_hcir_graph, hbllm_compute)
are present, match expected symbol signatures, and execute without ABI divergence.
"""


def test_native_graph_module_abi():
    """Verify hbllm_hcir_graph module symbols and basic functionality."""
    import hbllm_hcir_graph

    # Required symbols
    for sym in (
        "NativeGraph",
        "blake3_hash",
        "topological_connected_components",
        "fast_geodesic_astar",
        "fast_collision_cone_sweep",
    ):
        assert hasattr(hbllm_hcir_graph, sym), f"Missing required Rust symbol: {sym}"

    # Verify NativeGraph instantiation and edge addition
    g = hbllm_hcir_graph.NativeGraph()
    g.add_node("n1", "concept", "data1")
    g.add_node("n2", "concept", "data2")
    g.add_edge("e1", "causes", ["n1"], ["n2"], 1.0, {"weight": "1.0"})
    assert g.has_node("n1")
    assert g.has_edge("e1")


def test_native_compute_kernel_abi():
    """Verify hbllm_compute module symbols and basic functionality."""
    import hbllm_compute

    # Required symbols
    for sym in (
        "UniversalEngine",
        "fast_astar_search",
        "kuramoto_phase_step_simd",
        "snn_step_lif_simd",
        "grid_cell_path_integrate_simd",
        "forecast_hazard_density_simd",
        "causal_attribution_simd",
        "bitboard_simulate_rollout",
    ):
        assert hasattr(hbllm_compute, sym), f"Missing required Rust symbol: {sym}"

    # Verify fast_astar_search execution (height, width, start, goal, barriers)
    path = hbllm_compute.fast_astar_search(5, 5, (0, 0), (4, 4), [(2, 2)])
    assert path is not None
    assert len(path) > 0
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
