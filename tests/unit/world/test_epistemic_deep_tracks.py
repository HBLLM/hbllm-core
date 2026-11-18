"""Unit tests for the 4 Advanced Epistemic & Cortical tracks.

Track 1: Metacognitive Simulation Budget Governor & Entorhinal Path Integration Loop
Track 2: Domain-General Temporal Dependency & Delayed-Effect Pearl Calculus
Track 3: Relational Graph Isomorphism & Zero-Shot Structural Analogy Transfer
Track 4: Betti Homology & Topological Cul-de-Sac Cavity Pruning in Manifold Routing
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.grounding.concepts.relational_graph_matcher import (
    RelationalGraphMatcher as VentralRelationalGraphMatcher,
)


class TestRelationalGraphAndAnalogyMatching:
    """Track 3: Relational scene graph Hungarian matching and object mutation detection."""

    def test_scene_graph_construction_and_matching(self) -> None:
        grid_a = np.zeros((8, 8), dtype=np.int32)
        grid_a[1:3, 1:3] = 2  # object A (red block)
        grid_a[5:7, 5:7] = 3  # object B (green block)

        grid_b = np.zeros((8, 8), dtype=np.int32)
        grid_b[1:3, 1:3] = 4  # recolored object A (blue block)
        grid_b[5:7, 5:7] = 5  # recolored object B (yellow block)

        graph_a = VentralRelationalGraphMatcher.build_scene_graph(grid_a, bg_color=0)
        graph_b = VentralRelationalGraphMatcher.build_scene_graph(grid_b, bg_color=0)

        assert len(graph_a.nodes) == 2
        assert len(graph_b.nodes) == 2

        correspondences = VentralRelationalGraphMatcher.match_graphs(
            graph_a, graph_b, ignore_color=True, ignore_scale=True
        )
        assert len(correspondences) == 2
        for corr in correspondences:
            assert corr.match_score >= 0.70

    def test_detect_object_mutations(self) -> None:
        grid_prev = np.zeros((8, 8), dtype=np.int32)
        grid_prev[1:3, 1:3] = 2  # object 1 at (1,1)

        grid_curr = np.zeros((8, 8), dtype=np.int32)
        grid_curr[3:5, 1:3] = 2  # object 1 moved downwards by 2 rows

        graph_prev = VentralRelationalGraphMatcher.build_scene_graph(grid_prev, bg_color=0)
        graph_curr = VentralRelationalGraphMatcher.build_scene_graph(grid_curr, bg_color=0)

        mutations = VentralRelationalGraphMatcher.detect_object_mutations(graph_prev, graph_curr)
        assert len(mutations["moved"]) == 1
        assert abs(mutations["moved"][0]["delta"][0] - 2.0) < 0.1
