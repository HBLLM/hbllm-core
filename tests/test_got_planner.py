"""Tests for the Graph-of-Thoughts (GoT) planner data structures."""

import pytest

from hbllm.brain.planner_node import ThoughtGraph, ThoughtNode


def test_thought_graph_add_root():
    """Verify adding root nodes to the graph."""
    g = ThoughtGraph()
    root = g.add_root("Initial idea", score=0.8)

    assert root.id in g.nodes
    assert root.content == "Initial idea"
    assert root.score == 0.8
    assert root.depth == 0
    assert root.is_leaf
    assert len(g.root_ids) == 1


def test_thought_graph_branching():
    """Verify branching creates child nodes with correct parent links."""
    g = ThoughtGraph()
    root = g.add_root("Root thought")

    child1 = g.branch(root.id, "Branch A", score=0.7)
    child2 = g.branch(root.id, "Branch B", score=0.9)

    assert len(root.children_ids) == 2
    assert child1.parent_ids == [root.id]
    assert child2.parent_ids == [root.id]
    assert child1.depth == 1
    assert not root.is_leaf
    assert child1.is_leaf


def test_thought_graph_merge():
    """Verify merging combines multiple parents into a synthesis node."""
    g = ThoughtGraph()
    r1 = g.add_root("Path A", score=0.6)
    r2 = g.add_root("Path B", score=0.8)

    merged = g.merge([r1.id, r2.id], "Synthesis of A + B", score=0.95)

    assert merged.is_merged
    assert merged.depth == 1
    assert set(merged.parent_ids) == {r1.id, r2.id}
    assert merged.id in r1.children_ids
    assert merged.id in r2.children_ids


def test_thought_graph_best_path():
    """Verify best_path traces from root to highest-scoring leaf."""
    g = ThoughtGraph()
    root = g.add_root("Root", score=0.5)

    good = g.branch(root.id, "Good branch", score=0.9)
    g.branch(root.id, "Bad branch", score=0.2)

    best = g.best_path()
    assert len(best) == 2
    assert best[0].id == root.id
    assert best[1].id == good.id


def test_thought_graph_prune():
    """Verify pruning removes low-scoring leaf nodes."""
    g = ThoughtGraph()
    root = g.add_root("Root", score=0.5)

    good = g.branch(root.id, "Good", score=0.8)
    weak = g.branch(root.id, "Weak", score=0.1)

    pruned = g.prune(min_score=0.3)

    assert pruned == 1
    assert weak.id not in g.nodes
    assert good.id in g.nodes


def test_thought_graph_summary():
    """Verify graph summary returns correct statistics."""
    g = ThoughtGraph()
    r1 = g.add_root("A", score=0.6)
    r2 = g.add_root("B", score=0.8)
    c1 = g.branch(r1.id, "A1", score=0.7)
    g.merge([c1.id, r2.id], "Merged", score=0.9)

    s = g.summary()
    assert s["total_nodes"] == 4
    assert s["root_count"] == 2
    assert s["leaf_count"] == 1  # Only merged is a leaf
    assert s["max_depth"] == 2
    assert s["merged_count"] == 1
    assert s["avg_score"] > 0


def test_thought_graph_multi_depth():
    """Verify multi-depth graph reasoning works correctly."""
    g = ThoughtGraph()
    root = g.add_root("Start", score=0.4)

    # Depth 1
    d1a = g.branch(root.id, "D1-A", score=0.6)
    d1b = g.branch(root.id, "D1-B", score=0.3)

    # Depth 2 (only from better branch)
    d2a = g.branch(d1a.id, "D2-A refinement", score=0.85)

    # Merge d2a with d1b
    g.merge([d2a.id, d1b.id], "Final synthesis", score=0.95)

    path = g.best_path()
    assert len(path) >= 2
    assert path[-1].score == 0.95  # Merged node should win
    assert path[-1].is_merged


def test_thought_node_defaults():
    """Verify ThoughtNode has sane defaults."""
    node = ThoughtNode()
    assert node.content == ""
    assert node.score == 0.0
    assert node.depth == 0
    assert node.parent_ids == []
    assert node.children_ids == []
    assert not node.is_merged
    assert node.is_leaf


def test_prune_doesnt_remove_roots():
    """Verify pruning never removes root nodes even if low-scoring."""
    g = ThoughtGraph()
    root = g.add_root("Low root", score=0.1)

    pruned = g.prune(min_score=0.5)

    assert pruned == 0
    assert root.id in g.nodes


def test_thought_graph_mcts_stats():
    """Verify backpropagate correctly updates visits and cumulative_reward."""
    g = ThoughtGraph()
    root = g.add_root("Root Phase 2")

    child1 = g.branch(root.id, "Branch 1")
    g.backpropagate(child1.id, 1.0)

    assert child1.visits == 1
    assert child1.cumulative_reward == 1.0
    assert child1.q_value == 1.0

    assert root.visits == 1
    assert root.cumulative_reward == pytest.approx(0.95)  # 1.0 * decay(0.95)
    assert root.q_value == pytest.approx(0.95)

    child2 = g.branch(root.id, "Branch 2")
    g.backpropagate(child2.id, 0.0)

    assert child2.visits == 1
    assert child2.q_value == 0.0

    assert root.visits == 2
    assert root.cumulative_reward == pytest.approx(0.95)  # 0.95 + 0.0*0.95
    assert root.q_value == pytest.approx(0.475)  # 0.95 / 2


def test_thought_graph_uct_selection():
    """Verify UCT selects unvisited nodes first, then balances Q-value and exploration."""
    g = ThoughtGraph()
    root = g.add_root("Root")

    c1 = g.branch(root.id, "C1")
    c2 = g.branch(root.id, "C2")
    c3 = g.branch(root.id, "C3")

    # Selection should prioritize unvisited nodes
    first_sel = g.select_leaf_uct(root.id)
    assert first_sel.visits == 0
    assert first_sel.id in [c1.id, c2.id, c3.id]

    # Manually visit all children with different rewards
    g.backpropagate(c1.id, 1.0)  # q=1.0
    g.backpropagate(c2.id, 0.5)  # q=0.5
    g.backpropagate(c3.id, 0.0)  # q=0.0

    # C1 has highest Q, should be selected next (exploration term is equal for all)
    next_sel = g.select_leaf_uct(root.id)
    assert next_sel.id == c1.id

    # Visit C1 a lot with 0 reward to drop its Q heavily and increase its N
    for _ in range(5):
        g.backpropagate(c1.id, 0.0)

    # Now C2 should be favored due to higher Q (0.5 vs ~0.16) and higher exploration boost
    third_sel = g.select_leaf_uct(root.id)
    assert third_sel.id != c1.id
