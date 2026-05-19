"""Tests for Cognitive Compaction Engine."""

from hbllm.brain.compaction.engine import (
    CognitiveCompactionEngine,
    MemoryImportanceScorer,
    MemoryNode,
    SemanticFolder,
)


def test_memory_importance_scoring():
    scorer = MemoryImportanceScorer()

    # Mundane node
    node1 = MemoryNode(node_id="1", content="tick")
    assert scorer.score(node1) == 0.5  # Only recurrence=1 * 0.5

    # Critical failure node
    node2 = MemoryNode(node_id="2", content="crash", failure_association=1.0)
    assert scorer.score(node2) == 3.0  # 0.5 + 2.5

    # Highly emotional routine
    node3 = MemoryNode(node_id="3", content="morning", emotional_weight=0.8, recurrence=20)
    # 1.6 + 5.0 (max recur) = 6.6
    assert scorer.score(node3) > 6.0


def test_attention_decay_and_compaction():
    engine = CognitiveCompactionEngine(decay_rate=0.5)

    # Add a boring node
    engine.nodes["boring"] = MemoryNode(node_id="boring", content="yawn", activation_strength=0.3)

    # Add an important node
    engine.nodes["important"] = MemoryNode(
        node_id="important", content="fire", failure_association=1.0, activation_strength=0.3
    )

    engine.compact(current_memory_pressure=0.8)

    # Decay applied (0.3 * 0.5 = 0.15)

    # Boring should be archived (score < 3.0 and act < 0.2)
    assert "boring" not in engine.nodes
    assert "boring" in engine.cold_storage

    # Important should survive (score >= 3.0)
    assert "important" in engine.nodes
    assert "important" not in engine.cold_storage


def test_heuristic_folding():
    folder = SemanticFolder()

    events = [
        MemoryNode("1", {"action": "code"}),
        MemoryNode("2", {"action": "code"}),
        MemoryNode("3", {"action": "code"}),
        MemoryNode("4", {"action": "sleep"}),
    ]

    folder.heuristic_clustering(events)
    assert len(folder.heuristic_clusters["code"]) == 3
    assert len(folder.heuristic_clusters["sleep"]) == 1

    labels = folder.label_clusters()
    # Code gets labeled because it has >= 3 occurrences
    assert "code" in labels
    assert "sleep" not in labels
