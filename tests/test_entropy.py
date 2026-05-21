"""Tests for Cognitive Entropy Engine."""

from hbllm.brain.compaction.entropy import CognitiveEntropyEngine, EntropyMetrics


def test_entropy_score():
    engine = CognitiveEntropyEngine()

    # Healthy system
    metrics = EntropyMetrics(
        graph_density=0.1,
        stale_node_ratio=0.1,
        unused_memory_ratio=0.05,
        causal_drift=0.01,
        simulation_branch_explosion=3.0,
    )
    engine.update_metrics(metrics)
    score = engine.get_system_entropy_score()
    assert score < 0.3  # Low entropy

    # Unhealthy system
    metrics = EntropyMetrics(
        graph_density=0.9,
        stale_node_ratio=0.8,
        unused_memory_ratio=0.7,
        causal_drift=0.5,
        simulation_branch_explosion=25.0,
    )
    engine.update_metrics(metrics)
    score = engine.get_system_entropy_score()
    assert score > 0.6  # High entropy
