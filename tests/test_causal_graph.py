"""Tests for the Phase 4.2 Causal Cognition Graph."""

from __future__ import annotations

import time

import pytest

from hbllm.brain.causality.causal_graph import CausalGraph


@pytest.fixture
def temp_causal_graph(tmp_path):
    """Provides a temporary SQLite CausalGraph."""
    return CausalGraph(data_dir=tmp_path)


def test_calculate_causal_probability(temp_causal_graph):
    """Test the causal scoring function with various inputs."""
    # Strong causal link (recent, high trust, exact match, system intervention)
    strong_prob = temp_causal_graph.calculate_causal_probability(
        temporal_distance_s=2.0,  # Very recent
        source_trust=1.0,  # High trust
        event_match_score=0.9,  # Expected outcome
        state_alignment_score=1.0,  # Matches world state perfectly
        intervention_signal_strength=1.0,  # The system literally just pushed a button
    )
    assert strong_prob > 0.8

    # Weak causal link (long time ago, passive observation)
    weak_prob = temp_causal_graph.calculate_causal_probability(
        temporal_distance_s=150.0,  # Too old
        source_trust=0.5,  # Noisy sensor
        event_match_score=0.1,  # Unrelated
        state_alignment_score=0.2,  # Out of sync
        intervention_signal_strength=0.0,  # System did nothing
    )
    assert weak_prob < temp_causal_graph.hallucination_threshold


def test_hallucination_thresholding(temp_causal_graph):
    """Test that weak links are discarded and strong links are saved."""
    # This should be discarded
    link1 = temp_causal_graph.infer_and_store(
        source_id="task_idle",
        target_id="event_motion",
        temporal_distance_s=200.0,
        source_trust=0.3,
        event_match_score=0.1,
        state_alignment_score=0.0,
        intervention_signal_strength=0.0,
    )
    assert link1 is None

    # This should be saved
    link2 = temp_causal_graph.infer_and_store(
        source_id="task_turn_on_light",
        target_id="event_light_on",
        temporal_distance_s=1.5,
        source_trust=1.0,
        event_match_score=1.0,
        state_alignment_score=1.0,
        intervention_signal_strength=1.0,
    )
    assert link2 is not None
    assert link2.probability > 0.8


def test_causal_graph_queries(temp_causal_graph):
    """Test retrieving causes and effects from the SQLite store."""
    temp_causal_graph.infer_and_store(
        source_id="task_A",
        target_id="event_B",
        temporal_distance_s=5.0,
        source_trust=0.9,
        event_match_score=0.8,
        state_alignment_score=0.8,
        intervention_signal_strength=0.9,
        metadata={"reason": "test_chain_1"},
    )

    temp_causal_graph.infer_and_store(
        source_id="event_B",
        target_id="task_C",
        temporal_distance_s=2.0,
        source_trust=0.9,
        event_match_score=0.9,
        state_alignment_score=0.9,
        intervention_signal_strength=0.1,  # passive trigger
        metadata={"reason": "test_chain_2"},
    )

    # Check what caused event_B
    causes = temp_causal_graph.get_causes("event_B")
    assert len(causes) == 1
    assert causes[0].source_id == "task_A"

    # Check what effects event_B caused
    effects = temp_causal_graph.get_effects("event_B")
    assert len(effects) == 1
    assert effects[0].target_id == "task_C"
