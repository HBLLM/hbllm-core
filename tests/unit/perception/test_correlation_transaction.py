"""Tests for Correlation Transaction — §A8.3.

Validates that CorrelationCandidate → CORRELATES_WITH edge in HCIR.
"""

from __future__ import annotations

from hbllm.hcir.graph import CognitiveGraph, HCIREdgeType
from hbllm.perception.correlation_engine import CorrelationCandidate
from hbllm.perception.correlation_transaction import CorrelationTransaction


def _make_graph_with_observations() -> CognitiveGraph:
    """Create a graph with visual and audio observation nodes."""
    graph = CognitiveGraph()

    # Add visual observation node
    from hbllm.hcir.graph import VisualObservationNode

    visual_node = VisualObservationNode(
        label="person detected",
        embedding_ref="emb_vis_001",
        embedding_model="siglip",
    )
    visual_node.id = "vis_obs_001"
    graph.add_node(visual_node)

    # Add audio observation node
    from hbllm.hcir.graph import AudioObservationNode

    audio_node = AudioObservationNode(
        label="footsteps",
        embedding_ref="emb_aud_001",
        embedding_model="yamnet",
    )
    audio_node.id = "aud_obs_001"
    graph.add_node(audio_node)

    return graph


class TestCorrelationCommit:
    """CorrelationCandidate → CORRELATES_WITH edge."""

    def test_commit_creates_edge(self) -> None:
        graph = _make_graph_with_observations()
        tx = CorrelationTransaction(graph=graph)

        candidate = CorrelationCandidate(
            source_observation_id="vis_obs_001",
            target_observation_id="aud_obs_001",
            source_modality="vision",
            target_modality="audio",
            temporal_overlap=0.85,
            spatial_overlap=0.9,
            delta_time_ms=120.0,
            score=0.87,
        )

        edge = tx.commit_correlation(candidate)

        assert edge is not None
        assert edge.edge_type == HCIREdgeType.CORRELATES_WITH
        assert "vis_obs_001" in edge.sources
        assert "aud_obs_001" in edge.targets

    def test_edge_carries_metadata(self) -> None:
        graph = _make_graph_with_observations()
        tx = CorrelationTransaction(graph=graph)

        candidate = CorrelationCandidate(
            source_observation_id="vis_obs_001",
            target_observation_id="aud_obs_001",
            source_modality="vision",
            target_modality="audio",
            temporal_overlap=0.85,
            spatial_overlap=0.9,
            delta_time_ms=120.0,
            score=0.87,
        )

        edge = tx.commit_correlation(candidate)

        assert edge is not None
        assert edge.properties["confidence"] == 0.87
        assert edge.properties["temporal_overlap"] == 0.85
        assert edge.properties["spatial_overlap"] == 0.9
        assert edge.properties["delta_time_ms"] == 120.0
        assert edge.properties["source_modality"] == "vision"
        assert edge.properties["target_modality"] == "audio"
        assert edge.properties["created_by"] == "correlation_engine"

    def test_observations_remain_independent(self) -> None:
        """Observations must NOT be merged or deleted by correlation."""
        graph = _make_graph_with_observations()
        tx = CorrelationTransaction(graph=graph)

        candidate = CorrelationCandidate(
            source_observation_id="vis_obs_001",
            target_observation_id="aud_obs_001",
            source_modality="vision",
            target_modality="audio",
            temporal_overlap=1.0,
            spatial_overlap=None,
            delta_time_ms=0.0,
            score=0.95,
        )

        tx.commit_correlation(candidate)

        # Both nodes still exist independently
        assert graph.get_node("vis_obs_001") is not None
        assert graph.get_node("aud_obs_001") is not None

    def test_missing_node_returns_none(self) -> None:
        """If an observation doesn't exist in the graph, skip."""
        graph = CognitiveGraph()
        tx = CorrelationTransaction(graph=graph)

        candidate = CorrelationCandidate(
            source_observation_id="nonexistent_1",
            target_observation_id="nonexistent_2",
            source_modality="vision",
            target_modality="audio",
            temporal_overlap=1.0,
            spatial_overlap=None,
            delta_time_ms=0.0,
            score=0.9,
        )

        edge = tx.commit_correlation(candidate)
        assert edge is None


class TestContradiction:
    """Visual + audio contradiction: observations preserved independently."""

    def test_empty_room_plus_party_noise_both_preserved(self) -> None:
        """Visual: empty room. Audio: loud party. Both stay in HCIR."""
        graph = CognitiveGraph()

        from hbllm.hcir.graph import AudioObservationNode, VisualObservationNode

        visual = VisualObservationNode(
            label="empty room",
            embedding_ref="emb_empty",
            embedding_model="siglip",
        )
        visual.id = "vis_empty"
        graph.add_node(visual)

        audio = AudioObservationNode(
            label="party noise",
            embedding_ref="emb_party",
            embedding_model="yamnet",
        )
        audio.id = "aud_party"
        graph.add_node(audio)

        tx = CorrelationTransaction(graph=graph)

        # Low correlation score because the observations conflict
        candidate = CorrelationCandidate(
            source_observation_id="vis_empty",
            target_observation_id="aud_party",
            source_modality="vision",
            target_modality="audio",
            temporal_overlap=1.0,
            spatial_overlap=None,
            delta_time_ms=0.0,
            score=0.15,  # Low score — conflicting observations
        )

        edge = tx.commit_correlation(candidate)
        assert edge is not None

        # Both observations still exist — never suppressed
        assert graph.get_node("vis_empty") is not None
        assert graph.get_node("aud_party") is not None

        # The edge preserves the low confidence
        assert edge.properties["confidence"] == 0.15


class TestBatchCommit:
    """Batch correlation commitment."""

    def test_batch_filters_by_min_score(self) -> None:
        graph = _make_graph_with_observations()
        tx = CorrelationTransaction(graph=graph)

        candidates = [
            CorrelationCandidate(
                source_observation_id="vis_obs_001",
                target_observation_id="aud_obs_001",
                source_modality="vision",
                target_modality="audio",
                temporal_overlap=1.0,
                spatial_overlap=None,
                delta_time_ms=0.0,
                score=0.9,
            ),
            CorrelationCandidate(
                source_observation_id="vis_obs_001",
                target_observation_id="aud_obs_001",
                source_modality="vision",
                target_modality="audio",
                temporal_overlap=0.1,
                spatial_overlap=None,
                delta_time_ms=4000.0,
                score=0.1,  # Below threshold
            ),
        ]

        edges = tx.commit_batch(candidates, min_score=0.3)
        assert len(edges) == 1
        assert edges[0].properties["confidence"] == 0.9
