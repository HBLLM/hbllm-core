"""Audio Perception Transaction — HCIR commitment layer.

Commits audio assessment into HCIR graph atomically.
This is the ONLY layer that creates/modifies HCIR nodes.

Architecture:
    AudioPerceptionRuntime → AudioAssessment (evidence only)
    AudioPerceptionTransaction → HCIR nodes + edges + beliefs

Invariant:
    The runtime produces evidence.
    This layer commits observations.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from hbllm.hcir.graph import (
    AcousticConceptNode,
    AudioObservationNode,
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    _new_id,
)
from hbllm.perception.audio_memory import AudioMemory
from hbllm.perception.providers.audio_evidence import (
    AudioAssessment,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AudioPerceptionTransaction:
    """Commits audio perception results into HCIR — atomically.

    Usage::

        tx = AudioPerceptionTransaction(graph=graph, memory=memory)
        node = tx.commit_speech(assessment)
        concept = tx.commit_learning(assessment)

    """

    def __init__(
        self,
        graph: CognitiveGraph,
        memory: AudioMemory,
        provider_id: str = "unknown",
    ) -> None:
        self._graph = graph
        self._memory = memory
        self._provider_id = provider_id

    def commit_speech(
        self,
        assessment: AudioAssessment,
    ) -> AudioObservationNode | None:
        """Commit speech evidence as an HCIR observation.

        Args:
            assessment: Full audio assessment with speech evidence.

        Returns:
            AudioObservationNode if speech was present, None otherwise.

        """
        if assessment.speech is None:
            return None

        speech = assessment.speech
        obs = assessment.observation

        node = AudioObservationNode(
            id=_new_id("aobs"),
            label=f"speech: {speech.transcript[:50]}",
            embedding_ref=obs.embedding_ref or "",
            embedding_space=obs.embedding_space,
            embedding_model=self._provider_id,
            event_type="speech",
            event_id=obs.temporal.event_id,
            start_time=obs.temporal.start_time,
            end_time=obs.temporal.end_time,
            duration=obs.temporal.duration,
            transcript=speech.transcript,
            speaker_ref=(
                speech.speaker_ref.embedding_ref
                if speech.speaker_ref else ""
            ),
        )
        self._graph.add_node(node)
        return node

    def commit_event(
        self,
        assessment: AudioAssessment,
        event_index: int = 0,
    ) -> AudioObservationNode | None:
        """Commit a sound event as an HCIR observation.

        Args:
            assessment: Full audio assessment.
            event_index: Index of the event to commit.

        Returns:
            AudioObservationNode if event exists, None otherwise.

        """
        if event_index >= len(assessment.events):
            return None

        event = assessment.events[event_index]
        obs = assessment.observation

        node = AudioObservationNode(
            id=_new_id("aobs"),
            label=f"event: {event.event_type}",
            embedding_ref=obs.embedding_ref or "",
            embedding_space=obs.embedding_space,
            embedding_model=self._provider_id,
            event_type=event.event_type,
            event_id=obs.temporal.event_id,
            start_time=obs.temporal.start_time,
            end_time=obs.temporal.end_time,
            duration=obs.temporal.duration,
        )
        self._graph.add_node(node)
        return node

    def commit_learning(
        self,
        assessment: AudioAssessment,
    ) -> AcousticConceptNode | None:
        """Commit a learning event — creates a cognitive artifact.

        This does NOT train a model. It creates an AcousticConceptNode
        in HCIR that represents a learned sound pattern.

        Args:
            assessment: Assessment with proposed_label set.

        Returns:
            AcousticConceptNode if label is provided, None otherwise.

        """
        label = assessment.proposed_label
        if not label:
            return None

        obs = assessment.observation
        now = time.time()

        # Check if concept already exists
        existing = self._find_concept(label)
        if existing is not None:
            # Update existing concept
            existing.observation_count += 1
            existing.last_heard = now
            return existing

        # Create new concept
        concept = AcousticConceptNode(
            id=_new_id("acpt"),
            label=label,
            prototype_ref=obs.embedding_ref or "",
            embedding_space=obs.embedding_space,
            embedding_model=self._provider_id,
            observation_count=1,
            last_heard=now,
        )
        self._graph.add_node(concept)

        # Create observation node and link to concept
        obs_node = AudioObservationNode(
            id=_new_id("aobs"),
            label=f"exemplar: {label}",
            embedding_ref=obs.embedding_ref or "",
            embedding_space=obs.embedding_space,
            embedding_model=self._provider_id,
            event_type="learning",
            event_id=obs.temporal.event_id,
            start_time=obs.temporal.start_time,
        )
        self._graph.add_node(obs_node)

        # SUPPORTS edge: observation → concept
        edge = HCIREdge(
            sources=[obs_node.id],
            targets=[concept.id],
            edge_type=HCIREdgeType.SUPPORTS,
            weight=1.0,
        )
        self._graph.add_edge(edge)

        return concept

    def _find_concept(self, label: str) -> AcousticConceptNode | None:
        """Find an existing acoustic concept by label."""
        for node in self._graph.nodes_by_type(HCIRNodeType.ACOUSTIC_CONCEPT):
            if isinstance(node, AcousticConceptNode) and node.label == label:
                return node
        return None
