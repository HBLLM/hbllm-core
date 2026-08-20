"""Audio Perception Facade — unified API for audio perception.

Combines AudioPerceptionRuntime (evidence) and
AudioPerceptionTransaction (HCIR commitment) behind a simple API.

Usage::

    perception = AudioPerception(runtime, transaction)
    assessment = await perception.listen(audio)
    concept = await perception.learn_sound(audio, "my_doorbell")
"""

from __future__ import annotations

from hbllm.hcir.graph import AcousticConceptNode, AudioObservationNode
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.audio_perception_transaction import AudioPerceptionTransaction
from hbllm.perception.providers.audio_evidence import AudioAssessment
from hbllm.perception.providers.audio_types import AudioInput


class AudioPerception:
    """Unified audio perception API.

    Composes runtime (evidence-only) and transaction (HCIR commitment).
    """

    def __init__(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
    ) -> None:
        self._runtime = runtime
        self._transaction = transaction

    async def listen(self, audio: AudioInput) -> AudioAssessment:
        """Listen to audio and produce an assessment.

        Evidence-only — does NOT commit to HCIR.

        Args:
            audio: Raw audio input.

        Returns:
            Full AudioAssessment.

        """
        return await self._runtime.perceive(audio)

    async def learn_sound(
        self,
        audio: AudioInput,
        label: str,
    ) -> AcousticConceptNode | None:
        """Learn a sound — creates a cognitive artifact in HCIR.

        Does NOT train a model. Creates an AcousticConceptNode.

        Args:
            audio: Audio sample to learn from.
            label: Label for the concept.

        Returns:
            AcousticConceptNode if successful.

        """
        assessment = await self._runtime.perceive(audio, label=label)
        return self._transaction.commit_learning(assessment)

    async def recognize_speech(
        self,
        audio: AudioInput,
    ) -> AudioObservationNode | None:
        """Recognize and commit speech to HCIR.

        Args:
            audio: Audio input with speech.

        Returns:
            AudioObservationNode if speech detected.

        """
        assessment = await self._runtime.perceive(audio)
        return self._transaction.commit_speech(assessment)

    async def recognize_event(
        self,
        audio: AudioInput,
    ) -> AudioObservationNode | None:
        """Recognize and commit a sound event to HCIR.

        Args:
            audio: Audio input.

        Returns:
            AudioObservationNode if event detected.

        """
        assessment = await self._runtime.perceive(audio)
        if assessment.events:
            return self._transaction.commit_event(assessment, event_index=0)
        return None
