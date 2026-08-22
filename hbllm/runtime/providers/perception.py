"""Unified Perception Provider Protocol — HBLLM Cognitive Runtime.

Perception providers emit modality-specific observations.
They NEVER construct HCIR nodes — the ``EvidenceNormalizer``
handles the conversion to ``PerceptualEvidenceNode``.

Architecture::

    Perception Provider → PerceptualObservation → EvidenceNormalizer → HCIR
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hbllm.runtime.providers.capability import ProviderCapability


@runtime_checkable
class UnifiedPerceptionProvider(Protocol):
    """A perception provider that emits modality-specific observations.

    Providers NEVER construct HCIR nodes.  They produce typed
    observations that the ``EvidenceNormalizer`` converts to
    ``PerceptualEvidenceNode`` instances.

    This protocol extends the existing perception provider contracts
    (``SpeechProvider``, ``VisionProvider``, etc.) with a unified
    capability declaration and lifecycle.

    Note:
        The ``observe()`` method returns ``list[Any]`` rather than
        a specific type because each perception modality has its own
        observation type (``SpeechEvidence``, ``VisualAssessment``,
        sensor readings).  The ``EvidenceNormalizer`` handles type
        dispatch.
    """

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for this provider."""
        ...

    async def observe(self, input_data: Any) -> list[Any]:
        """Produce observations from input data.

        Args:
            input_data: Modality-specific input (audio frames,
                image data, sensor readings, etc.).

        Returns:
            List of modality-specific observation objects.
            These are NOT HCIR nodes — they are provider-native
            types that the ``EvidenceNormalizer`` will convert.
        """
        ...

    async def initialize(self) -> None:
        """Initialize provider resources (load models, etc.)."""
        ...

    async def shutdown(self) -> None:
        """Release provider resources."""
        ...
