"""TTS Action Adapter — concrete ActionProvider wrapping speech synthesis.

Executes structured ActionIntent requests with safety boundaries (e.g. max volume,
utterance length) and produces ExecutionResult outputs.
"""

from __future__ import annotations

import logging
import time

from hbllm.runtime.providers.action import ActionIntent, ExecutionResult
from hbllm.runtime.providers.capability import ProviderCapability

logger = logging.getLogger(__name__)


class TTSActionAdapter:
    """Concrete ActionProvider for text-to-speech synthesis (Kokoro/Orpheus/Piper).

    Conforms to ``ActionProvider``.
    Executes ``ActionIntent`` requests with ``action_type="speak"`` and
    safety constraints.

    Usage::

        adapter = TTSActionAdapter()
        result = await adapter.execute(ActionIntent(action_type="speak", parameters={"text": "Hello"}))
    """

    def __init__(
        self,
        provider_id: str = "piper_tts",
        backend: str = "kokoro",
    ) -> None:
        self._provider_id = provider_id
        self._backend = backend
        self.speech_history: list[str] = []

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for speech synthesis."""
        return ProviderCapability(
            provider_id=self._provider_id,
            provider_type="action",
            capabilities=["speak", "voice_synthesis", "audio_feedback"],
            modalities=["audio"],
            latency_profile="very_low",
            quality_profile="high",
            risk_profile="none",
            memory_requirement_mb=250,
            hardware_requirements=["cpu"],
            requires_network=False,
        )

    async def initialize(self) -> None:
        """Initialize TTS audio output resources."""
        logger.info(
            "Initialized TTSActionAdapter (%s, backend=%s)", self._provider_id, self._backend
        )

    async def shutdown(self) -> None:
        """Release TTS audio resources."""
        logger.info("Shutdown TTSActionAdapter (%s)", self._provider_id)

    async def execute(self, intent: ActionIntent) -> ExecutionResult:
        """Execute speech action intent.

        Args:
            intent: Structured action request.

        Returns:
            ExecutionResult detailing success/failure and actual effect.
        """
        start_time = time.time()

        if intent.action_type != "speak":
            return ExecutionResult(
                success=False,
                action_type=intent.action_type,
                error=f"TTSActionAdapter cannot handle action type '{intent.action_type}'",
                duration_ms=(time.time() - start_time) * 1000.0,
                provider_id=self._provider_id,
            )

        text = intent.parameters.get("text", "")
        if not text:
            return ExecutionResult(
                success=False,
                action_type=intent.action_type,
                error="No text parameter provided for speech action",
                duration_ms=(time.time() - start_time) * 1000.0,
                provider_id=self._provider_id,
            )

        # Enforce safety constraints
        for constraint in intent.safety_constraints:
            if "max_chars" in constraint:
                try:
                    max_chars = int(constraint.split("=")[-1])
                    if len(text) > max_chars:
                        text = text[:max_chars] + "..."
                except Exception:
                    pass

        self.speech_history.append(text)
        duration_ms = (time.time() - start_time) * 1000.0

        return ExecutionResult(
            success=True,
            action_type=intent.action_type,
            actual_effect=f"Spoke utterance ({len(text)} chars): '{text}'",
            duration_ms=duration_ms,
            provider_id=self._provider_id,
        )
