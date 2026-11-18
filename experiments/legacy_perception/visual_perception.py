"""Visual Perception — convenience facade over runtime + transaction.

Provides the high-level API: learn() and recognize().
Internally delegates to VisualPerceptionRuntime (evidence) and
VisualPerceptionTransaction (HCIR commitment).

Usage::

    perception = VisualPerception(runtime, transaction)

    # One-shot learning
    concept = await perception.learn(image, "screwdriver")

    # Recognition
    result = await perception.recognize(image)
    if result.matched:
        print(f"Recognized: {result.label}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hbllm.hcir.graph import VisualConceptNode
    from hbllm.perception.providers.base import ImageInput
    from hbllm.perception.visual_perception_runtime import VisualPerceptionRuntime
    from hbllm.perception.visual_perception_transaction import (
        VisualPerceptionTransaction,
        VisualRecognitionResult,
    )


class VisualPerception:
    """Convenience API: runtime.perceive() + transaction.commit_*().

    This is the public-facing interface for visual cognition.
    It composes the runtime (evidence production) with the
    transaction (HCIR commitment) into a clean two-method API.
    """

    def __init__(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
    ) -> None:
        self.runtime = runtime
        self.transaction = transaction

    async def learn(
        self,
        image: ImageInput,
        label: str,
        context: str = "",
    ) -> VisualConceptNode:
        """One-shot visual learning: perceive + commit as concept.

        1. Runtime: encode image → search memory → build assessment
        2. Transaction: create/update concept → link evidence → record belief
        """
        assessment = await self.runtime.perceive_with_label(image, label, context)
        return await self.transaction.commit_learning(assessment)

    async def recognize(
        self,
        image: ImageInput,
    ) -> VisualRecognitionResult:
        """Visual recognition: perceive + commit recognition result.

        1. Runtime: encode image → search memory → build assessment
        2. Transaction: match/ambiguous/novel → store evidence
        """
        assessment = await self.runtime.perceive(image)
        return await self.transaction.commit_recognition(assessment)
