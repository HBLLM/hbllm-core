"""One-Shot Visual Learning Evaluation Harness.

Tests recognition accuracy, concept separation, provenance tracing,
and ambiguity handling across the full perception pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OneShotEvalResult:
    """Results from a one-shot evaluation run."""

    total_concepts: int = 0
    total_recognition_attempts: int = 0
    correct_recognitions: int = 0
    ambiguous_recognitions: int = 0
    novel_recognitions: int = 0
    false_recognitions: int = 0
    concept_separation_scores: dict[str, float] = field(default_factory=dict)
    provenance_traceable: int = 0
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_recognition_attempts == 0:
            return 0.0
        return self.correct_recognitions / self.total_recognition_attempts

    @property
    def ambiguity_rate(self) -> float:
        if self.total_recognition_attempts == 0:
            return 0.0
        return self.ambiguous_recognitions / self.total_recognition_attempts

    def summary(self) -> dict[str, Any]:
        return {
            "total_concepts": self.total_concepts,
            "total_attempts": self.total_recognition_attempts,
            "accuracy": round(self.accuracy, 4),
            "ambiguity_rate": round(self.ambiguity_rate, 4),
            "correct": self.correct_recognitions,
            "ambiguous": self.ambiguous_recognitions,
            "novel": self.novel_recognitions,
            "false": self.false_recognitions,
            "provenance_traceable": self.provenance_traceable,
            "concept_separation": self.concept_separation_scores,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class OneShotEvaluator:
    """Evaluation harness for one-shot visual learning.

    Usage::

        evaluator = OneShotEvaluator(perception)

        # Phase 1: Teach
        await evaluator.teach("cup", [cup_image_1])
        await evaluator.teach("bottle", [bottle_image_1])

        # Phase 2: Evaluate
        result = await evaluator.evaluate([
            ("cup", cup_test_image),
            ("bottle", bottle_test_image),
        ])
        print(result.summary())
    """

    def __init__(self, perception: Any) -> None:
        """Initialize with a VisualPerception instance."""
        self.perception = perception
        self._taught_labels: list[str] = []

    async def teach(self, label: str, images: list[Any]) -> None:
        """Teach one or more images for a label."""
        for img in images:
            await self.perception.learn(img, label)
        self._taught_labels.append(label)
        logger.info("Taught '%s' with %d images", label, len(images))

    async def evaluate(
        self,
        test_cases: list[tuple[str, Any]],
    ) -> OneShotEvalResult:
        """Evaluate recognition on labeled test images.

        Args:
            test_cases: List of (expected_label, image) pairs.

        """
        result = OneShotEvalResult(
            total_concepts=len(set(self._taught_labels)),
        )

        start = time.time()

        for expected_label, image in test_cases:
            result.total_recognition_attempts += 1
            recognition = await self.perception.recognize(image)

            if recognition.matched:
                if recognition.label == expected_label:
                    result.correct_recognitions += 1
                else:
                    result.false_recognitions += 1
            elif recognition.is_ambiguous:
                result.ambiguous_recognitions += 1
            elif recognition.is_novel:
                result.novel_recognitions += 1

            # Check provenance
            if recognition.observation_node_id:
                result.provenance_traceable += 1

        result.elapsed_seconds = time.time() - start

        logger.info(
            "Evaluation complete: accuracy=%.2f%%, ambiguity=%.2f%%",
            result.accuracy * 100,
            result.ambiguity_rate * 100,
        )

        return result
