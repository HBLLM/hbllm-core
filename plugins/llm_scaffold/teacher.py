"""Parallel LLM Teacher Scaffold & Self-Detaching Autonomy Tracker.

Strictly isolated in ``plugins/llm_scaffold/`` (zero imports from here inside
``hbllm/hcir/``). Defaults offline to ``checkpoints/Llama-3.2-1B``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hbllm.hcir.nodes import BeliefNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_CHECKPOINT = Path("checkpoints/Llama-3.2-1B")


@dataclass
class AutonomyTracker:
    """Tracks Pure-HCIR vs LLM-scaffold autonomy rates across dialogue turns."""

    parse_attempts: int = 0
    parse_hcir_success: int = 0
    plan_attempts: int = 0
    plan_hcir_success: int = 0
    realize_attempts: int = 0
    realize_hcir_success: int = 0
    shadow_invocations: int = 0
    distilled_lessons: int = 0
    rejected_lessons: int = 0
    retirement_threshold: float = 0.98
    min_turns_for_retirement: int = 50

    @property
    def hcir_parse_autonomy_rate(self) -> float:
        return self.parse_hcir_success / self.parse_attempts if self.parse_attempts else 1.0

    @property
    def hcir_plan_autonomy_rate(self) -> float:
        return self.plan_hcir_success / self.plan_attempts if self.plan_attempts else 1.0

    @property
    def hcir_realize_autonomy_rate(self) -> float:
        return self.realize_hcir_success / self.realize_attempts if self.realize_attempts else 1.0

    @property
    def overall_autonomy_rate(self) -> float:
        total = self.parse_attempts + self.plan_attempts + self.realize_attempts
        if total == 0:
            return 1.0
        ok = self.parse_hcir_success + self.plan_hcir_success + self.realize_hcir_success
        return ok / total

    @property
    def can_retire_llm(self) -> bool:
        return (
            self.parse_attempts >= self.min_turns_for_retirement
            and self.hcir_parse_autonomy_rate >= self.retirement_threshold
            and self.hcir_realize_autonomy_rate >= self.retirement_threshold
        )

    def record_parse(self, hcir_ok: bool) -> None:
        self.parse_attempts += 1
        if hcir_ok:
            self.parse_hcir_success += 1

    def record_plan(self, hcir_ok: bool) -> None:
        self.plan_attempts += 1
        if hcir_ok:
            self.plan_hcir_success += 1

    def record_realize(self, hcir_ok: bool) -> None:
        self.realize_attempts += 1
        if hcir_ok:
            self.realize_hcir_success += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "hcir_parse_autonomy_rate": round(self.hcir_parse_autonomy_rate, 4),
            "hcir_plan_autonomy_rate": round(self.hcir_plan_autonomy_rate, 4),
            "hcir_realize_autonomy_rate": round(self.hcir_realize_autonomy_rate, 4),
            "overall_autonomy_rate": round(self.overall_autonomy_rate, 4),
            "shadow_invocations": self.shadow_invocations,
            "distilled_lessons": self.distilled_lessons,
            "rejected_lessons": self.rejected_lessons,
            "can_retire_llm": self.can_retire_llm,
        }


class LLMTeacherScaffold:
    """Offline-first teacher scaffold that distills verified lessons into HCIR."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_LOCAL_CHECKPOINT,
        tracker: AutonomyTracker | None = None,
        teacher_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.tracker = tracker or AutonomyTracker()
        self._teacher_fn = teacher_fn
        self.retired = False

    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint_path.exists()

    def advise_and_distill(
        self,
        utterance: str,
        workspace: Any,
        *,
        speaker_id: str = "user",
        confidence_floor: float = 0.65,
    ) -> dict[str, Any]:
        """Consult the teacher scaffold (if not retired) and distill verified lessons."""
        if self.retired or self.tracker.can_retire_llm:
            self.retired = True
            return {"retired": True, "response_text": "", "distilled": False}

        self.tracker.shadow_invocations += 1
        if self._teacher_fn is not None:
            suggestion = self._teacher_fn(utterance)
        else:
            # Deterministic offline placeholder when no live teacher callable is bound
            suggestion = {
                "response_text": "",
                "claim": "",
                "confidence": 0.0,
                "verified": False,
            }

        claim = str(suggestion.get("claim", "")).strip()
        confidence = float(suggestion.get("confidence", 0.0))
        verified = bool(suggestion.get("verified", False))
        distilled = False

        if claim and verified and confidence >= confidence_floor and workspace is not None:
            node = BeliefNode(
                id=f"scaffold_lesson_{uuid.uuid4().hex[:10]}",
                statement=claim,
                confidence=min(1.0, confidence),
                provenance=Provenance(
                    source="llm_scaffold",
                    confidence=min(1.0, confidence),
                    evidence=[f"speaker:{speaker_id}", f"utterance:{utterance[:80]}"],
                ),
            )
            op = TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node.id,
                node_data=node.model_dump(),
            )
            commit_fn = getattr(workspace, "commit_operations", None)
            if callable(commit_fn):
                receipt = commit_fn([op], author="llm_scaffold")
                distilled = bool(getattr(receipt, "is_committed", False))
            if distilled:
                self.tracker.distilled_lessons += 1
            else:
                self.tracker.rejected_lessons += 1
        elif claim:
            self.tracker.rejected_lessons += 1

        return {
            "retired": False,
            "response_text": str(suggestion.get("response_text", "")),
            "distilled": distilled,
            "checkpoint": str(self.checkpoint_path),
        }
