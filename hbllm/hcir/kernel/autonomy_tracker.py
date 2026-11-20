"""HCIR Autonomy Tracker & Graduation Telemetry.

Tracks Pure-HCIR parsing, planning, and realization autonomy rates
across dialogue and interaction turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    distilled_entities: int = 0
    distilled_lexicon_entries: int = 0
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
            "parse_attempts": self.parse_attempts,
            "hcir_parse_autonomy_rate": round(self.hcir_parse_autonomy_rate, 4),
            "hcir_plan_autonomy_rate": round(self.hcir_plan_autonomy_rate, 4),
            "hcir_realize_autonomy_rate": round(self.hcir_realize_autonomy_rate, 4),
            "overall_autonomy_rate": round(self.overall_autonomy_rate, 4),
            "shadow_invocations": self.shadow_invocations,
            "distilled_lessons": self.distilled_lessons,
            "distilled_entities": self.distilled_entities,
            "distilled_lexicon_entries": self.distilled_lexicon_entries,
            "rejected_lessons": self.rejected_lessons,
            "can_retire_llm": self.can_retire_llm,
        }
