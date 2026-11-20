"""Causal ontology evidence hierarchy, set views, and Spelke physical axioms."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from hbllm.hcir.world.percept import TerminalSignal


class OntologyEvidenceRank(IntEnum):
    """Hierarchy of empirical evidence strength for ontology induction."""

    UNVERIFIED = 0
    HEURISTIC = 10
    CANDIDATE = 20
    HYDRATION_DEFAULT = 40
    TRAVERSAL = 60
    COLLISION = 80
    INTERACTION = 85
    LETHAL = 95
    TERMINAL_REWARD = 100


def parse_evidence(evidence: Any) -> tuple[int, str]:
    """Parse raw evidence into (rank, descriptive_str)."""
    if evidence is None:
        return OntologyEvidenceRank.UNVERIFIED, "unverified"
    if (
        evidence == TerminalSignal.SUCCESS
        or evidence == "terminal_reward"
        or evidence == "terminal_signal"
    ):
        return OntologyEvidenceRank.TERMINAL_REWARD, "terminal_reward:success"
    if isinstance(evidence, TerminalSignal):
        return (
            OntologyEvidenceRank.TERMINAL_REWARD
            if evidence == TerminalSignal.SUCCESS
            else OntologyEvidenceRank.UNVERIFIED,
            f"terminal_signal:{evidence.value}",
        )
    if isinstance(evidence, (int, OntologyEvidenceRank)):
        return int(evidence), f"rank:{int(evidence)}"
    if isinstance(evidence, str):
        ev_lower = evidence.lower()
        if any(tok in ev_lower for tok in ("lethal", "death", "fatal", "hazard")):
            return OntologyEvidenceRank.LETHAL, evidence
        if "terminal" in ev_lower or "reward" in ev_lower or "success" in ev_lower:
            return OntologyEvidenceRank.TERMINAL_REWARD, evidence
        if (
            "interaction" in ev_lower
            or "contact" in ev_lower
            or "displacement" in ev_lower
            or "push" in ev_lower
        ):
            return OntologyEvidenceRank.INTERACTION, evidence
        if "collision" in ev_lower or "bump" in ev_lower or "wall" in ev_lower:
            return OntologyEvidenceRank.COLLISION, evidence
        if "hydration" in ev_lower or "prior" in ev_lower or "background" in ev_lower:
            return OntologyEvidenceRank.HYDRATION_DEFAULT, evidence
        if ev_lower.startswith("topology:"):
            return OntologyEvidenceRank.HYDRATION_DEFAULT + 10, evidence
        if (
            "traversal" in ev_lower
            or "step" in ev_lower
            or "walk" in ev_lower
            or "free_space" in ev_lower
        ):
            return OntologyEvidenceRank.TRAVERSAL, evidence
        if "candidate" in ev_lower or "hypothesis" in ev_lower or "heuristic" in ev_lower:
            return OntologyEvidenceRank.CANDIDATE, evidence
        return OntologyEvidenceRank.HEURISTIC, evidence
    if isinstance(evidence, dict):
        if (
            evidence.get("has_terminal_reward") is True
            or evidence.get("terminal") == TerminalSignal.SUCCESS
            or evidence.get("provenance") in ("terminal_reward", "terminal_reward:success")
        ):
            return OntologyEvidenceRank.TERMINAL_REWARD, str(
                evidence.get("provenance", "terminal_reward")
            )
        if "rank" in evidence:
            return int(evidence["rank"]), str(
                evidence.get("provenance", f"rank:{evidence['rank']}")
            )
    return OntologyEvidenceRank.HEURISTIC, str(evidence)


class GovernedSetView(set):
    """Set view that enforces ontology provenance by routing mutations through governed callbacks."""

    def __init__(
        self,
        items: Any,
        on_add: Any | None = None,
        on_discard: Any | None = None,
        on_clear: Any | None = None,
    ) -> None:
        super().__init__(items)
        self._on_add = on_add
        self._on_discard = on_discard
        self._on_clear = on_clear

    def add(self, item: Any) -> None:
        super().add(item)
        if self._on_add is not None:
            self._on_add(item)

    def discard(self, item: Any) -> None:
        super().discard(item)
        if self._on_discard is not None:
            self._on_discard(item)

    def remove(self, item: Any) -> None:
        super().remove(item)
        if self._on_discard is not None:
            self._on_discard(item)

    def clear(self) -> None:
        super().clear()
        if self._on_clear is not None:
            self._on_clear()

    def update(self, *s: Any) -> None:
        for items in s:
            for item in items:
                self.add(item)


class SpelkePhysicalAxiom(StrEnum):
    """Elizabeth Spelke's infant/primate Core Knowledge physical principles."""

    FREE_SPACE = "free_space"
    COHESION = "cohesion"
    SOLIDITY_CONTINUITY = "solidity_continuity"
    CONTACT_CAUSALITY = "contact_causality"
    DIRECTIONAL_INERTIA = "directional_inertia"


__all__ = [
    "GovernedSetView",
    "OntologyEvidenceRank",
    "SpelkePhysicalAxiom",
    "parse_evidence",
]
