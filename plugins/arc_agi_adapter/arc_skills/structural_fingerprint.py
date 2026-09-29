"""Structural Fingerprinting and Cross-Game Transfer for ARC-AGI-3.

Enables transfer of successful strategies between structurally similar games.
Games with similar action sets, entity layouts, panel structures, and symmetries
can share declarative and induced skills without requiring identical layouts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer
from plugins.arc_agi_adapter.arc_skills.perceptual_context import PerceptualSkillContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuralFingerprint:
    """Domain-agnostic game structure signature."""

    action_set: frozenset[int]
    entity_count_range: tuple[int, int]
    has_panels: bool
    color_count: int
    has_symmetry: bool
    dominant_entity_shape: str

    @classmethod
    def from_grid(
        cls,
        grid: np.ndarray,
        available_actions: list[int] | set[int],
    ) -> StructuralFingerprint:
        """Extract domain-agnostic structural fingerprint from a grid and action affordances."""
        pctx = PerceptualSkillContext.from_grid(grid)
        entities = pctx.entities

        # Symmetry evaluation
        sym_scores = VisualSymmetryAnalyzer.compute_symmetry_scores(grid)
        has_sym = any(v >= 0.80 for v in sym_scores.values())

        # Dominant entity shape classification
        dominant_shape = cls._classify_dominant_shape(entities)

        return cls(
            action_set=frozenset(available_actions),
            entity_count_range=(max(0, len(entities) - 3), len(entities) + 3),
            has_panels=pctx.dividers is not None and len(pctx.dividers) > 0,
            color_count=len(set(int(c) for c in grid.flatten())),
            has_symmetry=has_sym,
            dominant_entity_shape=dominant_shape,
        )

    @classmethod
    def _classify_dominant_shape(cls, entities: list[Any]) -> str:
        """Classify the dominant entity shape among entities."""
        if not entities:
            return "amorphous"

        counts = {"rectangular": 0, "linear": 0, "amorphous": 0}
        for e in entities:
            h = getattr(e, "height", None) or (e.max_r - e.min_r + 1)
            w = getattr(e, "width", None) or (e.max_c - e.min_c + 1)
            area = getattr(e, "area", len(getattr(e, "pixels", [])))

            if h == 1 or w == 1:
                counts["linear"] += 1
            elif area == h * w:
                counts["rectangular"] += 1
            else:
                counts["amorphous"] += 1

        return max(counts.items(), key=lambda x: x[1])[0]

    def similarity(self, other: StructuralFingerprint) -> float:
        """Compute 0.0 - 1.0 structural similarity score between two fingerprints."""
        score = 0.0

        # Action set match (weight: 0.30)
        if self.action_set == other.action_set:
            score += 0.30
        elif self.action_set.issubset(other.action_set) or other.action_set.issubset(
            self.action_set
        ):
            score += 0.15

        # Panel existence match (weight: 0.15)
        if self.has_panels == other.has_panels:
            score += 0.15

        # Color richness match (weight: 0.15)
        if abs(self.color_count - other.color_count) <= 2:
            score += 0.15
        elif abs(self.color_count - other.color_count) <= 4:
            score += 0.08

        # Symmetry property match (weight: 0.10)
        if self.has_symmetry == other.has_symmetry:
            score += 0.10

        # Dominant entity morphology match (weight: 0.15)
        if self.dominant_entity_shape == other.dominant_entity_shape:
            score += 0.15

        # Entity count range overlap (weight: 0.15)
        overlap = max(
            0,
            min(self.entity_count_range[1], other.entity_count_range[1])
            - max(self.entity_count_range[0], other.entity_count_range[0]),
        )
        if overlap > 0:
            score += 0.15

        return min(1.0, score)


class StructuralTransferRegistry:
    """Maps structural fingerprints to successful strategies for cross-game transfer."""

    def __init__(self) -> None:
        self.registry: dict[str, tuple[StructuralFingerprint, Any]] = {}

    def register_success(
        self,
        game_id: str,
        fingerprint: StructuralFingerprint,
        skill: Any,
    ) -> None:
        """Register a winning strategy fingerprint for a game."""
        self.registry[game_id] = (fingerprint, skill)
        logger.info(
            "Registered structural transfer strategy for game '%s' (actions=%s, panels=%s, shape=%s)",
            game_id,
            sorted(fingerprint.action_set),
            fingerprint.has_panels,
            fingerprint.dominant_entity_shape,
        )

    def find_similar(
        self,
        fingerprint: StructuralFingerprint,
        threshold: float = 0.65,
    ) -> list[tuple[float, Any, str]]:
        """Find skills from structurally similar games, ranked by similarity."""
        matches: list[tuple[float, Any, str]] = []
        for game_id, (fp, skill) in self.registry.items():
            sim = fingerprint.similarity(fp)
            if sim >= threshold:
                matches.append((sim, skill, game_id))

        matches.sort(key=lambda m: m[0], reverse=True)
        return matches

    def clear(self) -> None:
        self.registry.clear()


# Global singleton transfer registry
_GLOBAL_TRANSFER_REGISTRY: StructuralTransferRegistry | None = None


def get_global_transfer_registry() -> StructuralTransferRegistry:
    global _GLOBAL_TRANSFER_REGISTRY
    if _GLOBAL_TRANSFER_REGISTRY is None:
        _GLOBAL_TRANSFER_REGISTRY = StructuralTransferRegistry()
    return _GLOBAL_TRANSFER_REGISTRY
