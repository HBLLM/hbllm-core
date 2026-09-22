"""Morphological Shape Analysis — Rotation-invariant shape archetypes and concept tracking.

Provides translation- and rotation-invariant geometric representations of 2D spatial
entities. Used for disentangled concept learning: recognizing that a shape in one visual
form or position is the same archetype as a shape in another form or position.
Learned affordances and causal rules are discovered empirically through experience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hbllm.hcir.spatial_planner import EntityRole


@dataclass(frozen=True)
class ShapeArchetype:
    """Canonical, translation-invariant relative coordinate geometry of a 2D entity."""

    relative_coords: frozenset[tuple[int, int]]
    height: int
    width: int
    area: int
    canonical_id: tuple[tuple[int, int], ...]

    @classmethod
    def from_coords(cls, coords: set[tuple[int, int]] | list[tuple[int, int]]) -> ShapeArchetype:
        if not coords:
            return cls(frozenset(), 0, 0, 0, ())
        min_r = min(r for r, c in coords)
        min_c = min(c for r, c in coords)
        norm_coords = frozenset((r - min_r, c - min_c) for r, c in coords)
        h = max(r for r, c in norm_coords) + 1
        w = max(c for r, c in norm_coords) + 1
        area = len(norm_coords)

        orbit = cls._compute_rotations(norm_coords, h, w)
        canon = min(orbit, key=lambda s: sorted(s))
        return cls(
            relative_coords=norm_coords,
            height=h,
            width=w,
            area=area,
            canonical_id=tuple(sorted(canon)),
        )

    @classmethod
    def _compute_rotations(
        cls, coords: frozenset[tuple[int, int]], h: int, w: int
    ) -> list[frozenset[tuple[int, int]]]:
        """Compute all 4 orthogonal orientations normalized to (0, 0)."""
        rotations = [coords]
        curr = coords
        curr_h, curr_w = h, w
        for _ in range(3):
            rotated = frozenset((c, curr_h - 1 - r) for r, c in curr)
            curr_h, curr_w = curr_w, curr_h
            curr = rotated
            rotations.append(rotated)
        return rotations

    def is_rotation_of(self, other: ShapeArchetype) -> bool:
        """Check if this shape archetype is an orthogonal rotation of another."""
        if self.area != other.area:
            return False
        return self.canonical_id == other.canonical_id


class MorphologicalConcept:
    """Disentangled conceptual representation of an archetype, its semantic roles, and learned affordances.

    Generic cognitive model:
    - Shape archetypes represent structural geometry.
    - Features (colors, textures, tokens) are observed dynamically.
    - Affordances (rotatable, passable, interactive, switchable) are learned empirically
      via trial-and-error interaction rather than hardcoded domain slots.
    """

    def __init__(
        self,
        canonical_id: tuple[tuple[int, int], ...],
        archetype: ShapeArchetype,
        canonical_name: str,
        inferred_role: EntityRole = EntityRole.UNKNOWN,
        observed_features: set[Any] | None = None,
        learned_affordances: dict[str, Any] | None = None,
        interaction_count: int = 0,
        # Backward-compatible keyword arguments for domain adapters
        observed_colors: set[Any] | None = None,
        is_rotatable: bool = False,
        rotation_trigger: Any | None = None,
        is_color_switch: bool = False,
        color_transitions: dict[Any, Any] | None = None,
        passable_colors: set[Any] | None = None,
        barrier_colors: set[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.canonical_id = canonical_id
        self.archetype = archetype
        self.canonical_name = canonical_name
        self.inferred_role = inferred_role

        # Generic observed features
        self.observed_features: set[Any] = set(observed_features or [])
        if observed_colors:
            self.observed_features.update(observed_colors)

        # Generic learned affordances
        self.learned_affordances: dict[str, Any] = dict(learned_affordances or {})
        if is_rotatable or "is_rotatable" in kwargs:
            self.learned_affordances["rotatable"] = is_rotatable or kwargs.get(
                "is_rotatable", False
            )
        if rotation_trigger is not None:
            self.learned_affordances["rotation_trigger"] = rotation_trigger
        if is_color_switch or "is_color_switch" in kwargs:
            self.learned_affordances["feature_switch"] = is_color_switch or kwargs.get(
                "is_color_switch", False
            )
        if color_transitions is not None:
            self.learned_affordances["feature_transitions"] = dict(color_transitions)
        if passable_colors is not None:
            self.learned_affordances["passable_features"] = set(passable_colors)
        if barrier_colors is not None:
            self.learned_affordances["obstacle_features"] = set(barrier_colors)
        self.learned_affordances.update(kwargs)

        self.interaction_count = interaction_count

    # ── Learning Methods (Trial & Error) ──────────────────────────────────

    def learn_affordance(self, affordance: str, value: Any = True) -> None:
        """Empirically learn or update an affordance from interaction trial."""
        self.learned_affordances[affordance] = value

    def get_affordance(self, affordance: str, default: Any = None) -> Any:
        """Query whether an affordance has been learned for this concept."""
        return self.learned_affordances.get(affordance, default)

    def has_affordance(self, affordance: str) -> bool:
        """Check if an affordance is present and truthy."""
        return bool(self.learned_affordances.get(affordance, False))

    def record_observation(self, feature: Any) -> None:
        """Record an observed feature (visual token, color, texture)."""
        self.observed_features.add(feature)

    def record_interaction(self) -> None:
        """Record an empirical interaction attempt."""
        self.interaction_count += 1

    # ── Backward-Compatibility Accessors ──────────────────────────────────

    @property
    def observed_colors(self) -> set[Any]:
        return self.observed_features

    @observed_colors.setter
    def observed_colors(self, val: set[Any]) -> None:
        self.observed_features = set(val)

    @property
    def is_rotatable(self) -> bool:
        return bool(self.learned_affordances.get("rotatable", False))

    @is_rotatable.setter
    def is_rotatable(self, val: bool) -> None:
        self.learned_affordances["rotatable"] = val

    @property
    def rotation_trigger(self) -> Any | None:
        return self.learned_affordances.get("rotation_trigger")

    @rotation_trigger.setter
    def rotation_trigger(self, val: Any) -> None:
        self.learned_affordances["rotation_trigger"] = val

    @property
    def is_color_switch(self) -> bool:
        return bool(self.learned_affordances.get("feature_switch", False))

    @is_color_switch.setter
    def is_color_switch(self, val: bool) -> None:
        self.learned_affordances["feature_switch"] = val

    @property
    def color_transitions(self) -> dict[Any, Any]:
        return self.learned_affordances.setdefault("feature_transitions", {})

    @color_transitions.setter
    def color_transitions(self, val: dict[Any, Any]) -> None:
        self.learned_affordances["feature_transitions"] = dict(val)

    @property
    def passable_colors(self) -> set[Any]:
        return self.learned_affordances.setdefault("passable_features", set())

    @passable_colors.setter
    def passable_colors(self, val: set[Any]) -> None:
        self.learned_affordances["passable_features"] = set(val)

    @property
    def barrier_colors(self) -> set[Any]:
        return self.learned_affordances.setdefault("obstacle_features", set())

    @barrier_colors.setter
    def barrier_colors(self, val: set[Any]) -> None:
        self.learned_affordances["obstacle_features"] = set(val)

    def __repr__(self) -> str:
        return (
            f"MorphologicalConcept({self.canonical_name!r}, "
            f"role={self.inferred_role.name}, "
            f"features={len(self.observed_features)}, "
            f"affordances={list(self.learned_affordances.keys())})"
        )
