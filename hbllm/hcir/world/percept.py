from __future__ import annotations

"""HCIR Perceptual Frame & Sensory Lifting Protocol.

Provides universal sensory lifting abstractions decoupling domain-specific
sensory transducers (e.g. 2D ARC-AGI grids, AI2-THOR RGB-D, Crafter tiles,
robotics point-clouds) from core cognitive faculties (path planning, causal
induction, active inference, and executive decision-making).

In accordance with HBLLM Invariant 1.7 (General Intelligence Invariant),
the cognitive core never inspects raw domain pixels, color frequencies,
or benchmark-specific visual heuristics. Domain adapters lift raw observations
into a PerceptionFrame, and the core operates exclusively over these percepts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity


@dataclass
class PerceptionFrame:
    """Universal Perceptual Frame consumed by HCIR Cognitive Core faculties.

    Attributes:
        entities: Segmented spatial entities with bounding boxes, centroids, and roles.
        passable_topology: 2D boolean array (True=walkable/traversable, False=blocked)
            or an abstract topological navigation graph.
        avatar_pos: Detected current coordinates of the controllable agent / avatar.
        avatar_features: Set of feature/color IDs belonging to the avatar.
        candidate_goals: Known or structural candidate goal / exit coordinates.
        static_barriers: Coordinates of impassable static obstacles.
        lethal_cells: Coordinates of known hazardous / lethal positions.
        shape: Dimensions of the spatial observation space (e.g., (H, W)).
        diff: Optional observation difference from the previous frame.
        metadata: Domain-specific auxiliary metadata (e.g., tags, step limit).
    """

    entities: list[SpatialEntity] = field(default_factory=list)
    passable_topology: np.ndarray | None = None
    avatar_pos: tuple[int, int] | None = None
    avatar_features: set[int] = field(default_factory=set)
    candidate_goals: list[tuple[int, int]] = field(default_factory=list)
    static_barriers: set[tuple[int, int]] = field(default_factory=set)
    lethal_cells: set[tuple[int, int]] = field(default_factory=set)
    shape: tuple[int, int] = (0, 0)
    diff: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_passable(self, r: int, c: int) -> bool:
        """Check whether a given grid coordinate is walkable and safe."""
        if self.passable_topology is not None:
            H, W = self.passable_topology.shape
            if 0 <= r < H and 0 <= c < W:
                return bool(self.passable_topology[r, c])
            return False
        if (r, c) in self.static_barriers or (r, c) in self.lethal_cells:
            return False
        H, W = self.shape
        return 0 <= r < H and 0 <= c < W


class BasePerceptionAdapter(ABC):
    """Abstract Base Perception Adapter interface.

    Domain plugins implement this interface to lift raw sensory modalities
    into standardized PerceptionFrames for the cognitive blackbox.
    """

    @abstractmethod
    def lift_percept(
        self,
        raw_observation: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> PerceptionFrame:
        """Lift raw sensory input into a standardized PerceptionFrame."""
        ...


class DefaultPerceptionAdapter(BasePerceptionAdapter):
    """Domain-agnostic default perception adapter for spatial 2D grid observations.

    Uses connected-component analysis and context roles (goals, barriers, avatar)
    to lift raw arrays into typed SpatialEntity components and a passable topology.
    """

    def lift_percept(
        self,
        raw_observation: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> PerceptionFrame:
        """Lift raw 2D grid observation into a typed PerceptionFrame."""
        if isinstance(raw_observation, PerceptionFrame):
            return raw_observation
        if raw_observation is None:
            return PerceptionFrame()

        grid = np.asarray(raw_observation)
        if grid.ndim != 2:
            return PerceptionFrame()

        H, W = grid.shape
        bg = (
            int(kwargs.get("bg", 0))
            if "bg" in kwargs
            else int(np.bincount(grid.flatten().astype(int)).argmax())
        )

        from scipy import ndimage

        lethal_features: set[int] = set()
        lethal_positions: set[tuple[int, int]] = set()
        if context and hasattr(context, "hazard_tracker"):
            lethal_features = set(context.hazard_tracker.known_lethal_features)
            lethal_positions = set(context.hazard_tracker.static_lethal_positions)

        entities: list[SpatialEntity] = []
        unique_feats = [int(x) for x in np.unique(grid) if int(x) != bg]
        for feat in unique_feats:
            mask = grid == feat
            labeled, num_features = ndimage.label(mask)
            for lab in range(1, num_features + 1):
                comp_mask = labeled == lab
                coords = np.argwhere(comp_mask)
                cells = [tuple(int(x) for x in c) for c in coords]
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

                role = EntityRole.UNKNOWN
                if feat in lethal_features:
                    role = EntityRole.DYNAMIC_HAZARD
                elif context and hasattr(context, "symbolic_theory"):
                    if context.symbolic_theory.is_goal(feat):
                        role = EntityRole.GOAL
                    elif context.symbolic_theory.is_barrier(feat):
                        role = EntityRole.OBSTACLE
                if (
                    context
                    and hasattr(context, "avatar_feature")
                    and context.avatar_feature == feat
                ):
                    role = EntityRole.AGENT

                ent = SpatialEntity(
                    id=f"entity_{feat}_{r_min}_{c_min}",
                    role=role,
                    feature_id=feat,
                    area=len(cells),
                    centroid=centroid,
                    bounding_box=(int(r_min), int(r_max), int(c_min), int(c_max)),
                    grid_pos=(int(round(centroid[0])), int(round(centroid[1]))),
                    properties={"cells": cells},
                )
                entities.append(ent)

        barriers: set[tuple[int, int]] = set()
        goals: list[tuple[int, int]] = []
        avatar_pos: tuple[int, int] | None = None
        lethal_cells: set[tuple[int, int]] = set(lethal_positions)
        for e in entities:
            if e.role == EntityRole.DYNAMIC_HAZARD or (
                e.feature_id is not None and e.feature_id in lethal_features
            ):
                lethal_cells.update(e.properties.get("cells", []))
            elif e.role == EntityRole.OBSTACLE:
                barriers.update(e.properties.get("cells", []))
            elif e.role == EntityRole.GOAL:
                goals.append(e.grid_pos)
            elif e.role == EntityRole.AGENT:
                avatar_pos = e.grid_pos

        if context and hasattr(context, "learned_barriers"):
            barriers.update(context.learned_barriers)

        candidate_goals: set[tuple[int, int]] = set(goals)
        if context and hasattr(context, "learned_goal_positions"):
            candidate_goals.update(context.learned_goal_positions)

        # Detect structural exit markers, gateway notches, and target zones
        detect_goals_fn = getattr(context, "detect_structural_goals", None)
        if detect_goals_fn is None:
            try:
                from hbllm.hcir.world.cortex_perception import PerceptionEngine

                detect_goals_fn = PerceptionEngine.detect_structural_goals
            except Exception:
                detect_goals_fn = None
        if detect_goals_fn is not None:
            try:
                structural_goals = detect_goals_fn(grid=grid, bg=bg)
                for sg in structural_goals:
                    if sg.get("type") in (
                        "exit_marker",
                        "gateway_notch",
                        "target_zone",
                        "target_constellation",
                        "receptacle_dock",
                    ):
                        for cp in sg.get("cells", []):
                            candidate_goals.add(cp)
                        pos = sg.get("position")
                        if pos:
                            candidate_goals.add(pos)
            except Exception:
                pass

        if candidate_goals:
            for e in entities:
                if e.grid_pos in candidate_goals and e.role not in (
                    EntityRole.AGENT,
                    EntityRole.DYNAMIC_HAZARD,
                ):
                    e.role = EntityRole.GOAL

        passable = np.ones((H, W), dtype=bool)
        for r, c in barriers | lethal_cells:
            if 0 <= r < H and 0 <= c < W:
                passable[r, c] = False

        return PerceptionFrame(
            entities=entities,
            passable_topology=passable,
            avatar_pos=avatar_pos,
            candidate_goals=list(candidate_goals),
            static_barriers=barriers,
            lethal_cells=lethal_cells,
            shape=(H, W),
            metadata={"grid": grid, "bg": bg},
        )


class PerceptionRegistry:
    """Universal registry for sensory perception adapters.

    Allows external environment plugins (ARC-AGI, AI2-THOR, Robotics) to plug
    into the cognitive core without the core having any reverse dependencies on plugins.
    """

    _default_adapter: BasePerceptionAdapter | None = DefaultPerceptionAdapter()
    _adapters: dict[str, BasePerceptionAdapter] = {"default": DefaultPerceptionAdapter()}

    @classmethod
    def register(cls, name: str, adapter: BasePerceptionAdapter, set_default: bool = True) -> None:
        """Register an adapter by name, optionally setting it as the default."""
        cls._adapters[name] = adapter
        if set_default or cls._default_adapter is None:
            cls._default_adapter = adapter

    @classmethod
    def get(cls, name: str) -> BasePerceptionAdapter | None:
        """Get a registered adapter by name."""
        return cls._adapters.get(name)

    @classmethod
    def get_default(cls) -> BasePerceptionAdapter | None:
        """Get the active default perception adapter."""
        return cls._default_adapter

    @classmethod
    def clear(cls) -> None:
        """Reset the registry to default."""
        cls._adapters.clear()
        cls._default_adapter = DefaultPerceptionAdapter()
