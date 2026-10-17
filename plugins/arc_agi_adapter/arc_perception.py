from __future__ import annotations

"""ARC-AGI Sensory Perception Adapter.

Implements domain-specific perceptual lifting for ARC-AGI 2D grid environments.
Lifts raw 2D pixel matrices into formal HCIR PerceptionFrames (SpatialEntities,
topological passability maps, candidate exits, and avatar anchors).

All domain-specific visual heuristics (grid background estimation, perimeter
notch detection, ARC palette symmetries, corridor manifold continuity) live here
in the plugin layer, keeping the cognitive core 100% domain-agnostic.
"""

import logging
from typing import Any

import numpy as np

from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.percept import BasePerceptionAdapter, PerceptionFrame

logger = logging.getLogger(__name__)


class ARCPerceptionAdapter(BasePerceptionAdapter):
    """Domain Sensory Adapter lifting ARC-AGI 2D grids into standardized PerceptionFrames."""

    def __init__(self) -> None:
        pass

    def lift_percept(
        self,
        raw_observation: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> PerceptionFrame:
        """Lift a raw ARC observation into a standardized PerceptionFrame."""
        # 1. Normalize sensory array
        grid: np.ndarray
        if isinstance(raw_observation, np.ndarray):
            grid = raw_observation[-1] if raw_observation.ndim == 3 else raw_observation
        elif hasattr(raw_observation, "frame") and getattr(raw_observation, "frame"):
            f = raw_observation.frame
            grid = f[-1] if isinstance(f, np.ndarray) and f.ndim == 3 else np.asarray(f, dtype=int)
        elif hasattr(raw_observation, "raw_data"):
            rd = raw_observation.raw_data
            grid = (
                rd[-1] if isinstance(rd, np.ndarray) and rd.ndim == 3 else np.asarray(rd, dtype=int)
            )
        else:
            try:
                grid = np.asarray(raw_observation, dtype=int)
                if grid.ndim == 3 and len(grid) > 0:
                    grid = grid[-1]
            except Exception:
                grid = np.zeros((16, 16), dtype=int)

        H, W = grid.shape

        # 2. Extract context attributes safely
        bg = PerceptionEngine.estimate_background(grid)
        av_pos = getattr(context, "avatar_pos", None) or kwargs.get("avatar_pos")
        av_feature = getattr(context, "avatar_feature", None) or kwargs.get("avatar_feature")
        av_feats = set(
            getattr(context, "avatar_features", set()) or kwargs.get("avatar_features", set())
        )
        if av_feature is not None:
            av_feats.add(av_feature)

        symbolic_theory = getattr(context, "symbolic_theory", None)
        barrier_features = set()
        goal_features = set()
        walkable_features = set()
        if symbolic_theory is not None:
            barrier_features.update(symbolic_theory.barrier_features)
            goal_features.update(symbolic_theory.goal_features)
            walkable_features.update(symbolic_theory.walkable_features)

        learned_barriers = set(
            getattr(context, "learned_barriers", set()) or kwargs.get("learned_barriers", set())
        )
        learned_goals = set(
            getattr(context, "learned_goal_positions", set())
            or kwargs.get("learned_goal_positions", set())
        )
        known_lethals = set(
            getattr(context, "current_level_lethal_positions", set())
            or kwargs.get("lethal_cells", set())
        )
        if hasattr(context, "hazard_tracker") and hasattr(
            context.hazard_tracker, "static_lethal_positions"
        ):
            known_lethals.update(context.hazard_tracker.static_lethal_positions)

        known_lethal_features = set()
        if hasattr(context, "hazard_tracker") and hasattr(
            context.hazard_tracker, "known_lethal_features"
        ):
            known_lethal_features.update(context.hazard_tracker.known_lethal_features)

        # 3. Extract segmented spatial entities
        entities = PerceptionEngine.extract_entities(
            grid=grid,
            bg=bg,
            symbolic_theory=symbolic_theory,
            avatar_features=av_feats,
            state_mutations=getattr(context, "state_mutations", None),
            learned_barriers=learned_barriers,
            learned_goal_positions=learned_goals,
            avatar_pos=av_pos,
        )

        # 4. Detect structural exit and candidate goals
        candidate_goals: set[tuple[int, int]] = set(learned_goals)
        structural_goals = PerceptionEngine.detect_structural_goals(
            grid=grid,
            bg=bg,
            avatar_features=av_feats,
            avatar_feature=av_feature,
            barrier_features=barrier_features,
            known_lethal_features=known_lethal_features,
            avatar_pos=av_pos,
        )
        for sg in structural_goals:
            if sg.get("type") in ("exit_marker", "gateway_notch", "target_zone"):
                for cp in sg.get("cells", []):
                    candidate_goals.add(cp)
                pos = sg.get("position")
                if pos:
                    candidate_goals.add(pos)

        # 5. Synthesize Passable Topology Map
        # 1 = traversable/walkable, 0 = impassable barrier/lethal
        passable_topology = np.ones((H, W), dtype=bool)
        static_barriers: set[tuple[int, int]] = set(learned_barriers)

        # Mark static barrier features
        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if val in barrier_features or (symbolic_theory and symbolic_theory.is_barrier(val)):
                    static_barriers.add((r, c))

        # Corridor Manifold Continuity:
        # If foreground features are explicitly walkable and the background is NOT walkable,
        # treat background as impassable void (e.g. tu93 maze walls)
        walkable_and_safe = set(walkable_features) | set(
            getattr(context, "verified_safe_features", set()) or set()
        )
        if len(walkable_and_safe) > 0 and bg not in walkable_and_safe:
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) == bg and (r, c) not in candidate_goals:
                        static_barriers.add((r, c))

        for r, c in static_barriers:
            if 0 <= r < H and 0 <= c < W:
                passable_topology[r, c] = False

        for r, c in known_lethals:
            if 0 <= r < H and 0 <= c < W:
                passable_topology[r, c] = False

        # Compute observation difference if previous grid is available
        diff = None
        prev_grid = getattr(context, "prev_grid", None)
        if prev_grid is None:
            prev_grid = kwargs.get("prev_grid")
        if (
            prev_grid is not None
            and isinstance(prev_grid, np.ndarray)
            and prev_grid.shape == grid.shape
        ):
            diff = PerceptionEngine.compute_frame_diff(prev_grid, grid)

        return PerceptionFrame(
            entities=entities,
            passable_topology=passable_topology,
            avatar_pos=av_pos,
            avatar_features=av_feats,
            candidate_goals=sorted(list(candidate_goals)),
            static_barriers=static_barriers,
            lethal_cells=known_lethals,
            shape=(H, W),
            diff=diff,
            metadata={"bg": bg, "grid": grid},
        )
