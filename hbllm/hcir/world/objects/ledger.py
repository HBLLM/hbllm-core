"""Object State Ledger & Interaction History."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.world.objects.models import _scipy_label

logger = logging.getLogger(__name__)


class ObjectStateLedger:
    """Maintains the discrete state configuration across the K salient objects."""

    def __init__(self) -> None:
        self.interaction_counts: dict[str, int] = {}
        self.interacted_positions: set[tuple[int, int]] = set()
        self.interacted_features: set[int] = set()
        self.effective_features: set[int] = set()
        self.observed_mutations: list[Any] = []
        self.unlocked_barriers: set[tuple[int, int]] = set()
        self.object_causal_effects: dict[str, list[str]] = {}
        self.unlocked_by_object: dict[str, set[tuple[int, int]]] = {}
        self.initially_unreachable_cells: set[tuple[int, int]] = set()
        self.newly_reachable_cells: set[tuple[int, int]] = set()
        self._baseline_reach_initialized: bool = False
        self._last_connectivity_sig: tuple[tuple[int, int], tuple[int, int], bytes] | None = None

    def reset_context_epoch(self) -> None:
        """Reset transient context-specific interaction state."""
        self.interaction_counts.clear()
        self.interacted_positions.clear()
        self.interacted_features.clear()
        self.effective_features.clear()
        self.observed_mutations.clear()
        self.unlocked_barriers.clear()
        self.object_causal_effects.clear()
        self.unlocked_by_object.clear()
        self.initially_unreachable_cells.clear()
        self.newly_reachable_cells.clear()
        self._baseline_reach_initialized = False
        self._last_connectivity_sig = None

    def record_interaction(
        self,
        object_id: str,
        pos: tuple[int, int],
        feature_id: int | None = None,
    ) -> None:
        """Record an interaction event with an object."""
        self.interaction_counts[object_id] = self.interaction_counts.get(object_id, 0) + 1
        self.interacted_positions.add(pos)
        if feature_id is not None:
            self.interacted_features.add(feature_id)

    def record_mutation(self, mutation: Any) -> None:
        """Record an environmental state mutation triggered by interaction."""
        self.observed_mutations.append(mutation)

    def record_causal_effect(
        self,
        object_id: str,
        effect_type: str,
        affected_coords: set[tuple[int, int]] | None = None,
        feature_id: int | None = None,
    ) -> None:
        """Record an empirical causal effect attributed to an object interaction."""
        self.object_causal_effects.setdefault(object_id, []).append(effect_type)
        if feature_id is not None:
            self.effective_features.add(int(feature_id))
        if affected_coords:
            self.unlocked_barriers.update(affected_coords)
            self.unlocked_by_object.setdefault(object_id, set()).update(affected_coords)

    def observe_grid_connectivity(
        self,
        grid: np.ndarray | None,
        avatar_pos: tuple[int, int] | None,
        bg: int,
        walkable_feats: set[int],
        av_feats: set[int],
    ) -> None:
        """Track compartment connectivity so post-unlock reachable regions are flagged causally."""
        if grid is None or avatar_pos is None or grid.ndim != 2:
            return
        if self._baseline_reach_initialized and not self.initially_unreachable_cells:
            return
        H, W = grid.shape
        ar, ac = avatar_pos
        if not (0 <= ar < H and 0 <= ac < W):
            return
        passable_feats = set(walkable_feats) | set(av_feats) | {int(bg)}
        passable_mask = np.isin(grid, list(passable_feats))
        passable_mask[ar, ac] = True
        sig = ((H, W), (ar, ac), passable_mask.tobytes())
        if self._baseline_reach_initialized and self._last_connectivity_sig == sig:
            return
        self._last_connectivity_sig = sig

        labeled_cc, _ = _scipy_label(passable_mask)
        av_cc = int(labeled_cc[ar, ac])
        reached_mask = labeled_cc == av_cc
        adj_mask = reached_mask.copy()
        adj_mask[1:, :] |= reached_mask[:-1, :]
        adj_mask[:-1, :] |= reached_mask[1:, :]
        adj_mask[:, 1:] |= reached_mask[:, :-1]
        adj_mask[:, :-1] |= reached_mask[:, 1:]

        if not self._baseline_reach_initialized:
            self._baseline_reach_initialized = True
            unreach = (~adj_mask) & (grid != int(bg))
            if np.any(unreach):
                rs, cs = np.nonzero(unreach)
                self.initially_unreachable_cells = set(zip(rs.tolist(), cs.tolist()))
            else:
                self.initially_unreachable_cells = set()
            return

        if self.initially_unreachable_cells:
            for pos in self.initially_unreachable_cells:
                if adj_mask[pos[0], pos[1]]:
                    self.newly_reachable_cells.add(pos)

    def is_in_newly_reachable(self, cells: set[tuple[int, int]]) -> bool:
        """True if any cell belongs to a compartment that became reachable after an unlock."""
        return bool(
            self.newly_reachable_cells and any(c in self.newly_reachable_cells for c in cells)
        )

    def get_interaction_count(self, object_id: str) -> int:
        """Return the number of times an object has been interacted with."""
        return self.interaction_counts.get(object_id, 0)

    def is_interacted(self, object_id: str) -> bool:
        """True if the object has been probed at least once."""
        return self.get_interaction_count(object_id) > 0

    def is_inert(self, object_id: str) -> bool:
        """True if the object was interacted with at least once and produced no causal mutations."""
        return (
            self.get_interaction_count(object_id) >= 1
            and object_id not in self.object_causal_effects
        )

    def get_unlocked_barriers(self) -> set[tuple[int, int]]:
        """Return all barrier coordinates unlocked via object interactions."""
        return set(self.unlocked_barriers)
