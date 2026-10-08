"""Dorsolateral Prefrontal Cortex (dlPFC) & Intraparietal Sulcus (IPS) Visuospatial Working Memory.

Modeled on primate visuospatial working memory and Baddeley's episodic buffer:
1. Feature-Location Coordinate Binding: Retains item-location associations
   M_WM = {(r, c): (revealed_feature, step, visit_count)}.
2. CA3 Pattern Completion & Pair-Matching Induction:
   When an action (e.g. click at (r, c)) reveals feature F, immediately queries
   working memory for another location (r', c') known to hold feature F. If present,
   emits a high-priority motor target to complete the pair.
3. Causal Stencil Discovery: Learns operator receptive fields:
   - Point mutation: click(r, c) modifies only (r, c).
   - Cardinal cross: click(r, c) toggles (r, c) and its 4 von Neumann neighbors.
   - Row/Column stencil: click(r, c) inverts entire axis.
4. Systematic Epistemic Scanning: Coordinates saccadic fixation in structured raster
   order over affordance arrays to avoid chaotic stochastic jitter.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemorySlot:
    """A single bound item in visuospatial working memory."""

    pos: tuple[int, int]
    feature_id: int
    step_observed: int
    visit_count: int = 1
    properties: dict[str, Any] = field(default_factory=dict)


class VisuospatialWorkingMemory:
    """Dorsolateral prefrontal working memory buffer binding features to spatial coordinates."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        # Coordinate -> WorkingMemorySlot
        self.bound_slots: dict[tuple[int, int], WorkingMemorySlot] = {}
        # Feature -> set of coordinates where this feature was observed
        self.feature_to_coords: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        # Confirmed resolved/matched coordinates (e.g. paired cards successfully cleared)
        self.resolved_coords: set[tuple[int, int]] = set()
        # Most recently probed coordinate and its revealed feature
        self.last_probed_coord: tuple[int, int] | None = None
        self.last_probed_feature: int | None = None
        # Observed stencils: click_pos relative offsets that changed
        self.discovered_stencils: list[set[tuple[int, int]]] = []
        # Pair-matching task evidence counter
        self.matching_evidence: int = 0
        self.is_pair_matching_task: bool = False

    def reset_episode(self, retain_long_term: bool = False) -> None:
        """Reset transient slot bindings while optionally preserving task-level stencil theories."""
        self.bound_slots.clear()
        self.feature_to_coords.clear()
        self.resolved_coords.clear()
        self.last_probed_coord = None
        self.last_probed_feature = None
        if not retain_long_term:
            self.discovered_stencils.clear()
            self.matching_evidence = 0
            self.is_pair_matching_task = False

    def record_probe(
        self,
        pos: tuple[int, int],
        feature_id: int,
        step: int,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Bind an observed feature to its spatial coordinate in working memory."""
        if pos in self.bound_slots:
            slot = self.bound_slots[pos]
            # If feature changed, update coordinate index
            if slot.feature_id != feature_id:
                self.feature_to_coords[slot.feature_id].discard(pos)
                slot.feature_id = feature_id
                self.feature_to_coords[feature_id].add(pos)
            slot.step_observed = step
            slot.visit_count += 1
        else:
            if len(self.bound_slots) >= self.capacity:
                # Evict oldest un-matched item (recency decay)
                oldest_pos = min(
                    self.bound_slots.keys(), key=lambda p: self.bound_slots[p].step_observed
                )
                old_feat = self.bound_slots[oldest_pos].feature_id
                self.feature_to_coords[old_feat].discard(oldest_pos)
                del self.bound_slots[oldest_pos]

            slot = WorkingMemorySlot(
                pos=pos,
                feature_id=feature_id,
                step_observed=step,
                visit_count=1,
                properties=properties or {},
            )
            self.bound_slots[pos] = slot
            self.feature_to_coords[feature_id].add(pos)

        self.last_probed_coord = pos
        self.last_probed_feature = feature_id

    def find_matching_pair(
        self,
        current_pos: tuple[int, int],
        current_feature: int,
    ) -> tuple[int, int] | None:
        """Check working memory for a previously observed location with the same non-zero feature."""
        if current_feature <= 0:
            return None

        candidates = [
            p
            for p in self.feature_to_coords.get(current_feature, set())
            if p != current_pos and p not in self.resolved_coords
        ]
        if candidates:
            # Sort by most recent observation
            candidates.sort(key=lambda p: self.bound_slots[p].step_observed, reverse=True)
            logger.info(
                "VisuospatialWorkingMemory: Matching pair found! Feature %d at %s matches previous observation at %s.",
                current_feature,
                current_pos,
                candidates[0],
            )
            self.matching_evidence += 1
            if self.matching_evidence >= 2:
                self.is_pair_matching_task = True
            return candidates[0]

        return None

    def mark_resolved(self, coords: set[tuple[int, int]]) -> None:
        """Mark coordinates as resolved / cleared."""
        self.resolved_coords.update(coords)
        for c in coords:
            if c in self.bound_slots:
                feat = self.bound_slots[c].feature_id
                self.feature_to_coords[feat].discard(c)
                del self.bound_slots[c]

    def record_diff_stencil(
        self,
        center: tuple[int, int],
        changed_coords: set[tuple[int, int]],
    ) -> None:
        """Learn spatial stencil offsets: {(dr, dc)} altered by activating center."""
        if not changed_coords:
            return
        stencil = {(r - center[0], c - center[1]) for r, c in changed_coords}
        if stencil not in self.discovered_stencils:
            self.discovered_stencils.append(stencil)
            logger.info(
                "VisuospatialWorkingMemory: Learned action causal stencil with %d relative offsets.",
                len(stencil),
            )

    def get_systematic_unprobed_candidate(
        self,
        candidates: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Select next unvisited candidate in clean raster scan order (dlPFC saccadic pacing)."""
        unvisited = [
            c for c in candidates if c not in self.bound_slots and c not in self.resolved_coords
        ]
        if not unvisited:
            return None
        # Sort top-to-bottom, left-to-right
        unvisited.sort(key=lambda p: (p[0], p[1]))
        return unvisited[0]
