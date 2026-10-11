"""Spatial Containment & Transport Engine for HCIR World Kernel.

Implements domain-agnostic spatial relation detection (INSIDE, NEAR, ON),
synchronous containment transport schema induction, object permanence tracking,
silhouette packing optimization, and room topology extraction with doorway detection.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.ndimage import label

from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    BeliefTransitionType,
)

logger = logging.getLogger(__name__)


# ── Domain 1: Spatial Relations & Containment Schemas ─────────────────────────


class SpatialRelationType(StrEnum):
    """Categorical spatial relations between physical or cognitive entities."""

    ON = "ON"
    INSIDE = "INSIDE"
    NEAR = "NEAR"
    CONTAINS = "CONTAINS"
    CONTAINED_BY = "CONTAINED_BY"
    ADJACENT = "ADJACENT"
    DISJOINT = "DISJOINT"


@dataclass
class SpatialRelationFact:
    """Structured spatial fact induced from continuous geometric perception."""

    relation: SpatialRelationType | str = SpatialRelationType.ON
    subject_id: str = ""
    object_id: str = ""
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_symbolic_predicate(self) -> dict[str, Any]:
        """Convert fact into declarative symbolic predicate form."""
        rel_str = (
            self.relation.value
            if isinstance(self.relation, SpatialRelationType)
            else str(self.relation)
        )
        return {
            "relation": rel_str,
            "subject": self.subject_id,
            "object": self.object_id,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }


class BaseSpatialContainmentEngine:
    """Domain-agnostic spatial relations and containment transport engine."""

    def __init__(self) -> None:
        self.discovered_spatial_schemas: list[dict[str, Any]] = []
        self.belief_history: list[BeliefTransitionEvent] = []
        self.interventions_count: int = 0

    @staticmethod
    def detect_spatial_relations(
        percept_items: Sequence[dict[str, Any]],
        near_threshold: float = 0.5,
        inside_relation: SpatialRelationType | str = SpatialRelationType.INSIDE,
        near_relation: SpatialRelationType | str = SpatialRelationType.NEAR,
    ) -> list[SpatialRelationFact]:
        """Induce relational facts directly from geometric and perceptual observations."""
        facts: list[SpatialRelationFact] = []
        percept_map = {p["percept_id"]: p for p in percept_items if "percept_id" in p}

        # Check containment relations
        for p in percept_items:
            cid = p.get("contained_in")
            if cid and cid in percept_map:
                facts.append(
                    SpatialRelationFact(
                        relation=inside_relation,
                        subject_id=p["percept_id"],
                        object_id=cid,
                        confidence=1.0,
                        evidence={"source": "direct_percept"},
                    )
                )

        # Check proximity / NEAR relations between all pairs
        ids = list(percept_map.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                p1, p2 = percept_map[id1], percept_map[id2]
                coords1 = p1.get("spatial_coordinates")
                coords2 = p2.get("spatial_coordinates")
                if coords1 is not None and coords2 is not None:
                    dx = coords1[0] - coords2[0]
                    dy = coords1[1] - coords2[1]
                    dist = float((dx**2 + dy**2) ** 0.5)
                    if dist <= near_threshold:
                        confidence = (
                            max(0.0, 1.0 - (dist / near_threshold)) if near_threshold > 0 else 1.0
                        )
                        facts.append(
                            SpatialRelationFact(
                                relation=near_relation,
                                subject_id=id1,
                                object_id=id2,
                                confidence=confidence,
                                evidence={"distance": dist},
                            )
                        )

        return facts

    @staticmethod
    def evaluate_containment_transport_invariance(
        container_moved: bool,
        container_disp: float,
        inside_disp: float,
        outside_disp: float,
        tolerance: float = 0.05,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate synchronous containment transport invariance."""
        transport_confirmed = (
            container_moved
            and abs(inside_disp - container_disp) < tolerance
            and outside_disp < tolerance
        )
        schema = {
            "schema_id": "schema_containment_transport",
            "relation": "INSIDE",
            "action": "PUSH",
            "invariant": "SYNCHRONOUS_TRANSPORT",
            "confirmed": transport_confirmed,
            "container_displacement": container_disp,
            "contained_displacement": inside_disp,
            "outside_displacement": outside_disp,
        }
        return transport_confirmed, schema

    @staticmethod
    def evaluate_object_permanence(
        predicted_pos: tuple[float, float],
        actual_pos: tuple[float, float],
        tolerance: float = 0.05,
    ) -> dict[str, Any]:
        """Compute Euclidean distance error between predicted and actual position."""
        dx = predicted_pos[0] - actual_pos[0]
        dy = predicted_pos[1] - actual_pos[1]
        prediction_error = float((dx**2 + dy**2) ** 0.5)
        return {
            "predicted_position": predicted_pos,
            "actual_position": actual_pos,
            "prediction_error": prediction_error,
            "permanence_preserved": prediction_error < tolerance,
        }

    def record_spatial_schema(
        self,
        schema: dict[str, Any],
        container_id: str,
        target_store: list[dict[str, Any]] | None = None,
        step_index: int | None = None,
    ) -> BeliefTransitionEvent:
        """Register confirmed spatial schema and log belief transition event."""
        self.discovered_spatial_schemas.append(schema)
        if target_store is not None:
            target_store.append(schema)

        step = self.interventions_count if step_index is None else step_index
        event = BeliefTransitionEvent(
            event_type=BeliefTransitionType.SPATIAL_SCHEMA_INDUCED,
            step_index=step,
            hypothesis_id="schema_containment",
            variable="spatial_containment",
            condition=f"INSIDE(x, {container_id}) ∧ MOVE({container_id}) => MOVE(x)",
            prior_confidence=0.5,
            posterior_confidence=1.0,
            is_falsified=False,
            evidence={"schema": schema},
        )
        self.belief_history.append(event)
        logger.info(
            "Spatial schema induced: INSIDE(x, %s) ∧ MOVE(%s) => MOVE(x)",
            container_id,
            container_id,
        )
        return event

    @staticmethod
    def solve_silhouette_packing(
        silhouette_cells: set[tuple[int, int]],
        piece_shapes: list[set[tuple[int, int]]],
        allow_rotations: bool = False,
    ) -> list[tuple[int, int]] | None:
        """Find non-overlapping placement offsets for piece_shapes to exactly cover silhouette_cells.

        Given a target silhouette (set of 2D cell coordinates) and a list of piece shapes (each a set of
        relative 2D cell coordinates normalized to min_r=0, min_c=0), finds the translation offset (dr, dc)
        for each piece such that the pieces are mutually disjoint and their union exactly covers the silhouette.

        Returns:
            A list of (dr, dc) offsets for each piece, or None if no valid packing exists.
        """
        if not piece_shapes or not silhouette_cells:
            return None

        total_piece_cells = sum(len(p) for p in piece_shapes)
        if total_piece_cells != len(silhouette_cells):
            return None

        indexed_pieces = sorted(enumerate(piece_shapes), key=lambda x: len(x[1]), reverse=True)
        n = len(piece_shapes)
        assignments: list[tuple[int, int] | None] = [None] * n

        def backtrack(
            piece_idx: int, covered: set[tuple[int, int]]
        ) -> list[tuple[int, int]] | None:
            if piece_idx == n:
                if covered == silhouette_cells:
                    return [assignments[i] for i in range(n)]  # type: ignore[misc]
                return None

            orig_idx, shape = indexed_pieces[piece_idx]
            first_uncovered = next(
                (cell for cell in sorted(silhouette_cells) if cell not in covered), None
            )
            if first_uncovered is None:
                return None

            fur, fuc = first_uncovered
            for pr, pc in shape:
                dr = fur - pr
                dc = fuc - pc
                translated = {(r + dr, c + dc) for r, c in shape}
                if translated.issubset(silhouette_cells) and translated.isdisjoint(covered):
                    assignments[orig_idx] = (dr, dc)
                    res = backtrack(piece_idx + 1, covered | translated)
                    if res is not None:
                        return res
                    assignments[orig_idx] = None

            return None

        return backtrack(0, set())


# ── Domain 2: Room Topology Extraction & Doorway Gating ───────────────────────


@dataclass
class RoomDoor:
    """A doorway connecting two chambers detected from wall openings."""

    door_coord: tuple[int, int] = (0, 0)
    connects_rooms: tuple[int, int] = (0, 0)


@dataclass
class RoomTopologyGraph:
    """Topological graph of chamber connectivity and access portals."""

    rooms: dict[int, list[tuple[int, int]]]
    doors: list[RoomDoor]
    adjacency: dict[int, list[int]]

    def get_room_for_coord(self, coord: tuple[int, int]) -> int | None:
        """Find the room index containing a given coordinate."""
        for r_id, cells in self.rooms.items():
            if coord in cells:
                return r_id
        return None

    def are_connected(self, room_a: int, room_b: int) -> bool:
        """Check if two rooms share an open doorway."""
        return room_b in self.adjacency.get(room_a, [])


class RoomTopologyExtractor:
    """Decomposes walkable space into chambers/rooms and detects connecting doorways.

    Domain-agnostic: operates on any boolean occupancy grid (True = walkable, False = wall).
    """

    @staticmethod
    def extract_rooms_and_doors(
        occupancy_grid: np.ndarray,
        min_room_size: int = 4,
    ) -> tuple[dict[int, list[tuple[int, int]]], list[RoomDoor]]:
        """Partitions occupancy grid into rooms separated by walls and identifies connecting doorways.

        Args:
            occupancy_grid: Boolean 2D grid where True = walkable, False = wall.
            min_room_size: Minimum number of cells for a region to be considered a room.

        Returns:
            Tuple of (rooms dict mapping room_id to cell coordinates, list of RoomDoor instances).
        """
        h, w = occupancy_grid.shape
        door_coords: set[tuple[int, int]] = set()

        # Detect doorways: walkable cells flanked by walls on two opposite sides
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not occupancy_grid[r, c]:
                    continue
                h_door = (
                    not occupancy_grid[r - 1, c]
                    and not occupancy_grid[r + 1, c]
                    and occupancy_grid[r, c - 1]
                    and occupancy_grid[r, c + 1]
                )
                v_door = (
                    not occupancy_grid[r, c - 1]
                    and not occupancy_grid[r, c + 1]
                    and occupancy_grid[r - 1, c]
                    and occupancy_grid[r + 1, c]
                )
                if h_door or v_door:
                    door_coords.add((r, c))

        # Remove door cells to separate rooms into distinct connected components
        room_grid = occupancy_grid.copy()
        for dr, dc in door_coords:
            room_grid[dr, dc] = False

        # Fast 4-connected component labeling using scipy native substrate
        struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        labeled_rooms, num_rooms = label(room_grid, structure=struct4)

        rooms: dict[int, list[tuple[int, int]]] = {}
        room_id_map: dict[tuple[int, int], int] = {}
        room_counter = 0

        for comp_id in range(1, num_rooms + 1):
            coords = [
                (int(r), int(c)) for r, c in zip(*np.where(labeled_rooms == comp_id), strict=False)
            ]
            if len(coords) >= min_room_size or comp_id == 1:
                room_counter += 1
                rooms[room_counter] = coords
                for coord in coords:
                    room_id_map[coord] = room_counter

        # Resolve which rooms each door connects
        doors: list[RoomDoor] = []
        for dr, dc in door_coords:
            adjacent_rooms: set[int] = set()
            for off_r, off_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = dr + off_r, dc + off_c
                if (nr, nc) in room_id_map:
                    adjacent_rooms.add(room_id_map[(nr, nc)])
            if len(adjacent_rooms) == 2:
                r_list = sorted(list(adjacent_rooms))
                doors.append(RoomDoor(door_coord=(dr, dc), connects_rooms=(r_list[0], r_list[1])))

        return rooms, doors

    @staticmethod
    def build_adjacency_graph(
        rooms: dict[int, list[tuple[int, int]]],
        doors: list[RoomDoor],
    ) -> dict[int, list[int]]:
        """Builds topological graph of room adjacencies."""
        adj: dict[int, set[int]] = {r: set() for r in rooms}
        for door in doors:
            ra, rb = door.connects_rooms
            if ra in adj and rb in adj:
                adj[ra].add(rb)
                adj[rb].add(ra)
        return {r: sorted(list(neighbors)) for r, neighbors in adj.items()}

    @classmethod
    def extract_topology_graph(
        cls,
        occupancy_grid: np.ndarray,
        min_room_size: int = 4,
    ) -> RoomTopologyGraph:
        """Extract complete RoomTopologyGraph including rooms, doorways, and adjacency."""
        rooms, doors = cls.extract_rooms_and_doors(occupancy_grid, min_room_size=min_room_size)
        adjacency = cls.build_adjacency_graph(rooms, doors)
        return RoomTopologyGraph(rooms=rooms, doors=doors, adjacency=adjacency)
