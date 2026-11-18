"""
Epistemic Spatial Grid — 2D occupancy, visibility, and frontier representation over CognitiveGraph.

Maintains an allocentric spatial substrate for embodied agents, tracking:
1. Epistemic visibility (seen vs unobserved space).
2. Physical occupancy (walls, passable floors, doors, movable obstacles).
3. Exploration frontiers (portals and boundaries adjacent to unobserved regions).
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from dataclasses import dataclass, field

from hbllm.hcir.graph import CognitiveGraph, EntityLifecycle, PhysicalEntityNode


@dataclass
class SpatialCellPartition:
    """Classified spatial layout extracted from the CognitiveGraph."""

    walls: set[tuple[int, int]] = field(default_factory=set)
    closed_doors: set[tuple[int, int]] = field(default_factory=set)
    open_doors: dict[tuple[int, int], PhysicalEntityNode] = field(default_factory=dict)
    movable_obstacles: dict[tuple[int, int], PhysicalEntityNode] = field(default_factory=dict)
    entities_by_coord: dict[tuple[int, int], PhysicalEntityNode] = field(default_factory=dict)
    occupied_coords: set[tuple[int, int]] = field(default_factory=set)
    max_x: int = 16
    max_y: int = 16

    @property
    def blocking_cells(self) -> set[tuple[int, int]]:
        """Static walls plus movable physical obstacles."""
        return self.walls | set(self.movable_obstacles.keys())


class EpistemicSpatialGrid:
    """Allocentric 2D spatial memory and visibility field over a CognitiveGraph."""

    def __init__(
        self,
        graph: CognitiveGraph | None = None,
        default_bounds: int | tuple[int, int] = 16,
    ) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        if isinstance(default_bounds, tuple):
            self.width, self.height = default_bounds
        else:
            self.width = self.height = default_bounds

        self.visited_cells: set[tuple[int, int]] = set()
        self.seen_cells: set[tuple[int, int]] = set()

    def reset(self) -> None:
        """Clear exploration history."""
        self.visited_cells.clear()
        self.seen_cells.clear()

    def mark_visited(self, pos: tuple[int, int]) -> None:
        """Record an agent step onto an allocentric grid position."""
        self.visited_cells.add(pos)

    def mark_seen(self, cells: Iterable[tuple[int, int]]) -> None:
        """Record observed grid coordinates."""
        self.seen_cells.update(cells)

    def sync_to_agent_node(self, agent_id: str = "agent_primary") -> None:
        """Persist visited and seen sets onto the agent's PhysicalEntityNode."""
        if self.graph.has_node(agent_id):
            node = self.graph.get_node(agent_id)
            if isinstance(node, PhysicalEntityNode):
                node.properties["visited_cells"] = set(self.visited_cells)
                node.properties["seen_cells"] = set(self.seen_cells)

    def sync_from_agent_node(self, agent_id: str = "agent_primary") -> None:
        """Restore visited and seen sets from the agent's PhysicalEntityNode."""
        if self.graph.has_node(agent_id):
            node = self.graph.get_node(agent_id)
            if isinstance(node, PhysicalEntityNode):
                self.visited_cells.update(node.properties.get("visited_cells", []))
                self.seen_cells.update(node.properties.get("seen_cells", []))

    def get_bounds(self) -> tuple[int, int]:
        """Compute the spatial bounding box containing default size and all observed entities."""
        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in self.graph.all_nodes()
            if isinstance(n, PhysicalEntityNode) and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])
        return max_x, max_y

    def partition_cells(
        self,
        ignored_entity_ids: set[str] | None = None,
        carrying_key: str | None = None,
    ) -> SpatialCellPartition:
        """Categorize all known coordinates in the CognitiveGraph into functional partitions."""
        max_x, max_y = self.get_bounds()
        partition = SpatialCellPartition(max_x=max_x, max_y=max_y)

        for node in self.graph.all_nodes():
            if (
                not isinstance(node, PhysicalEntityNode)
                or node.entity_lifecycle == EntityLifecycle.FORGOTTEN
                or node.entity_type == "agent"
            ):
                continue

            if ignored_entity_ids and node.id in ignored_entity_ids:
                continue

            coords = node.properties.get("coords")
            if not coords:
                continue

            partition.entities_by_coord[coords] = node
            partition.occupied_coords.add(coords)

            if node.properties.get("is_door"):
                door_state = node.properties.get("state")
                door_col = node.properties.get("color")
                if door_state == "open":
                    partition.open_doors[coords] = node
                elif door_state == "locked":
                    if carrying_key is not None and (carrying_key == door_col or door_col is None):
                        partition.closed_doors.add(coords)
                    else:
                        partition.walls.add(coords)
                else:  # closed
                    partition.closed_doors.add(coords)
            elif node.properties.get("pickupable", False):
                if (
                    node.entity_type == "key"
                    and carrying_key is not None
                    and (node.properties.get("color") == carrying_key or carrying_key is None)
                ):
                    pass
                else:
                    partition.movable_obstacles[coords] = node
            elif not node.properties.get("passable", False):
                partition.walls.add(coords)

        return partition

    def find_frontiers(
        self,
        agent_pos: tuple[int, int],
        ignored_entity_ids: set[str] | None = None,
        visited_cells: Iterable[tuple[int, int]] | None = None,
    ) -> tuple[tuple[int, int], PhysicalEntityNode | None] | None:
        """Find an untraversed open portal or nearest passable cell bordering unobserved space.

        Hierarchy:
        1. Open portals/doors leading into unvisited cells, sorted by proximity to agent.
        2. Open door cells themselves that haven't been stepped on.
        3. BFS for nearest passable cell bordering unseen space.
        """
        self.sync_from_agent_node()
        visited = set(self.visited_cells)
        if visited_cells:
            visited.update(visited_cells)
        visited.add(agent_pos)
        seen = set(self.seen_cells)

        partition = self.partition_cells(ignored_entity_ids=ignored_entity_ids)
        max_x, max_y = partition.max_x, partition.max_y
        blocking = partition.blocking_cells

        # Sort open doors by Manhattan proximity to agent
        sorted_doors = sorted(
            partition.open_doors.items(),
            key=lambda item: abs(item[0][0] - agent_pos[0]) + abs(item[0][1] - agent_pos[1]),
        )

        # Priority 1: Open doors leading into unvisited cells (closest first)
        for dc, door_node in sorted_doors:
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nc = (dc[0] + ddx, dc[1] + ddy)
                if (
                    0 < nc[0] < max_x - 1
                    and 0 < nc[1] < max_y - 1
                    and nc not in visited
                    and nc not in blocking
                    and nc not in partition.closed_doors
                ):
                    return (nc, door_node)

        # Priority 2: Open door cells themselves that haven't been stepped on
        for dc, door_node in sorted_doors:
            if dc not in visited:
                return (dc, door_node)

        # Priority 3: BFS for nearest passable cell bordering unseen space
        queue = collections.deque([agent_pos])
        bfs_visited = {agent_pos}
        while queue:
            cx, cy = queue.popleft()
            borders_unseen = False
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    0 < nx < max_x - 1
                    and 0 < ny < max_y - 1
                    and (nx, ny) not in blocking
                    and (nx, ny) not in seen
                ):
                    borders_unseen = True
                    break

            if borders_unseen and (cx, cy) != agent_pos:
                return ((cx, cy), None)

            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    0 < nx < max_x - 1
                    and 0 < ny < max_y - 1
                    and (nx, ny) not in bfs_visited
                    and (nx, ny) not in blocking
                    and (nx, ny) not in partition.closed_doors
                ):
                    bfs_visited.add((nx, ny))
                    queue.append((nx, ny))

        return None
