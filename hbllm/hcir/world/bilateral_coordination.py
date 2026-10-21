"""Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring).

Modeled on the mammalian corpus callosum and Supplementary Motor Area (SMA)
for bimanual effector coordination and bilateral spatial integration:
1. Bilateral Agent Detection: Identifies paired controllable entities moving synchronously
   or reflectionally across visual field axes (vertical, horizontal, or anti-phase).
2. Joint State Tracking: Maintains coupled bilateral state (p1, p2) in working memory.
3. Callosal Convergence Engine: Conducts joint forward search over (p1, p2) space,
   utilizing selective asymmetric barrier collisions (wall slip) to minimize bilateral
   distance min d(p1, p2) or dock both agents simultaneously onto complementary target zones.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.cortex_causal import ActionAffordance

logger = logging.getLogger(__name__)


class SymmetryAxis(StrEnum):
    """Symmetry reflection axis connecting bilateral effectors."""

    VERTICAL = "vertical"  # Reflection across vertical midline: dc2 = -dc1, dr2 = dr1
    HORIZONTAL = "horizontal"  # Reflection across horizontal midline: dr2 = -dr1, dc2 = dc1
    POINT = "point"  # Anti-phase / 180° rotation: dr2 = -dr1, dc2 = -dc1
    PARALLEL = "parallel"  # Synchronous parallel dual-agents: dr2 = dr1, dc2 = dc1
    QUAD = "quad"  # 4-way D4 reflection: p0:(dr, dc), p1:(dr, -dc), p2:(-dr, dc), p3:(-dr, -dc)


@dataclass
class BilateralState:
    """Coupled spatial state of bilateral controllable entities."""

    pos1: tuple[int, int]
    pos2: tuple[int, int]
    symmetry_axis: SymmetryAxis
    features: set[int] = field(default_factory=set)
    confidence: float = 1.0
    quad_positions: list[tuple[int, int]] = field(default_factory=list)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export bilateral state as a declarative causal predicate."""
        return CausalPredicate(
            variable="bilateral_symmetry",
            operator="==",
            value=self.symmetry_axis.value,
            relation=f"{self.pos1}->{self.pos2}",
            target_entity="dual_agent" if self.symmetry_axis != SymmetryAxis.QUAD else "quad_agent",
        )

    def to_affordance(self, action_id: Any = "STEP") -> ActionAffordance:
        """Synthesize coordinated bilateral action affordance."""
        from hbllm.hcir.world.cortex_causal import ActionAffordance

        return ActionAffordance(
            action_id=action_id,
            name=f"bilateral_{self.symmetry_axis.value}",
            confidence=self.confidence,
            metadata={
                "pos1": self.pos1,
                "pos2": self.pos2,
                "symmetry_axis": self.symmetry_axis.value,
                "quad_positions": self.quad_positions,
            },
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording coupled bilateral state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="bilateral_effector_state",
                node_data={
                    "pos1": self.pos1,
                    "pos2": self.pos2,
                    "symmetry_axis": self.symmetry_axis.value,
                    "features": sorted(list(self.features)),
                    "confidence": self.confidence,
                    "quad_positions": self.quad_positions,
                },
            )
        ]


class BilateralCoordinateIntegrator:
    """Corpus callosum bilateral coordinate integrator and joint planner."""

    def __init__(self) -> None:
        self.active_bilateral_state: BilateralState | None = None
        self.convergence_achieved: bool = False

    def detect_bilateral_pairing(
        self,
        moved_entities: list[
            tuple[tuple[int, int], tuple[int, int], int]
        ],  # (centroid, delta, feat)
        grid_shape: tuple[int, int],
    ) -> BilateralState | None:
        """Examine moved entity displacements to detect bilateral symmetry pairing.

        Args:
            moved_entities: List of (pos, (dr, dc), feature_id) for entities displaced in frame.
            grid_shape: (H, W) visual frame dimensions.

        Returns:
            BilateralState if a clear bilateral pair is identified, else None.
        """
        if len(moved_entities) < 2:
            return None

        H, W = grid_shape

        if len(moved_entities) >= 4:
            cand_quads = [e for e in moved_entities if e[1] != (0, 0)]
            if len(cand_quads) >= 4:
                import itertools

                for four_combo in itertools.combinations(cand_quads[:8], 4):
                    positions = [e[0] for e in four_combo]
                    if all(
                        abs(positions[a][0] - positions[b][0])
                        + abs(positions[a][1] - positions[b][1])
                        > 3
                        for a in range(4)
                        for b in range(a + 1, 4)
                    ):
                        deltas = [e[1] for e in four_combo]
                        is_quad = False
                        d0 = deltas[0]
                        dr, dc = d0
                        if dr == 0 and dc != 0:
                            pos_c = sum(1 for d in deltas if d[1] == dc)
                            neg_c = sum(1 for d in deltas if d[1] == -dc)
                            if pos_c == 2 and neg_c == 2:
                                is_quad = True
                        elif dc == 0 and dr != 0:
                            pos_r = sum(1 for d in deltas if d[0] == dr)
                            neg_r = sum(1 for d in deltas if d[0] == -dr)
                            if pos_r == 2 and neg_r == 2:
                                is_quad = True
                        elif dr != 0 and dc != 0:
                            expected = {(dr, dc), (dr, -dc), (-dr, dc), (-dr, -dc)}
                            if set(deltas) == expected:
                                is_quad = True

                        if is_quad:
                            sorted_by_r = sorted(positions, key=lambda p: (p[0], p[1]))
                            top_two = sorted(sorted_by_r[:2], key=lambda p: p[1])
                            bot_two = sorted(sorted_by_r[2:], key=lambda p: p[1])
                            ordered_pos = [top_two[0], top_two[1], bot_two[0], bot_two[1]]
                            features = {e[2] for e in four_combo}
                            state = BilateralState(
                                pos1=ordered_pos[0],
                                pos2=ordered_pos[1],
                                symmetry_axis=SymmetryAxis.QUAD,
                                features=features,
                                confidence=0.90,
                                quad_positions=ordered_pos,
                            )
                            self.active_bilateral_state = state
                            logger.info(
                                "BilateralCoordinateIntegrator: Detected Quad-Lateral 4-agent group at %s",
                                ordered_pos,
                            )
                            return state

        # Test all pairs of displaced entities
        for i in range(len(moved_entities)):
            for j in range(i + 1, len(moved_entities)):
                pos1, delta1, f1 = moved_entities[i]
                pos2, delta2, f2 = moved_entities[j]

                dr1, dc1 = delta1
                dr2, dc2 = delta2

                # Two components of the same sprite or adjacent cells cannot be bilateral agents
                if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) <= 2:
                    continue

                if (dr1, dc1) == (0, 0) and (dr2, dc2) == (0, 0):
                    continue

                axis: SymmetryAxis | None = None

                # Vertical reflection: dr2 == dr1 and dc2 == -dc1
                if dr1 == dr2 and dc1 == -dc2 and (dc1 != 0 or dr1 != 0):
                    axis = SymmetryAxis.VERTICAL
                # Horizontal reflection: dr2 == -dr1 and dc2 == dc1
                elif dr1 == -dr2 and dc1 == dc2 and (dr1 != 0 or dc1 != 0):
                    axis = SymmetryAxis.HORIZONTAL
                # Point reflection: dr2 == -dr1 and dc2 == -dc1
                elif dr1 == -dr2 and dc1 == -dc2 and (dr1 != 0 or dc1 != 0):
                    axis = SymmetryAxis.POINT
                # Parallel dual-agent: dr1 == dr2 and dc1 == dc2
                elif dr1 == dr2 and dc1 == dc2 and (dr1 != 0 or dc1 != 0):
                    axis = SymmetryAxis.PARALLEL

                if axis is not None:
                    state = BilateralState(
                        pos1=pos1,
                        pos2=pos2,
                        symmetry_axis=axis,
                        features={f1, f2},
                        confidence=0.85,
                    )
                    self.active_bilateral_state = state
                    logger.info(
                        "BilateralCoordinateIntegrator: Detected bilateral pair at %s and %s (axis=%s)",
                        pos1,
                        pos2,
                        axis.value,
                    )
                    return state

        return None

    def transform_action_for_agent2(
        self,
        delta: tuple[int, int],
        axis: SymmetryAxis,
    ) -> tuple[int, int]:
        """Compute the coupled displacement delta for the mirrored second agent."""
        dr, dc = delta
        if axis in (SymmetryAxis.VERTICAL, SymmetryAxis.QUAD):
            return (dr, -dc)
        elif axis == SymmetryAxis.HORIZONTAL:
            return (-dr, dc)
        elif axis == SymmetryAxis.POINT:
            return (-dr, -dc)
        elif axis == SymmetryAxis.PARALLEL:
            return (dr, dc)
        return (dr, dc)

    def transform_actions_for_quad(
        self,
        delta: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Compute 4-way coupled displacement deltas for D4 quad reflection."""
        dr, dc = delta
        return [(dr, dc), (dr, -dc), (-dr, dc), (-dr, -dc)]

    def plan_convergence_sequence(
        self,
        pos1: tuple[int, int],
        pos2: tuple[int, int],
        axis: SymmetryAxis,
        passable_mask: np.ndarray,
        action_deltas: dict[Any, tuple[int, int]],
        target_positions: list[tuple[int, int]] | None = None,
        max_search_depth: int = 1500,
    ) -> list[Any] | None:
        """Search for a joint action sequence that brings dual agents into convergence.

        In bimanual puzzle dynamics, one agent can be walked into a barrier (slipping)
        while the other moves freely, allowing asymmetric adjustments until both agents meet.

        Args:
            pos1: Coordinates of agent 1 (r, c).
            pos2: Coordinates of agent 2 (r, c).
            axis: Symmetry axis relation.
            passable_mask: 2D boolean array of walkable coordinates.
            action_deltas: Mapping of action_id -> (dr, dc) for agent 1.
            target_positions: Optional specific goal coordinates for agents to dock into.
            max_search_depth: Maximum nodes explored in joint state space.

        Returns:
            List of actions leading to convergence, or None if no path found.
        """
        H, W = passable_mask.shape

        def _is_walkable(r: int, c: int) -> bool:
            return 0 <= r < H and 0 <= c < W and bool(passable_mask[r, c])

        if axis == SymmetryAxis.QUAD:
            quad_start = (
                tuple(self.active_bilateral_state.quad_positions)
                if (
                    self.active_bilateral_state
                    and len(self.active_bilateral_state.quad_positions) == 4
                )
                else (pos1, pos2, (H - 1 - pos1[0], pos1[1]), (H - 1 - pos2[0], pos2[1]))
            )

            def _step_quad(
                pts: tuple[tuple[int, int], ...],
                d: tuple[int, int],
            ) -> tuple[tuple[int, int], ...]:
                agent_deltas = self.transform_actions_for_quad(d)
                next_pts = []
                for p, ad in zip(pts, agent_deltas):
                    np_pos = (p[0] + ad[0], p[1] + ad[1])
                    next_pts.append(np_pos if _is_walkable(np_pos[0], np_pos[1]) else p)
                return tuple(next_pts)

            def _quad_heuristic(pts: tuple[tuple[int, int], ...]) -> int:
                if target_positions:
                    return sum(
                        min(abs(p[0] - tg[0]) + abs(p[1] - tg[1]) for tg in target_positions)
                        for p in pts
                    )
                cr = sum(p[0] for p in pts) // len(pts)
                cc = sum(p[1] for p in pts) // len(pts)
                return sum(abs(p[0] - cr) + abs(p[1] - cc) for p in pts)

            initial_h = _quad_heuristic(quad_start)
            frontier_quad: list[tuple[int, int, tuple[tuple[int, int], ...], list[Any]]] = [
                (initial_h, 0, quad_start, [])
            ]
            visited_quad: set[tuple[tuple[int, int], ...]] = {quad_start}
            explored = 0

            while frontier_quad and explored < max_search_depth:
                explored += 1
                f, g, curr_quad, path = heapq.heappop(frontier_quad)

                if target_positions:
                    if all(p in target_positions for p in curr_quad):
                        return path
                else:
                    cr = sum(p[0] for p in curr_quad) // 4
                    cc = sum(p[1] for p in curr_quad) // 4
                    if all(abs(p[0] - cr) + abs(p[1] - cc) <= 1 for p in curr_quad):
                        return path

                for act, delta in action_deltas.items():
                    next_quad = _step_quad(curr_quad, delta)
                    if next_quad == curr_quad:
                        continue
                    if next_quad not in visited_quad:
                        visited_quad.add(next_quad)
                        next_g = g + 1
                        next_h = _quad_heuristic(next_quad)
                        heapq.heappush(
                            frontier_quad,
                            (next_g + next_h, next_g, next_quad, path + [act]),
                        )
            return None

        def _step_joint(
            p1: tuple[int, int],
            p2: tuple[int, int],
            d1: tuple[int, int],
        ) -> tuple[tuple[int, int], tuple[int, int]]:
            d2 = self.transform_action_for_agent2(d1, axis)
            np1 = (p1[0] + d1[0], p1[1] + d1[1])
            np2 = (p2[0] + d2[0], p2[1] + d2[1])

            # Wall slip: if next cell is blocked, agent stays in place
            res1 = np1 if _is_walkable(np1[0], np1[1]) else p1
            res2 = np2 if _is_walkable(np2[0], np2[1]) else p2
            return res1, res2

        def _heuristic(p1: tuple[int, int], p2: tuple[int, int]) -> int:
            if target_positions:
                if len(target_positions) >= 2:
                    t_list = list(target_positions)
                    best_sum = float("inf")
                    for i in range(len(t_list)):
                        for j in range(len(t_list)):
                            if i == j:
                                continue
                            t1, t2 = t_list[i], t_list[j]
                            d = (
                                abs(p1[0] - t1[0])
                                + abs(p1[1] - t1[1])
                                + abs(p2[0] - t2[0])
                                + abs(p2[1] - t2[1])
                            )
                            if d < best_sum:
                                best_sum = d
                    return int(best_sum)
                else:
                    tg = next(iter(target_positions))
                    return (
                        abs(p1[0] - tg[0])
                        + abs(p1[1] - tg[1])
                        + abs(p2[0] - tg[0])
                        + abs(p2[1] - tg[1])
                    )
            # Mutual convergence: Manhattan distance between agents
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Priority queue for A* joint search: (f_score, g_score, (p1, p2), path)
        start_state = (pos1, pos2)
        initial_h = _heuristic(pos1, pos2)
        frontier: list[tuple[int, int, tuple[tuple[int, int], tuple[int, int]], list[Any]]] = [
            (initial_h, 0, start_state, [])
        ]
        visited: set[tuple[tuple[int, int], tuple[int, int]]] = {start_state}

        explored = 0
        while frontier and explored < max_search_depth:
            explored += 1
            f, g, (curr1, curr2), path = heapq.heappop(frontier)

            # Check convergence goal condition
            if target_positions:
                if len(target_positions) >= 2:
                    if curr1 in target_positions and curr2 in target_positions and curr1 != curr2:
                        return path
                else:
                    if curr1 in target_positions and curr2 in target_positions:
                        return path
            else:
                dist = abs(curr1[0] - curr2[0]) + abs(curr1[1] - curr2[1])
                if dist <= 1:
                    return path

            for act, delta in action_deltas.items():
                next1, next2 = _step_joint(curr1, curr2, delta)
                if next1 == curr1 and next2 == curr2:
                    continue  # Both blocked, no progress

                next_state = (next1, next2)
                if next_state not in visited:
                    visited.add(next_state)
                    next_g = g + 1
                    next_h = _heuristic(next1, next2)
                    heapq.heappush(frontier, (next_g + next_h, next_g, next_state, path + [act]))

        return None

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export active bilateral state as declarative causal predicates."""
        if not self.active_bilateral_state:
            return []
        return [self.active_bilateral_state.to_causal_predicate()]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting bilateral integrator state."""
        if not self.active_bilateral_state:
            return []
        return self.active_bilateral_state.to_transaction_operations()
