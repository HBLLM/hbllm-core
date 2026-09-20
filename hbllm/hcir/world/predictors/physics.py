"""
Physics Predictor — Deterministic Physics Differential & State Transition Calculator.

Supports:
1. Continuous thermodynamics & pressure differentials (scalar variables).
2. Discrete 2D spatial kinematics, obstacle collisions, push mechanics,
   geodesic path metrics, and corner deadlock detection.
"""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from typing import Any

import numpy as np

from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class PhysicsPredictor:
    """Physics-based forward state transition predictor."""

    name: str = "physics"

    def predict_state(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
    ) -> tuple[dict[str, Any], float]:
        """Compute physics forward state transition and return (predicted_variables, confidence)."""
        predicted = dict(snapshot.variables)
        time_sec = horizon_ms / 1000.0

        # 1. Scalar variables (thermodynamics / pressure) for backward compatibility
        for var_name, var_value in snapshot.variables.items():
            if isinstance(var_value, (int, float)):
                if "temp" in var_name.lower():
                    if "cool" in action_intent.lower() or "reduce" in action_intent.lower():
                        predicted[var_name] = max(20.0, var_value - 0.1 * time_sec)
                    else:
                        predicted[var_name] = var_value + 0.05 * time_sec
                elif "pressure" in var_name.lower():
                    if "vent" in action_intent.lower():
                        predicted[var_name] = max(1.0, var_value - 0.02 * time_sec)

        # 2. Spatial Kinematics Simulation (if spatial entities or grid are present)
        spatial_entities = snapshot.variables.get("spatial_entities") or snapshot.variables.get(
            "entities"
        )
        if spatial_entities and isinstance(spatial_entities, dict):
            spatial_outcome, confidence = self.simulate_spatial_kinematics(
                spatial_entities=spatial_entities,
                variables=snapshot.variables,
                action_intent=action_intent,
            )
            predicted["spatial_outcome"] = spatial_outcome
            # Update spatial_entities dict with updated positions
            updated_entities = {
                k: dict(v) if isinstance(v, dict) else v for k, v in spatial_entities.items()
            }
            for eid, e_upd in spatial_outcome.get("updated_entities", {}).items():
                if eid in updated_entities and isinstance(updated_entities[eid], dict):
                    updated_entities[eid].update(e_upd)
            predicted["spatial_entities"] = updated_entities
            return predicted, confidence

        logger.debug(
            "PhysicsPredictor calculated state for action '%s' confidence=0.92", action_intent
        )
        return predicted, 0.92

    # ─────────────────────────────────────────────────────────────────────────
    # Spatial Kinematics & Physical Mechanics
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def parse_action_displacement(
        cls,
        action_intent: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        """Extract displacement vector (dr, dc) from action intent string or variables."""
        if variables:
            if "delta_r" in variables and "delta_c" in variables:
                return int(variables["delta_r"]), int(variables["delta_c"])
            if "dr" in variables and "dc" in variables:
                return int(variables["dr"]), int(variables["dc"])

        intent_lower = action_intent.lower()

        # Check explicit dr=..., dc=...
        m_dr = re.search(r"dr\s*=\s*([+-]?\d+)", action_intent, re.IGNORECASE)
        m_dc = re.search(r"dc\s*=\s*([+-]?\d+)", action_intent, re.IGNORECASE)
        if m_dr and m_dc:
            return int(m_dr.group(1)), int(m_dc.group(1))

        # Check tuple (dr, dc)
        m_tuple = re.search(r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)", action_intent)
        if m_tuple:
            return int(m_tuple.group(1)), int(m_tuple.group(2))

        # Directional semantic keywords
        if "up" in intent_lower or "north" in intent_lower:
            return -1, 0
        if "down" in intent_lower or "south" in intent_lower:
            return 1, 0
        if "left" in intent_lower or "west" in intent_lower:
            return 0, -1
        if "right" in intent_lower or "east" in intent_lower:
            return 0, 1

        return 0, 0

    @classmethod
    def simulate_spatial_kinematics(
        cls,
        spatial_entities: dict[str, dict[str, Any]],
        variables: dict[str, Any],
        action_intent: str,
    ) -> tuple[dict[str, Any], float]:
        """Deterministic simulation of 2D translation, barrier collisions, and push propagation."""
        dr, dc = cls.parse_action_displacement(action_intent, variables)
        grid_shape = tuple(variables.get("grid_shape", (64, 64)))
        H, W = int(grid_shape[0]), int(grid_shape[1])

        # 1. Parse barriers & target zones
        barrier_cells: set[tuple[int, int]] = set()
        raw_barriers = variables.get("barrier_cells")
        if raw_barriers:
            for b in raw_barriers:
                barrier_cells.add((int(b[0]), int(b[1])))

        target_positions: set[tuple[int, int]] = set()
        raw_targets = variables.get("target_positions")
        if raw_targets:
            for t in raw_targets:
                target_positions.add((int(t[0]), int(t[1])))

        # Extract entities
        avatar_id: str | None = None
        avatar_pos: tuple[int, int] | None = None
        pushable_entities: dict[str, tuple[int, int]] = {}
        target_entities: dict[str, tuple[int, int]] = {}

        for eid, edata in spatial_entities.items():
            pos = edata.get("position")
            if pos is None:
                continue
            r_c = (int(pos[0]), int(pos[1]))
            is_avatar = (
                edata.get("is_avatar", False)
                or edata.get("entity_type") == "agent"
                or eid == "avatar"
                or "avatar" in edata.get("name", "").lower()
            )
            is_movable = edata.get("movable", True)
            is_passable = edata.get("passable", False)
            affordances = edata.get("affordances", [])

            if is_avatar:
                avatar_id = eid
                avatar_pos = r_c
            elif "PUSHABLE" in affordances or (is_movable and not is_passable):
                pushable_entities[eid] = r_c
            elif is_passable or edata.get("is_goal", False) or "goal" in eid.lower():
                target_entities[eid] = r_c
                target_positions.add(r_c)
            elif not is_movable and not is_passable:
                barrier_cells.add(r_c)

        if avatar_pos is None:
            # Fallback if no avatar designated
            return {
                "success": False,
                "collision": False,
                "deadlock": False,
                "reason": "no_controllable_avatar",
            }, 0.50

        # 2. Compute motion differential
        curr_r, curr_c = avatar_pos
        next_r = curr_r + dr
        next_c = curr_c + dc

        collision = False
        collision_type: str | None = None
        deadlock = False
        deadlocked_entities: list[str] = []
        pushed_entities: list[str] = []
        updated_entities: dict[str, dict[str, Any]] = {}

        # Check boundary collision
        if not (0 <= next_r < H and 0 <= next_c < W):
            collision = True
            collision_type = "BOUNDARY_COLLISION"
            simulated_avatar_pos = (curr_r, curr_c)
        elif (next_r, next_c) in barrier_cells:
            collision = True
            collision_type = "BARRIER_COLLISION"
            simulated_avatar_pos = (curr_r, curr_c)
        else:
            # Check if stepping into a pushable object
            pushed_box_id: str | None = None
            for p_id, p_pos in pushable_entities.items():
                if p_pos == (next_r, next_c):
                    pushed_box_id = p_id
                    break

            if pushed_box_id is not None:
                box_next_r = next_r + dr
                box_next_c = next_c + dc
                # Check if box destination is clear
                box_blocked = False
                if not (0 <= box_next_r < H and 0 <= box_next_c < W):
                    box_blocked = True
                elif (box_next_r, box_next_c) in barrier_cells:
                    box_blocked = True
                elif any(
                    other_pos == (box_next_r, box_next_c)
                    for other_id, other_pos in pushable_entities.items()
                    if other_id != pushed_box_id
                ):
                    box_blocked = True

                if box_blocked:
                    collision = True
                    collision_type = "PUSH_OBSTRUCTED"
                    simulated_avatar_pos = (curr_r, curr_c)
                else:
                    # Successful tandem push
                    simulated_avatar_pos = (next_r, next_c)
                    updated_entities[pushed_box_id] = {"position": (box_next_r, box_next_c)}
                    pushed_entities.append(pushed_box_id)

                    # Check corner deadlock on pushed object
                    if cls.is_corner_deadlock(
                        box_pos=(box_next_r, box_next_c),
                        barrier_cells=barrier_cells,
                        target_positions=target_positions,
                        grid_shape=(H, W),
                    ):
                        deadlock = True
                        deadlocked_entities.append(pushed_box_id)
            else:
                # Clear translation
                simulated_avatar_pos = (next_r, next_c)

        if avatar_id is not None:
            updated_entities[avatar_id] = {"position": simulated_avatar_pos}

        # 3. Calculate Goal Progress & Metric
        initial_goal_dist = float("inf")
        simulated_goal_dist = float("inf")
        reference_pos_curr = curr_r, curr_c
        reference_pos_sim = simulated_avatar_pos

        # If there are pushable objects, track distance of the pushable object to target
        if pushable_entities and pushed_entities:
            pid = pushed_entities[0]
            reference_pos_curr = pushable_entities[pid]
            reference_pos_sim = updated_entities[pid]["position"]

        if target_positions:
            for t_r, t_c in target_positions:
                d_init = math.hypot(t_r - reference_pos_curr[0], t_c - reference_pos_curr[1])
                d_sim = math.hypot(t_r - reference_pos_sim[0], t_c - reference_pos_sim[1])
                if d_init < initial_goal_dist:
                    initial_goal_dist = d_init
                if d_sim < simulated_goal_dist:
                    simulated_goal_dist = d_sim

        progress = (
            (initial_goal_dist - simulated_goal_dist)
            if initial_goal_dist != float("inf") and simulated_goal_dist != float("inf")
            else 0.0
        )
        goal_reached = simulated_goal_dist == 0.0

        # 4. Epistemic Confidence Calibration
        if deadlock:
            confidence = 0.05
        elif collision:
            confidence = 0.25
        elif goal_reached:
            confidence = 0.99
        elif progress > 0.0:
            confidence = 0.95
        elif progress == 0.0:
            confidence = 0.70
        else:
            confidence = 0.45

        outcome = {
            "success": not collision and not deadlock,
            "collision": collision,
            "collision_type": collision_type,
            "deadlock": deadlock,
            "deadlocked_entities": deadlocked_entities,
            "pushed_entities": pushed_entities,
            "avatar_position": simulated_avatar_pos,
            "updated_entities": updated_entities,
            "goal_reached": goal_reached,
            "goal_distance": simulated_goal_dist if simulated_goal_dist != float("inf") else -1.0,
            "progress": progress,
        }

        return outcome, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # Topological Deadlock & Geodesic Calculation
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def is_corner_deadlock(
        cls,
        box_pos: tuple[int, int],
        barrier_cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
        target_positions: set[tuple[int, int]] | frozenset[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
    ) -> bool:
        """Detect if an object is trapped in an irreversible non-target corner."""
        r, c = box_pos
        H, W = grid_shape

        if box_pos in target_positions:
            return False

        blocked_up = (r - step_size < 0) or ((r - step_size, c) in barrier_cells)
        blocked_down = (r + step_size >= H) or ((r + step_size, c) in barrier_cells)
        blocked_left = (c - step_size < 0) or ((r, c - step_size) in barrier_cells)
        blocked_right = (c + step_size >= W) or ((r, c + step_size) in barrier_cells)

        is_corner = (
            (blocked_up and blocked_left)
            or (blocked_up and blocked_right)
            or (blocked_down and blocked_left)
            or (blocked_down and blocked_right)
        )
        return is_corner

    @classmethod
    def is_line_deadlock(
        cls,
        box_pos: tuple[int, int],
        barrier_cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
        target_positions: set[tuple[int, int]] | frozenset[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
    ) -> bool:
        """Checks if a pushable entity is pressed against a continuous flat wall without targets.

        If a block is against a continuous wall, it can only be pushed parallel to that wall.
        If there is no target along that continuous wall segment, and the ends of the segment
        are blocked by barriers/corners, the block is in an irreversible line deadlock.
        """
        if box_pos in target_positions:
            return False

        H, W = grid_shape
        br, bc = box_pos

        walls: list[tuple[tuple[int, int], list[tuple[int, int]]]] = [
            ((-step_size, 0), [(0, -step_size), (0, step_size)]),
            ((step_size, 0), [(0, -step_size), (0, step_size)]),
            ((0, -step_size), [(-step_size, 0), (step_size, 0)]),
            ((0, step_size), [(-step_size, 0), (step_size, 0)]),
        ]

        for wall_delta, move_deltas in walls:
            wr, wc = br + wall_delta[0], bc + wall_delta[1]
            if (wr, wc) in barrier_cells or wr < 0 or wr >= H or wc < 0 or wc >= W:
                d1, d2 = move_deltas[0], move_deltas[1]
                targets_along_wall = False

                curr_r, curr_c = br, bc
                blocked_d1 = False
                for _ in range(max(H, W)):
                    curr_r += d1[0]
                    curr_c += d1[1]
                    if (curr_r, curr_c) in barrier_cells or not (
                        0 <= curr_r < H and 0 <= curr_c < W
                    ):
                        blocked_d1 = True
                        break
                    adj_wall = (curr_r + wall_delta[0], curr_c + wall_delta[1])
                    if adj_wall not in barrier_cells and (
                        0 <= adj_wall[0] < H and 0 <= adj_wall[1] < W
                    ):
                        break
                    if (curr_r, curr_c) in target_positions:
                        targets_along_wall = True
                        break

                curr_r, curr_c = br, bc
                blocked_d2 = False
                for _ in range(max(H, W)):
                    curr_r += d2[0]
                    curr_c += d2[1]
                    if (curr_r, curr_c) in barrier_cells or not (
                        0 <= curr_r < H and 0 <= curr_c < W
                    ):
                        blocked_d2 = True
                        break
                    adj_wall = (curr_r + wall_delta[0], curr_c + wall_delta[1])
                    if adj_wall not in barrier_cells and (
                        0 <= adj_wall[0] < H and 0 <= adj_wall[1] < W
                    ):
                        break
                    if (curr_r, curr_c) in target_positions:
                        targets_along_wall = True
                        break

                if blocked_d1 and blocked_d2 and not targets_along_wall:
                    return True

        return False

    @classmethod
    def simulate_joint_displacement(
        cls,
        avatar_pos: tuple[int, int],
        action_delta: tuple[int, int],
        movable_entities: dict[str, tuple[int, int]],
        barrier_cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> tuple[tuple[int, int], dict[str, tuple[int, int]], bool]:
        """Simulate joint displacement of avatar and pushable entities.

        Returns (new_avatar_pos, new_entity_positions, is_blocked).
        Handles multi-body cascade pushes (e.g. pushing box A which pushes box B, or blocked by barrier).
        """
        H, W = grid_shape
        dr, dc = action_delta
        new_avatar_pos = (avatar_pos[0] + dr, avatar_pos[1] + dc)

        if not (0 <= new_avatar_pos[0] < H and 0 <= new_avatar_pos[1] < W):
            return avatar_pos, movable_entities, True

        if new_avatar_pos in barrier_cells:
            return avatar_pos, movable_entities, True

        pos_to_id = {pos: eid for eid, pos in movable_entities.items()}
        if new_avatar_pos not in pos_to_id:
            return new_avatar_pos, movable_entities, False

        chain: list[str] = []
        curr_pos = new_avatar_pos
        while curr_pos in pos_to_id:
            chain.append(pos_to_id[curr_pos])
            curr_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
            if not (0 <= curr_pos[0] < H and 0 <= curr_pos[1] < W):
                return avatar_pos, movable_entities, True
            if curr_pos in barrier_cells:
                return avatar_pos, movable_entities, True

        updated_entities = dict(movable_entities)
        for eid in chain:
            old_r, old_c = updated_entities[eid]
            updated_entities[eid] = (old_r + dr, old_c + dc)

        return new_avatar_pos, updated_entities, False

    @classmethod
    def compute_geodesic_path(
        cls,
        start: tuple[int, int],
        goal: tuple[int, int],
        barrier_cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
        footprint_offsets: list[tuple[int, int]] | set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Breadth-first search for shortest obstacle-clearing geodesic path, accounting for composite footprints."""
        H, W = grid_shape
        start_r, start_c = start
        goal_r, goal_c = goal

        if start == goal:
            return [start]

        effective_barriers = set(barrier_cells)
        if (goal_r, goal_c) in effective_barriers:
            effective_barriers.remove((goal_r, goal_c))

        offsets: list[tuple[int, int]] = list(footprint_offsets) if footprint_offsets else [(0, 0)]

        def is_footprint_valid(r: int, c: int) -> bool:
            for off_r, off_c in offsets:
                fr = r + off_r
                fc = c + off_c
                if not (0 <= fr < H and 0 <= fc < W):
                    return False
                if (fr, fc) in effective_barriers:
                    return False
            return True

        queue: deque[tuple[int, int, list[tuple[int, int]]]] = deque(
            [(start_r, start_c, [(start_r, start_c)])]
        )
        visited = {(start_r, start_c)}
        delta = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]

        best_partial_path = [(start_r, start_c)]
        min_dist_to_goal = math.hypot(goal_r - start_r, goal_c - start_c)
        iters = 0
        max_iterations = 2500

        while queue and iters < max_iterations:
            iters += 1
            r, c, path = queue.popleft()

            dist = math.hypot(goal_r - r, goal_c - c)
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                best_partial_path = path

            if math.hypot(goal_r - r, goal_c - c) <= (step_size * 0.9):
                if is_footprint_valid(goal_r, goal_c):
                    return path + [(goal_r, goal_c)]

            for dr, dc in delta:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    if is_footprint_valid(nr, nc):
                        queue.append((nr, nc, path + [(nr, nc)]))

        return best_partial_path

    @staticmethod
    def find_solitaire_jump_sequence(
        pegs: set[tuple[int, int]],
        valid_holes: set[tuple[int, int]],
        step_delta: int = 1,
        target_peg_count: int = 1,
        target_final_pos: tuple[int, int] | None = None,
        max_states: int = 50000,
    ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] | None:
        """Finds a sequence of jumps to reduce solitaire board pegs via graph search.

        A jump is valid if:
        - peg at from_pos (r, c)
        - peg at mid_pos (r + dr*step_delta, c + dc*step_delta)
        - empty hole at to_pos (r + 2*dr*step_delta, c + 2*dc*step_delta) in valid_holes
        Returns a list of (from_pos, mid_pos, to_pos) jumps, or None if no sequence found.
        """
        initial_state = frozenset(pegs)
        visited: set[frozenset[tuple[int, int]]] = set()
        states_explored = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def dfs(
            current_state: frozenset[tuple[int, int]],
            path: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
        ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] | None:
            nonlocal states_explored
            states_explored += 1
            if states_explored > max_states:
                return None

            if len(current_state) <= target_peg_count:
                if target_final_pos is None or target_final_pos in current_state:
                    return path

            if current_state in visited:
                return None
            visited.add(current_state)

            # Generate all legal jumps from current_state
            candidate_jumps = []
            for p in current_state:
                r, c = p
                for dr, dc in directions:
                    mid = (r + dr * step_delta, c + dc * step_delta)
                    dest = (r + 2 * dr * step_delta, c + 2 * dc * step_delta)
                    if mid in current_state and dest in valid_holes and dest not in current_state:
                        candidate_jumps.append((p, mid, dest))

            # Heuristic sort: prefer jumps that clear peripheral pegs or converge toward center
            if target_final_pos is not None:
                tr, tc = target_final_pos
                candidate_jumps.sort(key=lambda j: math.hypot(j[2][0] - tr, j[2][1] - tc))

            for jump in candidate_jumps:
                frm, mid, to_pos = jump
                next_state = (current_state - {frm, mid}) | {to_pos}
                res = dfs(next_state, path + [jump])
                if res is not None:
                    return res

            return None

        return dfs(initial_state, [])

    @staticmethod
    def simulate_telescopic_step(
        base_pos: tuple[int, int],
        length: int,
        axis: tuple[int, int],
        action_dir: tuple[int, int],
        min_length: int = 1,
        max_length: int = 10,
        rails: set[tuple[int, int]] | None = None,
        barriers: set[tuple[int, int]] | None = None,
    ) -> tuple[tuple[int, int], int]:
        """Simulate telescopic piston extension/retraction or rail translation.

        - If action_dir == axis: extend length by 1 (if within bounds and unblocked).
        - If action_dir == (-axis[0], -axis[1]): retract length by 1 (if >= min_length).
        - If action_dir is orthogonal to axis: translate base_pos along rails if supported.
        Returns: (new_base_pos, new_length).
        """
        all_barriers = barriers or set()

        if action_dir == axis:
            # Extension along axis
            new_tip = (
                base_pos[0] + (length + 1) * axis[0],
                base_pos[1] + (length + 1) * axis[1],
            )
            if length < max_length and new_tip not in all_barriers:
                return base_pos, length + 1
            return base_pos, length

        if action_dir == (-axis[0], -axis[1]):
            # Retraction
            if length > min_length:
                return base_pos, length - 1
            return base_pos, length

        # Orthogonal movement: check rail support across full piston body
        new_base = (base_pos[0] + action_dir[0], base_pos[1] + action_dir[1])
        if rails is not None and new_base not in rails:
            return base_pos, length

        # Check that translating the entire body does not collide with barriers
        for i in range(length + 1):
            cell = (new_base[0] + i * axis[0], new_base[1] + i * axis[1])
            if cell in all_barriers:
                return base_pos, length

        return new_base, length

    @staticmethod
    def find_lattice_track_path(
        grid: np.ndarray,
        start_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        track_color: int = 2,
        stride: int = 6,
        patch_size: int = 3,
    ) -> list[int] | None:
        """Find optimal action sequence to navigate a rail/track network on a discrete 2D lattice.

        Performs breadth-first search across lattice intersections spaced by `stride`,
        verifying that intermediate conduit/rail segments contain `track_color`.

        Args:
            grid: 2D numpy array representing the environment visual frame.
            start_pos: (start_r, start_c) coordinates on the grid.
            goal_pos: (goal_r, goal_c) target coordinates on the grid.
            track_color: Pixel color identifying valid track/conduit bridges.
            stride: Distance in pixels between adjacent lattice intersections (default 6).
            patch_size: Spatial dimensions of the conduit patch (default 3).

        Returns:
            List of action integers (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT) or None if unreachable.
        """
        H, W = grid.shape
        queue = deque([(start_pos[0], start_pos[1], [])])
        visited = {(start_pos[0], start_pos[1])}
        moves = [
            (-stride, 0, -stride // 2, 0, 1),  # UP
            (stride, 0, stride // 2, 0, 2),  # DOWN
            (0, -stride, 0, -stride // 2, 3),  # LEFT
            (0, stride, 0, stride // 2, 4),  # RIGHT
        ]
        while queue:
            cr, cc, path = queue.popleft()
            if (cr, cc) == (goal_pos[0], goal_pos[1]):
                return path
            for dr, dc, br_dr, br_dc, act in moves:
                nr, nc = cr + dr, cc + dc
                br_r, br_c = cr + br_dr, cc + br_dc
                if 0 <= br_r < H and 0 <= br_c < W:
                    patch = grid[br_r : br_r + patch_size, br_c : br_c + patch_size]
                    if np.any(patch == track_color):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc, path + [act]))
        return None
