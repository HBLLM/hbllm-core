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
        barrier_cells: set[tuple[int, int]],
        target_positions: set[tuple[int, int]],
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
    def compute_geodesic_path(
        cls,
        start: tuple[int, int],
        goal: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
    ) -> list[tuple[int, int]]:
        """Breadth-first search for shortest obstacle-clearing geodesic path."""
        H, W = grid_shape
        start_r, start_c = start
        goal_r, goal_c = goal

        if start == goal:
            return [start]

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
                return path + [(goal_r, goal_c)]

            for dr, dc in delta:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        if (nr, nc) not in barrier_cells:
                            queue.append((nr, nc, path + [(nr, nc)]))

        return best_partial_path
