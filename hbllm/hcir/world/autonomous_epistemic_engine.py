"""Autonomous Epistemic World Engine — Domain-Agnostic Exploration, Causal Induction, and Mental Simulation.

Implements the fundamental cognitive loop:
1. Observe raw pixel frames I_t and available actions A.
2. Analyze pixel difference layouts Δ(I_t, I_{t-1}) to ground motor dynamics (avatar self-identification)
   and environmental state mutations (switches, gates, consumables).
3. If current knowledge is insufficient to reach the target (WIN), generate causal hypotheses
   and explore unknown entities via active curiosity-driven trial and error.
4. When sufficient rules are confirmed, simulate candidate action plans purely in mental imagination
   (forward search over learned world transition models).
5. Switch to exploitation and execute the validated winning plan directly.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    CausalHypothesis,
)
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)

logger = logging.getLogger(__name__)


class EpistemicPhase(StrEnum):
    """Cognitive lifecycle phases for autonomous exploration and problem solving."""

    MOTOR_GROUNDING = "motor_grounding"  # Self-identification & basic action calibration
    EPISTEMIC_EXPLORATION = (
        "epistemic_exploration"  # Trial-and-error hypothesis testing on unknown objects
    )
    MENTAL_SIMULATION = "mental_simulation"  # Forward planning in mind
    EXPLOITATION = "exploitation"  # Direct execution of verified solution
    REPLANNING = "replanning"  # Recovering from unexpected contradiction


@dataclass
class EpistemicObservationDiff:
    """Detailed structural difference between consecutive observation frames."""

    changed_pixel_count: int = 0
    changed_mask: np.ndarray | None = None
    displaced_entities: list[tuple[SpatialEntity, tuple[int, int]]] = field(default_factory=list)
    mutated_pixels: list[tuple[int, int, int, int]] = field(
        default_factory=list
    )  # (r, c, old_val, new_val)
    disappeared_features: set[int] = field(default_factory=set)
    appeared_features: set[int] = field(default_factory=set)


@dataclass
class MentalSimulationStep:
    """A step planned purely within internal mental simulation."""

    action: int
    action_data: dict[str, Any] | None = None
    predicted_avatar_pos: tuple[int, int] = (0, 0)
    expected_mutation: str | None = None


class AutonomousEpistemicEngine:
    """Domain-agnostic cognitive engine for autonomous exploration, hypothesis testing,

    and forward mental simulation.
    """

    def __init__(
        self,
        exploration_budget: int = 150,
        enable_logging: bool = True,
    ) -> None:
        self.exploration_budget = exploration_budget
        self.enable_logging = enable_logging

        # Cognitive state
        self.phase: EpistemicPhase = EpistemicPhase.MOTOR_GROUNDING
        self.step_counter: int = 0

        # Memory of frames & actions
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_action_data: dict[str, Any] | None = None
        self.consecutive_quiescent_actions: int = 0

        # Avatar self-model (Motor Grounding)
        self.avatar_feature: int | None = None
        self.avatar_pos: tuple[int, int] | None = None
        self.avatar_size: int = 1
        self.action_dynamics: dict[int, ActionDynamicsModel] = {}
        self.tested_actions: set[int] = set()

        # World Model & Environmental Topology
        self.bg_feature: int = 0
        self.learned_barriers: set[tuple[int, int]] = set()
        self.learned_barrier_features: set[int] = set()
        self.learned_walkable_features: set[int] = set()
        self.learned_goal_positions: set[tuple[int, int]] = set()
        self.learned_goal_features: set[int] = set()
        self.learned_cargo_features: set[int] = set()
        self.learned_receptacle_positions: set[tuple[int, int]] = set()
        self.learned_receptacle_features: set[int] = set()

        # Causal Hypotheses & State Mutations
        self.hypotheses: list[CausalHypothesis] = []
        self.active_hypothesis: CausalHypothesis | None = None
        self.state_mutations: list[StateMutationModel] = []
        self.belief_history: list[BeliefTransitionEvent] = []

        # Curiosity & Exploration tracking
        self.entity_visit_counts: dict[str, int] = {}
        self.position_visit_counts: dict[tuple[int, int], int] = {}
        self.quiescent_click_targets: set[tuple[int, int]] = set()
        self.effective_click_targets: set[tuple[int, int]] = set()
        self.active_probe_target: tuple[int, int] | None = None
        self.active_probe_id: str | None = None
        self.probe_target_steps: int = 0
        self.probed_entity_ids: set[str] = set()
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=16)

        # Mental Imagination & Precomputed Execution Queue
        self.mental_plan: deque[MentalSimulationStep] = deque()

    def reset_episode(self, retain_dynamics: bool = True) -> None:
        """Reset episodic state upon level transition or death."""
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = None
        self.consecutive_quiescent_actions = 0
        self.mental_plan.clear()
        self.active_hypothesis = None
        self.active_probe_target = None
        self.active_probe_id = None
        self.probe_target_steps = 0
        self.recent_positions.clear()

        if not retain_dynamics:
            self.avatar_feature = None
            self.avatar_pos = None
            self.action_dynamics.clear()
            self.tested_actions.clear()
            self.learned_barriers.clear()
            self.learned_barrier_features.clear()
            self.learned_goal_positions.clear()
            self.learned_goal_features.clear()
            self.learned_cargo_features.clear()
            self.learned_receptacle_positions.clear()
            self.learned_receptacle_features.clear()
            self.state_mutations.clear()
            self.phase = EpistemicPhase.MOTOR_GROUNDING
        else:
            # Retain general physics, motor mappings, and mutation rules across levels
            self.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if self.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING
            )

    def is_motor_grounded(self) -> bool:
        """True if the agent has identified its avatar and calibrated directional actions."""
        return self.avatar_feature is not None and any(
            dyn.confidence >= 0.7 for dyn in self.action_dynamics.values()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Perception & Structural Difference Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_background(self, grid: np.ndarray) -> int:
        """Estimate the background feature from the perimeter mode distribution,
        accounting for enclosed boundary frames."""
        perimeter = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        vals, counts = np.unique(perimeter, return_counts=True)
        if len(vals) == 0:
            return 0

        # Check if perimeter is an enclosing outer frame (e.g. wall)
        # If the perimeter has high homogeneity (>=70%) or top_val is known barrier
        top_val = int(vals[np.argmax(counts)])
        top_count = int(np.max(counts))
        if (
            (top_count / len(perimeter) >= 0.70 or top_val in self.learned_barrier_features)
            and grid.shape[0] >= 3
            and grid.shape[1] >= 3
        ):
            interior = grid[1:-1, 1:-1]
            int_vals, int_counts = np.unique(interior, return_counts=True)
            if len(int_vals) > 0:
                valid_int = [
                    i for i, v in enumerate(int_vals) if int(v) not in self.learned_barrier_features
                ]
                if valid_int:
                    int_top_val = int(int_vals[valid_int[np.argmax(int_counts[valid_int])]])
                    if int_top_val != top_val:
                        self.learned_barrier_features.add(top_val)
                        return int_top_val

        # If top_val is a known barrier, avoid picking it as background
        if top_val in self.learned_barrier_features and len(vals) > 1:
            non_barriers = [
                i for i, v in enumerate(vals) if int(v) not in self.learned_barrier_features
            ]
            if non_barriers:
                return int(vals[non_barriers[np.argmax(counts[non_barriers])]])

        # If avatar is already hypothesized, avoid picking avatar color as background
        if self.avatar_feature is not None and len(vals) > 1:
            other_idx = [i for i, v in enumerate(vals) if v != self.avatar_feature]
            if other_idx:
                return int(vals[other_idx[np.argmax(counts[other_idx])]])
        return top_val

    def extract_entities(self, grid: np.ndarray, bg: int) -> list[SpatialEntity]:
        """Domain-agnostic connected-component entity segmentation."""
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        entities: list[SpatialEntity] = []

        for r in range(H):
            for c in range(W):
                if visited[r, c]:
                    continue
                val = int(grid[r, c])
                if val == bg:
                    visited[r, c] = True
                    continue

                # Flood fill connected component
                cells: list[tuple[int, int]] = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop()
                    cells.append((cr, cc))
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                            if int(grid[nr, nc]) == val:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                min_r = min(p[0] for p in cells)
                max_r = max(p[0] for p in cells)
                min_c = min(p[1] for p in cells)
                max_c = max(p[1] for p in cells)
                area = len(cells)
                centroid = (sum(p[0] for p in cells) / area, sum(p[1] for p in cells) / area)
                grid_pos = (int(round(centroid[0])), int(round(centroid[1])))

                # Border frame / partition detection
                is_border = (
                    (min_r <= 1 and max_r >= H - 2 and min_c <= 1 and max_c >= W - 2)
                    or (max_r - min_r >= H - 2 and max_c - min_c >= W - 2)
                    or (area > H * W * 0.35)
                )

                if is_border or val in self.learned_barrier_features:
                    role = EntityRole.OBSTACLE
                elif val == self.avatar_feature:
                    role = EntityRole.AGENT
                elif any(
                    m.trigger_feature == val or m.trigger_pos == grid_pos
                    for m in self.state_mutations
                ):
                    role = EntityRole.ACTUATOR
                elif val in self.learned_cargo_features:
                    role = EntityRole.MANIPULABLE
                elif (
                    val in self.learned_goal_features
                    or val in self.learned_receptacle_features
                    or grid_pos in self.learned_goal_positions
                    or grid_pos in self.learned_receptacle_positions
                ):
                    role = EntityRole.GOAL
                elif area <= 16:
                    # Gestalt visual saliency: small foreground objects start as candidate interactives
                    role = EntityRole.UNKNOWN
                else:
                    role = EntityRole.UNKNOWN

                ent = SpatialEntity(
                    id=f"ent_{val}_{len(entities)}_{min_r}_{min_c}",
                    role=role,
                    centroid=centroid,
                    grid_pos=grid_pos,
                    area=area,
                    bounding_box=(min_r, max_r, min_c, max_c),
                    feature_id=val,
                    properties={"cells": cells, "feature": val},
                )
                entities.append(ent)

        return entities

    def compute_frame_diff(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray
    ) -> EpistemicObservationDiff:
        """Compute structural difference between two observation frames."""
        diff_mask = prev_grid != curr_grid
        changed_count = int(np.sum(diff_mask))

        diff = EpistemicObservationDiff(
            changed_pixel_count=changed_count,
            changed_mask=diff_mask,
        )
        if changed_count == 0:
            return diff

        H, W = curr_grid.shape
        for r in range(H):
            for c in range(W):
                if diff_mask[r, c]:
                    diff.mutated_pixels.append((r, c, int(prev_grid[r, c]), int(curr_grid[r, c])))

        prev_features = set(int(v) for v in np.unique(prev_grid))
        curr_features = set(int(v) for v in np.unique(curr_grid))
        diff.disappeared_features = prev_features - curr_features
        diff.appeared_features = curr_features - prev_features

        return diff

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Epistemic Assimilation & Bayesian Hypothesis Updating
    # ─────────────────────────────────────────────────────────────────────────

    def assimilate_feedback(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        is_win: bool = False,
        is_lost: bool = False,
    ) -> None:
        """Assimilate sensory feedback from previous action into empirical world models."""
        if self.prev_grid is None or self.last_action is None:
            return

        diff = self.compute_frame_diff(self.prev_grid, curr_grid)
        action = self.last_action
        action_data = self.last_action_data

        # Track quiescent (ineffective) actions
        if diff.changed_pixel_count == 0:
            self.consecutive_quiescent_actions += 1
            if action_data and "x" in action_data and "y" in action_data:
                target_coord = (int(action_data["y"]), int(action_data["x"]))
                self.quiescent_click_targets.add(target_coord)

            # If directional action produced 0 change, infer blocked movement
            if action in [1, 2, 3, 4] and self.avatar_pos is not None:
                exp_delta = (
                    self.action_dynamics[action].get_displacement()
                    if action in self.action_dynamics
                    else {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
                )
                blocked_r = self.avatar_pos[0] + exp_delta[0]
                blocked_c = self.avatar_pos[1] + exp_delta[1]
                H, W = curr_grid.shape
                if 0 <= blocked_r < H and 0 <= blocked_c < W:
                    self.learned_barriers.add((blocked_r, blocked_c))
                    blocked_feature = int(curr_grid[blocked_r, blocked_c])
                    if blocked_feature != self.bg_feature:
                        self.learned_barrier_features.add(blocked_feature)
                    logger.debug(
                        "AutonomousEpistemicEngine: Grounded barrier at (%d, %d) with feature %d",
                        blocked_r,
                        blocked_c,
                        blocked_feature,
                    )
            return

        # Action produced state change! Reset quiescent counter
        self.consecutive_quiescent_actions = 0
        if action_data and "x" in action_data and "y" in action_data:
            target_coord = (int(action_data["y"]), int(action_data["x"]))
            self.effective_click_targets.add(target_coord)

        # ── A. Motor Calibration & Avatar Grounding ──────────────────────────
        # Check if an entity shifted with this action
        H, W = curr_grid.shape
        bg = self.estimate_background(curr_grid)
        prev_entities = self.extract_entities(self.prev_grid, bg)
        curr_entities = self.extract_entities(curr_grid, bg)

        # Check all entity pairs that moved
        moved_entities: list[tuple[SpatialEntity, tuple[int, int]]] = []
        for pe in prev_entities:
            for ce in curr_entities:
                if pe.feature_id == ce.feature_id and pe.area == ce.area and pe.area <= 64:
                    dr = ce.grid_pos[0] - pe.grid_pos[0]
                    dc = ce.grid_pos[1] - pe.grid_pos[1]
                    if (dr != 0 or dc != 0) and abs(dr) <= 3 and abs(dc) <= 3:
                        moved_entities.append((ce, (dr, dc)))
                        break

        for ce, (dr, dc) in moved_entities:
            if self.avatar_feature is None or self.avatar_feature == ce.feature_id:
                self.avatar_feature = ce.feature_id
                self.avatar_pos = ce.grid_pos
                self.avatar_size = ce.area
                if action not in self.action_dynamics:
                    self.action_dynamics[action] = ActionDynamicsModel(
                        action_id=action,
                        delta_r=dr,
                        delta_c=dc,
                        confidence=0.6,
                        probes_tested=1,
                    )
                else:
                    self.action_dynamics[action].update_from_trial(
                        (dr, dc), success=True, learning_rate=0.5
                    )
                logger.debug(
                    "AutonomousEpistemicEngine: Avatar identified (feat=%d, size=%d). Calibrated action %d -> delta=(%d, %d)",
                    self.avatar_feature,
                    self.avatar_size,
                    action,
                    dr,
                    dc,
                )
            elif self.avatar_feature is not None and ce.feature_id != self.avatar_feature:
                # Entity pushed by avatar!
                self.learned_cargo_features.add(ce.feature_id)
                logger.info(
                    "AutonomousEpistemicEngine: Discovered PUSHABLE CARGO (feat=%d, size=%d)",
                    ce.feature_id,
                    ce.area,
                )

        # ── B. Environmental Mutation Induction ──────────────────────────────
        # If pixels changed at coordinates outside the avatar's movement trajectory
        distant_mutations = [
            (r, c, old_v, new_v)
            for r, c, old_v, new_v in diff.mutated_pixels
            if self.avatar_pos is None or (r, c) != self.avatar_pos
        ]
        # Ignore global screen animations / camera scrolling changes (>64 pixels) for localized door mutation induction
        if distant_mutations and len(distant_mutations) <= 64:
            trigger_pos = (
                (int(action_data["y"]), int(action_data["x"]))
                if action_data and "x" in action_data
                else self.avatar_pos
            )
            trigger_feat = (
                int(self.prev_grid[trigger_pos[0], trigger_pos[1]])
                if trigger_pos and 0 <= trigger_pos[0] < H and 0 <= trigger_pos[1] < W
                else None
            )

            # De-duplicate: update existing model instead of creating duplicates
            existing_mut = next(
                (
                    m
                    for m in self.state_mutations
                    if m.trigger_pos == trigger_pos
                    and m.trigger_feature == trigger_feat
                    and m.prior_value == distant_mutations[0][2]
                ),
                None,
            )
            if existing_mut is not None:
                existing_mut.record_observation(distant_mutations[0][3])
            else:
                mutation_model = StateMutationModel(
                    trigger_type="ACTION" if action >= 5 else "CONTACT",
                    trigger_pos=trigger_pos,
                    trigger_feature=trigger_feat,
                    mutation_type="ENVIRONMENTAL_TOGGLE",
                    prior_value=distant_mutations[0][2],
                    posterior_value=distant_mutations[0][3],
                    confidence=0.8,
                    occurrences=1,
                )
                self.state_mutations.append(mutation_model)
                logger.info(
                    "AutonomousEpistemicEngine: Induced StateMutationModel! Trigger %s at %s changed %d distant pixels.",
                    mutation_model.trigger_type,
                    trigger_pos,
                    len(distant_mutations),
                )

        # ── C. Win / Goal Grounding & Loss / Hazard Avoidance ────────────────
        if is_win:
            if self.avatar_pos is not None:
                self.learned_goal_positions.add(self.avatar_pos)
            if self.active_hypothesis:
                self.active_hypothesis.confirmed = True
                self.active_hypothesis.confidence = 1.0

        if is_lost:
            if self.avatar_pos is not None:
                self.learned_barriers.add(self.avatar_pos)
            if self.active_hypothesis:
                self.active_hypothesis.confidence = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Forward Mental Simulation (Planning in Imagination)
    # ─────────────────────────────────────────────────────────────────────────

    def simulate_in_mind(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
    ) -> list[MentalSimulationStep] | None:
        """Simulate candidate action sequences internally in memory without taking physical steps.

        Supports:
        1. Direct spatial navigation to goal.
        2. Multi-object cargo manipulation & Sokoban pushing.
        3. Hierarchical switch actuation and door unblocking via StateMutationModels.
        """
        if not self.is_motor_grounded() or self.avatar_pos is None:
            return None

        H, W = curr_grid.shape
        bg = self.estimate_background(curr_grid)
        entities = self.extract_entities(curr_grid, bg)

        # 1. Identify goals & receptacles
        goals: list[tuple[int, int]] = [
            e.grid_pos
            for e in entities
            if e.role == target_role
            or e.feature_id in self.learned_goal_features
            or e.feature_id in self.learned_receptacle_features
            or e.grid_pos in self.learned_goal_positions
            or e.grid_pos in self.learned_receptacle_positions
        ]
        if not goals and self.learned_goal_positions:
            goals = [g for g in self.learned_goal_positions if 0 <= g[0] < H and 0 <= g[1] < W]
        if not goals:
            # Gestalt fallback: rare, small foreground entities (non-avatar, non-barrier)
            candidate_goals = [
                e.grid_pos
                for e in entities
                if e.role == EntityRole.UNKNOWN
                and 1 <= e.area <= 9
                and e.feature_id != bg
                and e.feature_id != self.avatar_feature
            ]
            if candidate_goals:
                goals = candidate_goals

        if not goals:
            return None

        # 2. Identify candidate pushable blocks / cargo
        pushable_blocks: list[tuple[int, int]] = [
            e.grid_pos
            for e in entities
            if (
                e.feature_id in self.learned_cargo_features
                or (
                    e.role == EntityRole.MANIPULABLE
                    and 1 <= e.area <= 9
                    and e.feature_id != bg
                    and e.feature_id != self.avatar_feature
                    and e.grid_pos not in goals
                )
            )
        ]

        # 3. Identify barriers & doors with mutation triggers
        static_barriers: set[tuple[int, int]] = set(self.learned_barriers)
        for r in range(H):
            for c in range(W):
                if int(curr_grid[r, c]) in self.learned_barrier_features:
                    static_barriers.add((r, c))

        # Check known state mutation triggers (switches that open doors)
        mutation_triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for m in self.state_mutations:
            opened_cells: set[tuple[int, int]] = set()
            for r in range(H):
                for c in range(W):
                    if int(curr_grid[r, c]) == m.prior_value and (
                        m.posterior_value == bg or m.posterior_value == 0
                    ):
                        opened_cells.add((r, c))
            if not opened_cells:
                continue

            # Map specific trigger pos if known
            if (
                m.trigger_pos is not None
                and len(m.trigger_pos) >= 2
                and 0 <= m.trigger_pos[0] < H
                and 0 <= m.trigger_pos[1] < W
            ):
                tr_p = (int(m.trigger_pos[0]), int(m.trigger_pos[1]))
                mutation_triggers.setdefault(tr_p, set()).update(opened_cells)

            # Map any grid cell matching trigger_feature (cross-level generalization)
            if m.trigger_feature is not None:
                for r in range(H):
                    for c in range(W):
                        if int(curr_grid[r, c]) == m.trigger_feature:
                            mutation_triggers.setdefault((r, c), set()).update(opened_cells)

        # 4. Available directional movements in mental model
        movable_actions: list[tuple[int, int, int]] = []
        for act in available_actions:
            if act in self.action_dynamics and self.action_dynamics[act].confidence >= 0.5:
                dr, dc = self.action_dynamics[act].get_displacement()
                if dr != 0 or dc != 0:
                    movable_actions.append((act, dr, dc))

        if not movable_actions:
            for act, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
                if act in available_actions:
                    movable_actions.append((act, dr, dc))

        start_pos = self.avatar_pos
        init_blocks = frozenset(pushable_blocks)
        init_open: frozenset[tuple[int, int]] = frozenset()

        is_block_delivery = bool(
            pushable_blocks
            and any(
                e.feature_id in self.learned_cargo_features or e.role == EntityRole.MANIPULABLE
                for e in entities
            )
        )

        def heuristic(pos: tuple[int, int], blocks: frozenset[tuple[int, int]]) -> float:
            if is_block_delivery and blocks:
                b_list = list(blocks)
                goal_dist = sum(
                    min(abs(g[0] - b[0]) + abs(g[1] - b[1]) for b in b_list) for g in goals
                )
                avatar_to_b = min(abs(pos[0] - b[0]) + abs(pos[1] - b[1]) for b in b_list)
                return float(goal_dist * 2.0 + avatar_to_b)
            else:
                return float(min(abs(g[0] - pos[0]) + abs(g[1] - pos[1]) for g in goals))

        # A* State: (f_score, g_cost, counter, pos, blocks, open_barriers, path)
        counter = 0
        open_set: list[
            tuple[
                float,
                int,
                int,
                tuple[int, int],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                list[MentalSimulationStep],
            ]
        ] = []
        h0 = heuristic(start_pos, init_blocks)
        heapq.heappush(open_set, (h0, 0, counter, start_pos, init_blocks, init_open, []))

        visited_states: set[
            tuple[tuple[int, int], frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
        ] = set()
        max_expansions = 4000

        while open_set and max_expansions > 0:
            max_expansions -= 1
            f_score, g_cost, _, cur_pos, cur_blocks, cur_open, path = heapq.heappop(open_set)

            state_key = (cur_pos, cur_blocks, cur_open)
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            # Check termination
            if is_block_delivery:
                if all(g in cur_blocks for g in goals):
                    logger.info(
                        "AutonomousEpistemicEngine: Compound Mental Simulation SUCCEEDED! Synthesized %d-step block delivery plan.",
                        len(path),
                    )
                    return path
            else:
                if cur_pos in goals:
                    logger.info(
                        "AutonomousEpistemicEngine: Mental Simulation SUCCEEDED! Synthesized %d-step path to goal %s.",
                        len(path),
                        cur_pos,
                    )
                    return path

            for act, dr, dc in movable_actions:
                nr, nc = cur_pos[0] + dr, cur_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue

                # Check if cell is barrier
                if (nr, nc) in static_barriers and (nr, nc) not in cur_open:
                    continue

                new_blocks = cur_blocks
                new_pos = (nr, nc)

                # Check block pushing
                if (nr, nc) in cur_blocks:
                    pushed_r, pushed_c = nr + dr, nc + dc
                    if not (0 <= pushed_r < H and 0 <= pushed_c < W):
                        continue
                    if (
                        pushed_r,
                        pushed_c,
                    ) in static_barriers and (pushed_r, pushed_c) not in cur_open:
                        continue
                    if (pushed_r, pushed_c) in cur_blocks:
                        continue

                    # Valid push! Update block positions
                    b_set = set(cur_blocks)
                    b_set.remove((nr, nc))
                    b_set.add((pushed_r, pushed_c))
                    new_blocks = frozenset(b_set)

                # Check state mutations (did new_pos or pushed block trigger a switch?)
                new_open = cur_open
                if new_pos in mutation_triggers:
                    new_open = cur_open | frozenset(mutation_triggers[new_pos])
                if is_block_delivery:
                    for b_pos in new_blocks:
                        if b_pos in mutation_triggers:
                            new_open = new_open | frozenset(mutation_triggers[b_pos])

                next_state_key = (new_pos, new_blocks, new_open)
                if next_state_key in visited_states:
                    continue

                new_cost = g_cost + 1
                h_val = heuristic(new_pos, new_blocks)
                counter += 1
                step = MentalSimulationStep(
                    action=act,
                    predicted_avatar_pos=new_pos,
                )
                heapq.heappush(
                    open_set,
                    (
                        new_cost + h_val,
                        new_cost,
                        counter,
                        new_pos,
                        new_blocks,
                        new_open,
                        path + [step],
                    ),
                )

        # Fallback to Hierarchical Subgoal Decomposition if compound A* did not reach the goals
        if not is_block_delivery:
            subgoal_plan = self._plan_hierarchical_subgoals(
                curr_grid=curr_grid,
                available_actions=available_actions,
                start_pos=start_pos,
                goals=set(goals),
                static_barriers=static_barriers,
                mutation_triggers=mutation_triggers,
                movable_actions=movable_actions,
            )
            if subgoal_plan:
                return subgoal_plan

        return None

    def _plan_hierarchical_subgoals(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        start_pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        mutation_triggers: dict[tuple[int, int], set[tuple[int, int]]],
        movable_actions: list[tuple[int, int, int]],
    ) -> list[MentalSimulationStep] | None:
        """Recursive multi-stage subgoal decomposition for locked barrier doors and remote mutation triggers.

        Decomposes high-level missions into sequential stages:
        1. Find path to an accessible switch/key that removes a blocking barrier.
        2. Actuate trigger and simulate environmental mutation (opening doors in mental model).
        3. Recurse from the updated world state toward subsequent switches or final goals.
        """
        H, W = curr_grid.shape
        cur_pos = start_pos
        open_barriers: set[tuple[int, int]] = set()
        accumulated_plan: list[MentalSimulationStep] = []
        max_stages = 8

        def find_shortest_path(
            s_pos: tuple[int, int],
            target_positions: set[tuple[int, int]],
            active_barriers: set[tuple[int, int]],
        ) -> list[MentalSimulationStep] | None:
            """BFS to find shortest path to any target position avoiding active barriers."""
            q: deque[tuple[tuple[int, int], list[MentalSimulationStep]]] = deque([(s_pos, [])])
            visited = {s_pos}
            while q:
                p, path = q.popleft()
                if p in target_positions:
                    return path
                for act, dr, dc in movable_actions:
                    nr, nc = p[0] + dr, p[1] + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in active_barriers or (nr, nc) in visited:
                        continue
                    visited.add((nr, nc))
                    step = MentalSimulationStep(action=act, predicted_avatar_pos=(nr, nc))
                    q.append(((nr, nc), path + [step]))
            return None

        for _ in range(max_stages):
            effective_barriers = static_barriers - open_barriers
            # Check if any goal is reachable directly
            goal_path = find_shortest_path(cur_pos, goals, effective_barriers)
            if goal_path is not None:
                accumulated_plan.extend(goal_path)
                logger.info(
                    "AutonomousEpistemicEngine: Hierarchical Subgoal Decomposition SUCCEEDED with %d total steps!",
                    len(accumulated_plan),
                )
                return accumulated_plan

            # Goal is not reachable. Find reachable mutation triggers (switches) that unlock new doors
            candidate_triggers: list[tuple[tuple[int, int], list[MentalSimulationStep], int]] = []
            for tr_pos, opened_set in mutation_triggers.items():
                unopened = opened_set - open_barriers
                if not unopened:
                    continue  # already opened
                # Check path to this switch
                tr_path = find_shortest_path(cur_pos, {tr_pos}, effective_barriers)
                if tr_path is not None:
                    candidate_triggers.append((tr_pos, tr_path, len(unopened)))

            if not candidate_triggers:
                # No accessible switch found that can unlock any barrier
                break

            # Prioritize switch that opens the most barriers or is closest
            candidate_triggers.sort(key=lambda x: (len(x[1]), -x[2]))
            chosen_pos, switch_path, _ = candidate_triggers[0]

            accumulated_plan.extend(switch_path)
            cur_pos = chosen_pos
            open_barriers.update(mutation_triggers[chosen_pos])

            # If mutation requires interaction action (e.g. 5) instead of CONTACT, append interaction
            matching_mutations = [
                m
                for m in self.state_mutations
                if (
                    m.trigger_pos == chosen_pos
                    or (
                        m.trigger_feature is not None
                        and int(curr_grid[chosen_pos[0], chosen_pos[1]]) == m.trigger_feature
                    )
                )
            ]
            if matching_mutations and matching_mutations[0].trigger_type == "ACTION":
                act_id = int(matching_mutations[0].metadata.get("action_id", 5))
                if act_id in available_actions:
                    accumulated_plan.append(
                        MentalSimulationStep(
                            action=act_id,
                            predicted_avatar_pos=cur_pos,
                            expected_mutation="ACTION_TRIGGER",
                        )
                    )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Active Curiosity-Driven Epistemic Exploration (Trial & Error)
    # ─────────────────────────────────────────────────────────────────────────

    def plan_epistemic_probe(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        """Select an exploratory probe action to resolve epistemic uncertainty

        via discovery economics (information gain / cost).
        """
        H, W = curr_grid.shape
        bg = self.estimate_background(curr_grid)

        # 1. Uncalibrated actions take absolute priority for motor grounding
        untested = [a for a in available_actions if a not in self.tested_actions]
        if untested:
            # Prioritize directional movement actions (1-4) first for avatar grounding
            directional_untested = [a for a in untested if a in [1, 2, 3, 4]]
            chosen_action = directional_untested[0] if directional_untested else untested[0]
            self.tested_actions.add(chosen_action)

            action_data = None
            if chosen_action >= 6:
                # Provide click coordinates at grid center or avatar pos
                cy, cx = self.avatar_pos if self.avatar_pos is not None else (H // 2, W // 2)
                action_data = {"x": cx, "y": cy}
            return chosen_action, action_data

        # 2. Curiosity over unknown entities & interactive objects
        entities = self.extract_entities(curr_grid, bg)
        unknown_entities = [
            e
            for e in entities
            if e.role in (EntityRole.UNKNOWN, EntityRole.ACTUATOR, EntityRole.MANIPULABLE)
            and e.area <= 100
        ]

        # Click Affordance Exploration (e.g. for action 6 games)
        if 6 in available_actions and not any(a in available_actions for a in [1, 2, 3, 4]):
            candidates: list[tuple[int, int, float]] = []
            for e in unknown_entities:
                cr, cc = e.grid_pos
                if (cr, cc) in self.quiescent_click_targets:
                    continue
                # Score candidate by novelty and effective history
                saliency = 100.0 / math.log2(2 + e.area)
                usage_pen = float(self.entity_visit_counts.get(e.id, 0)) * 25.0
                cand_score = saliency - usage_pen
                candidates.append((cr, cc, cand_score))

            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                best_r, best_c, _ = candidates[0]
                self.entity_visit_counts[f"click_{best_r}_{best_c}"] = (
                    self.entity_visit_counts.get(f"click_{best_r}_{best_c}", 0) + 1
                )
                return 6, {"x": best_c, "y": best_r}
            else:
                # Reset quiescent cache if all candidates are exhausted
                self.quiescent_click_targets.clear()
                return 6, {"x": W // 2, "y": H // 2}

        # Spatial Movement Curiosity: Probe least-visited unknown entities with target commitment & loop breaking
        if self.avatar_pos is not None:
            # 1. Update position tracking and record recent history
            self.recent_positions.append(self.avatar_pos)
            self.position_visit_counts[self.avatar_pos] = (
                self.position_visit_counts.get(self.avatar_pos, 0) + 1
            )

            # Check if any entity was reached and mark it as probed
            for e in entities:
                if (
                    abs(e.grid_pos[0] - self.avatar_pos[0])
                    + abs(e.grid_pos[1] - self.avatar_pos[1])
                    <= 1
                ):
                    self.probed_entity_ids.add(e.id)

            # Check if current committed probe target is reached or timed out
            if self.active_probe_target is not None:
                dist_to_target = abs(self.active_probe_target[0] - self.avatar_pos[0]) + abs(
                    self.active_probe_target[1] - self.avatar_pos[1]
                )
                self.probe_target_steps += 1
                if dist_to_target <= 1 or self.probe_target_steps > 12:
                    if self.active_probe_id:
                        self.probed_entity_ids.add(self.active_probe_id)
                    self.active_probe_target = None
                    self.active_probe_id = None
                    self.probe_target_steps = 0

            # 2. Check for oscillation loop (e.g. A <-> B ping-pong)
            is_oscillating = self.recent_positions.count(self.avatar_pos) >= 3
            if is_oscillating:
                logger.debug(
                    "AutonomousEpistemicEngine: Oscillation detected at %s! Breaking loop...",
                    self.avatar_pos,
                )
                self.active_probe_target = None
                self.active_probe_id = None
                best_act = available_actions[0]
                min_visits = float("inf")
                for act, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
                    if act in available_actions:
                        nr = self.avatar_pos[0] + dr
                        nc = self.avatar_pos[1] + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if (nr, nc) in self.learned_barriers or int(
                            curr_grid[nr, nc]
                        ) in self.learned_barrier_features:
                            continue
                        visits = self.position_visit_counts.get((nr, nc), 0)
                        recency_pen = 20.0 if (nr, nc) in list(self.recent_positions)[-6:] else 0.0
                        total_score = visits + recency_pen
                        if total_score < min_visits:
                            min_visits = total_score
                            best_act = act
                return best_act, None

            # 3. Filter candidate unprobed entities
            candidates = [
                e
                for e in unknown_entities
                if e.id not in self.probed_entity_ids
                and (
                    abs(e.grid_pos[0] - self.avatar_pos[0])
                    + abs(e.grid_pos[1] - self.avatar_pos[1])
                )
                > 1
            ]

            # If no active target committed, select a new candidate
            if self.active_probe_target is None and candidates:
                scored: list[tuple[SpatialEntity, float]] = []
                for e in candidates:
                    dist = abs(e.grid_pos[0] - self.avatar_pos[0]) + abs(
                        e.grid_pos[1] - self.avatar_pos[1]
                    )
                    visits = self.entity_visit_counts.get(e.id, 0)
                    info_val = 10.0 / (visits + 1.0)
                    cost = dist + 1.0
                    scored.append((e, info_val / cost))
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen = scored[0][0]
                self.active_probe_target = chosen.grid_pos
                self.active_probe_id = chosen.id
                self.probe_target_steps = 0
                self.entity_visit_counts[chosen.id] = self.entity_visit_counts.get(chosen.id, 0) + 1

            # 4. Navigate toward active target (or explore frontier if none available)
            target_pos = self.active_probe_target
            if target_pos is None:
                tr, tc = H // 2, W // 2
            else:
                tr, tc = target_pos

            best_action = available_actions[0]
            min_dist = float("inf")
            for act, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
                if act in available_actions:
                    dr_cal, dc_cal = (
                        self.action_dynamics[act].get_displacement()
                        if act in self.action_dynamics
                        and self.action_dynamics[act].confidence >= 0.5
                        else (dr, dc)
                    )
                    nr = self.avatar_pos[0] + dr_cal
                    nc = self.avatar_pos[1] + dc_cal
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in self.learned_barriers or int(
                        curr_grid[nr, nc]
                    ) in self.learned_barrier_features:
                        continue

                    visit_penalty = float(self.position_visit_counts.get((nr, nc), 0)) * 1.5
                    recency_penalty = 10.0 if (nr, nc) in list(self.recent_positions)[-4:] else 0.0
                    d = abs(tr - nr) + abs(tc - nc) + visit_penalty + recency_penalty
                    if d < min_dist:
                        min_dist = d
                        best_action = act

            return best_action, None

        # Fallback: cycle available actions with loop prevention
        action_idx = self.step_counter % len(available_actions)
        return available_actions[action_idx], None

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Master Autonomous Decision Loop
    # ─────────────────────────────────────────────────────────────────────────

    def decide(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        is_win: bool = False,
    ) -> tuple[int, dict[str, Any] | None]:
        """Unified cognitive decision function:

        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        """
        self.step_counter += 1

        # 1. Update background & assimilate sensory feedback from previous action
        self.bg_feature = self.estimate_background(curr_grid)
        if self.prev_grid is not None:
            self.assimilate_feedback(curr_grid, available_actions, is_win=is_win)

        # 2. Check active mental plan during EXPLOITATION phase
        if self.phase == EpistemicPhase.EXPLOITATION and self.mental_plan:
            next_step = self.mental_plan.popleft()
            if next_step.action in available_actions:
                self.prev_grid = curr_grid.copy()
                self.last_action = next_step.action
                self.last_action_data = next_step.action_data
                return next_step.action, next_step.action_data
            else:
                # Invalidation: simulated action no longer available in environment
                self.mental_plan.clear()
                self.phase = EpistemicPhase.REPLANNING

        # 3. If motor grounded, attempt Forward Mental Simulation (Planning in Imagination)
        if self.is_motor_grounded():
            simulated_plan = self.simulate_in_mind(curr_grid, available_actions)
            if simulated_plan:
                self.phase = EpistemicPhase.EXPLOITATION
                self.mental_plan = deque(simulated_plan)
                next_step = self.mental_plan.popleft()
                self.prev_grid = curr_grid.copy()
                self.last_action = next_step.action
                self.last_action_data = next_step.action_data
                return next_step.action, next_step.action_data

        # 4. If current knowledge is insufficient, fall back to Epistemic Curiosity Probing
        self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
        probe_action, probe_data = self.plan_epistemic_probe(curr_grid, available_actions)

        self.prev_grid = curr_grid.copy()
        self.last_action = probe_action
        self.last_action_data = probe_data
        return probe_action, probe_data
