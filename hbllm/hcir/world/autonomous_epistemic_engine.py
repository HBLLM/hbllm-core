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

        if not retain_dynamics:
            self.avatar_feature = None
            self.avatar_pos = None
            self.action_dynamics.clear()
            self.tested_actions.clear()
            self.learned_barriers.clear()
            self.learned_barrier_features.clear()
            self.learned_goal_positions.clear()
            self.learned_goal_features.clear()
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
        # If the perimeter has high homogeneity (>=75%), check interior mode
        top_val = int(vals[np.argmax(counts)])
        top_count = int(np.max(counts))
        if top_count / len(perimeter) >= 0.75 and grid.shape[0] > 4 and grid.shape[1] > 4:
            interior = grid[1:-1, 1:-1]
            int_vals, int_counts = np.unique(interior, return_counts=True)
            if len(int_vals) > 0:
                int_top_val = int(int_vals[np.argmax(int_counts)])
                # If interior dominant color differs from perimeter frame, interior is background
                if int_top_val != top_val and np.max(int_counts) / interior.size >= 0.20:
                    self.learned_barrier_features.add(top_val)
                    return int_top_val

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
                elif val in self.learned_goal_features or grid_pos in self.learned_goal_positions:
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

        moved_entity: SpatialEntity | None = None
        observed_delta: tuple[int, int] = (0, 0)

        for pe in prev_entities:
            for ce in curr_entities:
                if pe.feature_id == ce.feature_id and pe.area == ce.area and pe.area <= 64:
                    dr = ce.grid_pos[0] - pe.grid_pos[0]
                    dc = ce.grid_pos[1] - pe.grid_pos[1]
                    if (dr != 0 or dc != 0) and abs(dr) <= 3 and abs(dc) <= 3:
                        # Candidate controllable translation
                        moved_entity = ce
                        observed_delta = (dr, dc)
                        break
            if moved_entity is not None:
                break

        if moved_entity is not None:
            # If avatar is not yet identified, this entity is our candidate avatar
            if self.avatar_feature is None or self.avatar_feature == moved_entity.feature_id:
                self.avatar_feature = moved_entity.feature_id
                self.avatar_pos = moved_entity.grid_pos
                self.avatar_size = moved_entity.area

                # Calibrate ActionDynamicsModel for this action
                if action not in self.action_dynamics:
                    self.action_dynamics[action] = ActionDynamicsModel(
                        action_id=action,
                        delta_r=observed_delta[0],
                        delta_c=observed_delta[1],
                        confidence=0.6,
                        probes_tested=1,
                    )
                else:
                    self.action_dynamics[action].update_from_trial(
                        observed_delta, success=True, learning_rate=0.5
                    )
                logger.debug(
                    "AutonomousEpistemicEngine: Avatar identified (feat=%d, size=%d). Calibrated action %d -> delta=(%d, %d)",
                    self.avatar_feature,
                    self.avatar_size,
                    action,
                    observed_delta[0],
                    observed_delta[1],
                )

        # ── B. Environmental Mutation Induction ──────────────────────────────
        # If pixels changed at coordinates outside the avatar's movement trajectory
        distant_mutations = [
            (r, c, old_v, new_v)
            for r, c, old_v, new_v in diff.mutated_pixels
            if self.avatar_pos is None or (r, c) != self.avatar_pos
        ]
        if distant_mutations:
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

        # ── C. Win / Goal Grounding ──────────────────────────────────────────
        if is_win:
            if self.avatar_pos is not None:
                self.learned_goal_positions.add(self.avatar_pos)
            if self.active_hypothesis:
                self.active_hypothesis.confirmed = True
                self.active_hypothesis.confidence = 1.0

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

        Uses learned dynamics, barrier models, and state mutation triggers to plan
        a collision-free sequence directly to the target.
        """
        if not self.is_motor_grounded() or self.avatar_pos is None:
            return None

        H, W = curr_grid.shape
        bg = self.estimate_background(curr_grid)
        entities = self.extract_entities(curr_grid, bg)

        # Find target goals
        goals = [
            e.grid_pos
            for e in entities
            if e.role == target_role or e.feature_id in self.learned_goal_features
        ]
        if not goals and self.learned_goal_positions:
            goals = [g for g in self.learned_goal_positions if 0 <= g[0] < H and 0 <= g[1] < W]
        if not goals:
            # If no explicit goal known, look for rare Gestalt candidate
            candidate_goals = [
                e.grid_pos
                for e in entities
                if e.role == EntityRole.UNKNOWN and 1 <= e.area <= 9 and e.feature_id != bg
            ]
            if candidate_goals:
                goals = candidate_goals

        if not goals:
            return None

        # Build mental obstacle map from confirmed barriers
        static_barriers = set(self.learned_barriers)
        for r in range(H):
            for c in range(W):
                if int(curr_grid[r, c]) in self.learned_barrier_features:
                    static_barriers.add((r, c))

        # Available directional movements in mental model
        movable_actions: list[tuple[int, int, int]] = []
        for act in available_actions:
            if act in self.action_dynamics and self.action_dynamics[act].confidence >= 0.5:
                dr, dc = self.action_dynamics[act].get_displacement()
                if dr != 0 or dc != 0:
                    movable_actions.append((act, dr, dc))

        if not movable_actions:
            # Default directional heuristic fallback
            for act, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
                if act in available_actions:
                    movable_actions.append((act, dr, dc))

        start_pos = self.avatar_pos
        target_pos = min(goals, key=lambda g: abs(g[0] - start_pos[0]) + abs(g[1] - start_pos[1]))

        # A* Search in mental space
        open_set: list[tuple[float, int, tuple[int, int], list[MentalSimulationStep]]] = []
        initial_h = abs(target_pos[0] - start_pos[0]) + abs(target_pos[1] - start_pos[1])
        heapq.heappush(open_set, (initial_h, 0, start_pos, []))

        visited_costs: dict[tuple[int, int], int] = {start_pos: 0}
        max_expansions = 2500

        while open_set and max_expansions > 0:
            max_expansions -= 1
            f_score, g_cost, cur_pos, path = heapq.heappop(open_set)

            # Check if target reached in mental simulation
            if cur_pos == target_pos:
                logger.info(
                    "AutonomousEpistemicEngine: Mental Simulation SUCCEEDED! Synthesized %d-step path to goal %s.",
                    len(path),
                    target_pos,
                )
                return path

            for act, dr, dc in movable_actions:
                nr, nc = cur_pos[0] + dr, cur_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in static_barriers and (nr, nc) != target_pos:
                    continue

                new_cost = g_cost + 1
                if (nr, nc) not in visited_costs or new_cost < visited_costs[(nr, nc)]:
                    visited_costs[(nr, nc)] = new_cost
                    h_val = abs(target_pos[0] - nr) + abs(target_pos[1] - nc)
                    step = MentalSimulationStep(
                        action=act,
                        predicted_avatar_pos=(nr, nc),
                    )
                    heapq.heappush(open_set, (new_cost + h_val, new_cost, (nr, nc), path + [step]))

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

        # Spatial Movement Curiosity: Probe least-visited unknown entities
        if self.avatar_pos is not None and unknown_entities:
            scored_entities: list[tuple[SpatialEntity, float]] = []
            for e in unknown_entities:
                dist = abs(e.grid_pos[0] - self.avatar_pos[0]) + abs(
                    e.grid_pos[1] - self.avatar_pos[1]
                )
                visits = self.entity_visit_counts.get(e.id, 0)
                # Curiosity Formula = Information Value / Cost
                info_val = 10.0 / (visits + 1.0)
                cost = dist + 1.0
                curiosity_score = info_val / cost
                scored_entities.append((e, curiosity_score))

            scored_entities.sort(key=lambda x: x[1], reverse=True)
            target_entity = scored_entities[0][0]
            self.entity_visit_counts[target_entity.id] = (
                self.entity_visit_counts.get(target_entity.id, 0) + 1
            )

            # Move toward target entity
            tr, tc = target_entity.grid_pos
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

                    # Distance heuristic + visit count penalty to break pacing loops
                    visit_penalty = float(self.position_visit_counts.get((nr, nc), 0)) * 0.75
                    d = abs(tr - nr) + abs(tc - nc) + visit_penalty
                    if d < min_dist:
                        min_dist = d
                        best_action = act

            self.position_visit_counts[self.avatar_pos] = (
                self.position_visit_counts.get(self.avatar_pos, 0) + 1
            )
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
