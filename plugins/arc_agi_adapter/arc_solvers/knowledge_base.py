"""ARC-AGI Cross-Level Knowledge & Causal Inference.

Knowledge persistence, goal induction, trial memory, and causal
affordance engines that accumulate learning across game levels.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.subgoal_decomposer import HCIRSkill
from plugins.arc_agi_adapter.arc_skills import (
    CoupledControllableSkillAcquisition,
    KinematicMomentumSkillAcquisition,
    MorphologicalProgramSynthesis,
    PermutationAlgebraSkillAcquisition,
    RelationalAffordanceSkillAcquisition,
    SpatiotemporalSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_solvers.visual_analysis import (
    DiffType,
    FrameDiff,
    FrameDiffAnalyzer,
    PuzzleTypology,
)

logger = logging.getLogger(__name__)


@dataclass
class ActionAffordance:
    """Learned behavioral effect of a specific action ID."""

    action_id: int
    delta_r: int = 0
    delta_c: int = 0
    is_cycler: bool = False
    is_commit_or_stamp: bool = False
    is_click: bool = False
    confidence: float = 0.0
    times_tested: int = 0


@dataclass
class ControllableSignature:
    """Visual invariant signature of the agent's controllable entity or selector."""

    color: int | None = None
    area: int = 0
    shape_pattern: tuple[int, ...] = ()
    is_discrete_selector: bool = False
    active_slots: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ObjectInteractionRecipe:
    """Learned recipe for interacting with an object type.

    Captures the full interaction lifecycle: what action works on an object,
    what the outcome is, and where to deliver it if applicable.
    Persists across levels so the agent can skip exploration for known objects.
    """

    object_color: int  # Color that identifies this object type
    signature_key: str | None = None  # Compound signature 'f{c}_s{shape}_z{size}'
    shape_category: str | None = None  # 'point', 'line_h', 'line_v', 'rect', etc.
    size_bucket: str | None = None  # 'point', 'small', 'medium', 'large'
    object_area_range: tuple[int, int] = (1, 100)  # (min_area, max_area) observed
    interaction_action: int = 5  # Action that works (5=pickup/interact, 6=click)
    outcome: str = "unknown"  # 'pickup', 'destroy', 'transform', 'toggle'
    delivery_zone_color: int | None = None  # If pickup, where to deliver
    delivery_zone_bounds: tuple[int, int, int, int] | None = None  # (min_r, max_r, min_c, max_c)
    approach_direction: str = "nearest"  # 'nearest', 'above', 'below', 'left', 'right'
    times_confirmed: int = 0  # How many times this recipe succeeded
    confidence: float = 0.0  # Confidence in recipe validity


class CrossLevelKnowledgeBase:
    """Holds accumulated causal models and invariants across levels of a puzzle."""

    def __init__(self) -> None:
        self.puzzle_typology: PuzzleTypology = PuzzleTypology.UNKNOWN
        self.action_affordances: dict[int, ActionAffordance] = {}
        self.controllable_signature: ControllableSignature = ControllableSignature()
        self.goal_reference_pattern: np.ndarray | None = None
        self.goal_target_zone: tuple[int, int, int, int] | None = None
        self.walkable_colors: set[int] = set()
        self.barrier_colors: set[int] = set()
        self.hazard_colors: set[int] = set()
        self.discrete_state_transitions: dict[
            tuple[int, int], int
        ] = {}  # (slot_idx, action) -> next_slot
        self.levels_solved: int = 0
        self.total_epistemic_probes: int = 0

        # Object Interaction Memory: learned recipes for interacting with objects
        # Maps object_color -> ObjectInteractionRecipe
        self.object_recipes: dict[int, ObjectInteractionRecipe] = {}
        # Maps signature_key -> ObjectInteractionRecipe
        self.signature_recipes: dict[str, ObjectInteractionRecipe] = {}
        # Snapshot of grid when level completes (for learning delivery zones)
        self.last_completion_grid: np.ndarray | None = None
        # Track objects near avatar when action 5 is used (for learning pickup)
        self._pending_interaction: dict | None = None
        # Learned procedural skills transferred across levels
        self.skills: list[HCIRSkill] = []

        # Multi-Paradigm Skill Acquisition Engines
        self.spatiotemporal: SpatiotemporalSkillAcquisition = SpatiotemporalSkillAcquisition()
        self.permutation: PermutationAlgebraSkillAcquisition = PermutationAlgebraSkillAcquisition()
        self.kinematics: KinematicMomentumSkillAcquisition = KinematicMomentumSkillAcquisition()
        self.relational: RelationalAffordanceSkillAcquisition = (
            RelationalAffordanceSkillAcquisition()
        )
        self.morphology: MorphologicalProgramSynthesis = MorphologicalProgramSynthesis()
        self.coupled: CoupledControllableSkillAcquisition = CoupledControllableSkillAcquisition()

    def register_skill(self, skill: HCIRSkill) -> None:
        """Store an acquired skill for reuse in subsequent levels.

        Reaching the same effect again reinforces confidence and keeps
        whichever route was more efficient, rather than just overwriting.
        """
        for i, s in enumerate(self.skills):
            if s.skill_id == skill.skill_id:
                better = skill if len(skill.action_sequence) <= len(s.action_sequence) else s
                self.skills[i] = HCIRSkill(
                    skill_id=skill.skill_id,
                    preconditions=better.preconditions,
                    action_sequence=better.action_sequence,
                    action_data_sequence=better.action_data_sequence,
                    expected_effect=better.expected_effect,
                    confidence=min(1.0, s.confidence + 0.15),
                    times_executed=s.times_executed + 1,
                    times_succeeded=s.times_succeeded + 1,
                )
                return
        self.skills.append(skill)

    def register_hazard(
        self,
        color: int,
        pos: tuple[int, int] | None = None,
        action: int | None = None,
    ) -> None:
        """Record a discovered lethal hazard or negative constraint.

        Persists across levels and retries so known fatal features are never
        stepped onto or interacted with again.
        """
        self.hazard_colors.add(color)
        self.barrier_colors.add(color)
        self.walkable_colors.discard(color)
        if color in self.object_recipes:
            del self.object_recipes[color]

    def register_observation(
        self,
        prev_grid: np.ndarray,
        action: int,
        curr_grid: np.ndarray,
        diff: FrameDiff,
    ) -> None:
        """Assimilate a visual transition into the persistent knowledge base."""
        self.total_epistemic_probes += 1

        # Retrieve or initialize affordance
        aff = self.action_affordances.setdefault(action, ActionAffordance(action_id=action))
        aff.times_tested += 1

        if diff.diff_type == DiffType.TRANSLATION and diff.translation_delta:
            dr, dc = diff.translation_delta
            aff.delta_r = dr
            aff.delta_c = dc
            aff.confidence = min(1.0, aff.confidence + 0.35)
            self.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION

            if self.controllable_signature.color is None and diff.moved_object_color is not None:
                self.controllable_signature.color = diff.moved_object_color
                self.controllable_signature.area = diff.moved_object_size

            # Learn walkable floor colors from vacated and newly entered pixels
            if self.controllable_signature.color is not None:
                for r, c in diff.mutated_coords:
                    old_c = diff.old_colors.get((r, c))
                    new_c = diff.new_colors.get((r, c))
                    if old_c == self.controllable_signature.color and new_c is not None:
                        self.walkable_colors.add(new_c)
                    elif new_c == self.controllable_signature.color and old_c is not None:
                        self.walkable_colors.add(old_c)

            # Detect interaction events: object appearing/disappearing near avatar
            if self.controllable_signature.color is not None:
                avatar_color = self.controllable_signature.color
                # Count non-background, non-avatar objects in both frames
                bg = int(np.bincount(prev_grid.flatten()).argmax())
                prev_obj_colors = set(
                    int(c) for c in np.unique(prev_grid) if c != 0 and c != bg and c != avatar_color
                )
                curr_obj_colors = set(
                    int(c) for c in np.unique(curr_grid) if c != 0 and c != bg and c != avatar_color
                )
                gained = curr_obj_colors - prev_obj_colors
                lost = prev_obj_colors - curr_obj_colors
                if gained or lost:
                    aff.is_click = True  # action has interaction side-effects

        elif diff.diff_type == DiffType.INDEX_CYCLE:
            aff.is_cycler = True
            aff.confidence = min(1.0, aff.confidence + 0.3)
            if self.puzzle_typology == PuzzleTypology.UNKNOWN:
                self.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            self.controllable_signature.is_discrete_selector = True
            if diff.bounding_box:
                slot = (diff.bounding_box[0], diff.bounding_box[2])
                if slot not in self.controllable_signature.active_slots:
                    self.controllable_signature.active_slots.append(slot)

        elif diff.diff_type == DiffType.CANVAS_TRANSFORMATION:
            aff.is_commit_or_stamp = True
            aff.confidence = min(1.0, aff.confidence + 0.4)
            if self.puzzle_typology not in (
                PuzzleTypology.SPATIAL_NAVIGATION,
                PuzzleTypology.DISCRETE_PERMUTATION,
            ):
                self.puzzle_typology = PuzzleTypology.CANVAS_STAMPING

        elif diff.diff_type == DiffType.NO_CHANGE:
            aff.confidence = max(0.0, aff.confidence - 0.1)

            # Trial-and-error barrier learning: if this action was previously
            # observed to cause translation, a NO_CHANGE means we hit a barrier.
            # Learn the barrier color from cells in the expected direction.
            if aff.confidence > 0.0 and (aff.delta_r != 0 or aff.delta_c != 0):
                if self.controllable_signature.color is not None:
                    avatar_pts = np.argwhere(prev_grid == self.controllable_signature.color)
                    if len(avatar_pts) > 0:
                        ar = float(np.mean(avatar_pts[:, 0]))
                        ac = float(np.mean(avatar_pts[:, 1]))
                        # Check cells in the expected movement direction
                        check_r = int(round(ar + aff.delta_r))
                        check_c = int(round(ac + aff.delta_c))
                        H, W = prev_grid.shape
                        if 0 <= check_r < H and 0 <= check_c < W:
                            blocking_color = int(prev_grid[check_r, check_c])
                            if (
                                blocking_color != 0
                                and blocking_color != self.controllable_signature.color
                                and blocking_color not in self.walkable_colors
                                and blocking_color not in self.object_recipes
                            ):
                                # Only treat as global barrier if it is structural terrain
                                # (e.g. maze wall, boundary), not a small movable item or crate
                                b_pts = np.argwhere(prev_grid == blocking_color)
                                color_count = len(b_pts)
                                if color_count >= 50:
                                    span_r = int(b_pts[:, 0].max() - b_pts[:, 0].min())
                                    span_c = int(b_pts[:, 1].max() - b_pts[:, 1].min())
                                    if span_r >= int(H * 0.35) or span_c >= int(W * 0.35):
                                        self.barrier_colors.add(blocking_color)

        # --- Object Interaction Recipe Learning (diff-type independent) ---
        # Detect object pickups, clicks, and transformations regardless of diff type.
        # This must be outside the diff-type branches because action 5 may produce
        # various diff types (TRANSLATION, NO_CHANGE, etc.) depending on game mechanics.
        if self.controllable_signature.color is not None and not np.array_equal(
            prev_grid, curr_grid
        ):
            avatar_color = self.controllable_signature.color
            bg = int(np.bincount(prev_grid.flatten()).argmax())
            excluded = {0, bg, avatar_color} | self.walkable_colors
            prev_obj_colors = {int(c) for c in np.unique(prev_grid) if int(c) not in excluded}
            curr_obj_colors = {int(c) for c in np.unique(curr_grid) if int(c) not in excluded}
            lost_colors = prev_obj_colors - curr_obj_colors
            gained_colors = curr_obj_colors - prev_obj_colors

            # Detect PICKUP: action 5 near an object causes its color to disappear
            if action == 5 and lost_colors:
                avatar_pts = np.argwhere(prev_grid == avatar_color)
                if len(avatar_pts) > 0:
                    ar = float(np.mean(avatar_pts[:, 0]))
                    ac = float(np.mean(avatar_pts[:, 1]))
                    for lost_color in lost_colors:
                        lost_pts = np.argwhere(prev_grid == lost_color)
                        if len(lost_pts) > 0:
                            lr = float(np.mean(lost_pts[:, 0]))
                            lc = float(np.mean(lost_pts[:, 1]))
                            dist = math.hypot(lr - ar, lc - ac)
                            if dist < 15:  # within interaction range
                                area = int(np.count_nonzero(prev_grid == lost_color))
                                min_r = int(np.min(lost_pts[:, 0]))
                                max_r = int(np.max(lost_pts[:, 0]))
                                min_c = int(np.min(lost_pts[:, 1]))
                                max_c = int(np.max(lost_pts[:, 1]))
                                ent_temp = SpatialEntity(
                                    id=f"lost_{lost_color}",
                                    role=EntityRole.MANIPULABLE,
                                    centroid=(lr, lc),
                                    grid_pos=(int(round(lr)), int(round(lc))),
                                    area=area,
                                    bounding_box=(min_r, max_r, min_c, max_c),
                                    color=lost_color,
                                )
                                sig_meta = ent_temp.get_compound_signature()
                                sig_key = ent_temp.get_signature_key()
                                recipe = self.object_recipes.get(lost_color)
                                if recipe is None:
                                    recipe = ObjectInteractionRecipe(
                                        object_color=lost_color,
                                        signature_key=sig_key,
                                        shape_category=sig_meta["shape_category"],
                                        size_bucket=sig_meta["size_bucket"],
                                        object_area_range=(max(1, area - 5), area + 5),
                                        interaction_action=5,
                                        outcome="pickup",
                                        confidence=0.5,
                                        times_confirmed=1,
                                    )
                                    self.object_recipes[lost_color] = recipe
                                    self.signature_recipes[sig_key] = recipe
                                    logger.info(
                                        "Recipe LEARNED: sig=%s (color=%d) + action 5 = pickup (area=%d)",
                                        sig_key,
                                        lost_color,
                                        area,
                                    )
                                else:
                                    recipe.times_confirmed += 1
                                    recipe.confidence = min(1.0, recipe.confidence + 0.2)
                                    if not recipe.signature_key:
                                        recipe.signature_key = sig_key
                                        recipe.shape_category = sig_meta["shape_category"]
                                        recipe.size_bucket = sig_meta["size_bucket"]
                                    self.signature_recipes[sig_key] = recipe

            # Detect CONTACT PICKUP: movement action causes object to disappear (walked over it)
            elif action in (1, 2, 3, 4) and lost_colors:
                avatar_pts = np.argwhere(curr_grid == avatar_color)
                if len(avatar_pts) > 0:
                    ar = float(np.mean(avatar_pts[:, 0]))
                    ac = float(np.mean(avatar_pts[:, 1]))
                    for lost_color in lost_colors:
                        lost_pts = np.argwhere(prev_grid == lost_color)
                        if len(lost_pts) > 0:
                            lr = float(np.mean(lost_pts[:, 0]))
                            lc = float(np.mean(lost_pts[:, 1]))
                            dist = math.hypot(lr - ar, lc - ac)
                            if dist < 10:  # avatar walked to where the object was
                                area = int(np.count_nonzero(prev_grid == lost_color))
                                min_r = int(np.min(lost_pts[:, 0]))
                                max_r = int(np.max(lost_pts[:, 0]))
                                min_c = int(np.min(lost_pts[:, 1]))
                                max_c = int(np.max(lost_pts[:, 1]))
                                ent_temp = SpatialEntity(
                                    id=f"lost_{lost_color}",
                                    role=EntityRole.MANIPULABLE,
                                    centroid=(lr, lc),
                                    grid_pos=(int(round(lr)), int(round(lc))),
                                    area=area,
                                    bounding_box=(min_r, max_r, min_c, max_c),
                                    color=lost_color,
                                )
                                sig_meta = ent_temp.get_compound_signature()
                                sig_key = ent_temp.get_signature_key()
                                recipe = self.object_recipes.get(lost_color)
                                if recipe is None:
                                    recipe = ObjectInteractionRecipe(
                                        object_color=lost_color,
                                        signature_key=sig_key,
                                        shape_category=sig_meta["shape_category"],
                                        size_bucket=sig_meta["size_bucket"],
                                        object_area_range=(max(1, area - 5), area + 5),
                                        interaction_action=0,  # 0 = contact (any movement)
                                        outcome="pickup",
                                        confidence=0.4,
                                        times_confirmed=1,
                                    )
                                    self.object_recipes[lost_color] = recipe
                                    self.signature_recipes[sig_key] = recipe
                                    logger.info(
                                        "Recipe LEARNED: sig=%s (color=%d) + contact = pickup (area=%d)",
                                        sig_key,
                                        lost_color,
                                        area,
                                    )
                                else:
                                    recipe.times_confirmed += 1
                                    recipe.confidence = min(1.0, recipe.confidence + 0.15)
                                    if not recipe.signature_key:
                                        recipe.signature_key = sig_key
                                        recipe.shape_category = sig_meta["shape_category"]
                                        recipe.size_bucket = sig_meta["size_bucket"]
                                    self.signature_recipes[sig_key] = recipe

            # Detect CLICK interaction: action 6 near object causes change
            elif action == 6 and (lost_colors or gained_colors):
                for changed_color in lost_colors | gained_colors:
                    ch_pts = np.argwhere(curr_grid == changed_color)
                    sig_key = None
                    shape_cat = None
                    size_b = None
                    if len(ch_pts) > 0:
                        min_r = int(np.min(ch_pts[:, 0]))
                        max_r = int(np.max(ch_pts[:, 0]))
                        min_c = int(np.min(ch_pts[:, 1]))
                        max_c = int(np.max(ch_pts[:, 1]))
                        ent_temp = SpatialEntity(
                            id=f"ch_{changed_color}",
                            role=EntityRole.ACTUATOR,
                            centroid=(float(np.mean(ch_pts[:, 0])), float(np.mean(ch_pts[:, 1]))),
                            grid_pos=(min_r, min_c),
                            area=len(ch_pts),
                            bounding_box=(min_r, max_r, min_c, max_c),
                            color=changed_color,
                        )
                        sig_meta = ent_temp.get_compound_signature()
                        sig_key = ent_temp.get_signature_key()
                        shape_cat = sig_meta["shape_category"]
                        size_b = sig_meta["size_bucket"]
                    if changed_color not in self.object_recipes:
                        recipe = ObjectInteractionRecipe(
                            object_color=changed_color,
                            signature_key=sig_key,
                            shape_category=shape_cat,
                            size_bucket=size_b,
                            interaction_action=6,
                            outcome="transform",
                            confidence=0.4,
                            times_confirmed=1,
                        )
                        self.object_recipes[changed_color] = recipe
                        if sig_key:
                            self.signature_recipes[sig_key] = recipe
                        logger.info(
                            "Recipe LEARNED: sig=%s (color=%d) + action 6 = transform",
                            sig_key,
                            changed_color,
                        )

    def is_world_model_grounded(self, available_actions: list[int]) -> bool:
        """Check if sufficient dynamics have been verified to switch to goal planning."""
        if not available_actions:
            return False
        grounded_count = sum(
            1
            for a in available_actions
            if a in self.action_affordances and self.action_affordances[a].confidence >= 0.6
        )
        return grounded_count >= min(len(available_actions), 3)

    def bind_to_new_level(self, initial_grid: np.ndarray) -> dict[str, Any]:
        """Transfer learned knowledge to a new level's initial grid.

        Returns:
            Dictionary with bound controllable centroid, goal target, and suggested mode.
        """
        bindings: dict[str, Any] = {
            "controllable_centroid": None,
            "goal_target": None,
            "ready_for_zero_shot": False,
        }

        if self.controllable_signature.color is not None:
            # Find the matching controllable entity in the new level
            matches = np.argwhere(initial_grid == self.controllable_signature.color)
            if len(matches) > 0:
                centroid = (float(np.mean(matches[:, 0])), float(np.mean(matches[:, 1])))
                bindings["controllable_centroid"] = centroid
                bindings["ready_for_zero_shot"] = self.is_world_model_grounded([1, 2, 3, 4])

        # Transfer spatial knowledge to the new level
        bindings["barrier_colors"] = set(self.barrier_colors)
        bindings["walkable_colors"] = set(self.walkable_colors)
        bindings["puzzle_typology"] = self.puzzle_typology
        bindings["action_affordances"] = dict(self.action_affordances)

        logger.info(
            f"Knowledge Transfer to Level: Typology={self.puzzle_typology.value}, "
            f"ReadyForZeroShot={bindings['ready_for_zero_shot']}, "
            f"GroundedActions={len(self.action_affordances)}, "
            f"BarrierColors={self.barrier_colors}, "
            f"WalkableColors={self.walkable_colors}"
        )
        return bindings


# ─────────────────────────────────────────────────────────────────────────────
# 2a. Trial-and-Error Feedback & Goal Induction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrialOutcome:
    """Single trial-and-error observation."""

    action: int
    position: tuple[int, int]  # avatar position at time of action
    succeeded: bool  # whether visual change was observed
    diff_type: DiffType = DiffType.NO_CHANGE
    barrier_direction: tuple[int, int] | None = None  # direction that was blocked
    objects_affected: int = 0  # count of objects that changed
    item_gained: bool = False  # did avatar acquire an item
    item_lost: bool = False  # did avatar release an item


class TrialFeedbackMemory:
    """Persistent memory of trial-and-error outcomes.

    Records successful and failed actions at specific positions so the agent
    can learn spatial affordances inductively:
      - Which directions are blocked at which positions (barrier map)
      - Where interactions produce results (pickup/drop zones)
      - Which action sequences lead to progress
    """

    def __init__(self, max_history: int = 500) -> None:
        self.outcomes: deque[TrialOutcome] = deque(maxlen=max_history)
        self.barrier_positions: dict[tuple[int, int], set[int]] = {}  # pos → blocked actions
        self.interaction_zones: dict[tuple[int, int], list[int]] = {}  # pos → effective actions
        self.consecutive_no_change: int = 0
        self.total_trials: int = 0
        self.successful_trials: int = 0
        self.progress_events: list[dict[str, Any]] = []  # scored progress moments
        self._last_object_count: int | None = None

    def record(self, outcome: TrialOutcome) -> None:
        """Record a trial outcome and update spatial knowledge."""
        self.outcomes.append(outcome)
        self.total_trials += 1

        if outcome.succeeded:
            self.successful_trials += 1
            self.consecutive_no_change = 0
            # Record interaction zones where actions had effect
            if outcome.objects_affected > 0 or outcome.item_gained or outcome.item_lost:
                self.interaction_zones.setdefault(outcome.position, []).append(outcome.action)
        else:
            self.consecutive_no_change += 1
            # Record blocked direction as barrier
            if outcome.barrier_direction is not None:
                blocked = self.barrier_positions.setdefault(outcome.position, set())
                blocked.add(outcome.action)

    def record_progress(self, step: int, description: str, score_delta: float = 0.0) -> None:
        """Record a progress event (e.g. item delivered, zone reached)."""
        self.progress_events.append(
            {
                "step": step,
                "description": description,
                "score_delta": score_delta,
            }
        )

    def is_stuck(self, threshold: int = 8) -> bool:
        """Check if the agent appears stuck (no visual change for many steps)."""
        return self.consecutive_no_change >= threshold

    def get_blocked_actions_at(self, pos: tuple[int, int], radius: float = 2.0) -> set[int]:
        """Get actions known to be blocked at or near a position."""
        blocked: set[int] = set()
        for bpos, acts in self.barrier_positions.items():
            if math.hypot(bpos[0] - pos[0], bpos[1] - pos[1]) <= radius:
                blocked.update(acts)
        return blocked

    def exploration_score(self) -> float:
        """How much of the action space has been explored (0-1)."""
        if self.total_trials == 0:
            return 0.0
        return min(1.0, self.successful_trials / max(1, self.total_trials))

    def reset_episode(self) -> None:
        """Reset per-episode state while keeping learned barriers."""
        self.consecutive_no_change = 0
        self.total_trials = 0
        self.successful_trials = 0
        self.progress_events.clear()
        self._last_object_count = None


@dataclass
class GoalHypothesis:
    """A hypothesized goal predicate learned inductively."""

    description: str
    predicate_type: str  # "items_in_zone", "color_match", "position_reach", etc.
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    times_verified: int = 0
    times_falsified: int = 0

    def score(self) -> float:
        total = self.times_verified + self.times_falsified
        if total == 0:
            return self.confidence
        return self.confidence * (self.times_verified / total)


class GoalStateInductor:
    """Induces goal predicates from trial-and-error observations.

    Watches for level completions and reverse-engineers what made them happen
    by comparing pre-completion states against earlier frames. Hypothesizes
    goal predicates like:
      - "All color-X items must be within the color-Y zone"
      - "Avatar must reach position (r, c)"
      - "All cells in region must match target pattern"

    These hypotheses are then verified on subsequent levels for zero-shot
    transfer.
    """

    def __init__(self) -> None:
        self.hypotheses: list[GoalHypothesis] = []
        self.completion_snapshots: list[dict[str, Any]] = []
        self.pre_completion_frames: deque[np.ndarray] = deque(maxlen=5)
        self.step_count: int = 0

    def observe_frame(self, grid: np.ndarray) -> None:
        """Record a frame for later comparison when level completes."""
        self.pre_completion_frames.append(grid.copy())
        self.step_count += 1

    def observe_completion(self, final_grid: np.ndarray) -> None:
        """Called when a level completes. Analyze what changed to generate hypotheses."""
        snapshot = {
            "final_grid": final_grid.copy(),
            "step_count": self.step_count,
            "color_counts": {
                int(c): int(np.count_nonzero(final_grid == c)) for c in np.unique(final_grid)
            },
        }
        self.completion_snapshots.append(snapshot)

        # Hypothesis: Items-in-zone goal
        # Look for concentrated clusters of a specific color within a bounded region
        for color in np.unique(final_grid):
            if color == 0:
                continue
            positions = np.argwhere(final_grid == color)
            if 10 <= len(positions) <= 200:
                r_range = positions[:, 0].max() - positions[:, 0].min()
                c_range = positions[:, 1].max() - positions[:, 1].min()
                if 4 <= r_range <= 20 and 4 <= c_range <= 20:
                    # Could be a target zone with items
                    hyp = GoalHypothesis(
                        description=f"Items concentrated in color-{color} zone",
                        predicate_type="items_in_zone",
                        params={
                            "zone_color": int(color),
                            "r_min": int(positions[:, 0].min()),
                            "r_max": int(positions[:, 0].max()),
                            "c_min": int(positions[:, 1].min()),
                            "c_max": int(positions[:, 1].max()),
                        },
                        confidence=0.5,
                        times_verified=1,
                    )
                    # Don't add duplicate hypotheses
                    if not any(
                        h.predicate_type == hyp.predicate_type
                        and h.params.get("zone_color") == hyp.params["zone_color"]
                        for h in self.hypotheses
                    ):
                        self.hypotheses.append(hyp)

        # Compare first and final frames to find what changed
        if self.pre_completion_frames:
            initial = self.pre_completion_frames[0]
            if initial.shape == final_grid.shape:
                diff_mask = initial != final_grid
                changed = int(np.sum(diff_mask))
                if 0 < changed < initial.size * 0.3:
                    # Focused changes suggest a goal region
                    rows, cols = np.where(diff_mask)
                    hyp = GoalHypothesis(
                        description="Changes concentrated in goal region",
                        predicate_type="region_change",
                        params={
                            "r_min": int(rows.min()),
                            "r_max": int(rows.max()),
                            "c_min": int(cols.min()),
                            "c_max": int(cols.max()),
                            "changed_count": changed,
                        },
                        confidence=0.4,
                        times_verified=1,
                    )
                    self.hypotheses.append(hyp)

    def verify_hypothesis(self, grid: np.ndarray, completed: bool) -> None:
        """Verify hypotheses against a new level's outcome."""
        for hyp in self.hypotheses:
            if hyp.predicate_type == "items_in_zone":
                zone_color = hyp.params["zone_color"]
                positions = np.argwhere(grid == zone_color)
                has_zone = len(positions) >= 10
                if completed and has_zone:
                    hyp.times_verified += 1
                    hyp.confidence = min(1.0, hyp.confidence + 0.2)
                elif not completed and has_zone:
                    hyp.times_falsified += 1
                    hyp.confidence = max(0.0, hyp.confidence - 0.1)

    def get_best_hypothesis(self) -> GoalHypothesis | None:
        """Return the highest-scoring goal hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.score())

    def reset_episode(self) -> None:
        """Reset per-episode state, keep hypotheses."""
        self.pre_completion_frames.clear()
        self.step_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Generalized Dynamic Archetype Solvers (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CausalHypothesis:
    """A causal hypothesis explaining the functional effect of an action."""

    action_id: int
    typology: str  # "TRANSLATION", "TOGGLE_CLICK", "INDEX_CYCLE", "CANVAS_MUTATION", "NO_OP"
    delta: tuple[int, int] | None = None  # (dr, dc) for translation
    confidence: float = 0.0
    evidence_count: int = 0
    suggested_solver: str | None = None


class CausalAffordanceEngine:
    """Inductively generates and ranks causal action hypotheses from visual state transitions."""

    def __init__(self) -> None:
        self.hypotheses: dict[int, CausalHypothesis] = {}

    def reset_episode(self) -> None:
        """Clear active hypotheses."""
        self.hypotheses.clear()

    def hypothesize_from_transitions(
        self,
        transitions: list[tuple[np.ndarray, int, np.ndarray]],
    ) -> dict[int, CausalHypothesis]:
        """Analyzes a series of exploratory transitions and outputs the top-ranked hypothesis per action."""
        action_diffs: dict[int, list[FrameDiff]] = {}
        for prev, act, curr in transitions:
            diff = FrameDiffAnalyzer.analyze(prev, act, curr)
            action_diffs.setdefault(act, []).append(diff)

        best_hypotheses: dict[int, CausalHypothesis] = {}

        for act, diffs in action_diffs.items():
            total = len(diffs)
            if total == 0:
                continue

            trans_counts: dict[tuple[int, int], int] = {}
            cycle_count = 0
            click_count = 0
            canvas_count = 0
            noop_count = 0

            for d in diffs:
                if d.diff_type == DiffType.TRANSLATION and d.translation_delta:
                    trans_counts[d.translation_delta] = trans_counts.get(d.translation_delta, 0) + 1
                elif d.diff_type == DiffType.INDEX_CYCLE:
                    cycle_count += 1
                elif d.diff_type == DiffType.IN_PLACE_MUTATION:
                    click_count += 1
                elif d.diff_type == DiffType.CANVAS_TRANSFORMATION:
                    canvas_count += 1
                elif d.diff_type == DiffType.NO_CHANGE:
                    noop_count += 1

            if trans_counts:
                best_delta, count = max(trans_counts.items(), key=lambda item: item[1])
                conf = count / total
                best_hypotheses[act] = CausalHypothesis(
                    action_id=act,
                    typology="TRANSLATION",
                    delta=best_delta,
                    confidence=conf,
                    evidence_count=count,
                    suggested_solver="spatial_navigation",
                )
            elif click_count > total * 0.4 or (cycle_count > total * 0.4 and act == 6):
                best_hypotheses[act] = CausalHypothesis(
                    action_id=act,
                    typology="TOGGLE_CLICK",
                    confidence=max(click_count, cycle_count) / total,
                    evidence_count=max(click_count, cycle_count),
                    suggested_solver="lights_out",
                )
            elif cycle_count > total * 0.4:
                best_hypotheses[act] = CausalHypothesis(
                    action_id=act,
                    typology="INDEX_CYCLE",
                    confidence=cycle_count / total,
                    evidence_count=cycle_count,
                    suggested_solver="tumbler",
                )
            elif canvas_count > total * 0.4:
                best_hypotheses[act] = CausalHypothesis(
                    action_id=act,
                    typology="CANVAS_MUTATION",
                    confidence=canvas_count / total,
                    evidence_count=canvas_count,
                    suggested_solver="canvas_stamping",
                )
            else:
                best_hypotheses[act] = CausalHypothesis(
                    action_id=act,
                    typology="NO_OP",
                    confidence=noop_count / total,
                    evidence_count=noop_count,
                    suggested_solver=None,
                )

        self.hypotheses = best_hypotheses
        return best_hypotheses


class TemporalHazardTracker:
    """Discovers and tracks periodic hazard oscillations across temporal frames."""

    def __init__(self) -> None:
        self.hazard_history: dict[int, set[tuple[int, int]]] = {}
        self.inferred_period: int | None = None
        self.phase_hazard_sets: dict[int, set[tuple[int, int]]] = {}

    def reset_episode(self) -> None:
        """Clear recorded frames and inferred phases."""
        self.hazard_history.clear()
        self.inferred_period = None
        self.phase_hazard_sets.clear()

    def observe_frame(
        self,
        grid: np.ndarray,
        t: int,
        hazard_colors: set[int] | None = None,
    ) -> None:
        """Record hazard coordinates present at time step t."""
        if hazard_colors is not None:
            hazard_mask = np.isin(grid, list(hazard_colors))
            coords = set((int(r), int(c)) for r, c in np.argwhere(hazard_mask))
            self.hazard_history[t] = coords

    def record_hazard_coords(self, coords: set[tuple[int, int]], t: int) -> None:
        """Explicitly record known hazard coordinates at time step t."""
        self.hazard_history[t] = set(coords)

    def detect_periodicity(self, min_period: int = 2, max_period: int = 8) -> int | None:
        """Determines if recorded hazard occurrences exhibit a repeating period T."""
        if len(self.hazard_history) < 4:
            return None

        times = sorted(self.hazard_history.keys())
        for T in range(min_period, max_period + 1):
            is_valid = True
            phases: dict[int, set[tuple[int, int]]] = {}

            for t in times:
                phi = t % T
                hazards = self.hazard_history[t]
                if phi not in phases:
                    phases[phi] = hazards
                else:
                    if phases[phi] != hazards:
                        is_valid = False
                        break

            if is_valid and len(phases) == T:
                self.inferred_period = T
                self.phase_hazard_sets = phases
                return T

        return None

    def is_safe_at(self, r: int, c: int, t: int) -> bool:
        """Checks if coordinate (r, c) is predicted to be safe at step t."""
        if self.inferred_period is not None:
            phi = t % self.inferred_period
            return (r, c) not in self.phase_hazard_sets.get(phi, set())
        return (r, c) not in self.hazard_history.get(t, set())

    def get_safe_mask(self, shape: tuple[int, int], t: int) -> np.ndarray:
        """Returns 2D boolean mask where True = safe at time step t."""
        H, W = shape
        mask = np.ones((H, W), dtype=bool)
        if self.inferred_period is not None:
            phi = t % self.inferred_period
            for r, c in self.phase_hazard_sets.get(phi, set()):
                if 0 <= r < H and 0 <= c < W:
                    mask[r, c] = False
        else:
            for r, c in self.hazard_history.get(t, set()):
                if 0 <= r < H and 0 <= c < W:
                    mask[r, c] = False
        return mask


class DynamicPermutationSolver:
    """Solves discrete combinatorial puzzles, permutation locks, and cellular toggles.

    Includes Galois Field 2 (GF(2)) Gaussian Elimination for Lights-Out puzzles
    and cyclic permutation planners for multi-dial combination locks.
    """

    @staticmethod
    def solve_gf2_linear_system(
        A: np.ndarray,  # M x N binary matrix (0 or 1)
        b: np.ndarray,  # M-dim binary vector (0 or 1)
    ) -> np.ndarray | None:
        """Solves A * x = b (mod 2) via Gauss-Jordan elimination over GF(2).

        Returns binary vector x if a solution exists, else None.
        """
        M, N = A.shape
        if len(b) != M:
            raise ValueError(f"b length {len(b)} does not match A rows {M}")

        aug = np.hstack([A.astype(np.uint8) & 1, b.astype(np.uint8).reshape(-1, 1) & 1])

        pivot_row = 0
        pivot_cols: list[int] = []

        for c in range(N):
            if pivot_row >= M:
                break

            row_indices = np.where(aug[pivot_row:, c] == 1)[0]
            if len(row_indices) == 0:
                continue
            r = pivot_row + int(row_indices[0])

            if r != pivot_row:
                aug[[pivot_row, r]] = aug[[r, pivot_row]]

            for i in range(M):
                if i != pivot_row and aug[i, c] == 1:
                    aug[i] ^= aug[pivot_row]

            pivot_cols.append(c)
            pivot_row += 1

        # Check for inconsistency: row with all 0s in A but 1 in b
        for i in range(pivot_row, M):
            if aug[i, N] == 1:
                return None

        # Extract solution x
        x = np.zeros(N, dtype=np.uint8)
        for i, c in enumerate(pivot_cols):
            x[c] = aug[i, N]

        # Verify A @ x == b (mod 2)
        if not np.array_equal((A.astype(np.uint8) @ x) % 2, b.astype(np.uint8) % 2):
            return None

        return x

    @staticmethod
    def solve_lights_out_grid(
        grid: np.ndarray,  # H x W binary array: 1 = ON, 0 = OFF
        toggle_pattern: str = "cross",
    ) -> list[tuple[int, int]] | None:
        """Computes list of (r, c) cell coordinates to toggle to turn OFF all lights."""
        H, W = grid.shape
        N = H * W
        A = np.zeros((N, N), dtype=np.uint8)

        deltas = [(0, 0)]
        if toggle_pattern == "cross":
            deltas += [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif toggle_pattern == "full3x3":
            deltas += [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for r in range(H):
            for c in range(W):
                col_idx = r * W + c
                for dr, dc in deltas:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        row_idx = nr * W + nc
                        A[row_idx, col_idx] = 1

        b = (grid.flatten() != 0).astype(np.uint8)
        x = DynamicPermutationSolver.solve_gf2_linear_system(A, b)
        if x is None:
            return None

        toggles: list[tuple[int, int]] = []
        for idx in range(N):
            if x[idx] == 1:
                toggles.append((idx // W, idx % W))
        return toggles

    @staticmethod
    def solve_cyclic_dial(
        current_val: int,
        target_val: int,
        num_states: int,
        clockwise_action: int = 1,
        counter_clockwise_action: int = 2,
    ) -> list[int]:
        """Finds shortest directional rotation action sequence between cyclic dial states."""
        if current_val == target_val or num_states <= 1:
            return []
        cw_dist = (target_val - current_val) % num_states
        ccw_dist = (current_val - target_val) % num_states

        if cw_dist <= ccw_dist:
            return [clockwise_action] * cw_dist
        else:
            return [counter_clockwise_action] * ccw_dist

    @staticmethod
    def mod_inverse(a: int, m: int) -> int | None:
        """Compute the modular multiplicative inverse of a modulo m using Extended Euclidean Algorithm."""
        a = a % m
        if a == 0:
            return None
        t, new_t = 0, 1
        r, new_r = m, a
        while new_r != 0:
            quotient = r // new_r
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r - quotient * new_r
        if r > 1:
            return None
        return (t + m) % m

    @staticmethod
    def solve_modular_linear_system(
        A: np.ndarray,  # M x N matrix
        b: np.ndarray,  # M vector
        modulus: int,
    ) -> np.ndarray | None:
        """Solves A * x = b (mod modulus) via Gauss-Jordan elimination over Z_m.

        Returns integer vector x in [0, modulus-1] if solution exists, else None.
        """
        if modulus == 2:
            return DynamicPermutationSolver.solve_gf2_linear_system(A, b)

        M, N = A.shape
        m = modulus
        aug = np.hstack([A.astype(int) % m, b.astype(int).reshape(-1, 1) % m])

        pivot_row = 0
        pivot_cols: list[int] = []

        for c in range(N):
            if pivot_row >= M:
                break

            best_r = None
            for r in range(pivot_row, M):
                val = aug[r, c] % m
                if val != 0 and math.gcd(val, m) == 1:
                    best_r = r
                    break

            if best_r is None:
                for r in range(pivot_row, M):
                    if aug[r, c] % m != 0:
                        best_r = r
                        break

            if best_r is None:
                continue

            if best_r != pivot_row:
                aug[[pivot_row, best_r]] = aug[[best_r, pivot_row]]

            pivot_val = int(aug[pivot_row, c] % m)
            inv = DynamicPermutationSolver.mod_inverse(pivot_val, m)
            if inv is not None:
                aug[pivot_row] = (aug[pivot_row] * inv) % m
                for i in range(M):
                    if i != pivot_row and aug[i, c] % m != 0:
                        factor = aug[i, c] % m
                        aug[i] = (aug[i] - factor * aug[pivot_row]) % m
                pivot_cols.append(c)
                pivot_row += 1

        x = np.zeros(N, dtype=int)
        for i, c in reversed(list(enumerate(pivot_cols))):
            val = int(aug[i, N] % m)
            for j in range(c + 1, N):
                val = (val - int(aug[i, j] % m) * int(x[j])) % m
            p_val = int(aug[i, c] % m)
            inv = DynamicPermutationSolver.mod_inverse(p_val, m)
            if inv is not None:
                x[c] = (val * inv) % m
            else:
                for candidate in range(m):
                    if (candidate * p_val) % m == val % m:
                        x[c] = candidate
                        break

        if np.array_equal((A.astype(int) @ x) % m, b.astype(int) % m):
            return x

        # Bounded lattice search fallback for small composite systems
        if N <= 6 and (m**N) <= 65536:
            import itertools

            for cand in itertools.product(range(m), repeat=N):
                c_arr = np.array(cand, dtype=int)
                if np.array_equal((A.astype(int) @ c_arr) % m, b.astype(int) % m):
                    return c_arr

        return None

    @staticmethod
    def solve_multi_dial_combination(
        current_states: list[int],
        target_states: list[int],
        num_states: int,
        dial_actions: dict[int, tuple[int, int]],  # dial_idx -> (cw_action, ccw_action)
    ) -> list[int]:
        """Plans complete action sequence to align all dials to target states."""
        actions: list[int] = []
        for i, (cur, tgt) in enumerate(zip(current_states, target_states)):
            cw_act, ccw_act = dial_actions.get(i, (1, 2))
            actions.extend(
                DynamicPermutationSolver.solve_cyclic_dial(
                    cur, tgt, num_states, clockwise_action=cw_act, counter_clockwise_action=ccw_act
                )
            )
        return actions
