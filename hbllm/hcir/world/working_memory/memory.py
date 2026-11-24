"""Prefrontal Working Memory (dlPFC) Executive Controller."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.graph import FactNode, ProcedureNode, SkillNode
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_hypothesis import InductiveHypothesisEngine
from hbllm.hcir.world.cortex_morphology import MorphologicalSaliencyEngine
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema
from hbllm.hcir.world.frontopolar_subgoal_stack import FrontopolarSubgoalStack
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.inferotemporal_segmentation import InferotemporalSegmentationEngine
from hbllm.hcir.world.macro_actions import MacroActionPlan
from hbllm.hcir.world.percept import EpistemicPhase, ExogenousSignalTracker, PlaceLocus
from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAttributor
from hbllm.hcir.world.symbolic_constraints import SymbolicConstraintSolver
from hbllm.hcir.world.visuospatial_working_memory import VisuospatialWorkingMemory
from hbllm.hcir.world.working_memory.models import (
    DomainCompetenceProfile,
    HeldItem,
    HomeostaticState,
    SubgoalSchema,
    SubgoalStage,
    WorkingMemorySalienceState,
)

logger = logging.getLogger(__name__)


class PrefrontalWorkingMemory:
    """Biologically-modeled Prefrontal Working Memory (dlPFC) for latent state and subgoal reasoning."""

    @classmethod
    def get_mirrored_fields(cls) -> frozenset[str]:
        """Set of prefrontal fields mirrored to authoritative HCIR transactions."""
        return frozenset(
            {
                "phase",
                "step_counter",
                "avatar_pos",
                "avatar_feature",
                "avatar_features",
                "tested_actions",
                "exploration_budget",
                "consecutive_plan_failures",
                "consecutive_simulation_failures",
                "simulation_cooldown",
                "exploration_cooldown",
                "pipeline_runs",
                "homeostatic_state",
            }
        )

    def __init__(self) -> None:
        self.phase: EpistemicPhase = EpistemicPhase.MOTOR_GROUNDING
        self.pipeline_runs: int = 0
        self.last_decision_branch: str | None = None
        self.evidence_provenance_log: list[dict[str, Any]] = []

        self.homeostatic_state: HomeostaticState = HomeostaticState()
        self.domain_competence_profiles: dict[str, DomainCompetenceProfile] = {}
        self.held_items: list[HeldItem] = []
        self.active_schema: SubgoalSchema | None = None
        self.procedural_library: dict[str, DeclarativeNeuroSymbolicSkill] = {}
        self.unlocked_barriers: set[tuple[int, int]] = set()

        self.tool_barrier_affinities: dict[int, set[int]] = {}  # tool_feat -> set(barrier_feat)
        self.activated_triggers: set[tuple[int, int]] = set()
        self.topological_subgoals: list[tuple[int, int]] = []
        self.active_macro_goal: str | None = None
        self.focal_attention_point: tuple[int, int] | None = None
        self.simulated_goal: tuple[int, int] | None = None
        self.last_predicted_pos: tuple[int, int] | None = None
        self.feedback_assimilated: bool = False
        self.avatar_controllability_evidence: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        self.prev_oriented_threats: list[Any] = []
        self.constraint_solver: SymbolicConstraintSolver = SymbolicConstraintSolver()
        self.executive_directives: list[str] = []

        # Mental simulation plan & active causal hypothesis
        self.mental_plan: deque[Any] = deque()
        self.active_hypothesis: Any | None = None

        # Executive metacognitive cooldowns & failure counters
        self.consecutive_simulation_failures: int = 0
        self.simulation_cooldown: int = 0
        self.consecutive_plan_failures: int = 0
        self.exploration_cooldown: int = 0
        self.consecutive_stuck_steps: int = 0

        self.last_metacognitive_assessment: Any | None = None
        self.last_scene_graph: Any | None = None
        self.active_bound_bundles: list[Any] = []

        self.visuospatial: VisuospatialWorkingMemory = VisuospatialWorkingMemory()
        self.body_schema: ExtendedBodySchema = ExtendedBodySchema()
        self.subgoal_stack: FrontopolarSubgoalStack = FrontopolarSubgoalStack()
        self.macro_plan: MacroActionPlan = MacroActionPlan()
        self.remote_causal: RemoteCausalAttributor = RemoteCausalAttributor()
        self.habenular_ior: HabenularEpisodicIOR = HabenularEpisodicIOR()
        self.it_segmenter: InferotemporalSegmentationEngine = InferotemporalSegmentationEngine()
        self.acc_conflict: AnteriorCingulateConflictMonitor = AnteriorCingulateConflictMonitor()
        self.exogenous_tracker: ExogenousSignalTracker = ExogenousSignalTracker()
        self.hypothesis_engine: InductiveHypothesisEngine = InductiveHypothesisEngine()
        self.morphology: MorphologicalSaliencyEngine = MorphologicalSaliencyEngine()
        self._salience_baseline: WorkingMemorySalienceState = WorkingMemorySalienceState()
        self.concept_role_priors: dict[int, dict[str, float]] = {}

        # Continuous metric reference frame & sub-cell kinematic tracking
        self.continuous_agent_pos: np.ndarray | tuple[float, ...] | None = None
        self.continuous_goal_pos: np.ndarray | tuple[float, ...] | None = None
        self.continuous_displacement: float = 0.0
        self.continuous_delta: tuple[float, float] = (0.0, 0.0)
        self.last_continuous_moved: bool = False
        self.reference_frame: Any | None = None
        self.last_observation_metadata: dict[str, Any] = {}
        self.compiled_linguistic_hints: set[str] = set()
        self.active_plan_step: Any | None = None
        self.shroud_feature: int | None = None
        self.allocentric_belief_grid: np.ndarray | None = None
        self.prev_raw_sensory_grid: np.ndarray | None = None
        self.collected_inventory_features: set[int] = set()
        self.pending_gate_check: bool = False

        # Exploration & Curiosity Tracking (Single Source of Truth)
        self.active_probe_target: tuple[int, int] | None = None
        self.active_probe_id: str | None = None
        self.probe_target_steps: int = 0
        self.probed_entity_ids: set[str] = set()
        self.entity_visit_counts: dict[str, int] = {}
        self.position_visit_counts: dict[tuple[int, int], int] = {}
        self.visited_cells: set[tuple[int, int]] = set()
        self.active_epistemic_conditions: dict[tuple[int, int], dict[str, Any]] = {}
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=16)
        self.recent_actions: deque[Any] = deque(maxlen=16)
        self.exhausted_candidate_goals: set[tuple[int, int]] = set()
        self.quiescent_actuation_targets: set[tuple[int, int]] = set()
        self.effective_actuation_targets: set[tuple[int, int]] = set()
        self.last_effective_actuation_coord: tuple[int, int] | None = None
        self.consecutive_effective_actuations: int = 0
        self.actuation_affordances: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.last_effector_target_step: dict[tuple[int, int], int] = {}
        self.active_goal_converging_coord: tuple[int, int] | None = None
        self.consecutive_goal_converging_actuations: int = 0

        # Active Causal Empowerment & Contingency Testing
        self.active_causal_actuator_coord: tuple[int, int] | None = None
        self.consecutive_causal_probes: int = 0
        self.causal_actuator_streak_limit: int = 4

        # Feature-level interventional causal falsification
        self.quiescent_features: set[Any] = set()
        self.effective_features: set[Any] = set()
        self.feature_quiescent_counts: dict[Any, int] = {}

        # Metacognitive Refractory Action Inhibition
        self.inhibited_actions: dict[Any, int] = {}
        self.prior_avatar_feature: Any | None = None
        self.entity_action_failed_positions: dict[Any, set[tuple[int, int]]] = {}
        self.immobile_entity_actions: set[Any] = set()

        # Universal Dynamic Action Affordance Registry
        self.action_affordances: dict[Any, Any] = {}
        self.focus_switch_actions: set[Any] = set()
        self.undo_actions: set[Any] = set()
        self.index_selector_actions: set[Any] = set()
        self.value_mutator_actions: set[Any] = set()
        self.dial_mutation_streak: int = 0
        self.last_effector_target_coord: tuple[int, int] | None = None
        self.last_effector_target_entity_id: str | None = None
        self.last_non_displacement_action: Any | None = None
        self.context_epistemic_probes: int = 0
        self.total_epistemic_probes: int = 0

        # Object Knowledge Base and Atomic Macro-Task Stores (Rule 1.8)
        self.known_objects: dict[str, Any] = {}
        self.object_macro_tasks: dict[str, list[Any]] = {}
        self.object_trial_counts: dict[str, int] = {}
        self.object_success_counts: dict[str, int] = {}

    @property
    def level_epistemic_probes(self) -> int:
        """Deprecated alias for ``context_epistemic_probes``."""
        return self.context_epistemic_probes

    @level_epistemic_probes.setter
    def level_epistemic_probes(self, val: int) -> None:
        self.context_epistemic_probes = val

    @property
    def exogenous_uncontrollable(self) -> set[PlaceLocus]:
        """Loci empirically identified as uncontrollable/exogenous (I(DeltaS; A) ~ 0)."""
        return self.exogenous_tracker.exogenous_loci

    @exogenous_uncontrollable.setter
    def exogenous_uncontrollable(self, val: set[PlaceLocus]) -> None:
        self.exogenous_tracker.exogenous_loci = set(val)

    def is_exogenous_locus(
        self,
        locus: PlaceLocus,
        *,
        lattice: Any = None,
        bg: Any = 0,
        include_outer_edge: bool = False,
    ) -> bool:
        """True if locus is empirically uncontrollable or topologically partitioned."""
        return self.exogenous_tracker.is_exogenous(
            locus,
            lattice=lattice,
            bg=bg,
            include_outer_edge=include_outer_edge,
        )

    def inhibit_action(self, action: Any, duration: int = 4) -> None:
        """Apply bounded, decaying inhibition to an action (Rule 1.8 & P2-7)."""
        if action is not None:
            self.inhibited_actions[action] = max(
                self.inhibited_actions.get(action, 0), min(int(duration), 10)
            )

    def decay_inhibited_actions(self) -> None:
        """Decay all active action inhibitions by 1 step, evicting expired ones."""
        expired = [act for act, timer in self.inhibited_actions.items() if timer <= 1]
        for act in list(self.inhibited_actions.keys()):
            self.inhibited_actions[act] -= 1
        for act in expired:
            self.inhibited_actions.pop(act, None)

    def record_domain_episode(
        self,
        domain_id: str,
        success: bool,
        mean_error: float = 0.20,
        predicted_confidence: float = 0.80,
    ) -> DomainCompetenceProfile:
        """Record domain performance outcome and update metacognitive competence profile (G9)."""
        if domain_id not in self.domain_competence_profiles:
            self.domain_competence_profiles[domain_id] = DomainCompetenceProfile(
                domain_id=domain_id
            )

        profile = self.domain_competence_profiles[domain_id]
        profile.record_episode(
            success=success,
            mean_error=mean_error,
            predicted_confidence=predicted_confidence,
        )

        self.homeostatic_state.record_progress(profile.learning_progress_rate)
        return profile

    def get_domain_competence(self, domain_id: str) -> DomainCompetenceProfile:
        """Retrieve competence profile for domain, creating initial profile if absent."""
        if domain_id not in self.domain_competence_profiles:
            self.domain_competence_profiles[domain_id] = DomainCompetenceProfile(
                domain_id=domain_id
            )
        return self.domain_competence_profiles[domain_id]

    def get_highest_priority_practice_domain(self) -> str | None:
        """Identify which domain has the highest practice priority in the Zone of Proximal Development."""
        if not self.domain_competence_profiles:
            return None
        return max(
            self.domain_competence_profiles.keys(),
            key=lambda d: self.domain_competence_profiles[d].practice_priority,
        )

    def has_salience_shift(self, engine: Any) -> bool:
        """Evaluate whether semantic discoveries, dynamic patroller kinematics, or high surprise
        warrant an executive phase shift back to forward mental simulation.
        """
        n_act = len(get_engine_state_attr(engine, "action_dynamics", {}))
        n_goals = len(get_engine_state_attr(engine, "learned_goal_positions", set()))
        n_barriers = len(get_engine_state_attr(engine, "learned_barriers", set()))
        n_traj = 0
        n_threats = len(get_engine_state_attr(engine, "mobile_threat_features", set()))

        b = self._salience_baseline
        shift = (
            n_act > b.calibrated_actions
            or n_goals > b.goal_positions
            or n_barriers != b.barriers
            or n_traj > b.trajectories
            or n_threats > b.mobile_threats
            or get_engine_state_attr(engine, "last_surprise", 0.0) >= 0.4
        )
        if shift:
            self._salience_baseline = WorkingMemorySalienceState(
                calibrated_actions=n_act,
                goal_positions=n_goals,
                barriers=n_barriers,
                trajectories=n_traj,
                mobile_threats=n_threats,
            )
        return shift

    def update_salience_baseline(self, engine: Any) -> None:
        """Explicitly update the salience baseline snapshot to current counts."""
        n_act = len(get_engine_state_attr(engine, "action_dynamics", {}))
        n_goals = len(get_engine_state_attr(engine, "learned_goal_positions", set()))
        n_barriers = len(get_engine_state_attr(engine, "learned_barriers", set()))
        n_traj = 0
        n_threats = len(get_engine_state_attr(engine, "mobile_threat_features", set()))
        self._salience_baseline = WorkingMemorySalienceState(
            calibrated_actions=n_act,
            goal_positions=n_goals,
            barriers=n_barriers,
            trajectories=n_traj,
            mobile_threats=n_threats,
        )

    def reset_salience_baseline(self) -> None:
        """Reset the salience baseline snapshot to zeroes."""
        self._salience_baseline = WorkingMemorySalienceState()

    def load_instructions(self, instructions: Sequence[str] | str | None) -> None:
        """Store executive directives in working memory to guide cognitive policies."""
        if not instructions:
            return
        if isinstance(instructions, str):
            lines = [l.strip() for l in instructions.strip().split("\n") if l.strip()]
        else:
            lines = [str(l).strip() for l in instructions if str(l).strip()]
        self.executive_directives = lines
        logger.info("PrefrontalWorkingMemory: Loaded %d executive directives.", len(lines))

    def integrate_sensory_observation(
        self,
        raw_grid: Any,
        avatar_features: set[int],
        bg_hint: Any = None,
    ) -> Any:
        """Integrate partially observable egocentric frames into a persistent allocentric belief map."""
        if raw_grid is None:
            return None
        arr = np.asarray(raw_grid)
        if arr.ndim != 2 or arr.size < 25:
            return raw_grid
        H, W = arr.shape
        if self.allocentric_belief_grid is not None and self.allocentric_belief_grid.shape != (
            H,
            W,
        ):
            self.shroud_feature = None
            self.allocentric_belief_grid = None
            self.prev_raw_sensory_grid = None

        prev = self.prev_raw_sensory_grid
        if self.shroud_feature is None:
            if prev is not None and prev.shape == (H, W):
                diff_mask = prev != arr
                if int(np.count_nonzero(diff_mask)) >= 6:
                    corners = {
                        int(arr[0, 0]),
                        int(arr[0, W - 1]),
                        int(arr[H - 1, 0]),
                        int(arr[H - 1, W - 1]),
                    } | {
                        int(prev[0, 0]),
                        int(prev[0, W - 1]),
                        int(prev[H - 1, 0]),
                        int(prev[H - 1, W - 1]),
                    }
                    for cand_s in corners:
                        newly_hidden = int(np.count_nonzero((prev != cand_s) & (arr == cand_s)))
                        newly_revealed = int(np.count_nonzero((prev == cand_s) & (arr != cand_s)))
                        if newly_hidden >= 3 and newly_revealed >= 3:
                            self.shroud_feature = cand_s
                            self.allocentric_belief_grid = prev.copy()
                            self.exogenous_tracker.reset_episode(retain_long_term=False)
                            break
            elif H >= 7 and W >= 7:
                c0 = int(arr[0, 0])
                outer_uniform = (
                    bool(np.all(arr[0, :] == c0))
                    and bool(np.all(arr[H - 1, :] == c0))
                    and bool(np.all(arr[:, 0] == c0))
                    and bool(np.all(arr[:, W - 1] == c0))
                )
                if outer_uniform:
                    inner_rings = sum(
                        [
                            bool(np.all(arr[1, :] == c0)),
                            bool(np.all(arr[H - 2, :] == c0)),
                            bool(np.all(arr[:, 1] == c0)),
                            bool(np.all(arr[:, W - 2] == c0)),
                        ]
                    )
                    non_s = arr[arr != c0]
                    if inner_rings >= 2 and non_s.size >= 9 and len(np.unique(non_s)) >= 2:
                        self.shroud_feature = c0
                        self.allocentric_belief_grid = arr.copy()
                        self.exogenous_tracker.reset_episode(retain_long_term=False)

        self.prev_raw_sensory_grid = arr.copy()
        if self.shroud_feature is None:
            return raw_grid

        s_val = self.shroud_feature
        if self.allocentric_belief_grid is None:
            self.allocentric_belief_grid = arr.copy()
        vis_mask = arr != s_val
        if avatar_features:
            floor_fill = int(bg_hint) if (bg_hint is not None and int(bg_hint) != s_val) else 0
            stale_av = (~vis_mask) & np.isin(self.allocentric_belief_grid, list(avatar_features))
            if np.any(stale_av):
                self.allocentric_belief_grid[stale_av] = floor_fill
        self.allocentric_belief_grid[vis_mask] = arr[vis_mask]
        return self.allocentric_belief_grid.copy()

    def reset_episode(
        self, retain_long_term: bool = True, retain_inhibitions: bool = False
    ) -> None:
        """Reset transient working memory for a new trial while optionally retaining cross-trial tool affinities."""
        self.held_items.clear()
        self.active_schema = None
        self.unlocked_barriers.clear()
        self.activated_triggers.clear()
        self.topological_subgoals.clear()
        self.active_macro_goal = None
        self.focal_attention_point = None
        self.simulated_goal = None
        self.last_predicted_pos = None
        self.feedback_assimilated = False
        self.mental_plan.clear()
        self.active_hypothesis = None
        self.consecutive_simulation_failures = 0
        self.simulation_cooldown = 0
        self.consecutive_plan_failures = 0
        self.exploration_cooldown = 0
        self.consecutive_stuck_steps = 0
        self.constraint_solver.reset_episode()
        self.visuospatial.reset_episode(retain_long_term=retain_long_term)
        self.body_schema.reset_episode(retain_dynamics=retain_long_term)
        self.subgoal_stack.reset_episode()
        self.active_bound_bundles.clear()
        self.macro_plan.clear()
        self.remote_causal.reset_episode(retain_long_term=retain_long_term)
        self.acc_conflict.reset_episode(retain_long_term=retain_long_term)
        self.exogenous_tracker.reset_episode(retain_long_term=retain_long_term)
        self.hypothesis_engine.reset_episode(retain_confirmed=retain_long_term)
        self.active_probe_target = None
        self.active_probe_id = None
        self.probe_target_steps = 0
        self.continuous_agent_pos = None
        self.continuous_goal_pos = None
        self.continuous_displacement = 0.0
        self.continuous_delta = (0.0, 0.0)
        self.last_continuous_moved = False
        self.active_plan_step = None
        self.reference_frame = None
        self.last_observation_metadata.clear()
        self.compiled_linguistic_hints.clear()
        self.shroud_feature = None
        self.allocentric_belief_grid = None
        self.prev_raw_sensory_grid = None
        self.collected_inventory_features.clear()
        self.pending_gate_check = False
        self.recent_positions.clear()
        self.recent_actions.clear()
        self.visited_cells.clear()
        self.active_epistemic_conditions.clear()
        self.consecutive_effective_actuations = 0
        self.consecutive_goal_converging_actuations = 0
        self.active_causal_actuator_coord = None
        self.consecutive_causal_probes = 0
        self.context_epistemic_probes = 0
        self.dial_mutation_streak = 0
        self.last_effector_target_coord = None
        self.last_effector_target_entity_id = None
        if not retain_inhibitions:
            self.inhibited_actions.clear()
        if not retain_long_term:
            self.undo_actions.clear()
            self.index_selector_actions.clear()
            self.value_mutator_actions.clear()
            self.tool_barrier_affinities.clear()
            self.habenular_ior.reset()
            self.avatar_controllability_evidence.clear()
            self.prev_oriented_threats.clear()
            self._salience_baseline = WorkingMemorySalienceState()
            self.probed_entity_ids.clear()
            self.entity_visit_counts.clear()
            self.position_visit_counts.clear()
            self.exhausted_candidate_goals.clear()
            self.quiescent_actuation_targets.clear()
            self.effective_actuation_targets.clear()
            self.last_effective_actuation_coord = None
            self.actuation_affordances.clear()
            self.last_effector_target_step.clear()
            self.active_goal_converging_coord = None
            self.quiescent_features.clear()
            self.effective_features.clear()
            self.feature_quiescent_counts.clear()
            self.prior_avatar_feature = None
            self.entity_action_failed_positions.clear()
            self.immobile_entity_actions.clear()
            self.action_affordances.clear()
            self.focus_switch_actions.clear()
            self.last_non_displacement_action = None
            self.total_epistemic_probes = 0
            self.known_objects.clear()
            self.object_macro_tasks.clear()
            self.object_trial_counts.clear()
            self.object_success_counts.clear()
        else:
            self.habenular_ior.reset_episode(
                retain_long_term=True,
                retain_inhibitions=retain_inhibitions,
            )

    def update_continuous_goal_from_lattice(
        self,
        grid: np.ndarray | None,
        theory: Any,
        avatar_features: Any,
        avatar_feature: int | None,
        bg_feature: int | None,
    ) -> None:
        """Lift active lattice goal or singleton waypoint into continuous metric coordinates."""
        if self.reference_frame is None or self.continuous_agent_pos is None or grid is None:
            return
        if grid.ndim < 2:
            return
        H, W = grid.shape
        rf = self.reference_frame
        origin_desc = rf.origin_description or "bottom_left_metric"
        x_b = rf.bounds
        x_min = float(x_b[0][0]) if (x_b and len(x_b) >= 1) else 0.0
        x_max = float(x_b[0][1]) if (x_b and len(x_b) >= 1) else float(W)
        y_min = float(x_b[1][0]) if (x_b and len(x_b) >= 2) else 0.0
        y_max = float(x_b[1][1]) if (x_b and len(x_b) >= 2) else float(H)

        target_rc: tuple[int, int] | None = None
        for gf in theory.goal_features:
            coords = np.argwhere(grid == gf)
            if len(coords) > 0:
                target_rc = (int(coords[0][0]), int(coords[0][1]))
                break
        if target_rc is None:
            bg_v = int(bg_feature) if bg_feature is not None else int(grid[1, 1])
            av_s = set(avatar_features or ())
            if avatar_feature is not None:
                av_s.add(int(avatar_feature))
            av_s.add(int(grid[0, 0]))
            vals, cnts = np.unique(grid, return_counts=True)
            for v_np, c_np in sorted(
                zip(vals.tolist(), cnts.tolist(), strict=False), key=lambda vc: vc[1]
            ):
                vf = int(v_np)
                if (
                    vf != bg_v
                    and vf not in av_s
                    and 1 <= int(c_np) <= 2
                    and not theory.is_barrier(vf)
                ):
                    coords = np.argwhere(grid == vf)
                    if len(coords) > 0:
                        target_rc = (int(coords[0][0]), int(coords[0][1]))
                        break
        if target_rc is not None:
            gr, gc = target_rc
            cell_w = (x_max - x_min) / max(1, W)
            cell_h = (y_max - y_min) / max(1, H)
            cont_x = x_min + (float(gc) + 0.5) * cell_w
            cont_y = (
                y_max - (float(gr) + 0.5) * cell_h
                if "bottom_left" in origin_desc
                else y_min + (float(gr) + 0.5) * cell_h
            )
            self.continuous_goal_pos = np.array([cont_x, cont_y], dtype=float)
        # If target_rc is None (e.g. agent overlaps/occludes the goal on the discrete lattice),
        # retain self.continuous_goal_pos so continuous servoing can reach exact convergence.

    def register_cardinality_constraint(
        self,
        center: tuple[int, int],
        count: int,
        grid_shape: tuple[int, int],
        radius: int = 1,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Register a local numerical count constraint and run unit propagation."""
        self.constraint_solver.register_observation(center, count, grid_shape, radius)
        return self.constraint_solver.propagate_constraints()

    def get_unrevealed_safe_cells(self) -> set[tuple[int, int]]:
        """Query cells deduced by prefrontal constraint solver to be guaranteed safe."""
        return self.constraint_solver.get_unrevealed_safe_cells()

    def register_topological_doorway(self, door_coord: tuple[int, int], target_room: int) -> None:
        """Register a doorway transition as an active macro-subgoal in prefrontal memory."""
        self.topological_subgoals.append(door_coord)
        self.active_macro_goal = f"doorway_to_room_{target_room}"

    def acquire_item(
        self,
        feature_id: int,
        role: str,
        step: int,
        position: tuple[int, int],
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Store an acquired entity into prefrontal working memory (latent object permanence)."""
        if not any(item.feature_id == feature_id for item in self.held_items):
            self.held_items.append(
                HeldItem(
                    feature_id=feature_id,
                    role=role,
                    acquired_step=step,
                    original_position=position,
                    properties=properties or {},
                )
            )
            self.body_schema.acquire_tool(feature_id=feature_id, step=step, role=role)
            logger.info(
                "WorkingMemory: Acquired item [feature=%d, role=%s] into latent store at step %d.",
                feature_id,
                role,
                step,
            )

    def expend_item(self, feature_id: int) -> bool:
        """Remove a consumable item from working memory when applied to a barrier."""
        self.body_schema.expend_tool(feature_id)
        for i, item in enumerate(self.held_items):
            if item.feature_id == feature_id:
                self.held_items.pop(i)
                logger.info("WorkingMemory: Expended held item [feature=%d].", feature_id)
                return True
        return False

    def is_holding(self, feature_id: int) -> bool:
        """Check if an item of specified feature is currently held in working memory."""
        return any(item.feature_id == feature_id for item in self.held_items)

    def register_tool_unlock(self, tool_feature: int, barrier_feature: int) -> None:
        """Learn causal resonance: tool_feature unlocks barriers of barrier_feature."""
        self.tool_barrier_affinities.setdefault(tool_feature, set()).add(barrier_feature)
        self.body_schema.register_resonance(tool_feature, barrier_feature)
        logger.info(
            "WorkingMemory: Learned tool-barrier resonance: Tool %d unlocks Barrier %d.",
            tool_feature,
            barrier_feature,
        )

    @property
    def held_feature_ids(self) -> set[int]:
        """Set of feature IDs for all items currently retained in latent working memory."""
        return {item.feature_id for item in self.held_items}

    def formulate_tool_use_schema(
        self,
        avatar_pos: tuple[int, int],
        tools: Sequence[tuple[int, int, int]],  # (r, c, feature_id)
        barriers: Sequence[tuple[int, int, int]],  # (r, c, feature_id)
        goals: Sequence[tuple[int, int]],
    ) -> SubgoalSchema | None:
        """Formulate a 3-stage prefrontal behavioral schema:

        Stage 1: Acquire Tool/Key.
        Stage 2: Transport & Unlock Barrier.
        Stage 3: Navigate to Primary Goal.
        """
        if not goals:
            return None

        # If already holding a tool that matches an obstructing barrier:
        held_feats = self.held_feature_ids

        stages: list[SubgoalStage] = []

        # Find barrier that obstructs goals
        obstructing_barriers = [
            b for b in barriers if any(abs(b[0] - g[0]) + abs(b[1] - g[1]) <= 3 for g in goals)
        ]
        target_barrier_cells = {(b[0], b[1]) for b in obstructing_barriers} or {
            (b[0], b[1]) for b in barriers
        }

        if not held_feats and tools:
            # Stage 1: Acquire nearest tool
            best_tool = min(
                tools, key=lambda t: abs(t[0] - avatar_pos[0]) + abs(t[1] - avatar_pos[1])
            )
            stages.append(
                SubgoalStage(
                    stage_id="ACQUIRE_TOOL",
                    target_role="tool",
                    target_positions={(best_tool[0], best_tool[1])},
                    required_feature=best_tool[2],
                )
            )

        if target_barrier_cells:
            # Stage 2: Apply to Barrier
            stages.append(
                SubgoalStage(
                    stage_id="UNLOCK_BARRIER",
                    target_role="barrier",
                    target_positions=target_barrier_cells,
                )
            )

        # Stage 3: Reach Final Goal
        stages.append(
            SubgoalStage(
                stage_id="REACH_GOAL",
                target_role="goal",
                target_positions=set(goals),
            )
        )

        schema = SubgoalSchema(
            schema_name="TOOL_ASSISTED_NAVIGATION",
            stages=stages,
        )
        self.active_schema = schema
        return schema

    def synthesize_skill_from_schema(
        self,
        schema: SubgoalSchema,
        required_actions: set[int] | Sequence[int] | None = None,
    ) -> DeclarativeNeuroSymbolicSkill:
        """Synthesize a DeclarativeNeuroSymbolicSkill from an executed/learned SubgoalSchema."""
        subgoal_seq = schema.to_subgoal_sequence()
        signature_pred = (
            ActionAffordancePredicate(required=required_actions) if required_actions else AllOf()
        )

        skill_cls_name = f"SynthesizedSkill_{schema.schema_name}"
        skill_instance = type(
            skill_cls_name,
            (DeclarativeNeuroSymbolicSkill,),
            {
                "skill_name": f"skill_{schema.schema_name.lower()}",
                "description": f"Synthesized neuro-symbolic procedural skill for {schema.schema_name}",
                "signature": signature_pred,
                "program": subgoal_seq,
            },
        )()

        self.procedural_library[schema.schema_name] = skill_instance
        logger.info(
            "PrefrontalWorkingMemory: Synthesized declarative procedural skill '%s' (%d subgoals)",
            skill_instance.skill_name,
            len(subgoal_seq),
        )
        return skill_instance

    def register_procedural_skill(
        self,
        schema_name: str,
        skill: DeclarativeNeuroSymbolicSkill,
    ) -> None:
        """Register a procedural skill in working memory procedural library."""
        self.procedural_library[schema_name] = skill

    def query_procedural_skill(self, schema_name: str) -> DeclarativeNeuroSymbolicSkill | None:
        """Retrieve compiled procedural skill from working memory procedural library."""
        return self.procedural_library.get(schema_name)

    def compose_macro_skill(
        self,
        skill_names: Sequence[str],
        macro_name: str,
        description: str = "",
    ) -> DeclarativeNeuroSymbolicSkill | None:
        """Hierarchically compose multiple declarative skills from the procedural library into a macro-skill."""
        from hbllm.hcir.skills.declarative import CompositeDeclarativeSkill

        sub_skills = [
            self.procedural_library[name] for name in skill_names if name in self.procedural_library
        ]
        if not sub_skills:
            return None

        composite = CompositeDeclarativeSkill(
            skills=sub_skills,
            skill_name=macro_name,
            description=description,
        )
        self.procedural_library[macro_name] = composite
        logger.info(
            "PrefrontalWorkingMemory: Composed macro-skill '%s' from %d sub-skills: %s",
            macro_name,
            len(sub_skills),
            skill_names,
        )
        return composite

    def instantiate_skill_as_schema(
        self,
        skill_name: str,
        target_positions_override: dict[str, set[tuple[int, int]]] | None = None,
    ) -> SubgoalSchema | None:
        """Instantiate a stored DeclarativeNeuroSymbolicSkill as an active SubgoalSchema."""
        skill = self.procedural_library.get(skill_name)
        if skill is None:
            return None

        stages: list[SubgoalStage] = []
        if getattr(skill, "program", None) is not None and hasattr(skill.program, "subgoals"):
            for sg in skill.program.subgoals:
                q = getattr(sg, "target_query", {})
                stage_id = q.get("stage_id", "STAGE")
                override_targets = (target_positions_override or {}).get(stage_id)
                if override_targets is not None:
                    t_pos = set(override_targets)
                else:
                    t_pos = set(tuple(p) for p in q.get("target_positions", []))

                stages.append(
                    SubgoalStage(
                        stage_id=stage_id,
                        target_role=q.get("role", "goal"),
                        target_positions=t_pos,
                        required_feature=q.get("required_feature"),
                    )
                )

        if not stages:
            stages.append(
                SubgoalStage(
                    stage_id="EXECUTE_SKILL",
                    target_role="goal",
                    target_positions=set(),
                )
            )

        schema = SubgoalSchema(
            schema_name=skill_name,
            stages=stages,
        )
        self.active_schema = schema
        logger.info(
            "PrefrontalWorkingMemory: Instantiated procedural skill '%s' as active SubgoalSchema (%d stages)",
            skill_name,
            len(stages),
        )
        return schema

    def find_matching_skill(
        self,
        available_actions: Sequence[int] | set[int] | None = None,
        context: Any = None,
    ) -> tuple[str, DeclarativeNeuroSymbolicSkill] | None:
        """Match stored procedural skills against available actions and context signature."""
        curr_epoch = (
            context.get("context_epoch", context.get("current_level"))
            if isinstance(context, dict)
            else getattr(context, "context_epoch", getattr(context, "current_level", None))
        )
        for name, skill in self.procedural_library.items():
            proc_macro = getattr(skill, "_procedural_macro", None)
            if proc_macro is not None and curr_epoch is not None:
                macro_epoch = getattr(
                    proc_macro, "context_epoch", getattr(proc_macro, "level", None)
                )
                if macro_epoch != curr_epoch:
                    continue
            sig = getattr(skill, "signature", None)
            if sig is None:
                return name, skill
            if hasattr(sig, "matches"):
                action_set = set(available_actions) if available_actions is not None else None
                if action_set is not None and sig.matches(action_set):
                    return name, skill
            else:
                return name, skill
        return None

    def handle_schema_failure(self, failure_pos: tuple[int, int] | None = None) -> bool:
        """Record failure during active schema execution.
        Invalidates active schema if consecutive failures reach threshold (>= 2).
        Returns True if schema was invalidated, False otherwise.
        """
        if self.active_schema is None:
            return False

        self.active_schema.failure_count += 1
        self.active_schema.confidence = max(0.0, self.active_schema.confidence - 0.5)
        logger.warning(
            "PrefrontalWorkingMemory: Active schema '%s' encountered failure (count=%d, confidence=%.2f)",
            self.active_schema.schema_name,
            self.active_schema.failure_count,
            self.active_schema.confidence,
        )

        if self.active_schema.failure_count >= 2 or self.active_schema.confidence <= 0.0:
            logger.warning(
                "PrefrontalWorkingMemory: Invalidating active schema '%s' due to recurring failures.",
                self.active_schema.schema_name,
            )
            self.active_schema = None
            return True
        return False

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export working memory state into declarative causal predicates."""
        predicates = [item.to_causal_predicate() for item in self.held_items]
        for tool_feat, barrier_feats in self.tool_barrier_affinities.items():
            for bf in barrier_feats:
                predicates.append(
                    CausalPredicate(
                        variable="tool_barrier_affinity",
                        operator="==",
                        value=bf,
                        relation=f"tool_{tool_feat}",
                        target_entity="avatar",
                    )
                )
        return predicates

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions reflecting latent object permanence, active schemas, and procedural skills."""
        ops: list[TransactionOperation] = []
        for item in self.held_items:
            ops.extend(item.to_transaction_operations())
        if self.active_schema:
            node_id = f"active_schema_{self.active_schema.schema_name}"
            node = ProcedureNode(
                id=node_id,
                name=self.active_schema.schema_name,
                description=f"Active prefrontal subgoal schema {self.active_schema.schema_name}",
                steps=[s.stage_id for s in self.active_schema.stages],
                properties={
                    "schema_name": self.active_schema.schema_name,
                    "current_stage_idx": self.active_schema.current_stage_idx,
                    "is_finished": self.active_schema.is_finished,
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )
        for name, skill in self.procedural_library.items():
            node_id = f"procedural_skill_{name}"
            skill_name = getattr(skill, "skill_name", name) or name
            desc = getattr(skill, "description", "") or f"Procedural skill for {name}"
            prog = getattr(skill, "program", None)
            if prog is not None:
                subgoal_count = (
                    len(prog.subgoals)
                    if hasattr(prog, "subgoals")
                    else (len(prog) if hasattr(prog, "__len__") else 0)
                )
            else:
                subgoal_count = 0

            node = SkillNode(
                id=node_id,
                name=skill_name,
                description=desc,
                properties={
                    "schema_name": name,
                    "subgoal_count": subgoal_count,
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )
        for barrier_pos in self.unlocked_barriers:
            node_id = f"unlocked_barrier_{barrier_pos[0]}_{barrier_pos[1]}"
            node = FactNode(
                id=node_id,
                statement=f"Barrier at {barrier_pos} has been unlocked",
                properties={
                    "position": list(barrier_pos),
                    "unlocked": True,
                    "fact_type": "unlocked_barrier",
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )

        ops.extend(self.visuospatial.to_transaction_operations())
        ops.extend(self.habenular_ior.to_transaction_operations())
        ops.extend(self.homeostatic_state.to_transaction_operations())
        return ops

    # ── Homeostatic Regulation & Social Directives (WP10) ────────────

    @property
    def homeostatic(self) -> HomeostaticState:
        """Alias for homeostatic_state."""
        return self.homeostatic_state

    def get_homeostatic_state(self) -> HomeostaticState:
        """Return the current prefrontal homeostatic state."""
        return self.homeostatic_state

    def consume_energy(self, amount: float) -> float:
        """Consume energy budget."""
        return self.homeostatic_state.consume_energy(amount)

    def replenish_energy(self, amount: float) -> float:
        """Replenish energy budget."""
        return self.homeostatic_state.replenish_energy(amount)

    def add_social_instruction(self, instruction: dict[str, Any]) -> None:
        """Add an explicit social instruction."""
        self.homeostatic_state.add_instruction(instruction)

    def clear_social_instructions(self) -> None:
        """Clear all social instructions."""
        self.homeostatic_state.clear_instructions()
