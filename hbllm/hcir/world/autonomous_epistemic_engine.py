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

import logging
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.active_inference import ActiveInferenceEngine
from hbllm.hcir.world.bilateral_coordination import (
    BilateralCoordinateIntegrator,
    BilateralState,
)
from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    CausalHypothesis,
)
from hbllm.hcir.world.cerebellar_phase_clock import (
    CerebellarPhaseClock,
)
from hbllm.hcir.world.counterfactual_simulation import (
    CounterfactualDeadlockDetector,
)
from hbllm.hcir.world.extended_body_schema import (
    ExtendedBodySchema,
)
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoalStack,
)
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.inferotemporal_segmentation import (
    InferotemporalSegmentationEngine,
)
from hbllm.hcir.world.intuitive_physics import IntuitivePhysicsEngine
from hbllm.hcir.world.kinetic_stream import (
    DorsalKineticStream,
)
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.object_state_graph import ObjectStateGraphPlanner
from hbllm.hcir.world.optical_ray_projection import (
    OpticalRayProjector,
)
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.hcir.world.remote_causal_attribution import (
    RemoteCausalAttributor,
)
from hbllm.hcir.world.spatial_containment import RoomDoor, RoomTopologyExtractor
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones
from hbllm.hcir.world.spatiotemporal_tracker import SpatiotemporalHazardTracker
from hbllm.hcir.world.surprise_engine import SurpriseEngine, SurpriseEvaluation
from hbllm.perception.saccadic_attention import SaccadicAttentionSystem

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cognitive Phases and Cortical Faculties Modular Imports
# ─────────────────────────────────────────────────────────────────────────────


from hbllm.hcir.world.cortex_assimilator import (
    EpistemicFeedbackAssimilator,
    EpistemicPhase,
)
from hbllm.hcir.world.cortex_causal import (
    ActionAffordance,
    CausalInductionCortex,
    HCIRSymbolicWorldTheory,
)
from hbllm.hcir.world.cortex_episodic import (
    HippocampalEpisodicCortex,
)
from hbllm.hcir.world.cortex_motor import (
    ACCConflictMonitor,
    EpistemicCuriosityExplorer,
    MotorCortexEffector,
)
from hbllm.hcir.world.cortex_perception import (
    EpistemicObservationDiff,
    OrientedThreat,
    PerceptionEngine,
)
from hbllm.hcir.world.cortex_planner import (
    MentalSimulationPlanner,
    MentalSimulationStep,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Master Autonomous Epistemic Engine Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class AutonomousEpistemicEngine:
    """Domain-agnostic cognitive engine for autonomous exploration, hypothesis testing,
    and forward mental simulation.
    """

    MIN_PROBES_PER_ACTION: int = 3

    def __init__(
        self,
        exploration_budget: int = 150,
        enable_logging: bool = True,
        instructions: Sequence[str] | str | None = None,
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
        self._avatar_feature: int | None = None
        self.avatar_features: set[int] = set()
        self.avatar_pos: tuple[int, int] | None = None
        self.avatar_size: int = 1

        # Controllability-based avatar identification: accumulates evidence
        # across multiple frames/actions to distinguish the avatar (responds
        # to player input) from other moving entities (autonomous movement).
        # Maps feature_id -> list of (action_id, observed_delta) observations.
        self._avatar_controllability_evidence: dict[int, list[tuple[int, tuple[int, int]]]] = {}

        # Experiential verification: features the avatar has successfully
        # traversed without dying. Replaces hardcoded pixel-count thresholds
        # with empirical "I walked on this and survived" knowledge.
        self.verified_safe_features: set[int] = set()

        # Intensional Neuro-Symbolic World Theory
        self.symbolic_theory: HCIRSymbolicWorldTheory = HCIRSymbolicWorldTheory()

        # Motor calibration & dynamics
        self.action_dynamics: dict[int, ActionDynamicsModel] = {}
        self.tested_actions: set[int] = set()
        self.tested_action_positions: set[tuple[int, tuple[int, int] | None]] = set()

        # State Mutation Models (Switches, buttons, door toggles)
        self.state_mutations: list[StateMutationModel] = []

        # Episodic Ground Facts (Layout coordinates, preserved per-level across attempts)
        self._current_level: int = 0
        self.level_failed_transitions: dict[int, set[tuple[tuple[int, int], int]]] = {}
        self.level_learned_barriers: dict[int, set[tuple[int, int]]] = {}
        self.level_lethal_positions: dict[int, set[tuple[int, int]]] = {}
        self.learned_barriers: set[tuple[int, int]] = self.level_learned_barriers.setdefault(
            0, set()
        )
        self.learned_goal_positions: set[tuple[int, int]] = set()
        self.learned_receptacle_positions: set[tuple[int, int]] = set()

        # Mental simulation plan
        self.mental_plan: deque[MentalSimulationStep] = deque()
        self.active_hypothesis: CausalHypothesis | None = None
        self.failed_transitions: set[tuple[tuple[int, int], int]] = (
            self.level_failed_transitions.setdefault(0, set())
        )

        # Theory of Mind: creature kinds observed to patrol autonomously
        # (feature-level concept, retained across levels like a human would).
        self.mobile_threat_features: set[int] = set()
        self._prev_oriented_threats: list[OrientedThreat] = []
        self.consecutive_plan_failures: int = 0
        self.consecutive_stuck_steps: int = 0
        self.exploration_cooldown: int = 0
        self.consecutive_simulation_failures: int = 0
        self.simulation_cooldown: int = 0
        self._last_sim_goal_count: int = 0
        self._last_sim_barrier_count: int = 0
        self._last_sim_action_count: int = 0
        self.current_simulated_goal: tuple[int, int] | None = None
        self.last_predicted_pos: tuple[int, int] | None = None

        # Prefrontal working memory & biological hazard tracker
        self.working_memory: PrefrontalWorkingMemory = PrefrontalWorkingMemory()
        if instructions:
            self.load_instructions(instructions)
        self.hazard_tracker: SpatiotemporalHazardTracker = SpatiotemporalHazardTracker()
        self.saccadic_attention: SaccadicAttentionSystem = SaccadicAttentionSystem()
        self.physics_engine: IntuitivePhysicsEngine = IntuitivePhysicsEngine()

        # Faculty C: Dorsal Visual Stream (Area MT/V5) Kinetic Figure-Ground Segregation
        self.dorsal_kinetic_stream: DorsalKineticStream = DorsalKineticStream()

        # Faculty D: Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring)
        self.bilateral_integrator: BilateralCoordinateIntegrator = BilateralCoordinateIntegrator()
        self.bilateral_state: BilateralState | None = None

        # Faculty E: Basal Ganglia & Cerebellar Predictive Phase Entrainment
        self.cerebellar_clock: CerebellarPhaseClock = CerebellarPhaseClock()
        self.pending_phase_wait_steps: int = 0

        # Faculty: Premotor Spatiotemporal Collision Cones & Trajectory Extrapolation
        self.collision_cones: SpatiotemporalCollisionCones = SpatiotemporalCollisionCones()

        # Faculty: Orbitofrontal Cortex (OFC) Counterfactual Deadlock Detector
        self.deadlock_detector: CounterfactualDeadlockDetector = CounterfactualDeadlockDetector()

        # Faculty: Brodmann Area 10 (aPFC) Hierarchical Subgoal Stack & Cognitive Branching
        self.subgoal_stack: FrontopolarSubgoalStack = self.working_memory.subgoal_stack

        # Faculty: Inferior Parietal Lobule (IPL) Extended Body Schema & Tool Affordances
        self.body_schema: ExtendedBodySchema = self.working_memory.body_schema

        # Faculty: Anterior Mid-Cingulate Cortex (aMCC) & Lateral Habenula Episodic IOR
        self.habenular_ior: HabenularEpisodicIOR = self.working_memory.habenular_ior

        # Faculty: Ventromedial Prefrontal Cortex (vmPFC) Remote Causal Attributor
        self.remote_causal: RemoteCausalAttributor = self.working_memory.remote_causal

        # Faculty: Parieto-Occipital Mental Imagery (V6/MST) Optical Ray Projector
        self.optical_projector: OpticalRayProjector = self.working_memory.optical_projector

        # Faculty: Inferotemporal Cortex (IT / Ventral Stream) Affordance Centroid Segmentation
        self.it_segmenter: InferotemporalSegmentationEngine = self.working_memory.it_segmenter

        # Faculty: Hippocampal & Lateral Habenula Episodic Memory (Trial-and-Error Loop)
        self.episodic_cortex: HippocampalEpisodicCortex = HippocampalEpisodicCortex(
            habenular_ior=self.habenular_ior,
        )

        # Faculty: vmPFC Causal Induction Cortex (Pearl's do(a) intervention & tool resonance)
        self.causal_cortex: CausalInductionCortex = CausalInductionCortex(
            symbolic_theory=self.symbolic_theory,
            causal_attributor=self.remote_causal,
            body_schema=self.body_schema,
        )

        # Faculty: Anterior Cingulate Cortex (ACC) Motor Conflict Monitor
        self.acc_conflict_monitor: ACCConflictMonitor = ACCConflictMonitor()

        # Room topology & doorway subgoal reasoning (spatial containment)
        self.room_topology: RoomTopologyExtractor = RoomTopologyExtractor()

        # Object-Centric Macro-Action State Graph Planner (5 Executive Directives)
        self.object_planner: ObjectStateGraphPlanner = ObjectStateGraphPlanner()
        self.topology_rooms: dict[int, list[tuple[int, int]]] = {}
        self.topology_doors: list[RoomDoor] = []
        self.room_adjacency: dict[int, list[int]] = {}
        self.current_room_id: int | None = None

        # Confidence-scaled predictive coding surprise engine
        self.surprise_engine: SurpriseEngine = SurpriseEngine(surprise_threshold=0.15)
        self.last_surprise: float = 0.0
        self.last_surprise_eval: SurpriseEvaluation | None = None

        # Free energy active inference decision selection
        self.active_inference: ActiveInferenceEngine = ActiveInferenceEngine(
            w_reward=0.35,
            w_info_gain=0.30,
            w_future_val=0.15,
            w_risk=0.10,
            w_cost=0.10,
        )

        # Active exploration & curiosity state
        self.active_probe_target: tuple[int, int] | None = None
        self.active_probe_id: str | None = None
        self.probe_target_steps: int = 0
        self.probed_entity_ids: set[str] = set()
        self.entity_visit_counts: dict[str, int] = {}
        self.position_visit_counts: dict[tuple[int, int], int] = {}
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=16)
        self.recent_actions: deque[Any] = deque(maxlen=16)
        self.exhausted_candidate_goals: set[tuple[int, int]] = set()
        self.quiescent_click_targets: set[tuple[int, int]] = set()
        self.effective_click_targets: set[tuple[int, int]] = set()
        self.last_effective_click_coord: tuple[int, int] | None = None
        self.consecutive_effective_clicks: int = 0
        self.click_affordances: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.last_effector_target_step: dict[tuple[int, int], int] = {}
        self.active_goal_converging_coord: tuple[int, int] | None = None
        self.consecutive_goal_converging_clicks: int = 0

        # Feature-level interventional causal falsification (Piagetian Stage D3)
        self.quiescent_features: set[int] = set()
        self.effective_features: set[int] = set()

        # Metacognitive Refractory Action Inhibition (Stage D13)
        self.inhibited_actions: dict[Any, int] = {}
        self.prior_avatar_feature: int | None = None
        self.entity_action_failed_positions: dict[Any, set[tuple[int, int]]] = {}
        self.immobile_entity_actions: set[Any] = set()

        # Universal Dynamic Action Affordance Registry
        self.action_affordances: dict[Any, ActionAffordance] = {}

        self.bg_feature: int = 0
        self.level_epistemic_probes: int = 0
        self.total_epistemic_probes: int = 0
        self._feedback_assimilated: bool = False

        self.hypotheses: list[CausalHypothesis] = []
        self.belief_history: list[BeliefTransitionEvent] = []

    def is_action_sufficiently_probed(self, action_id: Any) -> bool:
        """Check if an action has been tested enough times to have a reliable dynamics model."""
        dyn = self.action_dynamics.get(action_id)
        if dyn is None:
            return False
        probes = getattr(dyn, "probes_tested", 0)
        return probes >= self.MIN_PROBES_PER_ACTION

    def register_action_space(self, action_specs: Sequence[Any]) -> None:
        """Register external action definitions dynamically from peripheral driver or device descriptor.

        Accepts:
        - List of dicts (from SynapticDeviceDescriptor.action_schema):
          e.g. [{'action_id': 1, 'name': 'MOVE_UP'}, {'action_id': 6, 'name': 'CLICK_CELL', 'parameters': {'x': 'int', 'y': 'int'}}]
        - List of ActionAffordance instances
        - List of bare action IDs (int, str, Enum)
        """
        for spec in action_specs:
            aff = self._parse_action_spec(spec)
            self.action_affordances[aff.action_id] = aff

    def _parse_action_spec(self, spec: Any) -> ActionAffordance:
        return MotorCortexEffector.parse_action_spec(spec)

    def is_spatial_effector(self, action: Any) -> bool:
        """Returns True if the action is configured or observed to accept spatial coordinates."""
        aff = self.action_affordances.get(action)
        if aff is not None:
            return aff.requires_spatial_target
        return False

    def is_displacement_action(self, action: Any) -> bool:
        """Returns True if this action causes physical self/ego-motion displacement."""
        dyn = self.action_dynamics.get(action)
        if dyn is not None and dyn.is_displacement_action():
            return True
        aff = self.action_affordances.get(action)
        if aff is not None and aff.is_displacement:
            return True
        return False

    def is_action_calibrated(self, action: Any) -> bool:
        """Returns True if empirical dynamics for this action have been observed."""
        return action in self.action_dynamics

    @staticmethod
    def normalize_sensory_input(raw: Any) -> np.ndarray:
        """Normalize arbitrary sensory observations (camera, depth, lidar, 2D grid) into a spatial 2D array."""
        return PerceptionEngine.normalize_sensory_input(raw)

    # ── Property Pass-Throughs to HCIRSymbolicWorldTheory ─────────────────────

    @property
    def avatar_feature(self) -> int | None:
        if self._avatar_feature is not None:
            return self._avatar_feature
        if self.avatar_features:
            return next(iter(self.avatar_features))
        return None

    @avatar_feature.setter
    def avatar_feature(self, val: int | None) -> None:
        self._avatar_feature = val
        if val is not None:
            if not self.avatar_features or val not in self.avatar_features:
                self.avatar_features = {val}
        else:
            self.avatar_features.clear()

    @property
    def learned_goal_features(self) -> set[int]:
        return self.symbolic_theory.goal_features

    @learned_goal_features.setter
    def learned_goal_features(self, val: set[int]) -> None:
        self.symbolic_theory.goal_features = val

    @property
    def learned_barrier_features(self) -> set[int]:
        return self.symbolic_theory.barrier_features

    @learned_barrier_features.setter
    def learned_barrier_features(self, val: set[int]) -> None:
        self.symbolic_theory.barrier_features = val

    @property
    def learned_walkable_features(self) -> set[int]:
        return self.symbolic_theory.walkable_features

    @learned_walkable_features.setter
    def learned_walkable_features(self, val: set[int]) -> None:
        self.symbolic_theory.walkable_features = val

    @property
    def learned_cargo_features(self) -> set[int]:
        return self.symbolic_theory.cargo_features

    @learned_cargo_features.setter
    def learned_cargo_features(self, val: set[int]) -> None:
        self.symbolic_theory.cargo_features = val

    @property
    def learned_receptacle_features(self) -> set[int]:
        return self.symbolic_theory.receptacle_features

    @learned_receptacle_features.setter
    def learned_receptacle_features(self, val: set[int]) -> None:
        self.symbolic_theory.receptacle_features = val

    # ── Epistemic status of feature values ──────────────────────────────────────
    UNVERIFIED_FEATURE_COST: float = 8.0

    def is_feature_unverified(self, feat: int, bg: int) -> bool:
        """True if the agent has neither experience nor theory about ``feat``.

        A feature is 'known' if it is background, part of the avatar, walked on
        and survived, or classified by the symbolic theory (walkable, barrier,
        goal, candidate goal, cargo). Anything else is epistemically unknown.
        """
        if feat == bg or feat in self.verified_safe_features:
            return False
        av = self.avatar_features or (
            {self.avatar_feature} if self.avatar_feature is not None else set()
        )
        if feat in av:
            return False
        th = self.symbolic_theory
        if th.is_walkable(feat) or th.is_barrier(feat):
            return False
        if (
            feat in th.goal_features
            or feat in th.candidate_goal_features
            or feat in th.cargo_features
        ):
            return False
        return True

    def _epistemic_uncertainty_cost(self, feat: int, bg: int) -> float:
        """Soft exploration cost for stepping onto an unverified feature."""
        return self.UNVERIFIED_FEATURE_COST if self.is_feature_unverified(feat, bg) else 0.0

    def load_instructions(self, instructions: Sequence[str] | str | None) -> None:
        """Load executive cognitive directives into working memory to guide epistemic policies."""
        self.working_memory.load_instructions(instructions)

    @property
    def current_level(self) -> int:
        return self._current_level

    @current_level.setter
    def current_level(self, level: int) -> None:
        self.set_current_level(level)

    def set_current_level(self, level: int) -> None:
        if self._current_level != level:
            self._current_level = level
            self.failed_transitions = self.level_failed_transitions.setdefault(level, set())
            self.learned_barriers = self.level_learned_barriers.setdefault(level, set())
            self.hazard_tracker.static_lethal_positions = self.level_lethal_positions.setdefault(
                level, set()
            )

    # ── Episodic State Management ─────────────────────────────────────────────

    def reset_episode(
        self,
        retain_dynamics: bool = True,
        is_new_level: bool = False,
        level: int | None = None,
    ) -> None:
        """Reset episodic state upon level transition or death."""
        if level is not None:
            self.set_current_level(level)
        self.prev_grid = None
        self._prev_oriented_threats = []
        self.avatar_pos = None
        self.last_action = None
        self.last_action_data = None
        self.last_predicted_pos = None
        self.bilateral_state = None
        self.pending_phase_wait_steps = 0
        self.consecutive_quiescent_actions = 0

        self.consecutive_stuck_steps = 0
        self.entity_action_failed_positions = {}
        self.immobile_entity_actions = set()
        self.last_effective_click_coord = None
        self.consecutive_effective_clicks = 0
        self.mental_plan.clear()
        self.active_hypothesis = None
        self.active_probe_target = None
        self.active_probe_id = None
        self.probe_target_steps = 0
        self.recent_positions.clear()
        if hasattr(self, "recent_actions"):
            self.recent_actions.clear()
        self.level_epistemic_probes = 0
        self.working_memory.reset_episode(retain_long_term=retain_dynamics)
        self.collision_cones.reset_episode()
        self.consecutive_plan_failures = 0
        self.exploration_cooldown = 0
        self.consecutive_simulation_failures = 0
        self.simulation_cooldown = 0
        self._last_sim_goal_count = 0
        self._last_sim_barrier_count = 0
        self._last_sim_action_count = 0
        self.current_simulated_goal = None
        self._feedback_assimilated = False

        if not retain_dynamics:
            self.level_failed_transitions.clear()
            self.level_learned_barriers.clear()
            self.level_lethal_positions.clear()
            self._current_level = 0 if level is None else level
            self.failed_transitions = self.level_failed_transitions.setdefault(
                self._current_level, set()
            )
            self.learned_barriers = self.level_learned_barriers.setdefault(
                self._current_level, set()
            )
            self.hazard_tracker.clear_lethal_positions()
            self.hazard_tracker.reset_episode()
            self.physics_engine.reset_episode()
            self.exhausted_candidate_goals.clear()
            self.tested_action_positions.clear()
            self.learned_goal_positions.clear()
            self.learned_receptacle_positions.clear()
            self.topology_rooms.clear()
            self.topology_doors.clear()
            self.room_adjacency.clear()
            self.current_room_id = None
            self.surprise_engine.reset_ledger()
            self.last_surprise = 0.0
            self.last_surprise_eval = None
        elif is_new_level:
            # Level transition: point episodic spatial memory to the target level
            self.failed_transitions = self.level_failed_transitions.setdefault(
                self._current_level, set()
            )
            self.learned_barriers = self.level_learned_barriers.setdefault(
                self._current_level, set()
            )
            self.hazard_tracker.static_lethal_positions = self.level_lethal_positions.setdefault(
                self._current_level, set()
            )
            self.hazard_tracker.reset_episode()
            self.physics_engine.reset_episode()
            self.exhausted_candidate_goals.clear()
            self.tested_action_positions.clear()
            self.learned_goal_positions.clear()
            self.learned_receptacle_positions.clear()
            self.topology_rooms.clear()
            self.topology_doors.clear()
            self.room_adjacency.clear()
            self.current_room_id = None
            self.surprise_engine.reset_ledger()
            self.last_surprise = 0.0
            self.last_surprise_eval = None
            self.collision_cones.reset_episode()
            self.mobile_threat_features.clear()
            self._prev_oriented_threats = []
            self.prev_grid = None
            self.last_action = None
            self.last_predicted_pos = None
            self.mental_plan.clear()
            self.consecutive_simulation_failures = 0
            self.simulation_cooldown = 0
        else:
            # Same level retry on death
            self.hazard_tracker.grid_history.clear()
            self.hazard_tracker.step_history.clear()
            self.tested_action_positions.clear()
            self.last_surprise = 0.0
            self.last_surprise_eval = None
            self.collision_cones.reset_episode()
            self.prev_grid = None
            self.last_action = None
            self.last_predicted_pos = None
            self.mental_plan.clear()
            self._prev_oriented_threats = []

        if is_new_level and retain_dynamics:
            # ── Principled Hypothesis Management on Level Transition ──────────
            #
            # The human brain carries forward hypotheses and VERIFIES them through experience.
            # 1. Avatar identity is retained as a prior hypothesis: in almost all multi-level games,
            #    the player character maintains their visual identity across level stages.
            #    We record prior_avatar_feature, keep avatar_feature as an active prior,
            #    and clear controllability evidence so discrepancy can trigger re-grounding if needed.
            if self.avatar_feature is not None:
                self.prior_avatar_feature = self.avatar_feature
            self._avatar_controllability_evidence.clear()

            # 2. Feature-value semantics are carried forward as PRIORS, not
            #    facts. People assume "same appearance → same meaning" in a new
            #    level until experience contradicts it. Contradictions are
            #    already handled online: colliding marks a barrier, dying marks
            #    a feature lethal, surviving a traversal marks it safe. Wiping
            #    priors would discard e.g. which feature is the goal.
            #    (symbolic_theory, verified_safe_features and known lethal
            #    features are intentionally retained.)
            #
            #    Action-conditioned state mutations are layout-specific
            #    (a particular switch at a particular place) and are cleared.
            self.state_mutations.clear()
            self.bilateral_state = None
            if hasattr(self, "bilateral_integrator"):
                self.bilateral_integrator.active_bilateral_state = None

            # 3. Spatial/episodic state is level-specific
            self.entity_visit_counts.clear()
            self.probed_entity_ids.clear()
            self.quiescent_click_targets.clear()
            self.effective_click_targets.clear()
            self.click_affordances.clear()
            self.quiescent_features.clear()
            self.last_effector_target_step.clear()
            self.active_goal_converging_coord = None
            self.consecutive_goal_converging_clicks = 0
            # self.effective_features is retained as semantic affordance prior across levels
            self.consecutive_quiescent_actions = 0
            self.inhibited_actions.clear()

            # 4. Action dynamics and affordances are RETAINED — they represent
            #    structural motor knowledge (e.g., "action 1 moves up by 6 pixels")
            #    that is typically level-invariant.
            self.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if self.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING
            )

        elif not retain_dynamics:
            # Full Cross-Game Isolation: Zero cross-game leakage
            self.total_epistemic_probes = 0
            self.avatar_feature = None
            self.avatar_features.clear()
            self.prior_avatar_feature = None
            self._avatar_controllability_evidence.clear()
            self.verified_safe_features.clear()
            self.avatar_pos = None
            self.action_dynamics.clear()
            self.action_affordances.clear()
            self.tested_actions.clear()
            self.symbolic_theory.clear()
            self.state_mutations.clear()
            self.entity_visit_counts.clear()
            self.position_visit_counts.clear()
            self.probed_entity_ids.clear()
            self.quiescent_click_targets.clear()
            self.effective_click_targets.clear()
            self.click_affordances.clear()
            self.quiescent_features.clear()
            self.effective_features.clear()
            self.consecutive_quiescent_actions = 0
            self.inhibited_actions.clear()
            self.topology_rooms.clear()
            self.topology_doors.clear()
            self.room_adjacency.clear()
            self.current_room_id = None
            self.surprise_engine.reset_ledger()
            self.last_surprise = 0.0
            self.last_surprise_eval = None
            self.hazard_tracker.known_lethal_features.clear()
            self.hazard_tracker.periodic_cells.clear()
            self.mobile_threat_features.clear()
            self.phase = EpistemicPhase.MOTOR_GROUNDING
        else:
            self.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if self.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING
            )

        if hasattr(self, "object_planner"):
            self.object_planner.reset_episode(is_new_level=is_new_level)

        # Cortical Faculties Episode Reset
        if hasattr(self, "episodic_cortex") and not retain_dynamics:
            self.episodic_cortex.reset()
        if hasattr(self, "causal_cortex"):
            self.causal_cortex.reset_episode(retain_dynamics=retain_dynamics)
        if hasattr(self, "acc_conflict_monitor"):
            self.acc_conflict_monitor.reset()

    def is_motor_grounded(self) -> bool:
        """True if the agent has identified its avatar and calibrated directional actions."""
        return (self.avatar_feature is not None or bool(self.avatar_features)) and any(
            dyn.confidence >= 0.7 for dyn in self.action_dynamics.values()
        )

    def compute_exploration_budget(self, n_actions: int) -> int:
        """Adaptive budget: more probing when more unknowns exist."""
        if not self.is_motor_grounded():
            calibrated = sum(1 for d in self.action_dynamics.values() if d.confidence >= 0.7)
            uncalibrated = max(0, n_actions - calibrated)
            return max(
                self.exploration_budget,
                uncalibrated * self.MIN_PROBES_PER_ACTION * 3,
            )
        confirmed_count = (
            len(self.symbolic_theory.goal_features)
            + len(self.symbolic_theory.barrier_features)
            + len(self.state_mutations)
        )
        scaling = max(0.2, 1.0 - 0.15 * confirmed_count)
        return max(15, int(self.exploration_budget * scaling))

    # ── Delegated Public Interfaces ───────────────────────────────────────────

    def estimate_background(self, grid: np.ndarray) -> int:
        return PerceptionEngine.estimate_background(
            grid,
            barrier_features=self.symbolic_theory.barrier_features,
            avatar_features=self.avatar_features,
            bg_feature=self.bg_feature,
        )

    def extract_entities(self, grid: np.ndarray, bg: int) -> list[SpatialEntity]:
        return PerceptionEngine.extract_entities(
            grid,
            bg,
            symbolic_theory=self.symbolic_theory,
            avatar_features=self.avatar_features,
            state_mutations=self.state_mutations,
            learned_barriers=self.learned_barriers,
            learned_goal_positions=self.learned_goal_positions,
            learned_receptacle_positions=self.learned_receptacle_positions,
        )

    def compute_frame_diff(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray
    ) -> EpistemicObservationDiff:
        return PerceptionEngine.compute_frame_diff(prev_grid, curr_grid)

    def detect_structural_goals(
        self, grid: np.ndarray, bg: int | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        bg_val = bg if bg is not None else self.bg_feature
        return PerceptionEngine.detect_structural_goals(
            grid,
            bg=bg_val,
            avatar_features=self.avatar_features,
            avatar_feature=self.avatar_feature,
            barrier_features=self.symbolic_theory.barrier_features,
            known_lethal_features=self.hazard_tracker.known_lethal_features,
            avatar_pos=self.avatar_pos,
            **kwargs,
        )

    def assimilate_feedback(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        is_win: bool = False,
        is_lost: bool = False,
        action: int | None = None,
        action_data: dict[str, Any] | None = None,
    ) -> None:
        EpistemicFeedbackAssimilator.assimilate(
            self,
            curr_grid,
            available_actions,
            is_win=is_win,
            is_lost=is_lost,
            action=action,
            action_data=action_data,
        )

    def simulate_in_mind(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        return MentalSimulationPlanner.simulate_in_mind(
            self,
            curr_grid,
            available_actions,
            target_role=target_role,
            blocked_cells=blocked_cells,
        )

    def plan_epistemic_probe(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        return EpistemicCuriosityExplorer.plan_epistemic_probe(self, curr_grid, available_actions)

    def update_room_topology(self, curr_grid: np.ndarray) -> None:
        """Extract rooms and doorways from current occupancy grid knowledge."""
        H, W = curr_grid.shape
        occupancy = np.ones((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if (
                    (r, c) in self.learned_barriers
                    or self.symbolic_theory.is_barrier(val)
                    or val in self.hazard_tracker.known_lethal_features
                ):
                    occupancy[r, c] = False

        rooms, doors = self.room_topology.extract_rooms_and_doors(occupancy, min_room_size=4)
        self.topology_rooms = rooms
        self.topology_doors = doors
        self.room_adjacency = self.room_topology.build_adjacency_graph(rooms, doors)

        # Localize current room
        self.current_room_id = None
        if self.avatar_pos is not None:
            for rid, coords in rooms.items():
                if self.avatar_pos in coords:
                    self.current_room_id = rid
                    break

    def _update_avatar_position_from_grid(
        self, curr_grid: np.ndarray, known_av_feats: set[int]
    ) -> None:
        """Visually localize the avatar position on the current grid."""
        H, W = curr_grid.shape
        from scipy.ndimage import label

        mask = np.isin(curr_grid, list(known_av_feats))
        if not np.any(mask):
            return

        labeled, num_features = label(mask)
        if num_features == 0:
            return

        # If avatar was already precisely localized by assimilate_feedback via motion tracking,
        # and current avatar_pos contains an avatar feature and matches avatar_size, verify and keep it
        if self.avatar_pos is not None:
            r, c = self.avatar_pos
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                comp_lbl = labeled[r, c]
                if comp_lbl > 0:
                    coords = np.argwhere(labeled == comp_lbl)
                    min_r, max_r = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
                    min_c, max_c = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))
                    is_edge_hud = (
                        min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
                    ) or (
                        min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4)
                    )
                    if not is_edge_hud:
                        comp_area = len(coords)
                        if self.avatar_size == 0 or (
                            0.25 * self.avatar_size <= comp_area <= 2.5 * self.avatar_size
                        ):
                            return

        best_pos = None
        min_score = float("inf")

        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            area = len(coords)
            if 1 <= area <= max(49, int(H * W * 0.15)):
                min_r, max_r = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
                min_c, max_c = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))
                # Filter out peripheral border lines / HUD gauges
                is_edge_hud = (
                    min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
                ) or (min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4))
                if is_edge_hud:
                    continue

                centroid = (
                    int(round(float(np.mean(coords[:, 0])))),
                    int(round(float(np.mean(coords[:, 1])))),
                )
                if self.avatar_pos is not None:
                    dist = abs(centroid[0] - self.avatar_pos[0]) + abs(
                        centroid[1] - self.avatar_pos[1]
                    )
                    size_diff = abs(area - self.avatar_size) if self.avatar_size > 0 else 0
                    score = dist + size_diff * 0.5
                else:
                    # Fresh localization: choose component best matching known avatar_size
                    score = abs(area - self.avatar_size) if self.avatar_size > 0 else 0
                if score < min_score:
                    min_score = score
                    best_pos = centroid

        if best_pos is not None:
            self.avatar_pos = best_pos

    def ground_effector_action(self, curr_grid: np.ndarray, action: Any = None) -> dict[str, Any]:
        """Spatially ground an allocentric effector command onto salient affordances."""
        return MotorCortexEffector.ground_effector_action(self, curr_grid, action)

    # ── Theory of Mind: Creature Motion Observation ──────────────────────────

    def infer_motor_step_size(self, available_actions: Sequence[Any]) -> int:
        """Largest calibrated displacement quantum (the world's 'cell' size)."""
        return MotorCortexEffector.infer_motor_step_size(available_actions, self.action_dynamics)

    def _observe_oriented_threat_motion(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        av_feats: set[int],
    ) -> None:
        """Learn which creature kinds patrol by watching them translate.

        A human watching a creature notices: "that one walked one cell along its
        facing direction while I moved — it patrols". Stationary sentries never
        translate. This is a feature-level concept, so it generalizes to other
        creatures of the same kind in later levels.
        """
        if not av_feats:
            self._prev_oriented_threats = []
            return
        try:
            bg = self.estimate_background(curr_grid)
            entities = self.extract_entities(curr_grid, bg)
            threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                entities,
                bg=bg,
                step_size=self.infer_motor_step_size(available_actions),
                avatar_pos=self.avatar_pos,
                avatar_features=av_feats,
            )
        except Exception:
            self._prev_oriented_threats = []
            return

        prev = self._prev_oriented_threats
        if prev:
            prev_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in prev:
                prev_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            curr_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in threats:
                curr_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            for feat, cur_positions in curr_by_feat.items():
                if feat in av_feats or feat in self.mobile_threat_features:
                    continue
                old_positions = prev_by_feat.get(feat, set())
                appeared = cur_positions - old_positions
                vanished = old_positions - cur_positions
                for a in appeared:
                    if any(
                        (a[0] == v[0]) != (a[1] == v[1])  # pure cardinal translation
                        for v in vanished
                    ):
                        self.mobile_threat_features.add(feat)
                        self.consecutive_simulation_failures = 0
                        self.simulation_cooldown = 0
                        self.mental_plan.clear()
                        self.phase = EpistemicPhase.MENTAL_SIMULATION
                        logger.info(
                            "AutonomousEpistemicEngine: Observed creature kind %d patrolling "
                            "autonomously — modeling its trajectory.",
                            feat,
                        )
                        break
        self._prev_oriented_threats = threats

    # ── Master Decision Loop ──────────────────────────────────────────────────

    def decide(
        self,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action_schemas: Sequence[Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision function:
        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        """
        self.step_counter += 1
        if action_schemas:
            self.register_action_space(action_schemas)

        curr_grid = self.normalize_sensory_input(curr_grid)

        # 1. Update background & assimilate sensory feedback from previous action
        self.bg_feature = self.estimate_background(curr_grid)
        if self.prev_grid is not None and not self._feedback_assimilated:
            self.assimilate_feedback(curr_grid, available_actions, is_win=is_win, is_lost=is_lost)
        self._feedback_assimilated = False

        # Phasic Salience Reset: Environmental discovery or high surprise awakens deliberate forward search
        n_calibrated_actions = len(self.action_dynamics)
        if (
            len(self.learned_goal_positions) > self._last_sim_goal_count
            or len(self.learned_barriers) != self._last_sim_barrier_count
            or n_calibrated_actions > getattr(self, "_last_sim_action_count", 0)
            or self.last_surprise >= 0.4
        ):
            self.simulation_cooldown = 0
            self.consecutive_simulation_failures = 0
            if n_calibrated_actions > getattr(self, "_last_sim_action_count", 0):
                self.mental_plan.clear()
                self.phase = (
                    EpistemicPhase.MENTAL_SIMULATION
                    if self.is_motor_grounded()
                    else EpistemicPhase.EPISTEMIC_EXPLORATION
                )
                self._last_sim_action_count = n_calibrated_actions
            self._last_sim_goal_count = len(self.learned_goal_positions)
            self._last_sim_barrier_count = len(self.learned_barriers)

        # Metacognitive Refractory Inhibition: decay refractory timers
        if self.simulation_cooldown > 0:
            self.simulation_cooldown -= 1
        to_uninhibited = [act for act, timer in self.inhibited_actions.items() if timer <= 1]
        for act in self.inhibited_actions:
            self.inhibited_actions[act] -= 1
        for act in to_uninhibited:
            self.inhibited_actions.pop(act, None)

        # Anterior Mid-Cingulate & Lateral Habenula: Limit Cycle / Oscillation Detection
        osc_act = self.working_memory.habenular_ior.detect_action_oscillation()
        if osc_act is not None:
            self.inhibited_actions[osc_act] = max(self.inhibited_actions.get(osc_act, 0), 4)

        # Faculty: Anterior Cingulate Cortex (ACC) Limit Cycle & Conflict Suppression
        if self.is_motor_grounded() and hasattr(self, "acc_conflict_monitor"):
            self.acc_conflict_monitor.record_step(self.last_action, self.avatar_pos)
            is_osc, cyclic_acts = self.acc_conflict_monitor.detect_oscillation()
            if is_osc:
                for cyc_a in cyclic_acts:
                    self.inhibited_actions[cyc_a] = max(self.inhibited_actions.get(cyc_a, 0), 4)

        # Visual avatar localization
        known_av_feats = set(self.avatar_features) if self.avatar_features else set()
        if self.avatar_feature is not None:
            known_av_feats.add(self.avatar_feature)
        elif self.prior_avatar_feature is not None:
            known_av_feats.add(self.prior_avatar_feature)
        if known_av_feats:
            self._update_avatar_position_from_grid(curr_grid, known_av_feats)
            if (
                self.avatar_pos is not None
                and self.avatar_feature is None
                and self.prior_avatar_feature is not None
            ):
                self.avatar_feature = self.prior_avatar_feature
                self.avatar_features.add(self.prior_avatar_feature)

        # Observe creature motion: oriented entities that translated since the
        # previous frame are autonomous patrollers, not stationary sentries.
        self._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        # Faculty E: Basal Ganglia Motor Gating & Phase Entrainment
        if self.pending_phase_wait_steps > 0:
            self.pending_phase_wait_steps -= 1
            non_disp = [
                a
                for a in available_actions
                if not self.is_displacement_action(a) and not self.is_spatial_effector(a)
            ]
            if non_disp:
                return non_disp[0], None

        # Faculty: Brodmann Area 10 (aPFC) Hierarchical Subgoal Stack & Cognitive Branching
        if self.subgoal_stack.has_pending_goals:
            cur_sg = self.subgoal_stack.current_subgoal
            if cur_sg is not None and self.subgoal_stack.is_subgoal_satisfied(
                cur_sg, set(), avatar_pos=self.avatar_pos
            ):
                self.subgoal_stack.pop_subgoal()

        chosen_action: Any
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None

        # Prefrontal Affordance Panel Sequence Chunking:
        # If an interactive control panel has unvisited pop-out/minority items, commit to completing the pattern.
        # Parietal Affordance Gating: Trigger if:
        # 1. Pure click environment (no displacement actions).
        # 2. Hybrid environment where avatar is not yet grounded or is trapped in a stuck/oscillation loop.
        active_panel_target: tuple[int, int] | None = None
        has_displacement_actions = any(
            self.is_displacement_action(a) for a in available_actions
        ) or any(
            not self.is_spatial_effector(a) and a not in self.action_dynamics
            for a in available_actions
        )
        spatial_effector_actions = [a for a in available_actions if self.is_spatial_effector(a)]
        has_immobile_failures = bool(
            getattr(self, "immobile_entity_actions", None)
            and len(self.recent_actions) >= 2
            and any(a in self.immobile_entity_actions for a in list(self.recent_actions)[-2:])
        )
        is_stuck_or_looping = (
            self.consecutive_stuck_steps >= 2
            or (self.avatar_pos is not None and self.recent_positions.count(self.avatar_pos) >= 2)
            or (self.avatar_pos is None and self.step_counter > 4)
            or has_immobile_failures
        )
        has_aligned_optical_subgoal = False
        if self.avatar_pos is not None and spatial_effector_actions:
            bg_eval = self.estimate_background(curr_grid)
            sgs = PerceptionEngine.detect_structural_goals(
                curr_grid,
                bg=bg_eval,
                avatar_features=known_av_feats,
                avatar_feature=self.avatar_feature,
            )
            for sg in sgs:
                if (
                    sg.get("type")
                    in ("relational_alignment", "optical_mirror_target", "reflection_target")
                    and sg.get("is_midpoint")
                    and (
                        self.avatar_feature is None
                        or sg.get("feature") == self.avatar_feature
                        or sg.get("feature") in self.avatar_features
                    )
                ):
                    mp = sg["position"]
                    if (
                        abs(mp[1] - self.avatar_pos[1]) <= 1
                        and abs(mp[0] - self.avatar_pos[0]) <= 2
                    ):
                        has_aligned_optical_subgoal = True
                        break

        discrete_transform_actions = [
            a
            for a in available_actions
            if not self.is_displacement_action(a) and not self.is_spatial_effector(a)
        ]

        if (
            (spatial_effector_actions or discrete_transform_actions)
            and (
                not has_displacement_actions
                or is_stuck_or_looping
                or (has_aligned_optical_subgoal and not self.mental_plan)
            )
            and self.step_counter > 1
        ):
            bg = self.estimate_background(curr_grid)
            entities = self.extract_entities(curr_grid, bg)
            panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
            for p in panels:
                unvisited_minority = [
                    m.grid_pos
                    for m in p["minority_items"]
                    if self.entity_visit_counts.get(f"click_{m.grid_pos[0]}_{m.grid_pos[1]}", 0)
                    == 0
                    and m.grid_pos not in self.quiescent_click_targets
                ]
                if unvisited_minority:
                    active_panel_target = unvisited_minority[0]
                    break

            if active_panel_target is None:
                # Target unvisited or alternate optical mirrors to switch effector control
                optical_goals = [
                    g
                    for g in PerceptionEngine.detect_structural_goals(
                        curr_grid,
                        bg=bg,
                        avatar_features=known_av_feats,
                        avatar_feature=self.avatar_feature,
                    )
                    if g.get("type")
                    in ("relational_alignment", "optical_mirror_target", "reflection_target")
                    and g.get("position") is not None
                    and not g.get("is_midpoint")
                ]
                optical_goals.sort(key=lambda g: g.get("confidence", 0.5), reverse=True)
                for og in optical_goals:
                    mp = og["position"]
                    if self.avatar_pos is not None:
                        dist_l1 = abs(mp[0] - self.avatar_pos[0]) + abs(mp[1] - self.avatar_pos[1])
                        same_col = abs(mp[1] - self.avatar_pos[1]) <= 3
                        same_row = abs(mp[0] - self.avatar_pos[0]) <= 3
                        if dist_l1 < 8 or same_col or same_row:
                            continue  # Already controlling this entity or its axis
                    if self.entity_visit_counts.get(f"click_{mp[0]}_{mp[1]}", 0) < 4:
                        active_panel_target = mp
                        break

        if active_panel_target is not None and spatial_effector_actions:
            chosen_action = spatial_effector_actions[0]
            tr, tc = active_panel_target
            self.entity_visit_counts[f"click_{tr}_{tc}"] = (
                self.entity_visit_counts.get(f"click_{tr}_{tc}", 0) + 1
            )
            aff = self.action_affordances.get(chosen_action)
            param_keys = aff.target_param_keys if aff else ("x", "y")
            chosen_data = {}
            for k in param_keys:
                if k in ("x", "col", "c", "column", "azimuth"):
                    chosen_data[k] = tc
                elif k in ("y", "row", "r", "elevation", "distance"):
                    chosen_data[k] = tr
                else:
                    chosen_data[k] = 0
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
        elif (
            (not spatial_effector_actions or active_panel_target is None)
            and is_stuck_or_looping
            and discrete_transform_actions
        ):
            uninhibited_discrete = [
                a for a in discrete_transform_actions if self.inhibited_actions.get(a, 0) == 0
            ]
            chosen_action = (
                uninhibited_discrete[0] if uninhibited_discrete else discrete_transform_actions[0]
            )
            chosen_data = None
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION

        # 2. Check exploration cooldown
        elif self.exploration_cooldown > 0:
            self.exploration_cooldown -= 1
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

        # 3. Check active mental plan during EXPLOITATION phase
        elif self.phase == EpistemicPhase.EXPLOITATION and self.mental_plan:
            next_step = self.mental_plan.popleft()
            if (
                not self.mental_plan
                and hasattr(self, "object_planner")
                and self.object_planner.active_target_object_id
            ):
                self.object_planner.ledger.record_interaction(
                    self.object_planner.active_target_object_id,
                    next_step.predicted_avatar_pos or self.avatar_pos or (0, 0),
                )
                self.object_planner.active_target_object_id = None

            # ── Reactive Safety Check ──────────────────────────────────────
            # Before executing each plan step, validate safety with TWO
            # complementary checks:
            #
            # A) STATIC CHECK: Verify no stationary lethal entity has appeared
            #    on a cell the plan will traverse (e.g. a sentry we hadn't
            #    seen before, or a barrier that opened/closed).
            #
            # B) DYNAMIC PATROL CHECK: Forward-simulate the current patroller
            #    positions step-by-step along the remaining plan.  The mental
            #    plan was computed with patrol prediction, but patrollers may
            #    have deviated from the predicted trajectory since then.
            #    Re-simulating from the ACTUAL current positions detects
            #    collisions the stale plan can't anticipate.
            plan_safe = True
            H, W = curr_grid.shape
            bg = self.estimate_background(curr_grid)
            all_future_steps = [next_step] + list(self.mental_plan)

            # (A) Static feature check along the planned path
            for future_step in all_future_steps:
                fr, fc = future_step.predicted_avatar_pos
                if 0 <= fr < H and 0 <= fc < W:
                    feat = int(curr_grid[fr, fc])
                    if feat in self.mobile_threat_features:
                        # Patrollers are modeled dynamically below; their
                        # CURRENT grid position says nothing about future safety.
                        continue
                    if feat in self.hazard_tracker.known_lethal_features:
                        plan_safe = False
                        break

            # (A2) Premotor Spatiotemporal Collision Cones check:
            # If the avatar is currently under impending collision course (looming threat),
            # or the immediate step collides with a projected kinetic entity:
            if plan_safe and self.avatar_pos is not None:
                ar, ac = self.avatar_pos
                is_stationary = (
                    next_step.predicted_avatar_pos is None
                    or next_step.predicted_avatar_pos == (ar, ac)
                )
                if is_stationary and any(
                    self.collision_cones.is_collision_hazard(
                        ar, ac, t_f, ignored_features=self.mobile_threat_features
                    )
                    for t_f in (1, 2)
                ):
                    plan_safe = False
                elif next_step.predicted_avatar_pos is not None:
                    nr, nc = next_step.predicted_avatar_pos
                    if self.collision_cones.is_collision_hazard(
                        nr, nc, 1, ignored_features=self.mobile_threat_features
                    ):
                        plan_safe = False

            # (B) Dynamic patrol trajectory simulation
            #     Re-detect current patroller positions and forward-simulate
            #     their movement along the planned avatar trajectory.
            if plan_safe and self.mobile_threat_features:
                try:
                    step_size = self.infer_motor_step_size(available_actions)
                    entities = self.extract_entities(curr_grid, bg)
                    av_feats = self.avatar_features or (
                        {self.avatar_feature} if self.avatar_feature is not None else set()
                    )
                    live_threats = PerceptionEngine.detect_oriented_threats(
                        curr_grid,
                        entities,
                        bg=bg,
                        step_size=step_size,
                        avatar_pos=self.avatar_pos,
                        avatar_features=av_feats,
                    )
                    live_patrols: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                    for t in live_threats:
                        if t.feature_id in self.mobile_threat_features:
                            live_patrols.add((t.pos, t.facing))

                    if live_patrols:
                        # Build patrol-aware wall set for corridor checks
                        half_step = max(1, step_size // 2)
                        barrier_feats = self.symbolic_theory.barrier_features
                        bg_is_void = (
                            len(self.symbolic_theory.walkable_features) > 0
                            and bg not in self.symbolic_theory.walkable_features
                        )
                        patrol_walls: set[tuple[int, int]] = set(self.learned_barriers)
                        for rr in range(H):
                            for cc in range(W):
                                v = int(curr_grid[rr, cc])
                                if (
                                    self.symbolic_theory.is_barrier(v)
                                    or v in barrier_feats
                                    or (bg_is_void and v == bg)
                                ):
                                    patrol_walls.add((rr, cc))

                        def _blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
                            probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
                            dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                            for q in (probe, dest):
                                if not (0 <= q[0] < H and 0 <= q[1] < W) or q in patrol_walls:
                                    return True
                            return False

                        sim_patrols = frozenset(live_patrols)
                        for future_step in all_future_steps:
                            avatar_dest = future_step.predicted_avatar_pos
                            new_patrol_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                            collision = False
                            for pat_pos, pat_facing in sim_patrols:
                                if pat_pos == avatar_dest:
                                    collision = True
                                    break
                                dest = (
                                    pat_pos[0] + pat_facing[0] * step_size,
                                    pat_pos[1] + pat_facing[1] * step_size,
                                )
                                if (
                                    not (0 <= dest[0] < H and 0 <= dest[1] < W)
                                    or dest in patrol_walls
                                ):
                                    new_f = (-pat_facing[0], -pat_facing[1])
                                    new_patrol_set.add((pat_pos, new_f))
                                    continue
                                if dest == avatar_dest:
                                    collision = True
                                    break
                                new_f = pat_facing
                                if _blocked(dest, pat_facing):
                                    new_f = (-pat_facing[0], -pat_facing[1])
                                new_patrol_set.add((dest, new_f))
                            if collision:
                                plan_safe = False
                                break
                            sim_patrols = frozenset(new_patrol_set)
                except Exception:
                    pass  # If patrol simulation fails, rely on the static check

            # Habenular Episodic Gating: Avoid repeating known fatal actions at this state
            if plan_safe and self.working_memory.habenular_ior.is_action_inhibited(
                self.avatar_pos, next_step.action
            ):
                plan_safe = False

            # Cerebellar Phase Gating & Rhythm-Locked Safe Windows
            if plan_safe and next_step.predicted_avatar_pos is not None:
                phase_gate = self.cerebellar_clock.evaluate_motion_hazard_gate(
                    current_step=self.step_counter,
                    avatar_pos=self.avatar_pos,
                    target_pos=next_step.predicted_avatar_pos,
                    hazard_tracker=self.hazard_tracker,
                    background_feature=self.bg_feature,
                    avatar_features=self.avatar_features,
                    walkable_features=self.symbolic_theory.walkable_features
                    | self.verified_safe_features,
                )
                if phase_gate.should_wait:
                    self.mental_plan.appendleft(next_step)
                    self.pending_phase_wait_steps = phase_gate.wait_steps_recommended
                    non_disp = [
                        a
                        for a in available_actions
                        if not self.is_displacement_action(a) and not self.is_spatial_effector(a)
                    ]
                    if non_disp:
                        return non_disp[0], None

            if not plan_safe:
                self.mental_plan.clear()
                self.phase = EpistemicPhase.REPLANNING
                # Attempt forward simulation from current state with updated patroller positions
                simulated_plan = self.simulate_in_mind(curr_grid, available_actions)
                if not simulated_plan:
                    # If direct path fails, try blocking the immediate hazardous step to seek an alternative branch
                    danger_cells = {next_step.predicted_avatar_pos}
                    simulated_plan = self.simulate_in_mind(
                        curr_grid, available_actions, blocked_cells=danger_cells
                    )
                if simulated_plan:
                    self.consecutive_simulation_failures = 0
                    self.simulation_cooldown = 0
                    self.phase = EpistemicPhase.EXPLOITATION
                    self.mental_plan = deque(simulated_plan)
                    next_step = self.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    self.consecutive_simulation_failures += 1
                    self.simulation_cooldown = min(
                        12, 2 ** min(self.consecutive_simulation_failures, 4)
                    )
                    macro_plan = (
                        self.object_planner.plan_macro_option(self, curr_grid, available_actions)
                        if hasattr(self, "object_planner")
                        else None
                    )
                    if macro_plan:
                        self.phase = EpistemicPhase.EXPLOITATION
                        self.mental_plan = deque(macro_plan)
                        next_step = self.mental_plan.popleft()
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                    else:
                        chosen_action, chosen_data = self.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
            elif next_step.action in available_actions:
                chosen_action = next_step.action
                chosen_data = next_step.action_data
                predicted_pos = next_step.predicted_avatar_pos
            else:
                self.mental_plan.clear()
                self.phase = EpistemicPhase.REPLANNING
                chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

        # 4. If motor grounded, attempt Forward Mental Simulation or Habitual Exploration
        elif self.is_motor_grounded():
            # Deliberate-to-Habitual Search Backoff (Daw, Niv, & Dayan; Dolan & Dayan):
            # When deliberate forward simulation previously failed to find a valid goal path,
            # back off for K steps. Executive control shifts to habitual exploration (object-centric
            # macro planning and curiosity-driven epistemic probing) instead of burning search budgets.
            can_simulate = self.simulation_cooldown == 0
            simulated_plan = None
            if can_simulate:
                simulated_plan = self.simulate_in_mind(curr_grid, available_actions)
                if simulated_plan:
                    self.consecutive_simulation_failures = 0
                    self.simulation_cooldown = 0
                    self.phase = EpistemicPhase.EXPLOITATION
                    self.mental_plan = deque(simulated_plan)
                    next_step = self.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    self.consecutive_simulation_failures += 1
                    # Bounded exponential backoff: 2, 4, 8, max 12 steps of habitual exploration
                    self.simulation_cooldown = min(
                        12, 2 ** min(self.consecutive_simulation_failures, 4)
                    )
                    self._last_sim_goal_count = len(self.learned_goal_positions)
                    self._last_sim_barrier_count = len(self.learned_barriers)

            if simulated_plan is None:
                # Direct path to goal blocked or simulation backed off;
                # consult Object-Centric Macro-Action State Graph Planner (5 Executive Directives)
                macro_plan = (
                    self.object_planner.plan_macro_option(self, curr_grid, available_actions)
                    if hasattr(self, "object_planner")
                    else None
                )
                if macro_plan:
                    self.phase = EpistemicPhase.EXPLOITATION
                    self.mental_plan = deque(macro_plan)
                    next_step = self.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    if (
                        not self.mental_plan
                        and hasattr(self, "object_planner")
                        and self.object_planner.active_target_object_id
                    ):
                        self.object_planner.ledger.record_interaction(
                            self.object_planner.active_target_object_id,
                            predicted_pos or self.avatar_pos or (0, 0),
                        )
                        self.object_planner.active_target_object_id = None
                else:
                    self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    chosen_action, chosen_data = self.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        # 5. Fallback to Epistemic Curiosity Probing
        else:
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

        # Effector Grounding Invariant: Effector actions MUST have spatial target coordinates
        if self.is_spatial_effector(chosen_action):
            aff = self.action_affordances.get(chosen_action)
            req_keys = aff.target_param_keys if aff else ("x", "y")
            if (
                chosen_data is None
                or not isinstance(chosen_data, dict)
                or not all(k in chosen_data for k in req_keys)
            ):
                chosen_data = self.ground_effector_action(curr_grid, chosen_action)

        self.prev_grid = curr_grid.copy()
        self.last_action = chosen_action
        self.last_action_data = chosen_data
        self.last_predicted_pos = predicted_pos
        return chosen_action, chosen_data
