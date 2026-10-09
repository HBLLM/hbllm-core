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
from hbllm.hcir.world.cortex_deliberation import PrefrontalDeliberationEngine
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

        # Faculty: Prefrontal Deliberation Engine & Cognitive Arbitration
        self.deliberation_engine: PrefrontalDeliberationEngine = PrefrontalDeliberationEngine()

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
        """Reset episodic state upon level transition or death.
        Delegated to HippocampalEpisodicCortex for schema consolidation.
        """
        self.episodic_cortex.consolidate_schema_and_reset(
            self,
            retain_dynamics=retain_dynamics,
            is_new_level=is_new_level,
            level=level,
        )

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
        Delegated to PrefrontalDeliberationEngine.
        """
        return self.deliberation_engine.decide(
            self,
            curr_grid=curr_grid,
            available_actions=available_actions,
            is_win=is_win,
            is_lost=is_lost,
            action_schemas=action_schemas,
        )
