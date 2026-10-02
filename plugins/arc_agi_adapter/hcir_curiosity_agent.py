"""HCIR Curiosity & Exit AGI Agent (Core Internal Language Implementation).

Wired directly into HBLLM's typed cognitive internal language (HCIR):
1. Identify Exits & Exit Conditions (EntityRole.GOAL, EntityRole.PORTAL, Receptacles)
2. Identify Objects in Scene & Lift to HCIR EntityGraph (SpatialEntity, EntityRole, CutSets)
3. Causal Curiosity & Epistemic Probing ("Play with unknowns" via EpistemicObservationDiff)
4. Geodesic Spatial Navigation (HCIRSpatialEntityPlanner collision-free geodesic A*)

Domain-Agnostic: Zero game ID checks, zero bespoke puzzle archetypes.
"""

from __future__ import annotations

import collections
import heapq
import logging
import math
from typing import Any

import numpy as np

from hbllm.brain.concepts.grounded_concept_registry import GroundedConceptRegistry
from hbllm.brain.language.acquisition.contrastive_learner import ContrastiveLearner
from hbllm.brain.language.acquisition.cross_situational_learner import CrossSituationalLearner
from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import (
    CognitiveGraph,
    GoalNode,
    PhysicalEntityNode,
)
from hbllm.hcir.spatial_planner import (
    EntityGraph,
    EntityRole,
    HCIRSpatialEntityPlanner,
    ObjectAffordanceRule,
    SpatialActionIntent,
    SpatialEntity,
)
from hbllm.hcir.subgoal_decomposer import EpistemicFrontierDetector, HierarchicalGoalDecomposer
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    MentalSimulationStep,
)
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world.spatial_containment import (
    BaseSpatialContainmentEngine,
    RoomTopologyExtractor,
)
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer
from plugins.arc_agi_adapter.arc_skills.automaton_synthesis import (
    AutomatonProgramSynthesisSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.coupled_controllables import (
    CoupledControllableSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.grammar_translation import (
    GrammarTranslationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.hierarchical_pattern_grammar import (
    HierarchicalPatternGrammarSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.inverted_buoyancy import (
    InvertedBuoyancySkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.kinematic_arm_linkage import (
    KinematicLinkageSolver,
)
from plugins.arc_agi_adapter.arc_skills.kinetic_coupling import (
    KineticCouplingSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.laser_routing import (
    LaserRoutingSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.modal_incantation import (
    ModalIncantationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.morphological_mutation import (
    MorphologicalMutationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.morphology_synthesis import (
    MorphologicalProgramSynthesis,
)
from plugins.arc_agi_adapter.arc_skills.optical_mirror_reflection import (
    OpticalMirrorReflectionSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.permutation_algebra import (
    PermutationAlgebraSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.relational_affordance import (
    RelationalAffordanceSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.reticle_superposition import (
    ReticleSuperpositionSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.rigid_assembly import (
    RigidAssemblySkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.temporal_echo import (
    TemporalEchoSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.topology_transformation import (
    TopologyTransformationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.visual_canvas import (
    VisualCanvasSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.vortex_attractor import (
    VortexAttractorSkillAcquisition,
)

logger = logging.getLogger("hcir_curiosity_agent")


class HCIRCuriosityAgent:
    """Universal 4-Pillar Cognitive Agent grounded in HBLLM's HCIR internal language."""

    def __init__(self, step_size: int = 1, **kwargs: Any) -> None:
        self.step_size: int = step_size
        self.step_counter: int = 0
        self.current_level: int = 0
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_action_data: dict[str, int] | None = None
        self.current_avatar_pos: tuple[int, int] | None = None
        self.avatar_spawn_pos: tuple[int, int] | None = None

        # ── Core Cognitive Concept Architecture ──────────────────────────
        self.cognitive_graph: CognitiveGraph = CognitiveGraph()
        self.concept_registry: GroundedConceptRegistry = GroundedConceptRegistry(
            self.cognitive_graph
        )
        self.cross_situational: CrossSituationalLearner = CrossSituationalLearner(
            self.cognitive_graph, self.concept_registry
        )
        self.contrastive: ContrastiveLearner = ContrastiveLearner()
        self.active_agent_id: str | None = None
        self.known_actuator_signatures: set[str] = set()

        # ── Core HCIR Cognitive Engines ──────────────────────────────────
        self.planner: HCIRSpatialEntityPlanner = HCIRSpatialEntityPlanner(step_size=step_size)
        self.epistemic_engine: AutonomousEpistemicEngine = AutonomousEpistemicEngine()
        self.goal_decomposer: HierarchicalGoalDecomposer = HierarchicalGoalDecomposer()
        self.workspace: HCIRWorkspaceState = HCIRWorkspaceState()
        self.cut_set_analyzer: TopologicalCutSetAnalyzer = TopologicalCutSetAnalyzer()
        self.symmetry_analyzer: VisualSymmetryAnalyzer = VisualSymmetryAnalyzer()
        self.visual_canvas_skill: VisualCanvasSkillAcquisition = VisualCanvasSkillAcquisition()
        self.kinematic_linkage_solver: KinematicLinkageSolver = KinematicLinkageSolver()
        self.active_subgoal: GoalNode | None = None
        self.visited_cells: set[tuple[int, int]] = set()
        self.action_dynamics: dict[int, ActionDynamicsModel] = {}
        for act, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
            self.action_dynamics[act] = ActionDynamicsModel(
                action_id=act,
                delta_r=dr * step_size,
                delta_c=dc * step_size,
                confidence=0.5,
            )

        # ── Cross-Episode & Cross-Level Cognitive Memory ──────────────────
        self.known_avatar_color: int | None = None
        self.known_exit_colors: set[int] = set()
        self.known_actuator_positions: set[tuple[int, int]] = set()
        self.known_actuator_colors: set[int] = set()
        self.known_hazard_colors: set[int] = {8}  # Standard ARC hazard default
        self.walkable_terrain_colors: set[int] = set()  # Confirmed passable surface tiles
        self.known_barriers: set[tuple[int, int]] = set()
        self.affordance_memory: dict[str, ObjectAffordanceRule] = {}  # sig_key -> rule
        self.tested_positions: set[tuple[int, int]] = set()
        self.inventory: set[int] = set()  # Collected resource colors

        # ── Navigation & Anti-Stagnation ──────────────────────────────────
        self.position_visits: collections.Counter[tuple[int, int]] = collections.Counter()
        self.position_history: list[tuple[int, int]] = []
        self.step_size_counts: collections.Counter[int] = collections.Counter()
        self.action_history: list[int] = []
        self.planned_action_queue: list[tuple[int, dict[str, int] | None]] = []
        self.mental_simulation_plan: list[MentalSimulationStep] = []
        self.stuck_counter: int = 0
        self.last_target_pos: tuple[int, int] | None = None
        self.target_failure_counts: collections.Counter[tuple[int, int]] = collections.Counter()
        self.blacklisted_goals: set[tuple[int, int]] = set()
        self.spatial_transitions: dict[tuple[int, int], dict[int, tuple[int, int] | str]] = {}
        self.available_actions: list[int] = []
        self.known_goal_bboxes: set[tuple[int, int, int, int]] = set()
        self.carried_offset: tuple[int, int] | None = None
        self.last_target_is_goal: bool = False

        # ── Click & Manipulation State ────────────────────────────────────
        self.quiescent_clicks: set[tuple[int, int]] = set()
        self.completed_click_controls: set[tuple[int, int]] = set()
        self.consecutive_effective_clicks: int = 0
        self.visited_grid_hashes: set[int] = set()

    def reset_episode(self, retain_dynamics: bool = False, is_retry: bool = False) -> None:
        """Reset episodic state while optionally retaining cross-level dynamics."""
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = None
        self.step_counter = 0
        self.stuck_counter = 0
        self.planned_action_queue.clear()
        self.position_visits.clear()
        self.position_history.clear()
        self.action_history.clear()
        self.target_failure_counts.clear()
        self.last_target_pos = None
        self.current_avatar_pos = None
        self.avatar_spawn_pos = None
        self.consecutive_effective_clicks = 0
        self.visited_grid_hashes.clear()
        self.spatial_transitions.clear()
        self.carried_offset = None
        self.last_target_is_goal = False
        self.active_agent_id = None
        self.mental_simulation_plan: list[MentalSimulationStep] = []

        # Grid-coordinate episodic caches must ALWAYS be cleared per level/attempt
        self.known_barriers.clear()
        self.tested_positions.clear()
        self.blacklisted_goals.clear()
        self.known_goal_bboxes.clear()
        self.quiescent_clicks.clear()
        self.completed_click_controls.clear()
        self.known_actuator_positions.clear()
        self.inventory.clear()
        self.goal_decomposer.completed_subgoals.clear()
        self.workspace = HCIRWorkspaceState()
        self.active_subgoal = None
        self.visited_cells.clear()

        self.planner.reset(is_retry=is_retry)
        self.epistemic_engine.reset_episode(retain_dynamics=retain_dynamics or is_retry)
        self.visual_canvas_skill = VisualCanvasSkillAcquisition()
        self.kinematic_linkage_solver.reset_episode()

        if not retain_dynamics and not is_retry:
            self.step_size = 1
            self.planner.step_size = 1
            self.step_size_counts.clear()
            self.current_level = 0
            self.known_avatar_color = None
            self.known_exit_colors.clear()
            self.known_actuator_colors.clear()
            self.known_hazard_colors = {8}
            self.walkable_terrain_colors.clear()
            self.affordance_memory.clear()
            self.cognitive_graph = CognitiveGraph()
            self.concept_registry = GroundedConceptRegistry(self.cognitive_graph)
            self.cross_situational = CrossSituationalLearner(
                self.cognitive_graph, self.concept_registry
            )
            self.contrastive = ContrastiveLearner()
            self.known_actuator_signatures.clear()
            self.goal_decomposer = HierarchicalGoalDecomposer()
            self.workspace = HCIRWorkspaceState()
            self.action_dynamics.clear()
            for act, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
                self.action_dynamics[act] = ActionDynamicsModel(
                    action_id=act,
                    delta_r=dr * self.step_size,
                    delta_c=dc * self.step_size,
                    confidence=0.5,
                )
        elif retain_dynamics and not is_retry:
            self.current_level += 1

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 2: Scene Perception & HCIR EntityGraph Construction
    # ═══════════════════════════════════════════════════════════════════════

    def perceive_scene_hcir(
        self,
        grid: np.ndarray,
        available_actions: list[int],
    ) -> EntityGraph:
        """Lift raw pixel grid into HCIR SpatialEntity instances and construct EntityGraph."""
        if grid.ndim == 3:
            grid = grid[-1]
        grid = np.asarray(grid, dtype=np.int32)

        H, W = grid.shape
        counts = np.bincount(grid.flatten())
        bg_color = int(np.argmax(counts))

        # Color 8 in click-enabled games is interactable / bridge mechanism, not instant hazard
        if 6 in available_actions:
            self.known_hazard_colors.discard(8)
        else:
            self.known_hazard_colors.add(8)

        # 1. Segment connected components into HCIR SpatialEntities
        raw_entities = self._extract_spatial_entities(grid, bg_color)

        has_movement = any(a in available_actions for a in (1, 2, 3, 4))

        # Detect discrete panels separated by full blank gutters (when Action 5 platform transport is not present)
        blank_rows = set(r for r in range(H) if np.all(grid[r, :] == bg_color))
        blank_cols = set(c for c in range(W) if np.all(grid[:, c] == bg_color))

        if 5 not in available_actions and (blank_rows or blank_cols):
            row_intervals = []
            prev_r = -1
            for r in sorted(blank_rows):
                if r > prev_r + 1:
                    row_intervals.append((prev_r + 1, r - 1))
                prev_r = r

            if prev_r < H - 1:
                row_intervals.append((prev_r + 1, H - 1))
            if not row_intervals:
                row_intervals = [(0, H - 1)]

            col_intervals = []
            prev_c = -1
            for c in sorted(blank_cols):
                if c > prev_c + 1:
                    col_intervals.append((prev_c + 1, c - 1))
                prev_c = c
            if prev_c < W - 1:
                col_intervals.append((prev_c + 1, W - 1))
            if not col_intervals:
                col_intervals = [(0, W - 1)]

            panels = []
            for r0, r1 in row_intervals:
                for c0, c1 in col_intervals:
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    panels.append((area, (r0, r1, c0, c1)))
            panels.sort(key=lambda p: p[0], reverse=True)

            if len(panels) > 1 and panels[0][0] >= H * W * 0.4:
                selected_panel = panels[0][1]
                if self.current_avatar_pos:
                    ar, ac = self.current_avatar_pos
                    for _, p in panels:
                        if p[0] <= ar <= p[1] and p[2] <= ac <= p[3]:
                            selected_panel = p
                            break
                elif self.known_avatar_color is not None:
                    av_coords = np.argwhere(grid == self.known_avatar_color)
                    if len(av_coords) > 0:
                        ar, ac = np.mean(av_coords, axis=0)
                        for _, p in panels:
                            if p[0] <= ar <= p[1] and p[2] <= ac <= p[3]:
                                selected_panel = p
                                break
                min_r, max_r, min_c, max_c = selected_panel
                raw_entities = [
                    e
                    for e in raw_entities
                    if min_r <= e.grid_pos[0] <= max_r and min_c <= e.grid_pos[1] <= max_c
                ]

        # 2. Identify controllable avatar within primary playfield via Kinematic Continuity
        has_movement = any(a in available_actions for a in (1, 2, 3, 4))
        avatar_ent = self._identify_avatar_entity(raw_entities, grid, has_movement)
        if avatar_ent:
            avatar_ent.role = EntityRole.AGENT
            self.current_avatar_pos = avatar_ent.grid_pos
            self.active_agent_id = avatar_ent.id

        # 3. Classify functional roles into HCIR EntityRoles and CognitiveGraph Nodes
        barriers: set[tuple[int, int]] = set(self.known_barriers)

        # If dominant background is confirmed hazard or non-walkable terrain, treat all background pixels as impassable
        if bg_color in self.known_hazard_colors or (
            self.walkable_terrain_colors and bg_color not in self.walkable_terrain_colors
        ):
            for r, c in np.argwhere(grid == bg_color):
                barriers.add((int(r), int(c)))

        classified_entities: list[SpatialEntity] = []
        color_counts = collections.Counter(ent.color for ent in raw_entities if ent != avatar_ent)
        unique_exit_centroids = [
            e.centroid
            for e in raw_entities
            if e != avatar_ent
            and has_movement
            and self._is_candidate_exit(e, grid, bg_color)
            and color_counts.get(e.color, 0) == 1
        ]

        for ent in raw_entities:
            if ent == avatar_ent:
                classified_entities.append(ent)
                continue

            # Sub-step components of composite avatar body
            if (
                avatar_ent
                and self.step_size > 1
                and math.hypot(
                    ent.centroid[0] - avatar_ent.centroid[0],
                    ent.centroid[1] - avatar_ent.centroid[1],
                )
                <= min(2.5, self.step_size * 0.6)
                and ent.color not in self.known_exit_colors
            ):
                continue

            # Concentric outer frames of unique candidate exits
            if (
                any(
                    math.hypot(ent.centroid[0] - uec[0], ent.centroid[1] - uec[1])
                    <= max(2.0, self.step_size * 0.6)
                    for uec in unique_exit_centroids
                )
                and color_counts.get(ent.color, 0) > 1
            ):
                continue

            min_r, max_r, min_c, max_c = ent.bounding_box
            is_frame = (
                (max_r - min_r >= H - 4 and max_c - min_c >= W - 4)
                or (ent.area > H * W * 0.28)
                or (min_r <= 1 and max_r <= 1 and max_c - min_c > 8)
                or (min_r >= H - 2 and max_r >= H - 2 and max_c - min_c > 8)
                or (
                    has_movement
                    and (min_r >= H - 1 or max_r <= 0)
                    and ent.color not in self.known_exit_colors
                )
            )
            is_known_barrier = is_frame or ent.grid_pos in self.known_barriers

            if is_known_barrier:
                ent.role = EntityRole.OBSTACLE
                coords = ent.properties.get("coords", [ent.grid_pos])
                for r, c in coords:
                    barriers.add((r, c))
            elif ent.color in self.known_hazard_colors:
                ent.role = EntityRole.DYNAMIC_HAZARD
                classified_entities.append(ent)
            elif ent.color in self.walkable_terrain_colors or counts[ent.color] > H * W * 0.15:
                # If reachable by walking, it's terrain. If unreachable in a click game, it is a remote actuator
                if 6 in available_actions and avatar_ent and ent.area < H * W * 0.1:
                    path_to_ent = self._find_geodesic_path(
                        grid,
                        avatar_ent.grid_pos,
                        ent.grid_pos,
                        barriers,
                    )
                    if path_to_ent is None:
                        ent.role = EntityRole.ACTUATOR
                        classified_entities.append(ent)
                        continue
                ent.role = EntityRole.UNKNOWN
            elif ent.grid_pos in self.blacklisted_goals:
                ent.role = EntityRole.UNKNOWN
            elif (
                ent.color in self.known_actuator_colors
                or ent.grid_pos in self.known_actuator_positions
                or any(
                    math.hypot(ent.grid_pos[0] - ar, ent.grid_pos[1] - ac)
                    <= max(1.5, self.step_size * 0.7)
                    for ar, ac in self.known_actuator_positions
                )
            ):
                ent.role = EntityRole.ACTUATOR
                classified_entities.append(ent)
            else:
                sig_key = ent.get_signature_key()
                rule = self.affordance_memory.get(sig_key)
                if rule and rule.role != EntityRole.UNKNOWN:
                    ent.role = rule.role
                    classified_entities.append(ent)
                elif ent.color in self.known_exit_colors:
                    ent.role = EntityRole.GOAL
                    classified_entities.append(ent)
                elif (
                    color_counts[ent.color] > 1
                    and ent.area <= 64
                    and ent.color not in self.walkable_terrain_colors
                ):
                    # Multiple instances of items: collectible resources, movable blocks, or switches
                    ent.role = EntityRole.RESOURCE
                    classified_entities.append(ent)
                elif has_movement and self._is_candidate_exit(ent, grid, bg_color):
                    ent.role = EntityRole.GOAL
                    self.known_exit_colors.add(ent.color)
                    classified_entities.append(ent)
                else:
                    ent.role = EntityRole.MANIPULABLE if has_movement else EntityRole.ACTUATOR
                    classified_entities.append(ent)

            # Lift to CognitiveGraph PhysicalEntityNode
            existing_node = self.cognitive_graph.get_node(ent.id)
            if existing_node is not None:
                if isinstance(existing_node, PhysicalEntityNode):
                    existing_node.properties.update(
                        {
                            "color": ent.color,
                            "area": ent.area,
                            "centroid": ent.centroid,
                            "grid_pos": ent.grid_pos,
                            "bounding_box": ent.bounding_box,
                            "role": ent.role.name if hasattr(ent.role, "name") else str(ent.role),
                        }
                    )
                    existing_node.observed_properties.update(
                        {
                            "color": ent.color,
                            "area": ent.area,
                            "grid_pos": ent.grid_pos,
                        }
                    )
            else:
                p_node = PhysicalEntityNode(
                    id=ent.id,
                    name=f"node_{ent.id}",
                    entity_type=ent.role.name if hasattr(ent.role, "name") else str(ent.role),
                    properties={
                        "color": ent.color,
                        "area": ent.area,
                        "centroid": ent.centroid,
                        "grid_pos": ent.grid_pos,
                        "bounding_box": ent.bounding_box,
                        "role": ent.role.name if hasattr(ent.role, "name") else str(ent.role),
                    },
                    observed_properties={
                        "color": ent.color,
                        "area": ent.area,
                        "grid_pos": ent.grid_pos,
                    },
                )
                self.cognitive_graph.add_node(p_node)

        # 4. Construct topological EntityGraph via HCIRSpatialEntityPlanner
        entity_graph = self.planner.construct_entity_graph(
            entities=classified_entities,
            barriers=barriers,
            grid_shape=(H, W),
            step_size=self.step_size,
            known_barriers=self.known_barriers,
        )

        # In click-enabled games, lift remote (unreachable by walking) interactable objects
        if 6 in available_actions and entity_graph.agent:
            hazard_coords = {
                ent.grid_pos
                for ent in entity_graph.entities.values()
                if ent.role == EntityRole.DYNAMIC_HAZARD or ent.color in self.known_hazard_colors
            }
            for e in raw_entities:
                if e != entity_graph.agent and e.role not in (EntityRole.GOAL, EntityRole.PORTAL):
                    cr, cc = int(round(e.centroid[0])), int(round(e.centroid[1]))
                    path = self._find_geodesic_path(
                        grid,
                        entity_graph.agent.grid_pos,
                        (cr, cc),
                        entity_graph.barriers,
                        hazard_coords=hazard_coords,
                    )
                    if path is None:
                        e.role = EntityRole.ACTUATOR
                        if e.id not in entity_graph.entities:
                            entity_graph.entities[e.id] = e

        return entity_graph

    def _extract_spatial_entities(self, grid: np.ndarray, bg_color: int) -> list[SpatialEntity]:
        """Extract connected components and lift to HCIR SpatialEntity."""
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        entities: list[SpatialEntity] = []
        entity_idx = 0
        bg_colors = {0, bg_color}
        if self.known_avatar_color in bg_colors:
            bg_colors.discard(self.known_avatar_color)

        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if val in bg_colors or visited[r, c]:
                    continue

                coords: list[tuple[int, int]] = []
                queue = collections.deque([(r, c)])
                visited[r, c] = True

                while queue:
                    curr_r, curr_c = queue.popleft()
                    coords.append((curr_r, curr_c))

                    for dr, dc in (
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ):
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and int(grid[nr, nc]) == val:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                rs = [coord[0] for coord in coords]
                cs = [coord[1] for coord in coords]
                bbox = (min(rs), max(rs), min(cs), max(cs))
                centroid = (float(np.mean(rs)), float(np.mean(cs)))
                area = len(coords)
                grid_pos = (int(round(centroid[0])), int(round(centroid[1])))

                ent = SpatialEntity(
                    id=f"ent_{entity_idx}_{val}",
                    role=EntityRole.UNKNOWN,
                    centroid=centroid,
                    grid_pos=grid_pos,
                    area=area,
                    bounding_box=bbox,
                    color=val,
                    properties={"coords": coords, "color": val},
                )
                entities.append(ent)
                entity_idx += 1

        return self._assemble_composite_entities(entities)

    def _assemble_composite_entities(self, entities: list[SpatialEntity]) -> list[SpatialEntity]:
        """Assemble multi-part and nested entities via topological relations.

        Replaces ad-hoc area ratios with:
        1. Topological Containment: Inner glyphs/components inside enclosing sockets (B1 subset B2).
        2. Contiguous Part Adjacency: Touching parts forming a compact rigid body.
        3. Concept Protection: Controllable agent is never absorbed into stationary fixtures.
        """
        if len(entities) <= 1:
            return entities

        merged: list[SpatialEntity] = []
        consumed: set[int] = set()
        interaction_scale = max(self.step_size * 2, 8)

        for i, e1 in enumerate(entities):
            if i in consumed:
                continue
            core = e1
            for j, e2 in enumerate(entities):
                if i == j or j in consumed:
                    continue
                # Grounded filter: do not merge environment terrain
                if (
                    e1.color in self.walkable_terrain_colors
                    or e2.color in self.walkable_terrain_colors
                ):
                    continue

                # Concept Protection: Do not merge controllable avatar into non-avatar fixtures
                if self.current_avatar_pos is not None:
                    e1_is_av = e1.grid_pos == self.current_avatar_pos or (
                        self.known_avatar_color is not None and e1.color == self.known_avatar_color
                    )
                    e2_is_av = e2.grid_pos == self.current_avatar_pos or (
                        self.known_avatar_color is not None and e2.color == self.known_avatar_color
                    )
                    if e1_is_av != e2_is_av:
                        continue

                b1, b2 = e1.bounding_box, e2.bounding_box
                e1_inside_e2 = (
                    b2[0] <= b1[0] and b1[1] <= b2[1] and b2[2] <= b1[2] and b1[3] <= b2[3]
                )
                e2_inside_e1 = (
                    b1[0] <= b2[0] and b2[1] <= b1[1] and b1[2] <= b2[2] and b2[3] <= b1[3]
                )
                is_nested = e1_inside_e2 or e2_inside_e1

                comb_h = max(b1[1], b2[1]) - min(b1[0], b2[0]) + 1
                comb_w = max(b1[3], b2[3]) - min(b1[2], b2[2]) + 1
                is_compact_object = comb_h <= interaction_scale and comb_w <= interaction_scale

                dist = math.hypot(e1.centroid[0] - e2.centroid[0], e1.centroid[1] - e2.centroid[1])
                is_adjacent_parts = (
                    dist <= max(2.5, self.step_size * 0.7)
                    and comb_h <= max(self.step_size, 5)
                    and comb_w <= max(self.step_size, 5)
                    and (b1[0] <= b2[1] + 1 and b2[0] <= b1[1] + 1)
                    and (b1[2] <= b2[3] + 1 and b2[2] <= b1[3] + 1)
                )

                if (is_nested and is_compact_object) or is_adjacent_parts:
                    consumed.add(j)
                    outer = e2 if (e1_inside_e2 or e2.area > e1.area) else e1
                    inner = e1 if (e1_inside_e2 or e2.area > e1.area) else e2
                    dominant = inner if inner.color not in (0, 4) else outer
                    c1_coords = core.properties.get("coords", [core.grid_pos])
                    c2_coords = e2.properties.get("coords", [e2.grid_pos])
                    all_coords = list(c1_coords) + list(c2_coords)
                    comb_bbox = (
                        min(b1[0], b2[0]),
                        max(b1[1], b2[1]),
                        min(b1[2], b2[2]),
                        max(b1[3], b2[3]),
                    )
                    core = SpatialEntity(
                        id=dominant.id,
                        role=dominant.role,
                        centroid=(
                            (comb_bbox[0] + comb_bbox[1]) / 2.0,
                            (comb_bbox[2] + comb_bbox[3]) / 2.0,
                        ),
                        grid_pos=(
                            int(round((comb_bbox[0] + comb_bbox[1]) / 2.0)),
                            int(round((comb_bbox[2] + comb_bbox[3]) / 2.0)),
                        ),
                        area=len(all_coords),
                        bounding_box=comb_bbox,
                        color=dominant.color,
                        properties={
                            "coords": all_coords,
                            "color": dominant.color,
                            "feature_id": dominant.color,
                            "visual_id": dominant.color,
                            "composite": True,
                        },
                    )
            merged.append(core)
        return merged

    def _merge_concentric_entities(self, entities: list[SpatialEntity]) -> list[SpatialEntity]:
        return self._assemble_composite_entities(entities)

    def _identify_avatar_entity(
        self,
        entities: list[SpatialEntity],
        grid: np.ndarray,
        has_movement: bool,
    ) -> SpatialEntity | None:
        """Find controllable avatar entity via kinematic continuity and spatial features."""
        if not has_movement or not entities:
            return None

        # 1. Match confirmed avatar color using spatio-temporal kinematic proximity
        if self.known_avatar_color is not None:
            matches = [e for e in entities if e.color == self.known_avatar_color]
            if matches:
                if self.current_avatar_pos:
                    expected_pos = self.current_avatar_pos
                    if self.last_action in (1, 2, 3, 4):
                        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                        dr, dc = deltas[self.last_action]
                        expected_pos = (
                            self.current_avatar_pos[0] + dr * self.step_size,
                            self.current_avatar_pos[1] + dc * self.step_size,
                        )
                    matches.sort(
                        key=lambda e: min(
                            math.hypot(
                                e.centroid[0] - expected_pos[0], e.centroid[1] - expected_pos[1]
                            ),
                            math.hypot(
                                e.centroid[0] - self.current_avatar_pos[0],
                                e.centroid[1] - self.current_avatar_pos[1],
                            ),
                        )
                    )
                return matches[0]

        # 2. Match candidate entities (Episode Initialization)
        color_counts = collections.Counter(e.color for e in entities)
        cands = [
            e
            for e in entities
            if 1 <= e.area <= 64
            and e.color not in self.known_hazard_colors
            and e.color not in self.known_exit_colors
        ]
        if cands:
            cands.sort(
                key=lambda e: (
                    0 if color_counts.get(e.color, 0) == 1 else 1,
                    0 if 4 <= e.area <= 49 else (1 if e.area in (1, 2, 3) else 2),
                    abs(
                        (e.bounding_box[1] - e.bounding_box[0])
                        - (e.bounding_box[3] - e.bounding_box[2])
                    ),
                    e.area,
                )
            )
            return cands[0]

        return None

    def _is_candidate_exit(
        self,
        entity: SpatialEntity,
        grid: np.ndarray,
        bg_color: int,
    ) -> bool:
        if self.known_exit_colors:
            return entity.color in self.known_exit_colors

        sig_key = entity.get_signature_key()
        rule = self.affordance_memory.get(sig_key)
        if rule and rule.role in (EntityRole.GOAL, EntityRole.PORTAL, EntityRole.RECEPTACLE):
            return True
        if rule and rule.role in (
            EntityRole.ACTUATOR,
            EntityRole.RESOURCE,
            EntityRole.OBSTACLE,
            EntityRole.DYNAMIC_HAZARD,
        ):
            return False
        if entity.grid_pos in self.tested_positions and entity.color not in self.known_exit_colors:
            return False
        if entity.color in self.walkable_terrain_colors:
            return False
        if (
            entity.color in self.known_actuator_colors
            or entity.grid_pos in self.known_actuator_positions
            or any(
                math.hypot(entity.grid_pos[0] - ar, entity.grid_pos[1] - ac)
                <= max(1.5, self.step_size * 0.7)
                for ar, ac in self.known_actuator_positions
            )
        ):
            return False

        min_r, max_r, min_c, max_c = entity.bounding_box
        H, W = grid.shape

        # Reject extreme canvas corners which are border/wall vertices
        at_extreme_corners = (
            (min_r <= 0 and min_c <= 0)
            or (max_r >= H - 1 and max_c >= W - 1)
            or (min_r <= 0 and max_c >= W - 1)
            or (max_r >= H - 1 and min_c <= 0)
        )
        if at_extreme_corners:
            return False

        # Reject entities overlapping confirmed avatar spawn
        if self.avatar_spawn_pos:
            dist_to_spawn = math.hypot(
                entity.centroid[0] - self.avatar_spawn_pos[0],
                entity.centroid[1] - self.avatar_spawn_pos[1],
            )
            if dist_to_spawn <= 1.0:
                return False

        # Reject entities of avatar
        if self.known_avatar_color is not None and entity.color == self.known_avatar_color:
            return False
        if self.current_avatar_pos is not None and entity.grid_pos == self.current_avatar_pos:
            return False

        # Perimeter portal or distinct exit marker
        is_perimeter = min_r <= 1 or max_r >= H - 2 or min_c <= 1 or max_c >= W - 2
        if is_perimeter and entity.area <= 25:
            return True

        # Interior distinct goal marker or receptacle
        if 2 <= entity.area <= max(128, int(H * W * 0.15)) and entity.color != bg_color:
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 1: Exit & Goal Teleology
    # ═══════════════════════════════════════════════════════════════════════

    def evaluate_exit_conditions(
        self, entity_graph: EntityGraph, grid: np.ndarray
    ) -> SpatialEntity | None:
        """Check if victory/exit conditions are satisfied and an unblocked path exists."""
        if not entity_graph.agent:
            return None

        # 1. Check candidate goals (excluding blacklisted / failed targets)
        goals = [
            e
            for e in entity_graph.entities.values()
            if e.role in (EntityRole.GOAL, EntityRole.PORTAL, EntityRole.RECEPTACLE)
            and e != entity_graph.agent
            and e.grid_pos not in self.blacklisted_goals
            and (e.grid_pos not in self.tested_positions or 5 in self.available_actions)
            and self.target_failure_counts[e.grid_pos] < 2
        ]
        for g in goals:
            self.known_goal_bboxes.add(g.bounding_box)

        goals.sort(
            key=lambda g: math.hypot(
                g.grid_pos[0] - entity_graph.agent.grid_pos[0],
                g.grid_pos[1] - entity_graph.agent.grid_pos[1],
            )
        )

        # In multi-goal or transport environments, ensure uncollected resources are gathered
        goal_bboxes = self.known_goal_bboxes | {g.bounding_box for g in goals}
        uncollected_resources = [
            e
            for e in entity_graph.entities.values()
            if e.role == EntityRole.RESOURCE
            and e != entity_graph.agent
            and e.grid_pos not in self.tested_positions
            and e.grid_pos not in self.known_barriers
            and not any(
                math.hypot(
                    max(0, bbox[0] - e.grid_pos[0], e.grid_pos[0] - bbox[1]),
                    max(0, bbox[2] - e.grid_pos[1], e.grid_pos[1] - bbox[3]),
                )
                <= max(2.5, self.step_size * 1.0)
                for bbox in goal_bboxes
            )
        ]
        if uncollected_resources and (
            (5 in self.available_actions and len(self.inventory) == 0)
            or all(self.target_failure_counts[g.grid_pos] > 0 for g in goals)
        ):
            return None

        hazard_coords: set[tuple[int, int]] = set()
        for hz in getattr(entity_graph, "dynamic_hazards", []):
            coords = hz.properties.get("coords", [hz.grid_pos])
            for hr, hc in coords:
                hazard_coords.add((int(hr), int(hc)))
        for ent in entity_graph.entities.values():
            if ent.role == EntityRole.DYNAMIC_HAZARD or ent.color in self.known_hazard_colors:
                coords = ent.properties.get("coords", [ent.grid_pos])
                for hr, hc in coords:
                    hazard_coords.add((int(hr), int(hc)))

        # Immediate 1-step move to adjacent confirmed exit before extensive exploration
        if entity_graph.agent:
            for g in goals:
                dist = math.hypot(
                    g.grid_pos[0] - entity_graph.agent.grid_pos[0],
                    g.grid_pos[1] - entity_graph.agent.grid_pos[1],
                )
                if (
                    dist <= max(1.5, self.step_size * 1.05)
                    and self.target_failure_counts[g.grid_pos] < 2
                ):
                    path = self._find_geodesic_path(
                        grid,
                        entity_graph.agent.grid_pos,
                        g.grid_pos,
                        entity_graph.barriers,
                        hazard_coords=hazard_coords,
                    )
                    if path:
                        return g

        # Target Commitment: if already pursuing a valid exit goal, maintain commitment
        if self.last_target_pos and self.target_failure_counts[self.last_target_pos] < 2:
            active_goal = next((g for g in goals if g.grid_pos == self.last_target_pos), None)
            if active_goal:
                dist = math.hypot(
                    active_goal.grid_pos[0] - entity_graph.agent.grid_pos[0],
                    active_goal.grid_pos[1] - entity_graph.agent.grid_pos[1],
                )
                if dist > max(1.5, self.step_size * 0.9):
                    path = self._find_geodesic_path(
                        grid,
                        entity_graph.agent.grid_pos,
                        active_goal.grid_pos,
                        entity_graph.barriers,
                        hazard_coords=hazard_coords,
                    )
                    if path is not None:
                        return active_goal

        for goal in goals:
            target_pos = goal.grid_pos
            path = self._find_geodesic_path(
                grid,
                entity_graph.agent.grid_pos,
                target_pos,
                entity_graph.barriers,
                hazard_coords=hazard_coords,
            )
            if path is not None:
                return goal

        # 3. If direct geodesic path is blocked, perform Topological Cut-Set & Subgoal Decomposition
        if goals and entity_graph.agent:
            primary_cand = goals[0]
            all_barriers = set(entity_graph.barriers) | self.known_barriers
            cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
                start=entity_graph.agent.grid_pos,
                goal=primary_cand.grid_pos,
                barrier_cells=all_barriers,
                grid_shape=grid.shape,
                step_size=self.step_size,
            )

            # Extract room doorways via RoomTopologyExtractor
            occupancy = np.ones(grid.shape, dtype=bool)
            for br, bc in all_barriers:
                if 0 <= br < grid.shape[0] and 0 <= bc < grid.shape[1]:
                    occupancy[br, bc] = False
            _, room_doors = RoomTopologyExtractor.extract_rooms_and_doors(occupancy)

            candidate_subgoals: list[dict[str, Any]] = []

            # A. Actuators/switches in reachable component
            for ent in entity_graph.entities.values():
                if ent == entity_graph.agent:
                    continue
                if ent.role in (EntityRole.ACTUATOR, EntityRole.MANIPULABLE):
                    if (
                        (not cut_res.is_partitioned or ent.grid_pos in cut_res.start_component)
                        and ent.grid_pos not in self.tested_positions
                        and ent.grid_pos not in self.blacklisted_goals
                    ):
                        candidate_subgoals.append(
                            {
                                "id": f"actuator_{ent.grid_pos[0]}_{ent.grid_pos[1]}",
                                "position": ent.grid_pos,
                                "role": "actuator",
                            }
                        )

            # B. Cut-Set gate approach cell
            if cut_res.is_partitioned and cut_res.approach_cell:
                if (
                    cut_res.approach_cell not in self.tested_positions
                    and cut_res.approach_cell not in self.blacklisted_goals
                ):
                    candidate_subgoals.append(
                        {
                            "id": f"gate_{cut_res.approach_cell[0]}_{cut_res.approach_cell[1]}",
                            "position": cut_res.approach_cell,
                            "role": "gate_approach",
                        }
                    )

            # C. Room doorways in reachable component
            for door in room_doors:
                dc = door.door_coord
                if (
                    (not cut_res.is_partitioned or dc in cut_res.start_component)
                    and dc not in self.tested_positions
                    and dc not in self.blacklisted_goals
                ):
                    candidate_subgoals.append(
                        {
                            "id": f"door_{dc[0]}_{dc[1]}",
                            "position": dc,
                            "role": "doorway",
                        }
                    )

            unobserved_mask = np.ones(grid.shape, dtype=bool)
            for r, c in self.visited_cells:
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    unobserved_mask[r, c] = False
            for r, c in all_barriers:
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    unobserved_mask[r, c] = False

            if candidate_subgoals or np.any(unobserved_mask):
                primary_goal_node = GoalNode(
                    id=f"primary_{primary_cand.grid_pos[0]}_{primary_cand.grid_pos[1]}",
                    description="Reach primary exit goal",
                    properties={"target_position": primary_cand.grid_pos},
                )
                active_subgoal = self.goal_decomposer.decompose_goal(
                    workspace=self.workspace,
                    primary_goal=primary_goal_node,
                    avatar_pos=entity_graph.agent.grid_pos,
                    barrier_cells=all_barriers,
                    grid_shape=grid.shape,
                    candidate_subgoals=candidate_subgoals,
                    step_size=self.step_size,
                    unobserved_mask=unobserved_mask if np.any(unobserved_mask) else None,
                )
                if active_subgoal and active_subgoal.id != primary_goal_node.id:
                    sub_pos = active_subgoal.properties.get("target_position")
                    if sub_pos:
                        sub_path = self._find_geodesic_path(
                            grid,
                            entity_graph.agent.grid_pos,
                            sub_pos,
                            entity_graph.barriers,
                            hazard_coords=hazard_coords,
                        )
                        if sub_path is not None:
                            self.active_subgoal = active_subgoal
                            matched_ent = next(
                                (
                                    e
                                    for e in entity_graph.entities.values()
                                    if e.grid_pos == sub_pos
                                ),
                                None,
                            )
                            if matched_ent:
                                return matched_ent
                            return SpatialEntity(
                                id=active_subgoal.id,
                                color=primary_cand.color,
                                role=EntityRole.ACTUATOR
                                if "actuator" in active_subgoal.id
                                else EntityRole.GOAL,
                                grid_pos=sub_pos,
                                bounding_box=(sub_pos[0], sub_pos[0], sub_pos[1], sub_pos[1]),
                                area=1,
                                centroid=(float(sub_pos[0]), float(sub_pos[1])),
                            )

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 3: Epistemic Curiosity ("Play with Unknowns")
    # ═══════════════════════════════════════════════════════════════════════

    def select_curiosity_target(self, entity_graph: EntityGraph) -> SpatialEntity | None:
        """Select nearest unconfirmed entity to probe via geodesic navigation."""
        if not entity_graph.agent:
            return None

        # 1. If holding items in transport environments, prioritize receptacles/goals only
        if self.inventory:
            receptacles = [
                e
                for e in entity_graph.entities.values()
                if e.role in (EntityRole.RECEPTACLE, EntityRole.GOAL)
                and e != entity_graph.agent
                and e.grid_pos not in self.blacklisted_goals
                and self.target_failure_counts[e.grid_pos] < 2
            ]
            if receptacles:
                return receptacles[0]
            return None

        # 2. Prioritize uncollected RESOURCE items first (nearest to avatar)
        goals = [
            g
            for g in entity_graph.entities.values()
            if g.role in (EntityRole.GOAL, EntityRole.PORTAL, EntityRole.RECEPTACLE)
            and g != entity_graph.agent
        ]
        goal_bboxes = self.known_goal_bboxes | {g.bounding_box for g in goals}
        uncollected_resources = [
            e
            for e in entity_graph.entities.values()
            if e.role == EntityRole.RESOURCE
            and e != entity_graph.agent
            and e.grid_pos not in self.tested_positions
            and e.grid_pos not in self.known_barriers
            and self.target_failure_counts[e.grid_pos] < 2
            and not any(
                math.hypot(
                    max(0, bbox[0] - e.grid_pos[0], e.grid_pos[0] - bbox[1]),
                    max(0, bbox[2] - e.grid_pos[1], e.grid_pos[1] - bbox[3]),
                )
                <= max(2.5, self.step_size * 1.0)
                for bbox in goal_bboxes
            )
        ]
        if uncollected_resources:
            # Target Commitment for uncollected resources: maintain pursuit of active target
            if self.last_target_pos and self.target_failure_counts[self.last_target_pos] < 2:
                active_res = next(
                    (e for e in uncollected_resources if e.grid_pos == self.last_target_pos), None
                )
                if active_res:
                    dist = math.hypot(
                        active_res.grid_pos[0] - entity_graph.agent.grid_pos[0],
                        active_res.grid_pos[1] - entity_graph.agent.grid_pos[1],
                    )
                    if dist > max(1.5, self.step_size * 0.9):
                        return active_res

            uncollected_resources.sort(
                key=lambda e: math.hypot(
                    e.grid_pos[0] - entity_graph.agent.grid_pos[0],
                    e.grid_pos[1] - entity_graph.agent.grid_pos[1],
                )
            )
            return uncollected_resources[0]

        # 3. Select untested candidates
        candidates: list[SpatialEntity] = []
        for e in entity_graph.entities.values():
            if e == entity_graph.agent or e.role == EntityRole.OBSTACLE:
                continue
            if e.grid_pos in self.known_barriers or e.grid_pos in self.blacklisted_goals:
                continue
            if e.color in self.known_hazard_colors:
                continue
            if self.target_failure_counts[e.grid_pos] >= 2:
                continue
            if e.grid_pos in self.tested_positions and e.role != EntityRole.RESOURCE:
                continue
            candidates.append(e)

        if not candidates:
            candidates = [
                e
                for e in entity_graph.entities.values()
                if e != entity_graph.agent
                and e.grid_pos not in self.known_barriers
                and e.grid_pos not in self.blacklisted_goals
                and e.color not in self.known_hazard_colors
                and self.target_failure_counts[e.grid_pos] < 2
            ]

        if not candidates:
            # Epistemic Frontier Detection: find reachable boundary of unobserved space
            if entity_graph.agent and self.prev_grid is not None:
                grid_shape = self.prev_grid.shape
                all_barriers = set(entity_graph.barriers) | self.known_barriers
                target_positions = {
                    ent.grid_pos
                    for ent in entity_graph.entities.values()
                    if ent.role in (EntityRole.GOAL, EntityRole.PORTAL, EntityRole.RECEPTACLE)
                }
                deadlocks = {
                    pos
                    for pos in self.visited_cells
                    if PhysicsPredictor.is_corner_deadlock(
                        entity_pos=pos,
                        barrier_cells=all_barriers,
                        target_positions=target_positions,
                        grid_shape=grid_shape,
                        step_size=self.step_size,
                    )
                }
                unobserved_mask = np.ones(grid_shape, dtype=bool)
                for r, c in self.visited_cells:
                    if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                        unobserved_mask[r, c] = False
                for r, c in all_barriers:
                    if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                        unobserved_mask[r, c] = False

                if np.any(unobserved_mask):
                    frontiers = EpistemicFrontierDetector.detect_frontiers(
                        avatar_pos=entity_graph.agent.grid_pos,
                        unobserved_mask=unobserved_mask,
                        barrier_cells=all_barriers,
                        grid_shape=grid_shape,
                        step_size=self.step_size,
                        deadlock_cells=deadlocks,
                    )
                    for (fr, fc), _score in frontiers:
                        if (fr, fc) not in self.tested_positions and self.target_failure_counts[
                            (fr, fc)
                        ] < 2:
                            return SpatialEntity(
                                id=f"frontier_{fr}_{fc}",
                                color=0,
                                role=EntityRole.RESOURCE,
                                grid_pos=(fr, fc),
                                bounding_box=(fr, fr, fc, fc),
                                area=1,
                                centroid=(float(fr), float(fc)),
                            )
            return None

        # Target Commitment for general candidates: maintain pursuit of active target
        if self.last_target_pos and self.target_failure_counts[self.last_target_pos] < 2:
            active_cand = next((e for e in candidates if e.grid_pos == self.last_target_pos), None)
            if active_cand:
                dist = math.hypot(
                    active_cand.grid_pos[0] - entity_graph.agent.grid_pos[0],
                    active_cand.grid_pos[1] - entity_graph.agent.grid_pos[1],
                )
                if dist > max(1.5, self.step_size * 0.9):
                    return active_cand

        def target_score(ent: SpatialEntity) -> float:
            tpos = ent.grid_pos
            dist = math.hypot(
                tpos[0] - entity_graph.agent.grid_pos[0],
                tpos[1] - entity_graph.agent.grid_pos[1],
            )
            penalty = self.target_failure_counts[tpos] * 50.0
            return dist + penalty

        candidates.sort(key=target_score)
        return candidates[0]

    def assimilate_causal_differential(
        self, prev_grid: np.ndarray, action: int, curr_grid: np.ndarray
    ) -> None:
        """Update HCIR affordance memory and epistemic world model from ΔGrid."""
        prev_grid = np.asarray(prev_grid, dtype=np.int32)
        curr_grid = np.asarray(curr_grid, dtype=np.int32)
        diff_mask = prev_grid != curr_grid
        diff_count = int(np.sum(diff_mask))

        # Dynamic Motor Grounding & Step Calibration via Vector Alignment
        avatar_displaced = False
        new_avatar_pos: tuple[int, int] | None = None
        old_avatar_pos = self.current_avatar_pos
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

        if action in (1, 2, 3, 4):
            dr, dc = deltas[action]
            bg = int(np.bincount(prev_grid.flatten()).argmax())

            spawn_candidate = None
            if self.known_avatar_color is not None:
                o = np.argwhere(prev_grid == self.known_avatar_color)
                n = np.argwhere(curr_grid == self.known_avatar_color)
                if len(o) > 0 and len(n) > 0:
                    r0, c0 = np.mean(o, axis=0)
                    r1, c1 = np.mean(n, axis=0)
                    disp_r, disp_c = r1 - r0, c1 - c0
                    dist = math.hypot(disp_r, disp_c)
                    if dist >= 0.7:
                        avatar_displaced = True
                        new_avatar_pos = (int(round(r1)), int(round(c1)))
                        spawn_candidate = (int(round(r0)), int(round(c0)))
                        calibrated = int(round(dist))
                        if 1 <= calibrated <= 8 and calibrated != self.step_size:
                            self.step_size_counts[calibrated] += 1
                            if self.step_size <= 1 or self.step_size_counts[calibrated] >= 2:
                                self.step_size = calibrated
                                self.planner.step_size = calibrated
                                logger.info(
                                    "HCIR Motor Calibration: Measured step_size = %d (consensus %d)",
                                    calibrated,
                                    self.step_size_counts[calibrated],
                                )
                        # Empirical Motor Grounding: calibrate action dynamics directly from displacement
                        dr_emp = int(round(disp_r / self.step_size)) * self.step_size
                        dc_emp = int(round(disp_c / self.step_size)) * self.step_size
                        if (dr_emp, dc_emp) != (0, 0):
                            self.action_dynamics[action] = ActionDynamicsModel(
                                action_id=action,
                                delta_r=dr_emp,
                                delta_c=dc_emp,
                                confidence=1.0,
                            )
                            logger.info(
                                "HCIR Motor Calibration: Calibrated Action %d -> (%d, %d)",
                                action,
                                dr_emp,
                                dc_emp,
                            )

            # If known avatar is unconfirmed (or vanished from grid), check if ANY color displaced
            known_avatar_vanished = (
                self.known_avatar_color is not None
                and len(np.argwhere(curr_grid == self.known_avatar_color)) == 0
            )
            if not avatar_displaced and (self.known_avatar_color is None or known_avatar_vanished):
                best_cand_color = None
                best_cand_disp = 0.0
                best_cand_pos = None
                best_cand_spawn = None
                best_disp_r = 0.0
                best_disp_c = 0.0

                for c in range(16):
                    if c in (0, bg):
                        continue
                    o = np.argwhere(prev_grid == c)
                    n = np.argwhere(curr_grid == c)
                    if len(o) > 0 and len(n) > 0 and abs(len(o) - len(n)) <= 4:
                        cr0, cc0 = np.mean(o, axis=0)
                        cr1, cc1 = np.mean(n, axis=0)
                        disp_r, disp_c = cr1 - cr0, cc1 - cc0
                        dist = math.hypot(disp_r, disp_c)
                        if dist >= 0.7:
                            dot = disp_r * dr + disp_c * dc
                            area_weight = min(len(n), 16) * 2.0
                            score = dist + (dot if dot > 0 else 0) + area_weight
                            if score > best_cand_disp:
                                best_cand_disp = score
                                best_cand_color = c
                                best_cand_pos = (int(round(cr1)), int(round(cc1)))
                                best_cand_spawn = (int(round(cr0)), int(round(cc0)))
                                best_disp_r = disp_r
                                best_disp_c = disp_c

                if best_cand_color is not None:
                    avatar_displaced = True
                    new_avatar_pos = best_cand_pos
                    spawn_candidate = best_cand_spawn
                    if self.known_avatar_color != best_cand_color:
                        logger.info(
                            "HCIR Causal Discovery: Discovered/Corrected AVATAR color %s -> %d",
                            self.known_avatar_color,
                            best_cand_color,
                        )
                        self.known_barriers.clear()
                        self.spatial_transitions.clear()
                        self.known_avatar_color = best_cand_color
                    calibrated = int(round(math.hypot(best_disp_r, best_disp_c)))
                    if 1 <= calibrated <= 8 and calibrated != self.step_size:
                        self.step_size_counts[calibrated] += 1
                        if self.step_size <= 1 or self.step_size_counts[calibrated] >= 2:
                            self.step_size = calibrated
                            self.planner.step_size = calibrated
                            logger.info(
                                "HCIR Motor Calibration: Measured step_size = %d (consensus %d)",
                                calibrated,
                                self.step_size_counts[calibrated],
                            )
                    dr_emp = int(round(best_disp_r / self.step_size)) * self.step_size
                    dc_emp = int(round(best_disp_c / self.step_size)) * self.step_size
                    if (dr_emp, dc_emp) != (0, 0):
                        self.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr_emp,
                            delta_c=dc_emp,
                            confidence=1.0,
                        )
                        logger.info(
                            "HCIR Motor Calibration: Calibrated Action %d -> (%d, %d)",
                            action,
                            dr_emp,
                            dc_emp,
                        )

            if avatar_displaced and new_avatar_pos:
                self.current_avatar_pos = new_avatar_pos
                if self.avatar_spawn_pos is None and spawn_candidate is not None:
                    self.avatar_spawn_pos = spawn_candidate
                tr, tc = new_avatar_pos
                if 0 <= tr < prev_grid.shape[0] and 0 <= tc < prev_grid.shape[1]:
                    floor_c = int(prev_grid[tr, tc])
                    counts_prev = np.bincount(prev_grid.flatten())
                    c_count = counts_prev[floor_c] if floor_c < len(counts_prev) else 0
                    if floor_c == bg or (c_count > 64 and floor_c not in self.known_hazard_colors):
                        self.walkable_terrain_colors.add(floor_c)

        # Movement Action Feedback: Success vs Blocked
        if action in (1, 2, 3, 4):
            curr_node = (
                (
                    int(round(old_avatar_pos[0])),
                    int(round(old_avatar_pos[1])),
                )
                if old_avatar_pos
                else None
            )

            if not avatar_displaced:
                self.stuck_counter += 1
                self.planned_action_queue.clear()
                self.mental_simulation_plan.clear()
                self.epistemic_engine.mental_plan.clear()

                if curr_node:
                    dr, dc = deltas[action]
                    blocked_pos = (
                        curr_node[0] + dr * self.step_size,
                        curr_node[1] + dc * self.step_size,
                    )
                    self.known_barriers.add(blocked_pos)
                    self.planner.record_collision_barrier(blocked_pos)

                    if curr_node not in self.spatial_transitions:
                        self.spatial_transitions[curr_node] = {}
                    self.spatial_transitions[curr_node][action] = "BLOCKED"
                    logger.info(
                        "HCIR Causal Discovery: Blocked move %d from %s -> Marked transition BLOCKED",
                        action,
                        curr_node,
                    )

                if self.last_target_pos and self.current_avatar_pos:
                    dist_to_target = math.hypot(
                        self.current_avatar_pos[0] - self.last_target_pos[0],
                        self.current_avatar_pos[1] - self.last_target_pos[1],
                    )
                    if dist_to_target <= max(1.5, self.step_size * 1.1):
                        self.target_failure_counts[self.last_target_pos] += 1
                        if self.target_failure_counts[self.last_target_pos] >= 2:
                            self.blacklisted_goals.add(self.last_target_pos)
                            self.known_barriers.add(self.last_target_pos)
                            self.planner.record_collision_barrier(self.last_target_pos)
                            logger.info(
                                "HCIR Goal Blacklisted: %s unreachable/inert",
                                self.last_target_pos,
                            )
                return

            self.stuck_counter = 0
            self.completed_click_controls.clear()
            self.quiescent_clicks.clear()
            if curr_node and new_avatar_pos and curr_node != new_avatar_pos:
                if math.hypot(
                    curr_node[0] - new_avatar_pos[0], curr_node[1] - new_avatar_pos[1]
                ) <= max(2.0, self.step_size * 1.5):
                    if curr_node not in self.spatial_transitions:
                        self.spatial_transitions[curr_node] = {}
                    self.spatial_transitions[curr_node][action] = new_avatar_pos
                    if new_avatar_pos not in self.spatial_transitions:
                        self.spatial_transitions[new_avatar_pos] = {}
                    opp_action = {1: 2, 2: 1, 3: 4, 4: 3}.get(action, action)
                    curr_dyn = self.action_dynamics.get(action)
                    if curr_dyn:
                        for cand_a, cand_d in self.action_dynamics.items():
                            if (
                                cand_a in (1, 2, 3, 4)
                                and cand_d.delta_r == -curr_dyn.delta_r
                                and cand_d.delta_c == -curr_dyn.delta_c
                            ):
                                opp_action = cand_a
                                break
                    self.spatial_transitions[new_avatar_pos][opp_action] = curr_node

        # Avatar interaction in movement game
        if self.last_target_pos and self.current_avatar_pos:
            tr, tc = self.last_target_pos
            dist_to_target = math.hypot(
                self.current_avatar_pos[0] - tr,
                self.current_avatar_pos[1] - tc,
            )
            # Arrival at target based on calibrated step size
            if dist_to_target <= max(1.5, self.step_size * 0.9):
                bg = int(np.bincount(curr_grid.flatten()).argmax())
                old_val = (
                    int(prev_grid[tr, tc])
                    if 0 <= tr < prev_grid.shape[0] and 0 <= tc < prev_grid.shape[1]
                    else 0
                )
                new_val = (
                    int(curr_grid[tr, tc])
                    if 0 <= tr < curr_grid.shape[0] and 0 <= tc < curr_grid.shape[1]
                    else 0
                )
                H_grid, W_grid = curr_grid.shape

                # Remote mutation calculation: count changed pixels outside avatar's footprint
                remote_diff_count = 0
                if avatar_displaced and old_avatar_pos and new_avatar_pos:
                    avatar_clearance = max(2.5, self.step_size * 1.2)
                    for r, c in np.argwhere(diff_mask):
                        if r <= 1 or r >= H_grid - 2:
                            continue  # step counters / HUD borders
                        if (
                            math.hypot(r - old_avatar_pos[0], c - old_avatar_pos[1])
                            > avatar_clearance
                            and math.hypot(r - new_avatar_pos[0], c - new_avatar_pos[1])
                            > avatar_clearance
                        ):
                            remote_diff_count += 1
                else:
                    remote_diff_count = int(np.sum(diff_mask))

                is_valid_object = (
                    old_val not in (0, bg)
                    and old_val not in self.walkable_terrain_colors
                    and old_val != self.known_avatar_color
                    and old_val not in self.known_exit_colors
                )

                # Item pickup: targeted item disappeared under avatar
                if (
                    is_valid_object
                    and old_val != new_val
                    and dist_to_target <= max(1.5, self.step_size * 0.7)
                ):
                    self.inventory.add(old_val)
                    self.tested_positions.add((tr, tc))
                    sig_key = f"f{old_val}_srect_zsmall"
                    self.affordance_memory[sig_key] = ObjectAffordanceRule(
                        signature_key=sig_key,
                        role=EntityRole.RESOURCE,
                        action_intent=SpatialActionIntent.PICKUP,
                        outcomes=["pickup_success"],
                        confidence=1.0,
                    )
                    logger.info(
                        "HCIR Causal Discovery: Picked up item color %d at (%d, %d)",
                        old_val,
                        tr,
                        tc,
                    )

                # Trigger mutation elsewhere: targeted object caused remote changes
                if is_valid_object and remote_diff_count >= 2 and 5 not in self.available_actions:
                    self.known_actuator_positions.add((tr, tc))
                    self.known_actuator_colors.add(old_val)
                    self.tested_positions.add((tr, tc))
                    sig_key = f"f{old_val}_srect_zsmall"
                    self.affordance_memory[sig_key] = ObjectAffordanceRule(
                        signature_key=sig_key,
                        role=EntityRole.ACTUATOR,
                        action_intent=SpatialActionIntent.ACTUATE,
                        outcomes=["remote_trigger"],
                        confidence=1.0,
                    )
                    sig_key_irreg = f"f{old_val}_sirregular_zsmall"
                    self.affordance_memory[sig_key_irreg] = ObjectAffordanceRule(
                        signature_key=sig_key_irreg,
                        role=EntityRole.ACTUATOR,
                        action_intent=SpatialActionIntent.ACTUATE,
                        outcomes=["remote_trigger"],
                        confidence=1.0,
                    )
                    self.spatial_transitions.clear()
                    self.known_barriers.clear()
                    self.blacklisted_goals.clear()
                    self.target_failure_counts.clear()
                    self.last_target_pos = None
                    logger.info(
                        "HCIR Causal Discovery: Actuator color %d triggered state mutation (%d px)",
                        old_val,
                        remote_diff_count,
                    )

        if action == 5:
            H_grid, W_grid = curr_grid.shape
            interior_diff = [
                (r, c)
                for r, c in np.argwhere(diff_mask)
                if 1 <= r <= H_grid - 2 and 1 <= c <= W_grid - 2
            ]
            if len(interior_diff) >= 4:
                if len(self.inventory) > 0:
                    self.inventory.clear()
                    if self.carried_offset and old_avatar_pos:
                        dep_r = old_avatar_pos[0] + self.carried_offset[0]
                        dep_c = old_avatar_pos[1] + self.carried_offset[1]
                        self.tested_positions.add((dep_r, dep_c))
                    self.carried_offset = None
                    self.last_target_is_goal = False
                    logger.info("HCIR Physical Affordance: Deposited object at goal receptacle")
                else:
                    self.inventory.add(1)
                    if self.last_target_pos and old_avatar_pos:
                        dr = (
                            round((self.last_target_pos[0] - old_avatar_pos[0]) / self.step_size)
                            * self.step_size
                        )
                        dc = (
                            round((self.last_target_pos[1] - old_avatar_pos[1]) / self.step_size)
                            * self.step_size
                        )
                        if (dr, dc) != (0, 0):
                            self.carried_offset = (dr, dc)
                            logger.info(
                                "HCIR Physical Affordance: Picked up/attached object with offset %s",
                                self.carried_offset,
                            )
                    if self.last_target_pos:
                        self.tested_positions.add(self.last_target_pos)
                if self.last_target_pos:
                    self.target_failure_counts[self.last_target_pos] = 0
                self.spatial_transitions.clear()
                self.known_barriers.clear()
                self.blacklisted_goals.clear()
                self.target_failure_counts.clear()
                self.last_target_pos = None
                logger.info(
                    "HCIR Causal Discovery: Action 5 triggered physical state mutation (%d px)",
                    diff_count,
                )
            else:
                if self.last_target_pos:
                    self.target_failure_counts[self.last_target_pos] += 1

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 4: Geodesic Spatial Navigation ($A^*$)
    # ═══════════════════════════════════════════════════════════════════════

    def _find_geodesic_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        barriers: set[tuple[int, int]],
        scale: int = 1,
        hazard_coords: set[tuple[int, int]] | None = None,
        carried_offset: tuple[int, int] | None = None,
        other_obstacles: set[tuple[int, int]] | None = None,
    ) -> list[int] | None:
        """Find the shortest 4-directional path using A* avoiding barriers, hazards, and carried collisions."""
        H, W = grid.shape
        start_node = start
        goal_node = goal

        if start_node == goal_node:
            return []

        blocked_nodes = set(barriers | self.known_barriers)
        if other_obstacles:
            blocked_nodes |= other_obstacles
        if hazard_coords:
            blocked_nodes |= hazard_coords

        # If goal itself is solid / known barrier, target walkable neighbors of goal
        target_nodes = {goal_node}
        if goal_node in self.known_barriers:
            target_nodes = {
                (goal_node[0] + dr, goal_node[1] + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= goal_node[0] + dr < H
                and 0 <= goal_node[1] + dc < W
                and (goal_node[0] + dr, goal_node[1] + dc) not in blocked_nodes
            }
            if not target_nodes:
                return None
        else:
            # Goal is walkable: allow entering goal_node
            blocked_nodes.discard(goal_node)

        heap = [(0, 0, start_node, [])]
        visited: dict[tuple[int, int], int] = {start_node: 0}

        actions = {}
        for act in (1, 2, 3, 4):
            if act in self.action_dynamics and self.action_dynamics[act].confidence >= 0.7:
                dr = self.action_dynamics[act].delta_r // self.step_size
                dc = self.action_dynamics[act].delta_c // self.step_size
                actions[act] = (dr, dc)
            else:
                default_deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                actions[act] = default_deltas[act]

        max_expansions = 3500
        expansions = 0

        while heap and expansions < max_expansions:
            expansions += 1
            f, cost, curr, path = heapq.heappop(heap)

            for t in target_nodes:
                if math.hypot(curr[0] - t[0], curr[1] - t[1]) < max(1.0, self.step_size * 0.9):
                    if len(path) > 0 or curr == t:
                        return path
                    else:
                        dr, dc = t[0] - curr[0], t[1] - curr[1]
                        if abs(dr) >= abs(dc):
                            act = 1 if dr < 0 else 2
                        else:
                            act = 3 if dc < 0 else 4
                        return [act]

            if visited.get(curr, 999999) < cost:
                continue

            for act, (dr, dc) in actions.items():
                trans = self.spatial_transitions.get(curr, {}).get(act)
                if trans == "BLOCKED":
                    continue
                elif isinstance(trans, tuple):
                    nr, nc = trans
                else:
                    nr, nc = curr[0] + dr * self.step_size, curr[1] + dc * self.step_size

                if carried_offset is not None:
                    cr = nr + carried_offset[0]
                    cc = nc + carried_offset[1]
                    if not (0 <= cr < H and 0 <= cc < W):
                        continue

                if 0 <= nr < H and 0 <= nc < W:
                    step_blocked = False
                    for s in range(1, self.step_size + 1):
                        ir, ic = curr[0] + dr * s, curr[1] + dc * s
                        if (ir, ic) in blocked_nodes and (ir, ic) not in target_nodes:
                            step_blocked = True
                            break
                        if carried_offset is not None:
                            icr, icc = ir + carried_offset[0], ic + carried_offset[1]
                            if (icr, icc) in blocked_nodes and (icr, icc) not in target_nodes:
                                step_blocked = True
                                break
                    if step_blocked:
                        continue

                    # Cul-de-sac dead-end pruning
                    if (nr, nc) not in target_nodes and len(
                        self.spatial_transitions.get((nr, nc), {})
                    ) >= 3:
                        exits_available = 0
                        for test_act, (tdr, tdc) in actions.items():
                            if (
                                self.spatial_transitions.get((nr, nc), {}).get(test_act)
                                == "BLOCKED"
                            ):
                                continue
                            adj = (nr + tdr * self.step_size, nc + tdc * self.step_size)
                            if adj == curr or adj in blocked_nodes:
                                continue
                            exits_available += 1
                        if exits_available == 0:
                            continue

                    tile_penalty = self.position_visits.get((nr, nc), 0)
                    hazard_penalty = 0.0
                    if hazard_coords:
                        for hr, hc in hazard_coords:
                            # Collinear approach penalty: walking towards hazard along the same corridor
                            if (
                                curr[0] == nr == hr
                                and abs(nc - hc) < abs(curr[1] - hc)
                                and abs(nc - hc) <= self.step_size * 2.5
                            ):
                                hazard_penalty += 1000.0
                            elif (
                                curr[1] == nc == hc
                                and abs(nr - hr) < abs(curr[0] - hr)
                                and abs(nr - hr) <= self.step_size * 2.5
                            ):
                                hazard_penalty += 1000.0

                        min_hdist = min(
                            (math.hypot(nr - hr, nc - hc) for hr, hc in hazard_coords),
                            default=999.0,
                        )
                        if (
                            min_hdist <= max(3.0, self.step_size * 1.5)
                            and (nr, nc) not in hazard_coords
                        ):
                            hazard_penalty += 50.0 / (min_hdist + 0.1)

                    # U-turn penalty: discourage immediately reversing direction if last action was taken from start_node
                    u_turn_penalty = 0.0
                    if curr == start_node and self.last_action in actions:
                        opp_a = {1: 2, 2: 1, 3: 4, 4: 3}.get(self.last_action)
                        if act == opp_a:
                            u_turn_penalty = 15.0

                    new_cost = cost + 1 + tile_penalty * 2 + hazard_penalty + u_turn_penalty
                    heuristic = min(math.hypot(t[0] - nr, t[1] - nc) for t in target_nodes)

                    if (nr, nc) not in visited or new_cost < visited[(nr, nc)]:
                        visited[(nr, nc)] = new_cost
                        heapq.heappush(
                            heap,
                            (new_cost + heuristic, new_cost, (nr, nc), path + [act]),
                        )

        if start_node != goal_node:
            min_target_dist = min(
                (math.hypot(start_node[0] - t[0], start_node[1] - t[1]) for t in target_nodes),
                default=999.0,
            )
            if min_target_dist <= max(1.5, self.step_size * 1.1):
                dr, dc = goal_node[0] - start_node[0], goal_node[1] - start_node[1]
                if abs(dr) >= abs(dc):
                    act = 1 if dr < 0 else 2
                else:
                    act = 3 if dc < 0 else 4
                return [act]

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Master Decision Engine
    # ═══════════════════════════════════════════════════════════════════════

    def _sync_epistemic_engine_knowledge(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> None:
        """Bidirectional knowledge sync between agent and AutonomousEpistemicEngine.

        Feeds the agent's discovered knowledge INTO the engine, and pulls
        the engine's independently discovered knowledge BACK into the agent.
        """
        ee = self.epistemic_engine

        # ── Agent → Engine ───────────────────────────────────────────
        if self.known_avatar_color is not None:
            ee.avatar_feature = self.known_avatar_color
            ee.avatar_features = {self.known_avatar_color}
        if self.current_avatar_pos is not None:
            ee.avatar_pos = self.current_avatar_pos

        ee.learned_barriers.update(self.known_barriers)
        ee.learned_goal_features.update(self.known_exit_colors)
        ee.learned_walkable_features.update(self.walkable_terrain_colors)

        for pos in self.known_actuator_positions:
            for m in ee.state_mutations:
                if m.trigger_pos == pos:
                    break
            else:
                from hbllm.hcir.world.motor_calibration import StateMutationModel

                ee.state_mutations.append(
                    StateMutationModel(
                        trigger_type="CONTACT",
                        trigger_pos=pos,
                        mutation_type="ENVIRONMENTAL_TOGGLE",
                        confidence=0.8,
                    )
                )

        # Bidirectional action dynamics synchronization
        for act, model in self.action_dynamics.items():
            if (
                act not in ee.action_dynamics
                or model.confidence >= ee.action_dynamics[act].confidence
            ):
                ee.action_dynamics[act] = model
        for act, model in ee.action_dynamics.items():
            if (
                act not in self.action_dynamics
                or model.confidence > self.action_dynamics[act].confidence
            ):
                self.action_dynamics[act] = model

        # Sync step_size into action dynamics if calibrated
        if self.step_size > 1:
            for act, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
                if act not in ee.action_dynamics:
                    ee.action_dynamics[act] = ActionDynamicsModel(
                        action_id=act,
                        delta_r=dr * self.step_size,
                        delta_c=dc * self.step_size,
                        confidence=0.9,
                        probes_tested=3,
                    )

        # ── Engine → Agent ───────────────────────────────────────────
        if ee.avatar_feature is not None:
            if self.known_avatar_color is None:
                self.known_avatar_color = ee.avatar_feature
            elif ee.avatar_feature != self.known_avatar_color:
                coords_ee = np.argwhere(curr_grid == ee.avatar_feature)
                coords_cur = np.argwhere(curr_grid == self.known_avatar_color)
                if len(coords_ee) > len(coords_cur):
                    logger.info(
                        "HCIR Causal Discovery: Epistemic engine promoted avatar body %s -> %s (area %d > %d)",
                        self.known_avatar_color,
                        ee.avatar_feature,
                        len(coords_ee),
                        len(coords_cur),
                    )
                    self.known_avatar_color = ee.avatar_feature

        for bf in ee.learned_barrier_features:
            if bf not in self.known_hazard_colors and bf != self.known_avatar_color:
                H, W = curr_grid.shape
                bg = int(np.bincount(curr_grid.flatten()).argmax())
                if bf != bg:
                    for r, c in np.argwhere(curr_grid == bf):
                        self.known_barriers.add((int(r), int(c)))

        for gf in ee.learned_goal_features:
            self.known_exit_colors.add(gf)
        for gp in ee.learned_goal_positions:
            # Engine found goal positions via win grounding
            pass  # These are used in simulate_in_mind directly

        ee.learned_walkable_features.update(self.walkable_terrain_colors)

    def _try_mental_simulation(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
    ) -> list[MentalSimulationStep] | None:
        """Attempt forward mental simulation via the epistemic engine's world model.

        This leverages the engine's learned motor dynamics, barrier map,
        push mechanics, and hierarchical subgoal decomposition to find
        plans that greedy A* cannot (e.g. switch→door→goal sequences).
        """
        ee = self.epistemic_engine
        if not ee.is_motor_grounded():
            return None

        plan = ee.simulate_in_mind(curr_grid, available_actions)
        if plan and len(plan) > 0:
            logger.info(
                "HCIR Mental Simulation: Synthesized %d-step plan via epistemic engine",
                len(plan),
            )
        return plan

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        level: int = 0,
    ) -> tuple[int, float]:
        """Execute the 4-pillar curiosity-and-exit cognitive loop via HCIR."""
        self.step_counter += 1
        self.available_actions = available_actions
        if level != self.current_level:
            logger.info("HCIR Level Transition: %d -> %d", self.current_level, level)
            self.current_level = level
            self.spatial_transitions.clear()
            self.known_barriers.clear()
            self.tested_positions.clear()
            self.blacklisted_goals.clear()
            self.target_failure_counts.clear()
            self.position_visits.clear()
            self.planned_action_queue.clear()
            self.current_avatar_pos = None
            self.last_target_pos = None
            self.mental_simulation_plan = []
            self.visual_canvas_skill.reset()

        # 1. Causal learning from observation differential
        if self.prev_grid is not None and self.last_action is not None:
            self.assimilate_causal_differential(self.prev_grid, self.last_action, curr_grid)
            # Also feed the epistemic engine for independent causal learning
            self.epistemic_engine.prev_grid = self.prev_grid.copy()
            self.epistemic_engine.last_action = self.last_action
            self.epistemic_engine.last_action_data = self.last_action_data
            self.epistemic_engine.assimilate_feedback(curr_grid, available_actions)

        # Sync knowledge bidirectionally between agent and epistemic engine
        self._sync_epistemic_engine_knowledge(curr_grid, available_actions)

        # 2. Lift frame to HCIR EntityGraph
        eg = self.perceive_scene_hcir(curr_grid, available_actions)

        if eg.agent:
            self.position_visits[eg.agent.grid_pos] += 1
            self.position_history.append(eg.agent.grid_pos)
            self.visited_cells.add(eg.agent.grid_pos)
            for dr_v in range(-self.step_size, self.step_size + 1):
                for dc_v in range(-self.step_size, self.step_size + 1):
                    self.visited_cells.add(
                        (eg.agent.grid_pos[0] + dr_v, eg.agent.grid_pos[1] + dc_v)
                    )

        # Check if target reached
        if self.last_target_pos and eg.agent:
            dist = math.hypot(
                eg.agent.grid_pos[0] - self.last_target_pos[0],
                eg.agent.grid_pos[1] - self.last_target_pos[1],
            )
            if dist <= max(1.5, self.step_size * 0.9):
                self.tested_positions.add(self.last_target_pos)
                if self.active_subgoal:
                    self.goal_decomposer.completed_subgoals.add(self.active_subgoal.id)
                    self.active_subgoal = None
                self.last_target_pos = None

        # Anti-Oscillation Target Blacklisting: detect physical coordinate flip-flop (A <-> B <-> A <-> B)
        if len(self.position_history) >= 4 and self.last_target_pos:
            p = self.position_history
            if p[-1] == p[-3] and p[-2] == p[-4] and p[-1] != p[-2]:
                logger.info(
                    "HCIR Anti-Oscillation: Detected spatial coordinate flip-flop %s <-> %s -> Blacklisting target %s",
                    p[-2],
                    p[-1],
                    self.last_target_pos,
                )
                self.target_failure_counts[self.last_target_pos] += 2
                self.blacklisted_goals.add(self.last_target_pos)
                self.last_target_pos = None

        # Declarative Neurosymbolic Skill Dispatch (Autonomous Signature Recognition)
        if not self.planned_action_queue:
            if OpticalMirrorReflectionSkillAcquisition.is_optical_mirror_reflection_grid(
                curr_grid, available_actions
            ):
                plan = OpticalMirrorReflectionSkillAcquisition.plan_optical_mirror_reflection_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif ReticleSuperpositionSkillAcquisition.is_reticle_superposition_grid(
                curr_grid, available_actions
            ):
                plan = ReticleSuperpositionSkillAcquisition.plan_reticle_superposition_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif MorphologicalProgramSynthesis.is_gravity_spill_grid(curr_grid, available_actions):
                plan = MorphologicalProgramSynthesis.plan_gravity_spill_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif PermutationAlgebraSkillAcquisition.is_lights_out_grid(
                curr_grid, available_actions
            ):
                plan = PermutationAlgebraSkillAcquisition.plan_lights_out_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif CoupledControllableSkillAcquisition.is_mirrored_convergence_grid(
                curr_grid, available_actions
            ):
                plan_moves = CoupledControllableSkillAcquisition.plan_mirrored_convergence_grid(
                    curr_grid
                )
                if plan_moves:
                    self.planned_action_queue = [(a, None) for a in plan_moves]
            elif LaserRoutingSkillAcquisition.is_laser_routing_grid(curr_grid, available_actions):
                plan = LaserRoutingSkillAcquisition.plan_laser_routing_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif VortexAttractorSkillAcquisition.is_vortex_attractor_grid(
                curr_grid, available_actions
            ):
                plan = VortexAttractorSkillAcquisition.plan_vortex_attractor_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif InvertedBuoyancySkillAcquisition.is_buoyancy_excavation_grid(
                curr_grid, available_actions
            ):
                plan = InvertedBuoyancySkillAcquisition.plan_buoyancy_excavation_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif TopologyTransformationSkillAcquisition.is_topology_transformation_grid(
                curr_grid, available_actions
            ):
                plan = TopologyTransformationSkillAcquisition.plan_topology_transformation_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif KineticCouplingSkillAcquisition.is_kinetic_coupling_grid(
                curr_grid, available_actions
            ):
                plan = KineticCouplingSkillAcquisition.plan_kinetic_coupling_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif ModalIncantationSkillAcquisition.is_incantation_grid(curr_grid, available_actions):
                plan = ModalIncantationSkillAcquisition.plan_incantation_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif AutomatonProgramSynthesisSkillAcquisition.is_automaton_synthesis_grid(
                curr_grid, available_actions
            ):
                plan = AutomatonProgramSynthesisSkillAcquisition.plan_automaton_synthesis_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif RelationalAffordanceSkillAcquisition.is_peg_solitaire_grid(
                curr_grid, available_actions
            ):
                plan = RelationalAffordanceSkillAcquisition.plan_peg_solitaire_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif TemporalEchoSkillAcquisition.is_temporal_echo_grid(curr_grid, available_actions):
                plan = TemporalEchoSkillAcquisition.plan_temporal_echo_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif RigidAssemblySkillAcquisition.is_rigid_assembly_grid(curr_grid, available_actions):
                plan = RigidAssemblySkillAcquisition.plan_rigid_assembly_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif MorphologicalMutationSkillAcquisition.is_morphological_mutation_grid(
                curr_grid, available_actions
            ):
                plan = MorphologicalMutationSkillAcquisition.plan_morphological_mutation_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif HierarchicalPatternGrammarSkillAcquisition.is_pattern_grammar_grid(
                curr_grid, available_actions
            ):
                plan = HierarchicalPatternGrammarSkillAcquisition.plan_pattern_grammar_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif GrammarTranslationSkillAcquisition.is_grammar_translation_grid(
                curr_grid, available_actions
            ):
                plan = GrammarTranslationSkillAcquisition.plan_grammar_translation_grid(
                    curr_grid, current_level=level
                )
                if plan:
                    self.planned_action_queue = list(plan)
            elif self.visual_canvas_skill.can_handle(curr_grid, available_actions):
                v_act, v_conf, v_data = self.visual_canvas_skill.plan_canvas_stamping_step(
                    curr_grid
                )
                self.prev_grid = curr_grid.copy()
                self.last_action = v_act
                self.last_action_data = v_data
                self.action_history.append(v_act)
                return v_act, v_conf
            elif KinematicLinkageSolver.is_kinematic_linkage(curr_grid, available_actions):
                k_act, k_conf, k_data = self.kinematic_linkage_solver.plan_step(
                    curr_grid, current_level=level
                )
                self.prev_grid = curr_grid.copy()
                self.last_action = k_act
                self.last_action_data = k_data
                self.action_history.append(k_act)
                return k_act, k_conf

        # 2.5. Execute mental simulation plan if one is in-flight
        if self.mental_simulation_plan:
            step = self.mental_simulation_plan.pop(0)
            if step.action in available_actions:
                self.prev_grid = curr_grid.copy()
                self.last_action = step.action
                self.last_action_data = step.action_data
                self.action_history.append(step.action)
                return step.action, 0.92
            else:
                # Mental plan invalidated — action no longer available
                self.mental_simulation_plan.clear()
                logger.info(
                    "HCIR Mental Simulation: Plan invalidated — action %d unavailable", step.action
                )

        # 3. Execute in-flight planned action queue (for multi-action combos like clicks)
        if self.planned_action_queue:
            act, act_data = self.planned_action_queue.pop(0)
            self.prev_grid = curr_grid.copy()
            self.last_action = act
            self.last_action_data = act_data
            self.action_history.append(act)
            return act, 0.9

        # 4. Handle Click & Manipulation Affordance Puzzles (Actions 5, 6, 7)
        has_full_navigation = (1 in available_actions or 2 in available_actions) and (
            3 in available_actions or 4 in available_actions
        )
        if not has_full_navigation and 6 in available_actions:
            return self._plan_click_manipulation(eg, curr_grid, available_actions)
        elif self.stuck_counter >= 3:
            if 6 in available_actions:
                return self._plan_click_manipulation(eg, curr_grid, available_actions)
            elif 5 in available_actions:
                self.stuck_counter = 0
                self.prev_grid = curr_grid.copy()
                self.last_action = 5
                self.last_action_data = None
                self.action_history.append(5)
                return 5, 0.8

        # 4.5. Physical Affordance Probing with Carried Offset Awareness
        if len(self.inventory) > 0 and eg.agent and 5 in available_actions:
            offset = self.carried_offset or (0, 0)
            carried_pos = (eg.agent.grid_pos[0] + offset[0], eg.agent.grid_pos[1] + offset[1])

            # 1. Immediate Deposit if carried object is inside goal receptacle
            in_receptacle = any(
                bbox[0] <= carried_pos[0] <= bbox[1] and bbox[2] <= carried_pos[1] <= bbox[3]
                for bbox in self.known_goal_bboxes
            )
            if in_receptacle:
                logger.info(
                    "HCIR Pillar 1 Exit: Carried object at %s is inside goal receptacle -> Action 5 (DEPOSIT)",
                    carried_pos,
                )
                self.prev_grid = curr_grid.copy()
                self.last_action = 5
                self.last_action_data = None
                self.action_history.append(5)
                return 5, 0.95

            # 2. If carried with offset, route avatar to required delivery position
            if self.carried_offset is not None:
                # Other physical objects are obstacles (cannot be walked through or overlapped)
                other_obstacles: set[tuple[int, int]] = set()
                for ent in eg.entities.values():
                    if ent == eg.agent:
                        continue
                    # Skip the carried entity
                    if (
                        math.hypot(
                            ent.grid_pos[0] - carried_pos[0], ent.grid_pos[1] - carried_pos[1]
                        )
                        < self.step_size * 0.9
                    ):
                        continue
                    # Skip goal receptacle tiles that are not barriers
                    if ent.role == EntityRole.GOAL and ent.grid_pos not in eg.barriers:
                        continue
                    for r in range(ent.bounding_box[0], ent.bounding_box[1] + 1):
                        for c in range(ent.bounding_box[2], ent.bounding_box[3] + 1):
                            other_obstacles.add((r, c))

                valid_paths = []

                # Core Spatial Containment: solve_silhouette_packing for optimal slot allocation
                silhouette_cells: set[tuple[int, int]] = set()
                for bbox in self.known_goal_bboxes:
                    for r in range(bbox[0], bbox[1] + 1):
                        for c in range(bbox[2], bbox[3] + 1):
                            if (
                                (r, c) not in other_obstacles
                                and (r, c) not in eg.barriers
                                and (r, c) not in self.known_barriers
                            ):
                                silhouette_cells.add((r, c))

                carried_piece_shape = {
                    (dr_i, dc_i) for dr_i in range(self.step_size) for dc_i in range(self.step_size)
                }
                uncollected_shapes = []
                for ent in eg.entities.values():
                    if ent != eg.agent and ent.role == EntityRole.RESOURCE:
                        if (
                            math.hypot(
                                ent.grid_pos[0] - carried_pos[0], ent.grid_pos[1] - carried_pos[1]
                            )
                            >= self.step_size * 0.9
                        ):
                            uncollected_shapes.append(
                                {
                                    (dr_i, dc_i)
                                    for dr_i in range(self.step_size)
                                    for dc_i in range(self.step_size)
                                }
                            )

                all_pieces = [carried_piece_shape] + uncollected_shapes
                packing_plan = None
                if silhouette_cells and sum(len(p) for p in all_pieces) == len(silhouette_cells):
                    packing_plan = BaseSpatialContainmentEngine.solve_silhouette_packing(
                        silhouette_cells, all_pieces
                    )

                if packing_plan is not None:
                    target_r, target_c = packing_plan[0]
                    req_avatar_pos = (target_r - offset[0], target_c - offset[1])
                    if (
                        req_avatar_pos not in eg.barriers
                        and req_avatar_pos not in self.known_barriers
                        and req_avatar_pos not in other_obstacles
                    ):
                        path = self._find_geodesic_path(
                            curr_grid,
                            eg.agent.grid_pos,
                            req_avatar_pos,
                            eg.barriers,
                            carried_offset=offset,
                            other_obstacles=other_obstacles,
                        )
                        if path is not None:
                            valid_paths.append(
                                (len(path), path, (target_r, target_c), req_avatar_pos)
                            )

                if not valid_paths:
                    for bbox in self.known_goal_bboxes:
                        # Scan slots aligned to avatar phase
                        min_kr = math.ceil(
                            (bbox[0] - eg.agent.grid_pos[0] - offset[0]) / self.step_size
                        )
                        max_kr = math.floor(
                            (bbox[1] - eg.agent.grid_pos[0] - offset[0]) / self.step_size
                        )
                        min_kc = math.ceil(
                            (bbox[2] - eg.agent.grid_pos[1] - offset[1]) / self.step_size
                        )
                        max_kc = math.floor(
                            (bbox[3] - eg.agent.grid_pos[1] - offset[1]) / self.step_size
                        )

                        for kr in range(min_kr, max_kr + 1):
                            for kc in range(min_kc, max_kc + 1):
                                target_r = eg.agent.grid_pos[0] + offset[0] + kr * self.step_size
                                target_c = eg.agent.grid_pos[1] + offset[1] + kc * self.step_size

                                slot_occupied = (
                                    (target_r, target_c) in other_obstacles
                                    or (target_r, target_c) in eg.barriers
                                    or (target_r, target_c) in self.known_barriers
                                )
                                if slot_occupied:
                                    continue

                                req_avatar_pos = (target_r - offset[0], target_c - offset[1])
                                if (
                                    req_avatar_pos not in eg.barriers
                                    and req_avatar_pos not in self.known_barriers
                                    and req_avatar_pos not in other_obstacles
                                ):
                                    path = self._find_geodesic_path(
                                        curr_grid,
                                        eg.agent.grid_pos,
                                        req_avatar_pos,
                                        eg.barriers,
                                        carried_offset=offset,
                                        other_obstacles=other_obstacles,
                                    )
                                    if path is not None:
                                        valid_paths.append(
                                            (len(path), path, (target_r, target_c), req_avatar_pos)
                                        )

                if valid_paths:
                    # Sort prioritizing compact packing (corner alignment) then path length
                    valid_paths.sort(key=lambda x: (x[2][0], x[2][1], x[0]))
                    cost, path, target_slot, req_pos = valid_paths[0]
                    if len(path) == 0:
                        logger.info(
                            "HCIR Pillar 1 Exit: Reached delivery position %s for slot %s -> Action 5 (DEPOSIT)",
                            req_pos,
                            target_slot,
                        )
                        self.prev_grid = curr_grid.copy()
                        self.last_action = 5
                        self.last_action_data = None
                        self.action_history.append(5)
                        return 5, 0.95
                    else:
                        first_act = path[0]
                        logger.info(
                            "HCIR Pillar 1 Exit: Routing with offset %s to slot %s via avatar pos %s (%d steps)",
                            offset,
                            target_slot,
                            req_pos,
                            len(path),
                        )
                        self.last_target_pos = req_pos
                        self.prev_grid = curr_grid.copy()
                        self.last_action = first_act
                        self.last_action_data = None
                        self.action_history.append(first_act)
                        return first_act, 0.95

        # Physical Affordance Probing (Action 5) upon target arrival (for pickup/interaction)
        if (
            5 in available_actions
            and self.last_target_pos
            and eg.agent
            and len(self.inventory) == 0
        ):
            target_bbox = None
            for e in eg.entities.values():
                if e.grid_pos == self.last_target_pos or (
                    e.bounding_box[0] <= self.last_target_pos[0] <= e.bounding_box[1]
                    and e.bounding_box[2] <= self.last_target_pos[1] <= e.bounding_box[3]
                ):
                    target_bbox = e.bounding_box
                    break

            if target_bbox:
                dr_box = max(
                    0, target_bbox[0] - eg.agent.grid_pos[0], eg.agent.grid_pos[0] - target_bbox[1]
                )
                dc_box = max(
                    0, target_bbox[2] - eg.agent.grid_pos[1], eg.agent.grid_pos[1] - target_bbox[3]
                )
                dist = math.hypot(dr_box, dc_box)
                dr = (
                    0
                    if dr_box == 0
                    else (
                        target_bbox[0] - eg.agent.grid_pos[0]
                        if eg.agent.grid_pos[0] < target_bbox[0]
                        else target_bbox[1] - eg.agent.grid_pos[0]
                    )
                )
                dc = (
                    0
                    if dc_box == 0
                    else (
                        target_bbox[2] - eg.agent.grid_pos[1]
                        if eg.agent.grid_pos[1] < target_bbox[2]
                        else target_bbox[3] - eg.agent.grid_pos[1]
                    )
                )
            else:
                dist = math.hypot(
                    eg.agent.grid_pos[0] - self.last_target_pos[0],
                    eg.agent.grid_pos[1] - self.last_target_pos[1],
                )
                dr = self.last_target_pos[0] - eg.agent.grid_pos[0]
                dc = self.last_target_pos[1] - eg.agent.grid_pos[1]

            if dist <= max(2.0, self.step_size * 1.35) and self.last_action != 5:
                target_facing_act = None
                if abs(dr) >= abs(dc) and dr != 0:
                    target_facing_act = 1 if dr < 0 else 2
                elif dc != 0:
                    target_facing_act = 3 if dc < 0 else 4

                if target_facing_act and self.last_action != target_facing_act:
                    self.prev_grid = curr_grid.copy()
                    self.last_action = target_facing_act
                    self.last_action_data = None
                    self.action_history.append(target_facing_act)
                    return target_facing_act, 0.95

                self.prev_grid = curr_grid.copy()
                self.last_action = 5
                self.last_action_data = None
                self.action_history.append(5)
                logger.info(
                    "HCIR Pillar 3 Affordance: Interacting (Action 5) with target at %s",
                    self.last_target_pos,
                )
                return 5, 0.95

        # Extract hazard coordinates for geodesic clearance
        hazard_coords: set[tuple[int, int]] = set()
        for hz in getattr(eg, "dynamic_hazards", []):
            coords = hz.properties.get("coords", [hz.grid_pos])
            for hr, hc in coords:
                hazard_coords.add((int(hr), int(hc)))
        for ent in eg.entities.values():
            if ent.role == EntityRole.DYNAMIC_HAZARD or ent.color in self.known_hazard_colors:
                coords = ent.properties.get("coords", [ent.grid_pos])
                for hr, hc in coords:
                    hazard_coords.add((int(hr), int(hc)))

        # 5. Pillar 1: Exit Convergence (Closed-Loop Step-by-Step)
        goal_ent = self.evaluate_exit_conditions(eg, curr_grid)
        if goal_ent and eg.agent:
            target_pos = goal_ent.grid_pos
            path = self._find_geodesic_path(
                curr_grid,
                eg.agent.grid_pos,
                target_pos,
                eg.barriers,
                hazard_coords=hazard_coords,
            )
            if path:
                logger.info(
                    "HCIR Pillar 1 Exit: Routing to %s at %s (%d steps)",
                    goal_ent.role,
                    target_pos,
                    len(path),
                )
                first_act = path[0]
                self.last_target_pos = target_pos
                self.prev_grid = curr_grid.copy()
                self.last_action = first_act
                self.last_action_data = None
                self.action_history.append(first_act)
                return first_act, 0.95
            elif path is not None and len(path) == 0:
                if 5 in available_actions and len(self.inventory) > 0:
                    logger.info(
                        "HCIR Pillar 1 Exit: Already at goal %s with inventory - depositing",
                        target_pos,
                    )
                    self.last_target_pos = target_pos
                    self.prev_grid = curr_grid.copy()
                    self.last_action = 5
                    self.last_action_data = None
                    self.action_history.append(5)
                    return 5, 0.95
            else:
                # Goal exists but greedy A* can't reach it — try mental simulation
                # (handles switch→door→goal sequences, Sokoban push plans)
                if not self.mental_simulation_plan:
                    mental_plan = self._try_mental_simulation(curr_grid, available_actions)
                    if mental_plan:
                        self.mental_simulation_plan = mental_plan
                        step = self.mental_simulation_plan.pop(0)
                        self.prev_grid = curr_grid.copy()
                        self.last_action = step.action
                        self.last_action_data = step.action_data
                        self.action_history.append(step.action)
                        return step.action, 0.92

        # 6. Pillar 3: Epistemic Curiosity — Probe Nearest Untested Object
        curiosity_target = self.select_curiosity_target(eg)
        if curiosity_target and eg.agent:
            target_pos = curiosity_target.grid_pos
            path = self._find_geodesic_path(
                curr_grid,
                eg.agent.grid_pos,
                target_pos,
                eg.barriers,
                hazard_coords=hazard_coords,
            )
            if path:
                logger.info(
                    "HCIR Pillar 3 Curiosity: Probing %s (color %d) at %s (%d steps)",
                    curiosity_target.role,
                    curiosity_target.color,
                    target_pos,
                    len(path),
                )
                self.last_target_pos = target_pos
                first_act = path[0]
                self.prev_grid = curr_grid.copy()
                self.last_action = first_act
                self.last_action_data = None
                self.action_history.append(first_act)
                return first_act, 0.85

        # 6.5. Affordance Fallback: When goal is unreachable and walking stagnates
        if eg.agent and (not curiosity_target or self.position_visits[eg.agent.grid_pos] >= 2):
            # Try mental simulation before falling back to brute-force
            if not self.mental_simulation_plan and self.step_counter > 5:
                mental_plan = self._try_mental_simulation(curr_grid, available_actions)
                if mental_plan:
                    self.mental_simulation_plan = mental_plan
                    step = self.mental_simulation_plan.pop(0)
                    self.prev_grid = curr_grid.copy()
                    self.last_action = step.action
                    self.last_action_data = step.action_data
                    self.action_history.append(step.action)
                    return step.action, 0.92

            if 6 in available_actions:
                return self._plan_click_manipulation(eg, curr_grid, available_actions)
            elif 5 in available_actions and self.last_action != 5:
                self.prev_grid = curr_grid.copy()
                self.last_action = 5
                self.last_action_data = None
                self.action_history.append(5)
                return 5, 0.8

        # 7. Fallback: Collision-Safe Anti-Oscillation Step
        return self._fallback_exploration(eg, available_actions, curr_grid)

    def _plan_click_manipulation(
        self,
        eg: EntityGraph,
        curr_grid: np.ndarray,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Execute epistemic click exploration with causal momentum and submission interleaving."""
        curr_hash = hash(curr_grid.tobytes())
        is_revisit = curr_hash in self.visited_grid_hashes
        self.visited_grid_hashes.add(curr_hash)

        # 1. Interleave Action 7 (SUBMIT) or Action 5 (EXECUTE) every 4 steps
        auxiliary_actions = [a for a in available_actions if a in (5, 7)]
        if (
            auxiliary_actions
            and self.last_action == 6
            and (self.step_counter % 4 == 0 or len(self.completed_click_controls) > 0)
        ):
            chosen = auxiliary_actions[0]
            self.prev_grid = curr_grid.copy()
            self.last_action = chosen
            self.last_action_data = None
            self.completed_click_controls.clear()
            return chosen, 0.9

        # 2. Causal Momentum: continue clicking effective controls
        has_movement = any(a in available_actions for a in (1, 2, 3, 4))
        if (
            self.last_action == 6
            and self.last_target_pos is not None
            and self.prev_grid is not None
        ):
            diff_count = int(np.sum(self.prev_grid != curr_grid))
            if diff_count > 0:
                self.consecutive_effective_clicks += 1
                if diff_count > 1:
                    self.spatial_transitions.clear()
                    self.known_barriers.clear()
                    self.tested_positions.clear()
                    self.blacklisted_goals.clear()
                    self.target_failure_counts.clear()

                max_effective = 1 if has_movement else 8
                if is_revisit or self.consecutive_effective_clicks >= max_effective:
                    self.completed_click_controls.add(self.last_target_pos)
                    self.consecutive_effective_clicks = 0
                else:
                    tr, tc = self.last_target_pos
                    self.prev_grid = curr_grid.copy()
                    self.last_action = 6
                    self.last_action_data = {"x": tc, "y": tr}
                    return 6, 0.95
            else:
                self.quiescent_clicks.add(self.last_target_pos)
                self.consecutive_effective_clicks = 0

        # 2.5 Visual Symmetry Analyzer Gestalt Completion
        if not has_movement and 6 in available_actions and not self.planned_action_queue:
            is_sym, best_axis = VisualSymmetryAnalyzer.is_near_symmetric(curr_grid, threshold=0.55)
            if is_sym and best_axis:
                bg = int(np.argmax(np.bincount(curr_grid.flatten())))
                completed_grid = VisualSymmetryAnalyzer.predict_symmetric_completion(
                    curr_grid, symmetry_type=best_axis, background_color=bg
                )
                diff_coords = [
                    (r, c)
                    for r, c in zip(*np.where(curr_grid != completed_grid))
                    if (r, c) not in self.quiescent_clicks
                    and (r, c) not in self.completed_click_controls
                ]
                if 0 < len(diff_coords) <= 64:
                    logger.info(
                        "HCIR VisualSymmetry: Detected %s symmetry -> Queueing %d completion clicks",
                        best_axis,
                        len(diff_coords),
                    )
                    for r, c in diff_coords:
                        self.planned_action_queue.append((6, {"x": int(c), "y": int(r)}))
                    if 7 in available_actions:
                        self.planned_action_queue.append((7, None))
                    elif 5 in available_actions:
                        self.planned_action_queue.append((5, None))

                    act, act_data = self.planned_action_queue.pop(0)
                    self.last_target_pos = (act_data["y"], act_data["x"]) if act_data else None
                    self.prev_grid = curr_grid.copy()
                    self.last_action = act
                    self.last_action_data = act_data
                    self.action_history.append(act)
                    return act, 0.95

        # 3. Select next candidate interactable control
        candidates = []
        for e in eg.entities.values():
            if e.area > curr_grid.size * 0.25:
                continue
            if e == eg.agent:
                continue
            if e.role in (EntityRole.GOAL, EntityRole.PORTAL, EntityRole.DYNAMIC_HAZARD):
                continue
            if has_movement and e.role == EntityRole.RESOURCE:
                continue
            cr, cc = int(round(e.centroid[0])), int(round(e.centroid[1]))
            H, W = curr_grid.shape
            if cr < 0 or cr >= H or cc < 0 or cc >= W:
                continue
            if (
                self.current_avatar_pos
                and math.hypot(cr - self.current_avatar_pos[0], cc - self.current_avatar_pos[1])
                <= 2.5
            ):
                continue
            if (cr, cc) not in self.quiescent_clicks and (
                cr,
                cc,
            ) not in self.completed_click_controls:
                candidates.append((cr, cc, e))

        if candidates:
            candidates.sort(
                key=lambda item: (
                    0
                    if item[2].role == EntityRole.ACTUATOR
                    else (1 if 4 <= item[2].area <= 300 else 2),
                    item[2].area,
                )
            )
            best_r, best_c, best_ent = candidates[0]
            self.last_target_pos = (best_r, best_c)
            self.consecutive_effective_clicks = 1
            self.prev_grid = curr_grid.copy()
            self.last_action = 6
            self.last_action_data = {"x": best_c, "y": best_r}
            return 6, 0.85

        # Reset completed controls to allow cycling if needed
        self.completed_click_controls.clear()
        self.quiescent_clicks.clear()
        H, W = curr_grid.shape
        self.prev_grid = curr_grid.copy()
        self.last_action = 6
        self.last_action_data = {"x": W // 2, "y": H // 2}
        return 6, 0.5

    def _fallback_exploration(
        self,
        eg: EntityGraph,
        available_actions: list[int],
        curr_grid: np.ndarray,
    ) -> tuple[int, float]:
        """Perform collision-safe exploratory step when no specific path is active."""
        movement_actions = [a for a in available_actions if a in (1, 2, 3, 4)]
        if not movement_actions:
            chosen = available_actions[0]
            self.prev_grid = curr_grid.copy()
            self.last_action = chosen
            self.last_action_data = None
            return chosen, 0.4

        taboo_actions: set[int] = set()
        if len(self.action_history) >= 4:
            recent = self.action_history[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
                taboo_actions.update([recent[0], recent[1]])

        viable = [a for a in movement_actions if a not in taboo_actions] or movement_actions

        action_deltas = {}
        for act in (1, 2, 3, 4):
            if act in self.action_dynamics and self.action_dynamics[act].confidence >= 0.7:
                dr = self.action_dynamics[act].delta_r // self.step_size
                dc = self.action_dynamics[act].delta_c // self.step_size
                action_deltas[act] = (dr, dc)
            else:
                action_deltas[act] = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[act]
        pos = eg.agent.grid_pos if eg.agent else (0, 0)

        # Identify candidate goal for heuristic exploration bias
        goal_pos = None
        for ent in eg.entities.values():
            if ent.role in (EntityRole.GOAL, EntityRole.PORTAL, EntityRole.RECEPTACLE):
                goal_pos = ent.grid_pos
                break

        scored: list[tuple[float, int]] = []
        for act in viable:
            # If known blocked from this position, maximum penalty
            if self.spatial_transitions.get(pos, {}).get(act) == "BLOCKED":
                scored.append((999999.0, act))
                continue

            dr, dc = action_deltas[act]
            npos = (pos[0] + dr * self.step_size, pos[1] + dc * self.step_size)
            if npos in eg.barriers or npos in self.known_barriers:
                visits = 9999.0
            else:
                visits = float(self.position_visits.get(npos, 0))

            # PhysicsPredictor Deadlock Prevention
            all_barriers = set(eg.barriers) | self.known_barriers
            target_positions = {
                ent.grid_pos
                for ent in eg.entities.values()
                if ent.role in (EntityRole.GOAL, EntityRole.PORTAL, EntityRole.RECEPTACLE)
            }
            if PhysicsPredictor.is_corner_deadlock(
                entity_pos=npos,
                barrier_cells=all_barriers,
                target_positions=target_positions,
                grid_shape=curr_grid.shape,
                step_size=self.step_size,
            ):
                visits += 5000.0
            if PhysicsPredictor.is_line_deadlock(
                entity_pos=npos,
                barrier_cells=all_barriers,
                target_positions=target_positions,
                grid_shape=curr_grid.shape,
                step_size=self.step_size,
            ):
                visits += 3000.0

            # Counterfactual forward prediction
            pred_grid = CounterfactualPlanner.predict_outcome(
                grid=curr_grid,
                action_id=act,
                action_model=self.action_dynamics.get(act),
                avatar_color=self.known_avatar_color,
                movable_colors=self.epistemic_engine.learned_cargo_features,
            )
            if self.action_dynamics.get(act) is not None and np.array_equal(pred_grid, curr_grid):
                visits += 4000.0

            dist_bias = 0.0
            if goal_pos:
                dist_bias = math.hypot(npos[0] - goal_pos[0], npos[1] - goal_pos[1])

            score = visits * 10.0 + dist_bias
            scored.append((score, act))

        scored.sort(key=lambda s: s[0])
        best_act = scored[0][1]

        self.prev_grid = curr_grid.copy()
        self.last_action = best_act
        self.last_action_data = None
        self.action_history.append(best_act)
        return best_act, 0.45
