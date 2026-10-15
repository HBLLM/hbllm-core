"""Prefrontal Working Memory & Hierarchical Subgoal Schema System.

Modeled on the primate dorsolateral prefrontal cortex (dlPFC):
1. Latent State Retention (Object Permanence): Retains representations of items/tools acquired
   by the agent even when they vanish from the sensory grid upon contact/pickup.
2. Hierarchical Subgoal Decomposition: Formulates and manages multi-stage behavioral schemas:
   [Acquire Tool / Key] -> [Transport & Actuate Barrier] -> [Reach Primary Goal].
3. Tool-Affordance Resonance: Learns and associates tool features with barrier features they unlock.
4. Declarative Neuro-Symbolic Integration: Converts stages to SymbolicSubgoals and SubgoalSequences,
   retains atomic TransactionOperation proposals, and grounds Pearl causal predicates.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import ConceptNode, FactNode, ProcedureNode, SkillNode
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    SpatialActionIntent,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_hypothesis import InductiveHypothesisEngine
from hbllm.hcir.world.cortex_morphology import MorphologicalSaliencyEngine
from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema
from hbllm.hcir.world.frontopolar_subgoal_stack import FrontopolarSubgoalStack
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.inferotemporal_segmentation import InferotemporalSegmentationEngine
from hbllm.hcir.world.optical_ray_projection import OpticalRayProjector
from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAttributor
from hbllm.hcir.world.symbolic_constraints import SymbolicConstraintSolver
from hbllm.hcir.world.visuospatial_working_memory import VisuospatialWorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemorySalienceState:
    """Snapshot of working memory salient counts to detect phase shift events."""

    calibrated_actions: int = 0
    goal_positions: int = 0
    barriers: int = 0
    trajectories: int = 0
    mobile_threats: int = 0


@dataclass
class HeldItem:
    """An entity retained in latent working memory as carried / possessed by the agent."""

    feature_id: int
    role: str
    acquired_step: int
    original_position: tuple[int, int]
    properties: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export held item state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="held_item",
            operator="==",
            value=self.feature_id,
            relation=self.role,
            target_entity="avatar",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR state updates reflecting held item permanence."""
        node_id = f"held_item_{self.feature_id}"
        node = ConceptNode(
            id=node_id,
            label=f"held_item_{self.feature_id}",
            definition=f"Possessed tool or entity with feature {self.feature_id}",
            domain="body_schema",
            properties={
                "feature_id": self.feature_id,
                "role": self.role,
                "acquired_step": self.acquired_step,
                "original_position": list(self.original_position),
                "properties": dict(self.properties),
            },
        )
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data=node.model_dump(),
            )
        ]


@dataclass
class SubgoalStage:
    """A discrete stage in a hierarchical prefrontal behavioral schema."""

    stage_id: str
    target_role: str
    target_positions: set[tuple[int, int]]
    required_feature: int | None = None
    action_on_arrival: int | None = None  # None = CONTACT, or explicit action like 5
    is_completed: bool = False

    def to_symbolic_subgoal(self) -> SymbolicSubgoal:
        """Lift this stage to a declarative SymbolicSubgoal."""
        intent = (
            SpatialActionIntent.INTERACT
            if self.stage_id == "UNLOCK_BARRIER"
            else SpatialActionIntent.NAVIGATE
        )
        return SymbolicSubgoal(
            intent=intent,
            target_query={
                "stage_id": self.stage_id,
                "role": self.target_role,
                "target_positions": list(self.target_positions),
                "required_feature": self.required_feature,
            },
        )


@dataclass
class SubgoalSchema:
    """An active hierarchical plan composed of sequential stages."""

    schema_name: str
    stages: list[SubgoalStage]
    current_stage_idx: int = 0
    failure_count: int = 0
    confidence: float = 1.0

    @property
    def current_stage(self) -> SubgoalStage | None:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    def advance(self) -> SubgoalStage | None:
        if self.current_stage:
            self.current_stage.is_completed = True
        self.current_stage_idx += 1
        return self.current_stage

    @property
    def is_finished(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def to_subgoal_sequence(self) -> SubgoalSequence:
        """Convert all stages into a declarative HCIR SubgoalSequence."""
        return SubgoalSequence(*[stage.to_symbolic_subgoal() for stage in self.stages])

    def is_current_stage_satisfied(
        self,
        avatar_pos: tuple[int, int] | None,
        held_items: Sequence[HeldItem] | None = None,
        unlocked_barriers: set[tuple[int, int]] | None = None,
    ) -> bool:
        """Check if current stage target is satisfied by current epistemic state."""
        st = self.current_stage
        if st is None:
            return True
        if avatar_pos is not None and avatar_pos in st.target_positions:
            return True
        if st.stage_id == "ACQUIRE_TOOL" and held_items:
            if st.required_feature is None or any(
                h.feature_id == st.required_feature for h in held_items
            ):
                return True
        if st.stage_id == "UNLOCK_BARRIER" and unlocked_barriers:
            if any(p in unlocked_barriers for p in st.target_positions):
                return True
        return False


class PrefrontalWorkingMemory:
    """Biologically-modeled Prefrontal Working Memory (dlPFC) for latent state and subgoal reasoning."""

    def __init__(self) -> None:
        self.held_items: list[HeldItem] = []
        self.active_schema: SubgoalSchema | None = None
        self.procedural_library: dict[str, DeclarativeNeuroSymbolicSkill] = {}
        self.unlocked_barriers: set[tuple[int, int]] = set()

        self.tool_barrier_affinities: dict[int, set[int]] = {}  # tool_feat -> set(barrier_feat)
        self.activated_triggers: set[tuple[int, int]] = set()
        self.topological_subgoals: list[tuple[int, int]] = []
        self.active_macro_goal: str | None = None
        self.focal_attention_point: tuple[int, int] | None = None
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

        self.visuospatial: VisuospatialWorkingMemory = VisuospatialWorkingMemory()
        self.body_schema: ExtendedBodySchema = ExtendedBodySchema()
        self.subgoal_stack: FrontopolarSubgoalStack = FrontopolarSubgoalStack()
        self.remote_causal: RemoteCausalAttributor = RemoteCausalAttributor()
        self.habenular_ior: HabenularEpisodicIOR = HabenularEpisodicIOR()
        self.optical_projector: OpticalRayProjector = OpticalRayProjector()
        self.it_segmenter: InferotemporalSegmentationEngine = InferotemporalSegmentationEngine()
        self.acc_conflict: AnteriorCingulateConflictMonitor = AnteriorCingulateConflictMonitor()
        self.hypothesis_engine: InductiveHypothesisEngine = InductiveHypothesisEngine()
        self.morphology: MorphologicalSaliencyEngine = MorphologicalSaliencyEngine()
        self._salience_baseline: WorkingMemorySalienceState = WorkingMemorySalienceState()

        # Exploration & Curiosity Tracking (Single Source of Truth)
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

        # Active Causal Empowerment & Contingency Testing
        self.active_causal_actuator_coord: tuple[int, int] | None = None
        self.consecutive_causal_probes: int = 0
        self.causal_actuator_streak_limit: int = 4

        # Feature-level interventional causal falsification
        self.quiescent_features: set[int] = set()
        self.effective_features: set[int] = set()
        self.feature_quiescent_counts: dict[int, int] = {}

        # Metacognitive Refractory Action Inhibition
        self.inhibited_actions: dict[Any, int] = {}
        self.prior_avatar_feature: int | None = None
        self.entity_action_failed_positions: dict[Any, set[tuple[int, int]]] = {}
        self.immobile_entity_actions: set[Any] = set()

        # Universal Dynamic Action Affordance Registry
        self.action_affordances: dict[Any, Any] = {}
        self.focus_switch_actions: set[Any] = set()
        self.last_non_displacement_action: Any | None = None
        self.level_epistemic_probes: int = 0
        self.total_epistemic_probes: int = 0

    def has_salience_shift(self, engine: Any) -> bool:
        """Evaluate whether semantic discoveries, dynamic patroller kinematics, or high surprise
        warrant an executive phase shift back to forward mental simulation.
        """
        n_act = len(getattr(engine, "action_dynamics", {}))
        n_goals = len(getattr(engine, "learned_goal_positions", set()))
        n_barriers = len(getattr(engine, "learned_barriers", set()))
        n_traj = len(
            getattr(getattr(engine, "collision_cones", None), "projected_trajectories", [])
        )
        n_threats = len(getattr(engine, "mobile_threat_features", set()))

        b = self._salience_baseline
        shift = (
            n_act > b.calibrated_actions
            or n_goals > b.goal_positions
            or n_barriers != b.barriers
            or n_traj > b.trajectories
            or n_threats > b.mobile_threats
            or getattr(engine, "last_surprise", 0.0) >= 0.4
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
        n_act = len(getattr(engine, "action_dynamics", {}))
        n_goals = len(getattr(engine, "learned_goal_positions", set()))
        n_barriers = len(getattr(engine, "learned_barriers", set()))
        n_traj = len(
            getattr(getattr(engine, "collision_cones", None), "projected_trajectories", [])
        )
        n_threats = len(getattr(engine, "mobile_threat_features", set()))
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
        self.remote_causal.reset_episode(retain_long_term=retain_long_term)
        self.acc_conflict.reset_episode(retain_long_term=retain_long_term)
        self.hypothesis_engine.reset_episode(retain_confirmed=retain_long_term)
        self.active_probe_target = None
        self.active_probe_id = None
        self.probe_target_steps = 0
        self.recent_positions.clear()
        self.recent_actions.clear()
        self.consecutive_effective_clicks = 0
        self.consecutive_goal_converging_clicks = 0
        self.active_causal_actuator_coord = None
        self.consecutive_causal_probes = 0
        self.level_epistemic_probes = 0
        if not retain_inhibitions:
            self.inhibited_actions.clear()
        if not retain_long_term:
            self.tool_barrier_affinities.clear()
            self.habenular_ior.reset()
            self.avatar_controllability_evidence.clear()
            self.prev_oriented_threats.clear()
            self._salience_baseline = WorkingMemorySalienceState()
            self.probed_entity_ids.clear()
            self.entity_visit_counts.clear()
            self.position_visit_counts.clear()
            self.exhausted_candidate_goals.clear()
            self.quiescent_click_targets.clear()
            self.effective_click_targets.clear()
            self.last_effective_click_coord = None
            self.click_affordances.clear()
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
        else:
            self.habenular_ior.reset_episode(
                retain_long_term=True,
                retain_inhibitions=retain_inhibitions,
            )

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

    def orient_attention(self, coord: tuple[int, int]) -> None:
        """Orient prefrontal attentional focus to surprising or salient coordinates."""
        self.focal_attention_point = coord

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
        for name, skill in self.procedural_library.items():
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
        return ops
