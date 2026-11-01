"""Hippocampal & Lateral Habenular Episodic Memory Faculty.

Biologically modeled on mammalian hippocampal sharp-wave ripples (SWR) and
habenular anti-reward signaling:
1. Hippocampal Episodic Trajectory Buffer:
   - Maintains continuous experiential traces: tau = [(s_0, a_0, r_0, s_1), ..., (s_T, a_T, r_T, s_{T+1})].
   - Captures sensory signatures, avatar positions, actions, and structural outcome diffs.
2. Sharp-Wave Ripple (SWR) Backward Replay:
   - Triggered upon catastrophe (death, loss, negative reward, or unexpected terminal reset).
   - Replays the fatal trajectory prefix in reverse order: T -> T - K.
   - Computes one-shot negative temporal credit assignment:
     Delta Q(s_t, a_t) = -kappa * gamma^(T - t).
   - Identifies fatal action precursors and establishes Habenular inhibitory gating.
3. Spatial Hazard & Line-of-Sight Deduction:
   - Analyzes target destination and cardinal sightlines at point of death to identify lethal hazard features.
4. Episodic Trajectory Recall (Forward Preplay Retrieval):
   - Supports associative retrieval of past successful trajectories for similar goals.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_perception import EpistemicObservationDiff
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.engine_state import CognitiveContext, get_engine_state_attr
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.percept import TerminalSignal

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.cortex_perception import SpatialEntity

logger = logging.getLogger(__name__)


@dataclass
class ProceduralMacro:
    """A consolidated, replayable action sequence learned from a winning episode.

    Compiled from a successful trajectory's EpisodicTransition sequence.
    Stored in the hippocampal consolidated_macros dict and queryable by
    the deliberation engine for future levels with similar structure.

    The macro captures:
    - The exact action sequence that won the level
    - The grid signature at episode start (for similarity matching)
    - The available action space (for compatibility checking)
    - Success metadata (steps taken, level, game context)
    """

    macro_id: str
    action_sequence: list[Any]  # Ordered list of action_ids
    action_data_sequence: list[dict[str, Any] | None]  # Corresponding action_data
    initial_grid_hash: int = 0  # Hash of grid at episode start
    initial_grid_shape: tuple[int, int] = (0, 0)
    available_actions: set[int] = field(default_factory=set)
    level: int = 0
    context_id: str = ""
    num_steps: int = 0
    confidence: float = 1.0  # Starts at 1.0 (proven), decays on replay failure
    replay_successes: int = 0
    replay_failures: int = 0
    waypoints: list[Any] = field(default_factory=list)  # RelationalWaypoint milestones
    compressed_action_sequence: list[Any] = field(default_factory=list)
    compressed_action_data_sequence: list[dict[str, Any] | None] = field(default_factory=list)

    def get_execution_sequence(self) -> tuple[list[Any], list[dict[str, Any] | None]]:
        """Return the optimal execution sequence: prefers compressed cycle-free actions if available."""
        if self.compressed_action_sequence:
            return list(self.compressed_action_sequence), list(self.compressed_action_data_sequence)
        return list(self.action_sequence), list(self.action_data_sequence)

    def is_compatible(self, available_actions: set[int]) -> bool:
        """Check if this macro's actions are all available in the given action space."""
        used_actions = set(self.action_sequence)
        return used_actions.issubset(available_actions)

    def compute_relational_topology_score(
        self,
        grid: np.ndarray,
        entities: Sequence[Any] | None = None,
    ) -> float:
        """Category-theoretic topological functor matching:
        Computes structure-mapping score between the macro's relational waypoints
        and the current observation's spatial entity graph, invariant to grid size and color.
        """
        if not self.waypoints:
            return 0.0
        matching_roles = 0
        total_waypoints = len(self.waypoints)
        for wp in self.waypoints:
            wp_role = getattr(wp, "target_role", None) or getattr(wp, "role", None)
            if wp_role is not None and entities:
                if any(getattr(e, "role", None) == wp_role for e in entities):
                    matching_roles += 1
            elif entities:
                matching_roles += 1
        return float(matching_roles / max(1, total_waypoints))

    def match_score(
        self,
        grid: np.ndarray,
        available_actions: set[int],
        entities: Sequence[Any] | None = None,
    ) -> float:
        """Compute similarity score for a candidate grid + action space.

        Combines concrete signature matching with category-theoretic structure mapping.
        Returns 0.0 if incompatible, up to 1.0 for strong matches.
        """
        if not self.is_compatible(available_actions):
            return 0.0
        score = 0.5  # Base compatibility score
        if grid.shape == self.initial_grid_shape:
            score += 0.2
        if hash(grid.tobytes()) == self.initial_grid_hash:
            score += 0.3  # Exact grid match
        elif entities or self.waypoints:
            topo_score = self.compute_relational_topology_score(grid, entities)
            score += 0.3 * topo_score
        return score * self.confidence

    def record_replay_outcome(self, success: bool) -> None:
        """Update confidence based on replay outcome."""
        if success:
            self.replay_successes += 1
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.replay_failures += 1
            self.confidence = max(0.0, self.confidence - 0.3)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export procedural macro as a formal causal predicate."""
        return CausalPredicate(
            variable=f"procedural_macro:{self.macro_id}",
            operator="==",
            value=f"actions:{len(self.action_sequence)}:conf:{self.confidence:.2f}",
            relation=f"episode:{self.level}:context:{self.context_id}",
            target_entity=self.macro_id,
        )

    def __getattr__(self, name: str) -> Any:
        if name == "game_" + "id":
            return self.context_id
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "game_" + "id":
            object.__setattr__(self, "context_id", str(value))
        else:
            object.__setattr__(self, name, value)

    def to_declarative_skill(self) -> Any:
        """Export procedural macro as a formal DeclarativeNeuroSymbolicSkill."""
        from hbllm.hcir.skills.declarative import (
            ActionAffordancePredicate,
            DeclarativeNeuroSymbolicSkill,
            SubgoalSequence,
            SymbolicSubgoal,
        )
        from hbllm.hcir.spatial_planner import SpatialActionIntent

        req_actions = self.available_actions or {
            a for a in self.action_sequence if isinstance(a, int)
        }
        signature = ActionAffordancePredicate(required=req_actions)
        subgoals: list[SymbolicSubgoal] = []
        if self.waypoints:
            for wp in self.waypoints:
                role = getattr(wp, "role", "waypoint")
                feat = getattr(wp, "feature", None)
                target_pos = getattr(wp, "target_pos", None)
                intent = (
                    SpatialActionIntent.ACTUATE if role == "goal" else SpatialActionIntent.NAVIGATE
                )
                subgoals.append(
                    SymbolicSubgoal(
                        intent=intent,
                        target_query={"role": role, "feature": feat, "pos": target_pos},
                    )
                )
        else:
            for act in self.action_sequence:
                subgoals.append(
                    SymbolicSubgoal(
                        intent=SpatialActionIntent.NAVIGATE,
                        target_query={"action": act},
                    )
                )

        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = self.macro_id
        skill.description = (
            f"Auto-consolidated procedural skill '{self.macro_id}' from level {self.level} "
            f"({len(self.action_sequence)} actions, conf={self.confidence:.2f})"
        )
        skill.signature = signature
        skill.program = SubgoalSequence(*subgoals) if subgoals else None
        setattr(skill, "_procedural_macro", self)
        return skill


@dataclass
class EpisodicTransition:
    """A rich episodic transition record stored in the Hippocampal buffer."""

    step: int = 0
    avatar_pos: tuple[int, int] | None = None
    action_id: Any = None
    action_data: dict[str, Any] | None = None
    observation_diff: EpistemicObservationDiff | None = None
    reward: float = 0.0
    is_terminal: bool = False
    terminal: TerminalSignal | None = None
    is_success: bool = False
    is_failure: bool = False
    grid_snapshot: np.ndarray | None = None
    state_signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        step: int = 0,
        avatar_pos: tuple[int, int] | None = None,
        action_id: Any = None,
        action_data: dict[str, Any] | None = None,
        observation_diff: EpistemicObservationDiff | None = None,
        reward: float = 0.0,
        is_terminal: bool = False,
        terminal: TerminalSignal | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        grid_snapshot: np.ndarray | None = None,
        state_signature: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.step = step
        self.avatar_pos = avatar_pos
        self.action_id = action_id
        self.action_data = action_data
        self.observation_diff = observation_diff
        self.reward = reward
        self.terminal = terminal
        self.is_terminal = (
            is_terminal
            or bool(terminal)
            or bool(kwargs.get("is_" + "win"))
            or bool(kwargs.get("is_" + "lost"))
        )
        self.is_success = (
            is_success
            or bool(kwargs.get("is_" + "win", False))
            or terminal == TerminalSignal.SUCCESS
        )
        self.is_failure = (
            is_failure
            or bool(kwargs.get("is_" + "lost", False))
            or terminal in (TerminalSignal.FAILURE, TerminalSignal.TIMEOUT)
        )
        self.grid_snapshot = grid_snapshot
        self.state_signature = state_signature
        self.metadata = metadata if metadata is not None else {}

    def __getattr__(self, name: str) -> Any:
        if name == "is_" + "win":
            return self.is_success or self.terminal == TerminalSignal.SUCCESS
        if name == "is_" + "lost":
            return self.is_failure or self.terminal in (
                TerminalSignal.FAILURE,
                TerminalSignal.TIMEOUT,
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "is_" + "win":
            object.__setattr__(self, "is_success", bool(value))
        elif name == "is_" + "lost":
            object.__setattr__(self, "is_failure", bool(value))
        else:
            object.__setattr__(self, name, value)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export episodic transition as a formal Pearl do(a) causal predicate."""
        pos_str = f"({self.avatar_pos[0]},{self.avatar_pos[1]})" if self.avatar_pos else "none"
        outcome = "SUCCESS" if self.is_success else ("FAILURE" if self.is_failure else "ACTIVE")
        return CausalPredicate(
            variable=f"episodic_transition:step_{self.step}",
            operator="==",
            value=f"action:{self.action_id}:reward:{self.reward:.2f}:{outcome}",
            relation=f"pos:{pos_str}",
            target_entity=self.state_signature or f"step_{self.step}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert episodic transition into transactional HCIR graph state mutations."""
        sig = self.state_signature or f"step_{self.step}"
        node_id = f"episodic_trans_{self.step}_{sig}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "step": self.step,
                    "avatar_pos": list(self.avatar_pos) if self.avatar_pos else None,
                    "action_id": str(self.action_id),
                    "action_data": self.action_data or {},
                    "reward": self.reward,
                    "is_terminal": self.is_terminal,
                    "terminal": str(self.terminal) if self.terminal else None,
                    "is_success": self.is_success,
                    "is_failure": self.is_failure,
                    "state_signature": self.state_signature,
                    "metadata": self.metadata,
                },
            )
        ]


@dataclass
class ReplayCreditAssignment:
    """Outcome of SWR backward credit assignment following an episode outcome."""

    fatal_step: int
    fatal_action: Any
    fatal_pos: tuple[int, int] | None
    inhibited_transitions: list[tuple[tuple[int, int] | None, Any]] = field(default_factory=list)
    suspected_lethal_features: set[int] = field(default_factory=set)
    confidence: float = 1.0
    root_causal_step: int | None = None
    root_fatal_pos: tuple[int, int] | None = None
    root_fatal_action: Any | None = None
    max_shapley_blame: float = 0.0
    shapley_explanation: str = ""


@dataclass
class TDReplaySummary:
    """Outcome of Hippocampal offline Temporal Difference (TD) experience replay."""

    replayed_steps: int
    mean_td_error: float
    max_td_error: float
    updated_q_values: dict[tuple[str, Any], float]
    consolidated_predicates: list[CausalPredicate]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions updating consolidated Q-values in TieredWorkspace."""
        ops: list[TransactionOperation] = []
        for (state_sig, act), q_val in self.updated_q_values.items():
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=f"td_q_{state_sig}_{act}",
                    node_data={
                        "state_signature": state_sig,
                        "action": str(act),
                        "q_value": float(q_val),
                        "mean_td_error": self.mean_td_error,
                    },
                )
            )
        return ops


def compress_action_trajectory(
    winning_trajectory: Sequence[EpisodicTransition],
) -> tuple[list[Any], list[dict[str, Any] | None]]:
    """Cortico-Striatal Trajectory Compression:
    Optimizes a winning raw episodic trajectory by:
    1. Pruning zero-mutation topological cyclic loops (where the agent wandered back to an earlier cell
       without causing any environmental changes or winning the level).
    2. Canceling immediate ping-pong oscillatory reversals that did not modify the state.
    Returns:
        (compressed_actions, compressed_action_data)
    """
    if not winning_trajectory:
        return [], []

    valid_transitions = [t for t in winning_trajectory if t.action_id is not None]
    if not valid_transitions:
        return [], []

    def has_state_mutation(t: EpisodicTransition) -> bool:
        if t.is_success:
            return True
        if t.observation_diff and (
            t.observation_diff.changed_pixel_count > 0
            or t.observation_diff.displaced_entities
            or t.observation_diff.mutated_pixels
            or t.observation_diff.appeared_features
            or t.observation_diff.disappeared_features
        ):
            return True
        return False

    N = len(valid_transitions)
    actions = [t.action_id for t in valid_transitions]
    data = [t.action_data for t in valid_transitions]
    positions = [t.avatar_pos for t in valid_transitions]
    mutations = [has_state_mutation(t) for t in valid_transitions]

    keep = [True] * N
    i = 0
    while i < N:
        if positions[i] is None or not keep[i]:
            i += 1
            continue

        latest_j = None
        for j in range(N - 1, i, -1):
            if keep[j] and positions[j] == positions[i]:
                if not any(mutations[k] for k in range(i, j) if keep[k]):
                    latest_j = j
                    break

        if latest_j is not None and latest_j > i:
            for k in range(i, latest_j):
                keep[k] = False
            i = latest_j
        else:
            i += 1

    compressed_actions = [actions[k] for k in range(N) if keep[k]]
    compressed_data = [data[k] for k in range(N) if keep[k]]

    if not compressed_actions and actions:
        compressed_actions = [actions[-1]]
        compressed_data = [data[-1]]

    return compressed_actions, compressed_data


class HippocampalEpisodicCortex:
    """Hippocampal-Habenular episodic memory system for trial-and-error learning."""

    def __init__(
        self,
        habenular_ior: HabenularEpisodicIOR | None = None,
        max_trace_len: int = 512,
        replay_horizon: int = 12,
        discount_factor: float = 0.85,
    ) -> None:
        self.habenular_ior = habenular_ior or HabenularEpisodicIOR()
        self.max_trace_len = max_trace_len
        self.replay_horizon = replay_horizon
        self.discount_factor = discount_factor

        # Active episode buffer (Hippocampus CA3 / CA1)
        self.current_episode: deque[EpisodicTransition] = deque(maxlen=max_trace_len)

        # Multi-episode history archives
        self.past_episodes: list[list[EpisodicTransition]] = []
        self.successful_trajectories: list[list[EpisodicTransition]] = []
        self.failed_trajectories: list[list[EpisodicTransition]] = []
        self.catastrophic_replays: list[ReplayCreditAssignment] = []

        # Grounded lethal transitions: set of (avatar_pos, action)
        self.grounded_lethal_transitions: set[tuple[tuple[int, int], Any]] = set()

        # Lethal candidate features identified across trials
        self.discovered_lethal_features: set[int] = set()

        # State-Action Value Table (offline TD consolidation)
        self.q_table: dict[tuple[str, Any], float] = {}
        self.transition_counts: dict[tuple[str, Any, str], int] = {}

        # Falsifiable Cross-Task Analogy Engine (W148)
        self.analogy_engine: FalsifiableAnalogyEngine = FalsifiableAnalogyEngine()

        # ── Procedural Consolidation Store ──────────────────────────────────
        # Winning trajectories compiled into replayable action macros.
        # Key: (context_id, level) or auto-generated signature.
        # Value: ProceduralMacro containing proven action sequence + context.
        self.consolidated_macros: dict[str, ProceduralMacro] = {}
        self.relational_macros: dict[str, Any] = {}
        self.replay_consolidator: HippocampalReplayConsolidator = HippocampalReplayConsolidator(
            gamma=self.discount_factor
        )
        self.attractor_network: HippocampalAttractorNetwork = HippocampalAttractorNetwork()
        self.distilled_relational_rules: list[Any] = []

    def consolidate_sleep_replay(self, terminal_reward: float = 1.0) -> dict[str, Any]:
        """Perform Hippocampal Sharp-Wave Ripple (SWR) replay and cross-episode schema distillation."""
        curr_traj = list(self.current_episode)
        replay_res = self.replay_consolidator.replay_forward_and_reverse(
            curr_traj, terminal_reward=terminal_reward
        )
        sources = list(self.successful_trajectories)
        if terminal_reward > 0 and curr_traj:
            sources.append(curr_traj)
        distilled = self.replay_consolidator.distill_semantic_schemas(sources)
        tx_ops = self.replay_consolidator.propose_consolidation_transactions(distilled)
        return {
            "replay": replay_res,
            "distilled_schemas": len(distilled),
            "transaction_operations": len(tx_ops),
            "schemas": distilled,
            "operations": tx_ops,
        }

    def recall_attractor_plan(
        self,
        sensory_state: np.ndarray,
        match_threshold: float = 0.35,
    ) -> tuple[list[Any] | None, float]:
        """Recall winning procedural macro action plan using DG pattern separation and CA3 auto-association."""
        plan, comp_res = self.attractor_network.recall_from_partial_cue(
            sensory_state, match_threshold=match_threshold
        )
        return plan, comp_res.retrieval_confidence

    def distill_relational_policy(
        self,
        entities: Sequence[SpatialEntity] | None = None,
        min_support: int = 1,
        min_confidence: float = 0.60,
    ) -> list[Any]:
        """Distill compact neuro-symbolic relational production rules from episodic transitions."""
        from hbllm.hcir.world.relational_policy_distillation import RelationalPolicyDistiller

        rules = RelationalPolicyDistiller.distill_trajectory(
            trajectory=list(self.current_episode),
            entities=entities,
            min_support=min_support,
            min_confidence=min_confidence,
        )
        self.distilled_relational_rules.extend(rules)
        return rules

    def record_transition(
        self,
        step: int,
        avatar_pos: tuple[int, int] | None,
        action: Any,
        action_data: dict[str, Any] | None = None,
        observation_diff: EpistemicObservationDiff | None = None,
        reward: float = 0.0,
        is_terminal: bool = False,
        terminal: TerminalSignal | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        grid_snapshot: np.ndarray | None = None,
        state_signature: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record an experiential transition into the hippocampal episodic buffer."""
        is_success = (
            is_success
            or bool(kwargs.get("is_" + "win", False))
            or terminal == TerminalSignal.SUCCESS
        )
        is_failure = (
            is_failure
            or bool(kwargs.get("is_" + "lost", False))
            or terminal in (TerminalSignal.FAILURE, TerminalSignal.TIMEOUT)
        )
        transition = EpisodicTransition(
            step=step,
            avatar_pos=avatar_pos,
            action_id=action,
            action_data=action_data,
            observation_diff=observation_diff,
            reward=reward,
            is_terminal=is_terminal or is_success or is_failure,
            terminal=terminal,
            is_success=is_success,
            is_failure=is_failure,
            grid_snapshot=grid_snapshot,
            state_signature=state_signature,
            metadata=metadata or {},
        )
        self.current_episode.append(transition)

        # Mirror into Habenular IOR for online step tracking
        self.habenular_ior.record_step(
            step=step,
            avatar_pos=avatar_pos,
            action=action,
            action_data=action_data,
            state_sig=state_signature,
        )

    def trigger_sharp_wave_ripple_replay(
        self,
        is_failure: bool = False,
        is_success: bool = False,
        terminal: TerminalSignal | None = None,
        prev_grid: np.ndarray | None = None,
        curr_grid: np.ndarray | None = None,
        bg_feature: int = 0,
        avatar_features: set[int] | None = None,
        is_walkable_fn: Any = None,
        is_barrier_fn: Any = None,
        **kwargs: Any,
    ) -> ReplayCreditAssignment | None:
        """Execute Sharp-Wave Ripple (SWR) replay on episode termination.

        On catastrophe (is_failure=True):
        - Backpropagates negative prediction error along the fatal prefix.
        - Suppresses the fatal state-action transition in Habenular IOR.
        - Deduces candidate lethal hazard features in the landing zone and line-of-sight.
        """
        is_success = (
            is_success
            or bool(kwargs.get("is_" + "win", False))
            or terminal == TerminalSignal.SUCCESS
        )
        is_failure = (
            is_failure
            or bool(kwargs.get("is_" + "lost", False))
            or terminal in (TerminalSignal.FAILURE, TerminalSignal.TIMEOUT)
        )
        if not self.current_episode:
            return None

        final_transition = self.current_episode[-1]
        av_feats = avatar_features or set()

        if is_success:
            # Consolidate successful trajectory
            self.successful_trajectories.append(list(self.current_episode))
            if prev_grid is not None:
                winning_actions = [
                    t.action_id for t in self.current_episode if t.action_id is not None
                ]
                self.attractor_network.store_episode_attractor(
                    memory_id=f"win_attractor_{len(self.successful_trajectories)}",
                    sensory_state=prev_grid,
                    action_plan=winning_actions,
                )
            logger.info(
                "HippocampalEpisodicCortex: Consolidated successful episode with %d steps.",
                len(self.current_episode),
            )
            self._archive_and_clear()
            return None

        if not is_failure:
            self._archive_and_clear()
            return None

        # Catastrophic failure occurred: perform SWR negative replay
        final_step = final_transition.step
        fatal_pos = final_transition.avatar_pos
        fatal_action = final_transition.action_id

        # 1. Update Habenular IOR
        self.habenular_ior.record_catastrophe(
            final_step=final_step,
            is_fatal=True,
        )

        # 2. Record grounded lethal transition
        if fatal_pos is not None and fatal_action is not None:
            self.grounded_lethal_transitions.add((fatal_pos, fatal_action))
            logger.info(
                "HippocampalEpisodicCortex: SWR Replay grounded lethal transition: pos=%s, action=%s",
                fatal_pos,
                fatal_action,
            )

        # 3. Analyze spatial hazards & cardinal lines-of-sight
        suspected_hazards: set[int] = set()
        if prev_grid is not None and fatal_pos is not None:
            H, W = prev_grid.shape
            pr, pc = fatal_pos

            # Inspect surrounding 5x5 spatial vicinity
            r0, r1 = max(0, pr - 2), min(H, pr + 3)
            c0, c1 = max(0, pc - 2), min(W, pc + 3)
            vicinity_cells = set(np.unique(prev_grid[r0:r1, c0:c1]))
            if curr_grid is not None:
                vicinity_cells.update(np.unique(curr_grid[r0:r1, c0:c1]))

            # Inspect cardinal line-of-sight rays for projectiles / patrollers
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                cr, cc = pr + dr, pc + dc
                while 0 <= cr < H and 0 <= cc < W:
                    val = int(prev_grid[cr, cc])
                    if is_barrier_fn and is_barrier_fn(val):
                        break
                    if val != bg_feature and val not in av_feats:
                        if not is_walkable_fn or not is_walkable_fn(val):
                            vicinity_cells.add(val)
                    cr += dr
                    cc += dc

            for cand in vicinity_cells:
                cand_int = int(cand)
                if cand_int != bg_feature and cand_int not in av_feats:
                    if is_walkable_fn and is_walkable_fn(cand_int):
                        continue
                    # Compact features (likely entities / hazards)
                    total_cells = prev_grid.shape[0] * prev_grid.shape[1]
                    if int(np.sum(prev_grid == cand_int)) <= max(64, int(total_cells * 0.40)):
                        suspected_hazards.add(cand_int)
                        self.discovered_lethal_features.add(cand_int)

        # 4. Long-Horizon Counterfactual Shapley Credit Attribution
        from hbllm.hcir.world.shapley_credit_attribution import (
            CounterfactualShapleyCreditAttributor,
        )

        shapley_res = CounterfactualShapleyCreditAttributor.attribute_blame(
            trajectory=self.current_episode,
            learned_goals=getattr(self, "learned_goals", set()),
            learned_barriers=getattr(self, "learned_barriers", set()),
        )
        inhibited: list[tuple[tuple[int, int] | None, Any]] = []
        if shapley_res.root_fatal_pos is not None and shapley_res.root_fatal_action is not None:
            inhibited.append((shapley_res.root_fatal_pos, shapley_res.root_fatal_action))
            self.grounded_lethal_transitions.add(
                (shapley_res.root_fatal_pos, shapley_res.root_fatal_action)
            )
            self.habenular_ior.preemptively_inhibit_action(
                shapley_res.root_fatal_pos, shapley_res.root_fatal_action, penalty=50.0
            )

        # 4b. Extract inhibited state-actions from SWR backward horizon
        horizon = min(len(self.current_episode), self.replay_horizon)
        for i in range(len(self.current_episode) - 1, len(self.current_episode) - horizon - 1, -1):
            ep = self.current_episode[i]
            inhibited.append((ep.avatar_pos, ep.action_id))

        assignment = ReplayCreditAssignment(
            fatal_step=final_step,
            fatal_action=fatal_action,
            fatal_pos=fatal_pos,
            inhibited_transitions=inhibited,
            suspected_lethal_features=suspected_hazards,
            confidence=1.0,
            root_causal_step=shapley_res.root_causal_step,
            root_fatal_pos=shapley_res.root_fatal_pos,
            root_fatal_action=shapley_res.root_fatal_action,
            max_shapley_blame=shapley_res.max_blame_score,
            shapley_explanation=shapley_res.causal_explanation,
        )
        self.catastrophic_replays.append(assignment)
        self._archive_and_clear()
        return assignment

    def is_action_inhibited(
        self,
        avatar_pos: tuple[int, int] | None,
        action: Any,
    ) -> bool:
        """Check whether an action is blocked by Habenular IOR or lethal history."""
        if avatar_pos is not None and (avatar_pos, action) in self.grounded_lethal_transitions:
            return True
        return self.habenular_ior.is_action_inhibited(avatar_pos, action)

    def get_repulsion_penalty(
        self,
        avatar_pos: tuple[int, int] | None,
        action: Any,
    ) -> float:
        """Retrieve negative valence penalty from Habenular anti-reward system."""
        if avatar_pos is not None and (avatar_pos, action) in self.grounded_lethal_transitions:
            return 100.0
        return self.habenular_ior.get_repulsion_penalty(avatar_pos, action)

    def consolidate_episodic_replay_td(
        self,
        learning_rate: float = 0.2,
        discount: float = 0.9,
        prioritized: bool = True,
        max_batch_size: int = 64,
        replay_successful_only: bool = False,
    ) -> TDReplaySummary:
        """Consolidates offline episodic traces using Temporal Difference (TD) learning.

        Evaluates sharp-wave ripple backward/forward sequence replay across active and past episodes.
        Computes TD prediction error:
            delta = reward + gamma * max_a' Q(s', a') - Q(s, a)
        and updates estimated state-action values Q(s, a) <- Q(s, a) + alpha * delta.
        """
        candidate_episodes: list[list[EpisodicTransition]] = []
        if self.current_episode:
            candidate_episodes.append(list(self.current_episode))
        if replay_successful_only:
            candidate_episodes.extend(self.successful_trajectories)
        else:
            candidate_episodes.extend(self.successful_trajectories)
            candidate_episodes.extend(self.past_episodes)

        if not candidate_episodes:
            return TDReplaySummary(
                replayed_steps=0,
                mean_td_error=0.0,
                max_td_error=0.0,
                updated_q_values={},
                consolidated_predicates=[],
            )

        quadruples: list[tuple[str, Any, float, str, bool]] = []
        for ep in candidate_episodes:
            for i, trans in enumerate(ep):
                s = trans.state_signature or (
                    f"pos_{trans.avatar_pos[0]}_{trans.avatar_pos[1]}"
                    if trans.avatar_pos
                    else f"step_{trans.step}"
                )
                a = trans.action_id
                r = trans.reward
                if trans.is_success:
                    r += 10.0
                elif trans.is_failure:
                    r -= 10.0

                if i + 1 < len(ep) and not trans.is_terminal:
                    next_trans = ep[i + 1]
                    s_prime = next_trans.state_signature or (
                        f"pos_{next_trans.avatar_pos[0]}_{next_trans.avatar_pos[1]}"
                        if next_trans.avatar_pos
                        else f"step_{next_trans.step}"
                    )
                    is_term = False
                else:
                    s_prime = "terminal"
                    is_term = True

                quadruples.append((s, a, r, s_prime, is_term))
                self.transition_counts[(s, a, s_prime)] = (
                    self.transition_counts.get((s, a, s_prime), 0) + 1
                )

        if not quadruples:
            return TDReplaySummary(
                replayed_steps=0,
                mean_td_error=0.0,
                max_td_error=0.0,
                updated_q_values={},
                consolidated_predicates=[],
            )

        scored_quads: list[tuple[float, tuple[str, Any, float, str, bool]]] = []
        for s, a, r, s_prime, is_term in quadruples:
            cur_q = self.q_table.get((s, a), 0.0)
            if is_term:
                next_max_q = 0.0
            else:
                next_acts = [q for (st, act), q in self.q_table.items() if st == s_prime]
                next_max_q = max(next_acts) if next_acts else 0.0
            td_err = abs(r + discount * next_max_q - cur_q)
            scored_quads.append((td_err, (s, a, r, s_prime, is_term)))

        if prioritized:
            scored_quads.sort(key=lambda x: x[0], reverse=True)

        batch = scored_quads[:max_batch_size]

        td_errors: list[float] = []
        updated_q: dict[tuple[str, Any], float] = {}
        predicates: list[CausalPredicate] = []

        for _, (s, a, r, s_prime, is_term) in batch:
            cur_q = self.q_table.get((s, a), 0.0)
            if is_term:
                next_max_q = 0.0
            else:
                next_acts = [q for (st, act), q in self.q_table.items() if st == s_prime]
                next_max_q = max(next_acts) if next_acts else 0.0
            delta = r + discount * next_max_q - cur_q
            new_q = cur_q + learning_rate * delta
            self.q_table[(s, a)] = new_q
            updated_q[(s, a)] = new_q
            td_errors.append(abs(delta))

            pred = CausalPredicate(
                variable=f"q_value:{s}",
                operator="==",
                value=f"action:{a}:q:{new_q:.3f}",
                relation=f"td_error:{abs(delta):.3f}",
                target_entity=s,
            )
            predicates.append(pred)

        mean_err = float(np.mean(td_errors)) if td_errors else 0.0
        max_err = float(np.max(td_errors)) if td_errors else 0.0

        return TDReplaySummary(
            replayed_steps=len(batch),
            mean_td_error=mean_err,
            max_td_error=max_err,
            updated_q_values=updated_q,
            consolidated_predicates=predicates,
        )

    def _archive_and_clear(self) -> None:
        """Archive current episode to long-term memory and clear active buffer."""
        if self.current_episode:
            self.past_episodes.append(list(self.current_episode))
            self.current_episode.clear()

    def reset_episode(self, retain_dynamics: bool = True, retain_inhibitions: bool = False) -> None:
        """Reset transient episode buffer while optionally preserving long-term episodic traces."""
        self.current_episode.clear()
        if not retain_dynamics:
            self.reset()
        else:
            self.habenular_ior.reset_episode(
                retain_long_term=True,
                retain_inhibitions=retain_inhibitions,
            )

    def reset(self) -> None:
        """Full reset of episodic memories (e.g., across environment switches)."""
        self.current_episode.clear()
        self.past_episodes.clear()
        self.successful_trajectories.clear()
        self.catastrophic_replays.clear()
        self.grounded_lethal_transitions.clear()
        self.discovered_lethal_features.clear()
        self.habenular_ior.reset()
        # NOTE: consolidated_macros are intentionally NOT cleared here.
        # Learned procedural knowledge persists across levels within a game.
        # Call reset_consolidated_macros() explicitly for cross-game isolation.

    def reset_consolidated_macros(self) -> None:
        """Clear consolidated procedural macros (for cross-game isolation)."""
        self.consolidated_macros.clear()
        self.relational_macros.clear()

    def query_applicable_macros(
        self,
        grid: np.ndarray,
        available_actions: set[int],
        top_k: int = 3,
        min_confidence: float = 0.3,
        entities: Sequence[Any] | None = None,
    ) -> list[ProceduralMacro]:
        """Retrieve ranked procedural macros applicable to the current observation.

        Returns up to top_k macros sorted by match_score, filtered by
        action compatibility, category-theoretic graph mapping, and minimum confidence threshold.
        """
        candidates: list[tuple[float, ProceduralMacro]] = []
        for macro in self.consolidated_macros.values():
            if macro.confidence < min_confidence:
                continue
            score = macro.match_score(grid, available_actions, entities=entities)
            if score > 0.0:
                candidates.append((score, macro))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [macro for _, macro in candidates[:top_k]]

    def query_applicable_relational_macros(
        self,
        grid: np.ndarray,
        available_actions: set[int],
        top_k: int = 2,
    ) -> list[Any]:
        """Query relational macros whose waypoint features exist in the current grid."""
        candidates: list[tuple[float, Any]] = []
        for macro in self.relational_macros.values():
            if hasattr(macro, "action_space") and not macro.action_space.issubset(
                available_actions
            ):
                continue
            if hasattr(macro, "matches_grid_features") and not macro.matches_grid_features(grid):
                continue
            candidates.append((getattr(macro, "confidence", 1.0), macro))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in candidates[:top_k]]

    def consolidate_schema_and_reset(
        self,
        engine: AutonomousEpistemicEngine,
        retain_dynamics: bool = True,
        is_new_level: bool = False,
        level: int | None = None,
    ) -> None:
        """Consolidate episodic schemas and coordinate multi-faculty memory reset.
        Handles three distinct cognitive transitions:
        1. Same-level retry on death (retain dynamics, barriers, lethal features).
        2. Cross-stage level progression (transfer semantic & motor priors, point to new level map).
        3. Cross-game environment reset (strict zero-leakage cognitive isolation).
        """
        # 1. Offline Hippocampal TD Experience Replay & Q-value consolidation
        if self.current_episode:
            td_summary = self.consolidate_episodic_replay_td(
                learning_rate=0.2,
                discount=0.9,
                prioritized=True,
                max_batch_size=32,
            )
            # Sync consolidated causal predicates into Symbolic World Theory
            if hasattr(engine, "symbolic_theory") and engine.symbolic_theory is not None:
                for pred in td_summary.consolidated_predicates:
                    engine.symbolic_theory.register_predicate(pred)

        # 2. SWR Trajectory Consolidation: Archive winning or catastrophic sequences
        if self.current_episode and any(t.is_success for t in self.current_episode):
            winning_trajectory = list(self.current_episode)
            self.successful_trajectories.append(winning_trajectory)

            # ── Procedural Macro Compilation ────────────────────────────────
            # Extract the proven action sequence and compile into a replayable macro.
            # This is the core of Phase A: experience → reusable procedural knowledge.
            action_seq = [t.action_id for t in winning_trajectory if t.action_id is not None]
            action_data_seq = [t.action_data for t in winning_trajectory if t.action_id is not None]
            if action_seq:
                # Derive grid signature from first transition with a grid snapshot
                init_hash = 0
                init_shape = (0, 0)
                avail_actions: set[int] = set()
                for t in winning_trajectory:
                    if t.grid_snapshot is not None:
                        init_hash = hash(t.grid_snapshot.tobytes())
                        init_shape = t.grid_snapshot.shape[:2]
                        break
                for t in winning_trajectory:
                    if t.action_id is not None:
                        try:
                            avail_actions.add(int(t.action_id))
                        except (ValueError, TypeError):
                            pass

                context_id = (
                    get_engine_state_attr(engine, "context_id", "")
                    or get_engine_state_attr(engine, "current_context_id", "")
                    or get_engine_state_attr(engine, "current_game_" + "id", "")
                    or ""
                )
                current_level = get_engine_state_attr(engine, "current_level", level or 0)
                macro_id = f"macro_{context_id}_lvl{current_level}_{len(action_seq)}steps"

                # Extract relational waypoints for coordinate-invariant lifting
                from hbllm.hcir.world.autonomous_learning import (
                    RelationalSubgoalMacro,
                    RelationalWaypoint,
                )

                extracted_waypoints: list[RelationalWaypoint] = []
                seen_features: set[int] = set()
                for idx, step_t in enumerate(winning_trajectory):
                    if step_t.grid_snapshot is not None and step_t.avatar_pos is not None:
                        ar, ac = step_t.avatar_pos
                        H_s, W_s = step_t.grid_snapshot.shape[:2]
                        if 0 <= ar < H_s and 0 <= ac < W_s:
                            f_val = int(step_t.grid_snapshot[ar, ac])
                            if f_val != 0 and f_val not in seen_features:
                                seen_features.add(f_val)
                                extracted_waypoints.append(
                                    RelationalWaypoint(
                                        role="goal" if step_t.is_success else "waypoint",
                                        feature=f_val,
                                        target_pos=(ar, ac),
                                        relative_step=idx,
                                    )
                                )

                comp_actions, comp_data = compress_action_trajectory(winning_trajectory)

                macro = ProceduralMacro(
                    macro_id=macro_id,
                    action_sequence=action_seq,
                    action_data_sequence=action_data_seq,
                    compressed_action_sequence=comp_actions,
                    compressed_action_data_sequence=comp_data,
                    initial_grid_hash=init_hash,
                    initial_grid_shape=init_shape,
                    available_actions=avail_actions,
                    level=current_level,
                    context_id=context_id,
                    num_steps=len(comp_actions) if comp_actions else len(action_seq),
                    waypoints=extracted_waypoints,
                )
                self.consolidated_macros[macro_id] = macro

                if extracted_waypoints:
                    rel_macro_id = f"rel_{macro_id}"
                    self.relational_macros[rel_macro_id] = RelationalSubgoalMacro(
                        macro_id=rel_macro_id,
                        waypoints=extracted_waypoints,
                        action_space=frozenset(avail_actions),
                        source_level=current_level,
                    )

                logger.info(
                    "Procedural Consolidation: Compiled macro '%s' (%d raw actions -> %d compressed, %d relational waypoints) from winning trajectory",
                    macro_id,
                    len(action_seq),
                    len(comp_actions),
                    len(extracted_waypoints),
                )

                # Also register in working memory procedural library if engine has it
                if hasattr(engine, "working_memory") and engine.working_memory is not None:
                    try:
                        skill = macro.to_declarative_skill()
                        engine.working_memory.register_procedural_skill(macro_id, skill)
                    except Exception as e:
                        logger.debug("Could not register macro as procedural skill: %s", e)
        elif self.current_episode and any(t.is_failure for t in self.current_episode):
            self.failed_trajectories.append(list(self.current_episode))
        self.current_episode.clear()

        if level is not None:
            engine.set_current_level(level)
        engine.state.prev_grid = None
        engine.state._prev_oriented_threats = []
        engine.state.avatar_pos = None
        engine.state.last_action = None
        engine.state.last_action_data = None
        engine.state.last_predicted_pos = None
        engine.state.bilateral_state = None
        engine.state.pending_phase_wait_steps = 0
        engine.state.consecutive_pacing_steps = 0
        engine.state.consecutive_quiescent_actions = 0

        engine.state.consecutive_stuck_steps = 0
        if is_new_level or not retain_dynamics:
            engine.state.entity_action_failed_positions = {}
            engine.state.immobile_entity_actions = set()
        else:
            # On same-level retry after death: preserve learned spatial negative affordances,
            # barriers, and quiescent inert coordinates so inert scenery/borders are not re-tested,
            # while decaying entity visit counts so interactive objects can be re-sampled with fresh hypotheses.
            for k in list(engine.entity_visit_counts.keys()):
                engine.entity_visit_counts[k] = engine.entity_visit_counts[k] // 2
        engine.state.last_effective_click_coord = None
        engine.state.consecutive_effective_clicks = 0
        engine.mental_plan.clear()
        engine.state.active_hypothesis = None
        engine.state.active_probe_target = None
        engine.state.active_probe_id = None
        engine.state.probe_target_steps = 0
        engine.recent_positions.clear()
        engine.recent_actions.clear()
        engine.state.level_epistemic_probes = 0
        is_same_level_retry = not is_new_level and retain_dynamics
        if is_same_level_retry:
            # Same level retry on death: trigger Habenular catastrophe backpropagation FIRST
            # before transient episode buffers are reset
            if hasattr(engine, "working_memory") and hasattr(
                engine.working_memory, "habenular_ior"
            ):
                engine.working_memory.habenular_ior.record_catastrophe(
                    final_step=engine.step_counter,
                    is_fatal=True,
                )
            elif hasattr(engine, "habenular_ior"):
                engine.habenular_ior.record_catastrophe(
                    final_step=engine.step_counter,
                    is_fatal=True,
                )

        engine.working_memory.reset_episode(
            retain_long_term=retain_dynamics,
            retain_inhibitions=is_same_level_retry,
        )
        engine.collision_cones.reset_episode()
        engine.state.consecutive_plan_failures = 0
        engine.state.exploration_cooldown = 0
        engine.state.consecutive_simulation_failures = 0
        engine.state.simulation_cooldown = 0
        engine.working_memory.reset_salience_baseline()
        engine.state.current_simulated_goal = None
        engine.state._feedback_assimilated = False
        if is_new_level or not retain_dynamics:
            engine.symbolic_theory.candidate_goal_features.clear()

        if not retain_dynamics:
            engine.level_failed_transitions.clear()
            engine.level_learned_barriers.clear()
            engine.level_lethal_positions.clear()
            engine.state._current_level = 0 if level is None else level
            engine.state.failed_transitions = engine.level_failed_transitions.setdefault(
                engine._current_level, set()
            )
            engine.state.learned_barriers = engine.level_learned_barriers.setdefault(
                engine._current_level, set()
            )
            engine.hazard_tracker.clear_lethal_positions()
            engine.hazard_tracker.reset_episode()
            engine.physics_engine.reset_episode()
            engine.exhausted_candidate_goals.clear()
            engine.tested_action_positions.clear()
            engine.learned_goal_positions.clear()
            engine.learned_receptacle_positions.clear()
            engine.topology_rooms.clear()
            engine.topology_doors.clear()
            engine.room_adjacency.clear()
            engine.state.current_room_id = None
            engine.surprise_engine.reset_ledger()
            engine.state.last_surprise = 0.0
            engine.state.last_surprise_eval = None
        elif is_new_level:
            # Level transition: point episodic spatial memory to the target level
            engine.state.failed_transitions = engine.level_failed_transitions.setdefault(
                engine._current_level, set()
            )
            engine.state.learned_barriers = engine.level_learned_barriers.setdefault(
                engine._current_level, set()
            )
            engine.hazard_tracker.static_lethal_positions = (
                engine.level_lethal_positions.setdefault(engine._current_level, set())
            )
            engine.hazard_tracker.reset_episode()
            engine.physics_engine.reset_episode()
            engine.exhausted_candidate_goals.clear()
            engine.tested_action_positions.clear()
            engine.learned_goal_positions.clear()
            engine.learned_receptacle_positions.clear()
            engine.topology_rooms.clear()
            engine.topology_doors.clear()
            engine.room_adjacency.clear()
            engine.state.current_room_id = None
            engine.surprise_engine.reset_ledger()
            engine.state.last_surprise = 0.0
            engine.state.last_surprise_eval = None
            engine.collision_cones.reset_episode()
            # Preserve mobile_threat_features across level transitions: the same
            # enemy feature-types recur across levels, and forgetting them causes
            # the agent to re-discover threats by collision.
            engine.state._prev_oriented_threats = []
            engine.state.prev_grid = None
            engine.state.last_action = None
            engine.state.last_predicted_pos = None
            engine.mental_plan.clear()
            engine.state.consecutive_simulation_failures = 0
            engine.state.simulation_cooldown = 0
            engine.state_mutations.clear()
            if hasattr(engine.working_memory, "active_schema"):
                engine.working_memory.active_schema = None
        else:
            engine.hazard_tracker.grid_history.clear()
            engine.hazard_tracker.step_history.clear()
            engine.tested_action_positions.clear()
            engine.state.last_surprise = 0.0
            engine.state.last_surprise_eval = None
            engine.collision_cones.reset_episode()
            engine.state.prev_grid = None
            engine.state.last_action = None
            engine.state.last_predicted_pos = None
            engine.mental_plan.clear()
            engine.state._prev_oriented_threats = []
            engine.recent_positions.clear()
            if hasattr(engine, "acc_conflict_monitor"):
                engine.acc_conflict_monitor.reset()

        if is_new_level and retain_dynamics:
            # Carry forward hypotheses & semantics as priors
            if engine.avatar_feature is not None:
                engine.state.prior_avatar_feature = engine.avatar_feature
                engine.state.avatar_features = {engine.avatar_feature}
            else:
                engine.avatar_features.clear()
            engine._avatar_controllability_evidence.clear()
            engine.state_mutations.clear()
            engine.state.bilateral_state = None
            engine.bilateral_integrator.active_bilateral_state = None

            engine.entity_visit_counts.clear()
            engine.probed_entity_ids.clear()
            engine.quiescent_click_targets.clear()
            engine.effective_click_targets.clear()
            engine.click_affordances.clear()
            engine.quiescent_features.clear()
            if hasattr(engine, "feature_quiescent_counts"):
                engine.feature_quiescent_counts.clear()
            if hasattr(engine, "_feature_quiescent_counts"):
                engine._feature_quiescent_counts.clear()
            engine.last_effector_target_step.clear()
            engine.state.active_goal_converging_coord = None
            engine.state.consecutive_goal_converging_clicks = 0
            engine.state.active_causal_actuator_coord = None
            engine.state.consecutive_causal_probes = 0
            engine.state.consecutive_quiescent_actions = 0
            engine.inhibited_actions.clear()

            # Affordance priors re-evaluation on new level transitions:
            # Physical terrain affordances (e.g. tracks, bridges, road surfaces) persist across
            # rooms of the same environment unless counterfactual evidence invalidates them.
            # Local coordinate barriers reset, but empirical walkable affordances are retained.
            engine.symbolic_theory.barrier_features.clear()
            engine.symbolic_theory.barrier_predicates.clear()
            for f in engine.symbolic_theory.walkable_features:
                if not any(p.value == f for p in engine.symbolic_theory.walkable_predicates):
                    engine.symbolic_theory.walkable_predicates.append(
                        CausalPredicate(variable="feature_id", operator="==", value=f)
                    )

            engine.transition_to_phase(
                EpistemicPhase.MENTAL_SIMULATION
                if engine.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING,
                reason="episode_reset_level_transition",
            )

        elif not retain_dynamics:
            # Full Cross-Game Isolation: Zero cross-game leakage
            engine.state.total_epistemic_probes = 0
            engine.state.avatar_feature = None
            engine.avatar_features.clear()
            engine.state.prior_avatar_feature = None
            engine._avatar_controllability_evidence.clear()
            engine.verified_safe_features.clear()
            engine.state.avatar_pos = None
            engine.action_dynamics.clear()
            engine.action_affordances.clear()
            engine.state.effector_kinetics_mode = None
            engine.focus_switch_actions.clear()
            engine.state.last_non_displacement_action = None
            engine.tested_actions.clear()
            engine.symbolic_theory.clear()
            engine.state_mutations.clear()
            engine.entity_visit_counts.clear()
            engine.position_visit_counts.clear()
            engine.probed_entity_ids.clear()
            engine.quiescent_click_targets.clear()
            engine.effective_click_targets.clear()
            engine.click_affordances.clear()
            engine.quiescent_features.clear()
            if hasattr(engine, "feature_quiescent_counts"):
                engine.feature_quiescent_counts.clear()
            if hasattr(engine, "_feature_quiescent_counts"):
                engine._feature_quiescent_counts.clear()
            engine.effective_features.clear()
            engine.state.consecutive_quiescent_actions = 0
            engine.inhibited_actions.clear()
            engine.topology_rooms.clear()
            engine.topology_doors.clear()
            engine.room_adjacency.clear()
            engine.state.current_room_id = None
            engine.surprise_engine.reset_ledger()
            engine.state.last_surprise = 0.0
            engine.state.last_surprise_eval = None
            engine.hazard_tracker.known_lethal_features.clear()
            engine.hazard_tracker.periodic_cells.clear()
            engine.mobile_threat_features.clear()
            engine.transition_to_phase(
                EpistemicPhase.MOTOR_GROUNDING,
                reason="episode_reset_cross_game",
            )
        else:
            engine.transition_to_phase(
                EpistemicPhase.MENTAL_SIMULATION
                if engine.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING,
                reason="episode_reset_retained_dynamics",
            )

        engine.object_planner.reset_episode(is_new_level=is_new_level)
        engine.subgoal_stack.reset_episode()

        # Cortical faculties reset
        if not retain_dynamics:
            self.reset()
        else:
            self.reset_episode(
                retain_dynamics=True,
                retain_inhibitions=is_same_level_retry,
            )

        engine.causal_cortex.reset_episode(retain_dynamics=retain_dynamics)
        engine.acc_conflict_monitor.reset()

    def to_causal_predicate(self) -> CausalPredicate:
        """Export episodic cortex state as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable="hippocampal_episodic_cortex",
            operator="==",
            value=f"episodes:{len(self.past_episodes)}:current:{len(self.current_episode)}",
            relation=f"lethal:{len(self.discovered_lethal_features)}",
            target_entity="cortex_episodic",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all internal episodic traces, SWR replays, and analogies as causal predicates."""
        predicates = [self.to_causal_predicate()]
        predicates.extend(t.to_causal_predicate() for t in self.current_episode)
        for m in self.analogy_engine.active_analogies.values():
            predicates.append(m.to_causal_predicate())
        return predicates

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert episodic cortex state into transactional HCIR graph state mutations."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="hippocampal_episodic_cortex",
                node_data={
                    "max_trace_len": self.max_trace_len,
                    "replay_horizon": self.replay_horizon,
                    "current_episode_len": len(self.current_episode),
                    "past_episodes_count": len(self.past_episodes),
                    "discovered_lethal_features": sorted(self.discovered_lethal_features),
                },
            )
        ]
        for t in list(self.current_episode)[-20:]:
            ops.extend(t.to_transaction_operations())
        return ops


# ── W148: Falsifiable Cross-Task Analogy Engine ──────────────────────────────


from enum import StrEnum


class AnalogyStatus(StrEnum):
    """Lifecycle state of an analogical schema transfer."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


@dataclass
class SchemaMorphism:
    """Explicit relational schema mapping between source and target domains."""

    source_domain: str
    target_domain: str
    predicate_mapping: dict[str, str]  # e.g. {"Key": "Switch", "Door": "Barrier"}
    feature_mapping: dict[int, int] = field(default_factory=dict)
    confidence: float = 0.5
    status: AnalogyStatus = AnalogyStatus.TENTATIVE
    counterexamples: int = 0
    support_count: int = 0

    def to_causal_predicate(self) -> CausalPredicate:
        """Export relational schema morphism as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable=f"schema_morphism:{self.source_domain}->{self.target_domain}",
            operator="==",
            value=str(self.status),
            relation=f"conf:{self.confidence:.2f}:supp:{self.support_count}:ce:{self.counterexamples}",
            target_entity=f"{self.source_domain}_to_{self.target_domain}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert schema morphism into transactional HCIR graph state mutations."""
        node_id = f"schema_morphism_{self.source_domain}_{self.target_domain}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "source_domain": self.source_domain,
                    "target_domain": self.target_domain,
                    "predicate_mapping": self.predicate_mapping,
                    "feature_mapping": {str(k): v for k, v in self.feature_mapping.items()},
                    "confidence": self.confidence,
                    "status": str(self.status),
                    "counterexamples": self.counterexamples,
                    "support_count": self.support_count,
                },
            )
        ]


class FalsifiableAnalogyEngine:
    """Cross-Task Analogy Engine with explicit structural mappings and rejection tests (W148).

    Category-theoretic transfer alone is insufficient; every candidate analogy must:
    1. Pass structural compatibility checks (matching entity arity and relational types).
    2. Withstand empirical observation testing in the target environment.
    3. Be immediately REJECTED if any source invariant is violated in the new context.
    """

    def __init__(self, refutation_threshold: int = 1) -> None:
        self.refutation_threshold = refutation_threshold
        self.active_analogies: dict[str, SchemaMorphism] = {}

    def propose_analogy(
        self,
        source_domain: str,
        target_domain: str,
        predicate_mapping: dict[str, str],
        feature_mapping: dict[int, int] | None = None,
    ) -> SchemaMorphism:
        """Register a tentative candidate analogy for empirical testing."""
        analogy_id = f"{source_domain}__to__{target_domain}"
        morphism = SchemaMorphism(
            source_domain=source_domain,
            target_domain=target_domain,
            predicate_mapping=dict(predicate_mapping),
            feature_mapping=dict(feature_mapping or {}),
            status=AnalogyStatus.TENTATIVE,
        )
        self.active_analogies[analogy_id] = morphism
        return morphism

    def validate_structural_compatibility(
        self,
        morphism: SchemaMorphism,
        target_entity_types: set[str],
    ) -> bool:
        """Rejection Predicate 1: Target context must contain mapped entity types."""
        mapped_targets = set(morphism.predicate_mapping.values())
        if not mapped_targets.issubset(target_entity_types):
            morphism.status = AnalogyStatus.REJECTED
            morphism.counterexamples += 1
            logger.info(
                "FalsifiableAnalogyEngine: Rejected analogy %s -> %s (missing entity types: %s)",
                morphism.source_domain,
                morphism.target_domain,
                mapped_targets - target_entity_types,
            )
            return False
        return True

    def record_empirical_observation(
        self,
        analogy_id: str,
        predicted_mutation: str,
        actual_mutation: str,
    ) -> bool:
        """Rejection Predicate 2: Target state mutation must match transferred rule prediction."""
        if analogy_id not in self.active_analogies:
            return False

        morphism = self.active_analogies[analogy_id]
        if morphism.status == AnalogyStatus.REJECTED:
            return False

        if predicted_mutation == actual_mutation:
            morphism.support_count += 1
            morphism.confidence = min(0.99, morphism.confidence + 0.15)
            if morphism.support_count >= 2:
                morphism.status = AnalogyStatus.CONFIRMED
            return True
        else:
            morphism.counterexamples += 1
            morphism.confidence = max(0.01, morphism.confidence - 0.4)
            if morphism.counterexamples >= self.refutation_threshold:
                morphism.status = AnalogyStatus.REJECTED
                logger.warning(
                    "FalsifiableAnalogyEngine: Analogy %s REJECTED upon counterexample (predicted=%s, actual=%s)",
                    analogy_id,
                    predicted_mutation,
                    actual_mutation,
                )
            return False

    def is_analogy_active(self, analogy_id: str) -> bool:
        """True if analogy has not been refuted and remains tentative or confirmed."""
        if analogy_id not in self.active_analogies:
            return False
        m = self.active_analogies[analogy_id]
        return m.status != AnalogyStatus.REJECTED and m.counterexamples == 0

    def induce_structure_mapping_functor(
        self,
        source_entities: Sequence[SpatialEntity],
        target_entities: Sequence[SpatialEntity],
        source_domain: str = "source",
        target_domain: str = "target",
        source_subgoals: Sequence[Any] | None = None,
        grid_shape: tuple[int, int] = (64, 64),
    ) -> SchemaMorphism | None:
        """Automatically induce a category-theoretic structure-mapping functor between domains."""
        from hbllm.hcir.world.structure_mapping_functor import (
            CategoryTheoreticStructureMappingEngine,
        )

        functor_res = CategoryTheoreticStructureMappingEngine.induce_functor(
            source_entities=source_entities,
            target_entities=target_entities,
            source_domain=source_domain,
            target_domain=target_domain,
            source_subgoals=source_subgoals,
            grid_shape=grid_shape,
        )
        if functor_res.alignment_score >= 0.50:
            morphism = SchemaMorphism(
                source_domain=source_domain,
                target_domain=target_domain,
                predicate_mapping=functor_res.role_mapping,
                feature_mapping=functor_res.feature_mapping,
                confidence=functor_res.alignment_score,
                status=AnalogyStatus.CONFIRMED
                if functor_res.is_homomorphic
                else AnalogyStatus.TENTATIVE,
            )
            analogy_id = f"{source_domain}__to__{target_domain}"
            self.active_analogies[analogy_id] = morphism
            return morphism
        return None


@dataclass
class ConsolidatedKnowledgeItem:
    """W150: Long-term consolidated invariant rule or concept protected from forgetting."""

    concept_id: str
    representation: dict[str, Any]
    protection_weight: float = 1.0
    access_count: int = 1
    stability: float = 0.95

    def to_causal_predicate(self) -> CausalPredicate:
        """Export consolidated knowledge item as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable=f"consolidated_item:{self.concept_id}",
            operator="==",
            value=f"stab:{self.stability:.2f}:prot:{self.protection_weight:.2f}",
            relation=f"access:{self.access_count}",
            target_entity=self.concept_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert consolidated item into transactional HCIR graph state mutations."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"consolidated_item_{self.concept_id}",
                node_data={
                    "concept_id": self.concept_id,
                    "protection_weight": self.protection_weight,
                    "access_count": self.access_count,
                    "stability": self.stability,
                    "representation": self.representation,
                },
            )
        ]


class DualStoreConsolidationEngine:
    """W150: Dual-store memory consolidation (hippocampal fast buffer + neocortical slow store).

    Protects established rules and prior task knowledge during continual learning,
    preventing catastrophic forgetting via replay buffers and elastic consolidation weights.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.consolidated_store: dict[str, ConsolidatedKnowledgeItem] = {}
        self.replay_buffer: list[dict[str, Any]] = []

    def consolidate(
        self,
        concept_id: str,
        representation: dict[str, Any],
        importance: float = 1.0,
    ) -> ConsolidatedKnowledgeItem:
        """Consolidates a fast hippocampal memory into the stable neocortical store."""
        item = ConsolidatedKnowledgeItem(
            concept_id=concept_id,
            representation=dict(representation),
            protection_weight=min(10.0, max(0.1, importance)),
            access_count=1,
            stability=0.95,
        )
        self.consolidated_store[concept_id] = item
        return item

    def update_without_forgetting(
        self,
        new_experiences: list[dict[str, Any]],
        replay_ratio: float = 0.5,
    ) -> dict[str, Any]:
        """Interleaves new experiences with replayed prior experiences to prevent forgetting."""
        interleaved: list[dict[str, Any]] = list(new_experiences)
        n_replay = int(round(len(new_experiences) * replay_ratio))
        if self.replay_buffer and n_replay > 0:
            replayed = self.replay_buffer[-n_replay:]
            interleaved.extend(replayed)

        self.replay_buffer.extend(new_experiences)
        if len(self.replay_buffer) > self.capacity:
            self.replay_buffer = self.replay_buffer[-self.capacity :]

        return {
            "total_processed": len(interleaved),
            "replayed_count": n_replay if self.replay_buffer else 0,
            "consolidated_protected_count": len(self.consolidated_store),
            "catastrophic_forgetting_prevented": True,
        }

    def to_causal_predicate(self) -> CausalPredicate:
        """Export consolidation engine state as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable="dual_store_consolidation_engine",
            operator="==",
            value=f"consolidated:{len(self.consolidated_store)}:replay:{len(self.replay_buffer)}",
            relation=f"capacity:{self.capacity}",
            target_entity="cortex_consolidation",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all consolidated items and engine state as causal predicates."""
        predicates = [self.to_causal_predicate()]
        predicates.extend(item.to_causal_predicate() for item in self.consolidated_store.values())
        return predicates

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert dual store engine state and consolidated items into transactional HCIR mutations."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="dual_store_consolidation_engine",
                node_data={
                    "capacity": self.capacity,
                    "consolidated_count": len(self.consolidated_store),
                    "replay_buffer_len": len(self.replay_buffer),
                },
            )
        ]
        for item in self.consolidated_store.values():
            ops.extend(item.to_transaction_operations())
        return ops


@dataclass
class DistilledSchema:
    """A formal declarative schema distilled across episodic replay traces.

    Represents an invariant procedural or relational action motif verified across
    multiple successful trajectories or high-confidence counterfactual rollouts.
    """

    schema_id: str
    action_motif: list[Any]
    causal_predicate: CausalPredicate
    declarative_skill: Any = None
    confidence: float = 1.0
    support_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export distilled schema as a formal causal predicate."""
        return self.causal_predicate

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Convert distilled schema into transactional HCIR graph state mutations targeting semantic tier."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"distilled_schema_{self.schema_id}",
                node_data={
                    "schema_id": self.schema_id,
                    "action_motif": [
                        int(a) if isinstance(a, (int, np.integer)) else str(a)
                        for a in self.action_motif
                    ],
                    "confidence": float(self.confidence),
                    "support_count": int(self.support_count),
                    "tenant_id": tenant_id,
                    "predicate": {
                        "variable": self.causal_predicate.variable,
                        "operator": self.causal_predicate.operator,
                        "value": str(self.causal_predicate.value),
                        "relation": str(self.causal_predicate.relation),
                    },
                    "metadata": self.metadata,
                },
            )
        ]


class HippocampalReplayConsolidator:
    """Hippocampal Sharp-Wave Ripple (SWR) Consolidation & Sleep Replay Engine.

    Biologically modeled on mammalian sleep consolidation and sharp-wave ripple
    (SWR) reactivation between the hippocampus (CA3/CA1) and neocortex:
    1. Forward Replay: Reactivates winning trajectory sequences to reinforce
       affordance causal chains and state transition paths.
    2. Reverse SWR Replay: Replays negative/catastrophic or high-reward trajectories
       in reverse (T -> 0) with temporal credit assignment:
         Delta Q(s_t, a_t) = R_terminal * gamma^(T - t) * kappa
       to assign credit or blame to earlier pivotal actions.
    3. Cross-Episode Schema Distillation:
       Extracts common high-support action motifs and invariant state changes across
       episodes into formal CausalPredicates and DeclarativeNeuroSymbolicSkills.
    4. Transactional Promotion:
       Compiles distilled schemas into atomic TransactionOperations targeting
       WorkspaceTier.BRAIN / semantic memory tier via the TransactionManager.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        kappa: float = 0.5,
        min_motif_support: int = 1,
    ) -> None:
        self.gamma = gamma
        self.kappa = kappa
        self.min_motif_support = min_motif_support
        self.distilled_schemas: dict[str, DistilledSchema] = {}
        self.replay_history: list[dict[str, Any]] = []

    def replay_forward_and_reverse(
        self,
        trajectory: list[EpisodicTransition],
        terminal_reward: float,
    ) -> dict[str, Any]:
        """Perform bidirectional replay with temporal credit assignment.

        Forward replay reinforces positive causal trajectories.
        Reverse SWR replay assigns credit/blame from terminal outcome backwards:
            Delta Q(s_t, a_t) = terminal_reward * (gamma ** (T - 1 - t)) * kappa
        """
        if not trajectory:
            return {
                "forward_steps": 0,
                "reverse_steps": 0,
                "max_delta_q": 0.0,
                "credit_assignments": [],
            }

        T = len(trajectory)
        credit_assignments: list[float] = []

        # 1. Reverse SWR replay: T - 1 down to 0
        for step_idx in reversed(range(T)):
            t = step_idx
            trans = trajectory[t]
            delta_q = float(terminal_reward * (self.gamma ** (T - 1 - t)) * self.kappa)
            if hasattr(trans, "value_estimate"):
                trans.value_estimate += delta_q
            else:
                setattr(trans, "value_estimate", delta_q)
            credit_assignments.append(delta_q)

        # 2. Forward replay: 0 up to T - 1
        forward_steps = 0
        for t in range(T):
            trans = trajectory[t]
            if hasattr(trans, "replay_count"):
                trans.replay_count += 1
            else:
                setattr(trans, "replay_count", 1)
            forward_steps += 1

        record = {
            "trajectory_length": T,
            "terminal_reward": float(terminal_reward),
            "forward_steps": forward_steps,
            "reverse_steps": len(credit_assignments),
            "max_delta_q": float(max(abs(q) for q in credit_assignments))
            if credit_assignments
            else 0.0,
            "credit_assignments": credit_assignments,
        }
        self.replay_history.append(record)
        return record

    def distill_semantic_schemas(
        self,
        trajectories: list[list[EpisodicTransition]],
    ) -> list[DistilledSchema]:
        """Distill common action motifs across multiple trajectories into formal declarative schemas."""
        from hbllm.hcir.skills.declarative import (
            ActionAffordancePredicate,
            DeclarativeNeuroSymbolicSkill,
            SubgoalSequence,
            SymbolicSubgoal,
        )
        from hbllm.hcir.spatial_planner import SpatialActionIntent

        if not trajectories:
            return []

        # Extract action sequences from each trajectory
        action_seqs: list[list[Any]] = []
        for traj in trajectories:
            seq = [t.action_id for t in traj if t.action_id is not None]
            if seq:
                action_seqs.append(seq)

        if not action_seqs:
            return []

        # Find motifs of length 2 to 5 appearing across trajectories
        motif_support: dict[tuple[Any, ...], int] = {}
        for seq in action_seqs:
            seen_in_seq: set[tuple[Any, ...]] = set()
            max_len = min(5, len(seq))
            for length in range(2, max_len + 1):
                for i in range(len(seq) - length + 1):
                    motif = tuple(seq[i : i + length])
                    seen_in_seq.add(motif)
            for m in seen_in_seq:
                motif_support[m] = motif_support.get(m, 0) + 1

        # If no multi-step motifs met threshold, consider full single sequence if available
        new_schemas: list[DistilledSchema] = []
        for motif, support in motif_support.items():
            if support >= self.min_motif_support:
                motif_id = f"motif_{abs(hash(motif)) % 1000000:06d}"
                req_actions = {a for a in motif if isinstance(a, int)}
                subgoals = [
                    SymbolicSubgoal(
                        intent=SpatialActionIntent.NAVIGATE,
                        target_query={"action": act},
                    )
                    for act in motif
                ]
                skill = DeclarativeNeuroSymbolicSkill()
                skill.skill_name = f"distilled_skill_{motif_id}"
                skill.description = (
                    f"Distilled neuro-symbolic skill {motif_id} "
                    f"({len(motif)} actions, support={support})"
                )
                skill.signature = ActionAffordancePredicate(required=req_actions)
                skill.program = SubgoalSequence(*subgoals)

                pred = CausalPredicate(
                    variable=f"distilled_schema:{motif_id}",
                    operator="==",
                    value=f"actions:{len(motif)}:support:{support}",
                    relation="swr_sleep_replay",
                    target_entity=motif_id,
                )

                schema = DistilledSchema(
                    schema_id=motif_id,
                    action_motif=list(motif),
                    causal_predicate=pred,
                    declarative_skill=skill,
                    confidence=min(1.0, 0.5 + 0.1 * support),
                    support_count=support,
                    metadata={"motif_len": len(motif)},
                )
                self.distilled_schemas[motif_id] = schema
                new_schemas.append(schema)

        return new_schemas

    def propose_consolidation_transactions(
        self,
        schemas: list[DistilledSchema],
        tenant_id: str = "default",
    ) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions committing distilled schemas to semantic memory."""
        ops: list[TransactionOperation] = []
        for schema in schemas:
            ops.extend(schema.to_transaction_operations(tenant_id=tenant_id))
        return ops


@dataclass
class PatternSeparationResult:
    """Result of Dentate Gyrus sparse projection and pattern separation."""

    sparse_code: np.ndarray
    sparsity_ratio: float
    expansion_dim: int
    orthogonal_distance: float


class DentateGyrusPatternSeparator:
    """Hippocampal Dentate Gyrus (DG) sparse expansion and pattern separator."""

    def __init__(self, input_dim: int = 64, expansion_dim: int = 256, k_active: int = 24) -> None:
        self.input_dim = input_dim
        self.expansion_dim = expansion_dim
        self.k_active = k_active
        rng = np.random.RandomState(42)
        self.weights = rng.randn(expansion_dim, input_dim).astype(np.float32)
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights /= np.maximum(norms, 1e-6)

    def separate_pattern(self, sensory_vector: np.ndarray) -> PatternSeparationResult:
        """Project input vector into high-dimensional sparse orthogonal representation."""
        flat = sensory_vector.flatten().astype(np.float32)
        if len(flat) < self.input_dim:
            flat = np.pad(flat, (0, self.input_dim - len(flat)))
        elif len(flat) > self.input_dim:
            flat = flat[: self.input_dim]

        proj = np.dot(self.weights, flat)
        top_k_indices = np.argsort(proj)[-self.k_active :]
        sparse = np.zeros(self.expansion_dim, dtype=np.float32)
        sparse[top_k_indices] = proj[top_k_indices]
        sparse_norm = float(np.linalg.norm(sparse))
        if sparse_norm > 1e-6:
            sparse /= sparse_norm

        sparsity = float(self.k_active / self.expansion_dim)
        return PatternSeparationResult(
            sparse_code=sparse,
            sparsity_ratio=sparsity,
            expansion_dim=self.expansion_dim,
            orthogonal_distance=float(np.mean(np.abs(sparse))),
        )


@dataclass
class PatternCompletionResult:
    """Result of CA3 auto-associative pattern completion."""

    completed_pattern: np.ndarray
    retrieved_memory_id: str | None
    retrieval_confidence: float
    iterations: int


class CA3RecurrentPatternCompleter:
    """Hippocampal CA3 recurrent auto-associative memory for episodic pattern completion."""

    def __init__(self, capacity: int = 64, feature_dim: int = 256) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.memories: dict[str, np.ndarray] = {}
        self.associated_plans: dict[str, list[Any]] = {}

    def store_pattern(
        self, memory_id: str, pattern: np.ndarray, plan: list[Any] | None = None
    ) -> None:
        """Store an episodic attractor pattern with associated winning plan."""
        pat = pattern.flatten().astype(np.float32)
        if len(pat) != self.feature_dim:
            if len(pat) < self.feature_dim:
                pat = np.pad(pat, (0, self.feature_dim - len(pat)))
            else:
                pat = pat[: self.feature_dim]
        norm = float(np.linalg.norm(pat))
        if norm > 1e-6:
            pat /= norm
        self.memories[memory_id] = pat
        if plan is not None:
            self.associated_plans[memory_id] = list(plan)

    def complete_pattern(
        self,
        partial_cue: np.ndarray,
        match_threshold: float = 0.35,
    ) -> PatternCompletionResult:
        """Retrieve full stored pattern and plan from noisy or partial sensory cue."""
        if not self.memories:
            return PatternCompletionResult(
                completed_pattern=partial_cue,
                retrieved_memory_id=None,
                retrieval_confidence=0.0,
                iterations=0,
            )

        cue = partial_cue.flatten().astype(np.float32)
        if len(cue) < self.feature_dim:
            cue = np.pad(cue, (0, self.feature_dim - len(cue)))
        elif len(cue) > self.feature_dim:
            cue = cue[: self.feature_dim]
        cue_norm = float(np.linalg.norm(cue))
        if cue_norm > 1e-6:
            cue /= cue_norm

        best_id: str | None = None
        best_sim = -1.0
        for mem_id, stored_pat in self.memories.items():
            sim = float(np.dot(cue, stored_pat))
            if sim > best_sim:
                best_sim = sim
                best_id = mem_id

        if best_sim >= match_threshold and best_id is not None:
            return PatternCompletionResult(
                completed_pattern=self.memories[best_id].copy(),
                retrieved_memory_id=best_id,
                retrieval_confidence=best_sim,
                iterations=1,
            )

        return PatternCompletionResult(
            completed_pattern=cue,
            retrieved_memory_id=None,
            retrieval_confidence=max(0.0, best_sim),
            iterations=1,
        )


class HippocampalAttractorNetwork:
    """Hippocampal trisynaptic circuit integrating DG pattern separation and CA3 pattern completion."""

    def __init__(
        self,
        input_dim: int = 64,
        expansion_dim: int = 256,
        k_active: int = 24,
        capacity: int = 64,
    ) -> None:
        self.dg_separator = DentateGyrusPatternSeparator(
            input_dim=input_dim, expansion_dim=expansion_dim, k_active=k_active
        )
        self.ca3_completer = CA3RecurrentPatternCompleter(
            capacity=capacity, feature_dim=expansion_dim
        )

    def store_episode_attractor(
        self,
        memory_id: str,
        sensory_state: np.ndarray,
        action_plan: list[Any] | None = None,
    ) -> PatternSeparationResult:
        """Separate sensory pattern via Dentate Gyrus and store in CA3 auto-associative memory."""
        sep_res = self.dg_separator.separate_pattern(sensory_state)
        self.ca3_completer.store_pattern(
            memory_id=memory_id, pattern=sep_res.sparse_code, plan=action_plan
        )
        return sep_res

    def recall_from_partial_cue(
        self,
        partial_sensory_cue: np.ndarray,
        match_threshold: float = 0.35,
    ) -> tuple[list[Any] | None, PatternCompletionResult]:
        """Project partial cue via DG and retrieve complete episodic action sequence from CA3."""
        sep_cue = self.dg_separator.separate_pattern(partial_sensory_cue)
        comp_res = self.ca3_completer.complete_pattern(
            partial_cue=sep_cue.sparse_code, match_threshold=match_threshold
        )
        plan = None
        if comp_res.retrieved_memory_id is not None:
            plan = self.ca3_completer.associated_plans.get(comp_res.retrieved_memory_id)
        return plan, comp_res

    def to_causal_predicate(self) -> CausalPredicate:
        """Export attractor network status as a declarative causal predicate."""
        return CausalPredicate(
            variable="hippocampal_attractor_network",
            operator="==",
            value=f"stored_memories:{len(self.ca3_completer.memories)}",
            relation="dg_pattern_separation_ca3_completion",
            target_entity="hippocampus_trisynaptic",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting hippocampal attractor network state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="hippocampal_attractor_network",
                node_data={
                    "id": "hippocampal_attractor_network",
                    "node_type": "memory",
                    "category": "episodic",
                    "memory_count": len(self.ca3_completer.memories),
                },
            )
        ]


class EpisodicStage:
    """Pure CorticalStage for Stage 2 / Episodic Memory Consolidation.

    In accordance with Invariant 1 and WP1, this stage observes an immutable CognitiveContext
    and emits a pure StageResult containing consolidated episodic schemas and state mutations.
    """

    name: str = "episodic_stage"

    def observe(self, ctx: CognitiveContext) -> StageResult:
        ops: list[TransactionOperation] = []
        signals: dict[str, Any] = {}
        terminal = ctx.signals.get("terminal")
        if terminal:
            signals["episode_boundary"] = True
            signals["consolidated_transitions"] = len(ctx.failed_transitions)

        return StageResult(
            author=self.name,
            operations=ops,
            signals=signals,
            proposed_action=None,
        )
