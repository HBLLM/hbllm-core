"""Episodic memory transition representations, procedural macros, and trajectory models."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import EpistemicObservationDiff
from hbllm.hcir.world.percept import PlaceLocus, TerminalSignal, format_locus

logger = logging.getLogger(__name__)


@dataclass
class ProceduralMacro:
    """A consolidated, replayable action sequence learned from a winning episode.

    Compiled from a successful trajectory's EpisodicTransition sequence.
    Stored in the hippocampal consolidated_macros dict and queryable by
    the deliberation engine for future levels with similar structure.

    The macro captures:
    - The exact action sequence that solved the episode
    - The grid signature at episode start (for similarity matching)
    - The available action space (for compatibility checking)
    - Success metadata (steps taken, context epoch, environment context)
    """

    macro_id: str
    action_sequence: list[Any]  # Ordered list of action_ids
    action_data_sequence: list[dict[str, Any] | None]  # Corresponding action_data
    initial_grid_hash: int = 0  # Hash of grid at episode start
    initial_grid_shape: tuple[int, int] = (0, 0)
    available_actions: set[int] = field(default_factory=set)
    context_epoch: int = 0
    context_id: str = ""
    num_steps: int = 0
    confidence: float = 1.0  # Starts at 1.0 (proven), decays on replay failure
    replay_successes: int = 0
    replay_failures: int = 0
    waypoints: list[Any] = field(default_factory=list)  # RelationalWaypoint milestones
    compressed_action_sequence: list[Any] = field(default_factory=list)
    compressed_action_data_sequence: list[dict[str, Any] | None] = field(default_factory=list)

    def __init__(
        self,
        macro_id: str,
        action_sequence: list[Any],
        action_data_sequence: list[dict[str, Any] | None],
        initial_grid_hash: int = 0,
        initial_grid_shape: tuple[int, int] = (0, 0),
        available_actions: set[int] | None = None,
        context_epoch: int = 0,
        level: int | None = None,
        context_id: str = "",
        num_steps: int = 0,
        confidence: float = 1.0,
        replay_successes: int = 0,
        replay_failures: int = 0,
        waypoints: list[Any] | None = None,
        compressed_action_sequence: list[Any] | None = None,
        compressed_action_data_sequence: list[dict[str, Any] | None] | None = None,
    ) -> None:
        self.macro_id = macro_id
        self.action_sequence = action_sequence
        self.action_data_sequence = action_data_sequence
        self.initial_grid_hash = initial_grid_hash
        self.initial_grid_shape = initial_grid_shape
        self.available_actions = available_actions if available_actions is not None else set()
        resolved_epoch = level if level is not None else context_epoch
        self.context_epoch = resolved_epoch
        self.context_id = context_id
        self.num_steps = num_steps
        self.confidence = confidence
        self.replay_successes = replay_successes
        self.replay_failures = replay_failures
        self.waypoints = waypoints if waypoints is not None else []
        self.compressed_action_sequence = (
            compressed_action_sequence if compressed_action_sequence is not None else []
        )
        self.compressed_action_data_sequence = (
            compressed_action_data_sequence if compressed_action_data_sequence is not None else []
        )

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
            relation=f"episode:{self.context_epoch}:context:{self.context_id}",
            target_entity=self.macro_id,
        )

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
            f"Auto-consolidated procedural skill '{self.macro_id}' from context {self.context_epoch} "
            f"({len(self.action_sequence)} actions, conf={self.confidence:.2f})"
        )
        skill.signature = signature
        skill.program = SubgoalSequence(*subgoals) if subgoals else None
        setattr(skill, "_procedural_macro", self)
        return skill

    @property
    def level(self) -> int:
        """Compatibility alias for ``context_epoch``."""
        return self.context_epoch

    @level.setter
    def level(self, value: int) -> None:
        self.context_epoch = value


@dataclass
class EpisodicTransition:
    """A rich episodic transition record stored in the Hippocampal buffer."""

    step: int = 0
    avatar_pos: PlaceLocus | None = None
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
        avatar_pos: PlaceLocus | None = None,
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
    ) -> None:
        self.step = step
        self.avatar_pos = avatar_pos
        self.action_id = action_id
        self.action_data = action_data
        self.observation_diff = observation_diff
        self.reward = reward
        self.terminal = terminal
        self.is_terminal = is_terminal or bool(terminal)
        self.is_success = is_success or terminal == TerminalSignal.SUCCESS
        self.is_failure = is_failure or terminal in (
            TerminalSignal.FAILURE,
            TerminalSignal.TIMEOUT,
        )
        self.grid_snapshot = grid_snapshot
        self.state_signature = state_signature
        self.metadata = metadata if metadata is not None else {}

    def to_causal_predicate(self) -> CausalPredicate:
        """Export episodic transition as a formal Pearl do(a) causal predicate."""
        pos_str = format_locus(self.avatar_pos)
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
                    "avatar_pos": (
                        list(self.avatar_pos)
                        if isinstance(self.avatar_pos, tuple)
                        else self.avatar_pos
                    ),
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
    fatal_pos: PlaceLocus | None
    inhibited_transitions: list[tuple[PlaceLocus | None, Any]] = field(default_factory=list)
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
       without causing any environmental changes or solving the episode).
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


__all__ = [
    "EpisodicTransition",
    "ProceduralMacro",
    "ReplayCreditAssignment",
    "TDReplaySummary",
    "compress_action_trajectory",
]
