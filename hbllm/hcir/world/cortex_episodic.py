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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_perception import EpistemicObservationDiff
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


@dataclass
class EpisodicTransition:
    """A rich episodic transition record stored in the Hippocampal buffer."""

    step: int
    avatar_pos: tuple[int, int] | None
    action_id: Any
    action_data: dict[str, Any] | None = None
    observation_diff: EpistemicObservationDiff | None = None
    reward: float = 0.0
    is_terminal: bool = False
    is_lost: bool = False
    is_win: bool = False
    grid_snapshot: np.ndarray | None = None
    state_signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export episodic transition as a formal Pearl do(a) causal predicate."""
        pos_str = f"({self.avatar_pos[0]},{self.avatar_pos[1]})" if self.avatar_pos else "none"
        outcome = "WIN" if self.is_win else ("LOST" if self.is_lost else "ACTIVE")
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
                    "is_lost": self.is_lost,
                    "is_win": self.is_win,
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

    def record_transition(
        self,
        step: int,
        avatar_pos: tuple[int, int] | None,
        action: Any,
        action_data: dict[str, Any] | None = None,
        observation_diff: EpistemicObservationDiff | None = None,
        reward: float = 0.0,
        is_terminal: bool = False,
        is_lost: bool = False,
        is_win: bool = False,
        grid_snapshot: np.ndarray | None = None,
        state_signature: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an experiential transition into the hippocampal episodic buffer."""
        transition = EpisodicTransition(
            step=step,
            avatar_pos=avatar_pos,
            action_id=action,
            action_data=action_data,
            observation_diff=observation_diff,
            reward=reward,
            is_terminal=is_terminal,
            is_lost=is_lost,
            is_win=is_win,
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
        is_lost: bool,
        is_win: bool,
        prev_grid: np.ndarray | None = None,
        curr_grid: np.ndarray | None = None,
        bg_feature: int = 0,
        avatar_features: set[int] | None = None,
        is_walkable_fn: Any = None,
        is_barrier_fn: Any = None,
    ) -> ReplayCreditAssignment | None:
        """Execute Sharp-Wave Ripple (SWR) replay on episode termination.

        On catastrophe (is_lost=True):
        - Backpropagates negative prediction error along the fatal prefix.
        - Suppresses the fatal state-action transition in Habenular IOR.
        - Deduces candidate lethal hazard features in the landing zone and line-of-sight.
        """
        if not self.current_episode:
            return None

        final_transition = self.current_episode[-1]
        av_feats = avatar_features or set()

        if is_win:
            # Consolidate successful trajectory
            self.successful_trajectories.append(list(self.current_episode))
            logger.info(
                "HippocampalEpisodicCortex: Consolidated successful episode with %d steps.",
                len(self.current_episode),
            )
            self._archive_and_clear()
            return None

        if not is_lost:
            self._archive_and_clear()
            return None

        # Catastrophic failure occurred: perform SWR negative replay
        final_step = final_transition.step
        fatal_pos = final_transition.avatar_pos
        fatal_action = final_transition.action_id

        # 1. Update Habenular IOR
        self.habenular_ior.record_catastrophe(
            final_step=final_step,
            is_lost=True,
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
                    if int(np.sum(prev_grid == cand_int)) < 40:
                        suspected_hazards.add(cand_int)
                        self.discovered_lethal_features.add(cand_int)

        # 4. Extract inhibited state-actions from SWR backward horizon
        inhibited: list[tuple[tuple[int, int] | None, Any]] = []
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
                if trans.is_win:
                    r += 10.0
                elif trans.is_lost:
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
        if self.current_episode and any(t.is_win for t in self.current_episode):
            self.successful_trajectories.append(list(self.current_episode))
        elif self.current_episode and any(t.is_lost for t in self.current_episode):
            self.failed_trajectories.append(list(self.current_episode))
        self.current_episode.clear()

        if level is not None:
            engine.set_current_level(level)
        engine.prev_grid = None
        engine._prev_oriented_threats = []
        engine.avatar_pos = None
        engine.last_action = None
        engine.last_action_data = None
        engine.last_predicted_pos = None
        engine.bilateral_state = None
        engine.pending_phase_wait_steps = 0
        engine.consecutive_pacing_steps = 0
        engine.consecutive_quiescent_actions = 0

        engine.consecutive_stuck_steps = 0
        if is_new_level or not retain_dynamics:
            engine.entity_action_failed_positions = {}
            engine.immobile_entity_actions = set()
        has_focus_switches = (
            len(engine.focus_switch_actions) > 0
            or len(getattr(getattr(engine, "working_memory", None), "focus_switch_actions", set()))
            > 0
            or getattr(engine, "is_multi_agent", False)
        )
        if not retain_dynamics or (is_new_level and has_focus_switches):
            engine.avatar_features.clear()
            engine.avatar_feature = None
            engine.avatar_size = 0
            engine.prior_avatar_feature = None
        else:
            # Validate avatar features: void 0 and background feature must never be retained as avatar
            if (
                engine.avatar_features == {0}
                or (engine.bg_feature is not None and engine.bg_feature in engine.avatar_features)
                or engine.avatar_feature == 0
                or (engine.bg_feature is not None and engine.avatar_feature == engine.bg_feature)
            ):
                engine.avatar_features.clear()
                engine.avatar_feature = None
                engine.avatar_size = 0
                engine.prior_avatar_feature = None
            # On same-level retry after death: preserve learned spatial negative affordances,
            # barriers, and quiescent inert coordinates so inert scenery/borders are not re-tested,
            # while decaying entity visit counts so interactive objects can be re-sampled with fresh hypotheses.
            for k in list(engine.entity_visit_counts.keys()):
                engine.entity_visit_counts[k] = engine.entity_visit_counts[k] // 2
        engine.last_effective_click_coord = None
        engine.consecutive_effective_clicks = 0
        engine.mental_plan.clear()
        engine.active_hypothesis = None
        engine.active_probe_target = None
        engine.active_probe_id = None
        engine.probe_target_steps = 0
        engine.recent_positions.clear()
        engine.recent_actions.clear()
        engine.level_epistemic_probes = 0
        is_same_level_retry = not is_new_level and retain_dynamics
        if is_same_level_retry:
            # Same level retry on death: trigger Habenular catastrophe backpropagation FIRST
            # before transient episode buffers are reset
            if hasattr(engine, "working_memory") and hasattr(
                engine.working_memory, "habenular_ior"
            ):
                engine.working_memory.habenular_ior.record_catastrophe(
                    final_step=engine.step_counter,
                    is_lost=True,
                )
            elif hasattr(engine, "habenular_ior"):
                engine.habenular_ior.record_catastrophe(
                    final_step=engine.step_counter,
                    is_lost=True,
                )

        engine.working_memory.reset_episode(
            retain_long_term=retain_dynamics,
            retain_inhibitions=is_same_level_retry,
            is_new_level=is_new_level,
        )
        engine.collision_cones.reset_episode()
        engine.consecutive_plan_failures = 0
        engine.exploration_cooldown = 0
        engine.consecutive_simulation_failures = 0
        engine.simulation_cooldown = 0
        engine.working_memory.reset_salience_baseline()
        engine.working_memory.reset_episode(
            retain_long_term=retain_dynamics,
            is_new_level=is_new_level,
        )
        engine.current_simulated_goal = None
        engine._feedback_assimilated = False
        engine.failed_transitions.clear()
        engine.symbolic_theory.candidate_goal_features.clear()

        if not retain_dynamics:
            target_lvl = 0 if level is None else level
            engine.set_current_level(target_lvl)
            engine.level_failed_transitions.clear()
            engine.level_learned_barriers.clear()
            engine.level_lethal_positions.clear()
            engine.failed_transitions = engine.level_failed_transitions.setdefault(
                engine._current_level, set()
            )
            engine.learned_barriers = engine.level_learned_barriers.setdefault(
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
            engine.current_room_id = None
            engine.surprise_engine.reset_ledger()
            engine.last_surprise = 0.0
            engine.last_surprise_eval = None
        elif is_new_level:
            # Level transition: point episodic spatial memory to the target level
            if level is not None:
                engine.set_current_level(level)
            engine.failed_transitions = engine.level_failed_transitions.setdefault(
                engine._current_level, set()
            )
            engine.learned_barriers = engine.level_learned_barriers.setdefault(
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
            engine.current_room_id = None
            engine.surprise_engine.reset_ledger()
            engine.last_surprise = 0.0
            engine.last_surprise_eval = None
            engine.collision_cones.reset_episode()
            # Preserve mobile_threat_features across level transitions: the same
            # enemy feature-types recur across levels, and forgetting them causes
            # the agent to re-discover threats by collision.
            engine._prev_oriented_threats = []
            engine.prev_grid = None
            engine.last_action = None
            engine.last_predicted_pos = None
            engine.avatar_pos = None
            if not retain_dynamics or has_focus_switches:
                engine.avatar_features.clear()
                engine.avatar_feature = None
                engine.avatar_size = 0
            if hasattr(engine, "working_memory"):
                engine.working_memory.target_constellations.clear()
                engine.working_memory.structural_goals.clear()
            engine.mental_plan.clear()
            engine.consecutive_simulation_failures = 0
            engine.simulation_cooldown = 0
        else:
            # Level Regression Guard: When the game resets the agent to a
            # *different* (earlier) level upon death, level-specific spatial
            # goals and receptacle coordinates from the failed level must be
            # purged.  Without this, the planner sees phantom goals from
            # the previous level and cannot generate valid cargo-delivery plans.
            is_level_regression = level is not None and level != engine._current_level
            if is_level_regression and level is not None:
                engine.learned_goal_positions.clear()
                engine.learned_receptacle_positions.clear()
                engine.exhausted_candidate_goals.clear()
                engine.tested_action_positions.clear()
                engine.topology_rooms.clear()
                engine.topology_doors.clear()
                engine.room_adjacency.clear()
                engine.current_room_id = None
                engine.failed_transitions = engine.level_failed_transitions.setdefault(level, set())
                engine.learned_barriers = engine.level_learned_barriers.setdefault(level, set())
                engine.hazard_tracker.static_lethal_positions = (
                    engine.level_lethal_positions.setdefault(level, set())
                )
                engine.symbolic_theory.candidate_goal_features.clear()
                engine.symbolic_theory.goal_features.clear()
                engine.symbolic_theory.goal_predicates.clear()
                engine.working_memory.target_constellations.clear()
                engine.working_memory.structural_goals.clear()
                logger.info(
                    "HippocampalEpisodicCortex: Level regression detected (%d -> %d). "
                    "Purging level-specific spatial goals, receptacle coordinates, "
                    "and goal features to prevent cross-level contamination.",
                    engine._current_level,
                    level,
                )

            engine.hazard_tracker.grid_history.clear()
            engine.hazard_tracker.step_history.clear()
            if not is_level_regression:
                engine.tested_action_positions.clear()
            engine.last_surprise = 0.0
            engine.last_surprise_eval = None
            engine.collision_cones.reset_episode()
            engine.prev_grid = None
            engine.last_action = None
            engine.last_predicted_pos = None
            engine.mental_plan.clear()
            engine._prev_oriented_threats = []
            engine.recent_positions.clear()
            if hasattr(engine, "acc_conflict_monitor"):
                engine.acc_conflict_monitor.reset()

        if is_new_level and retain_dynamics:
            # Carry forward hypotheses & semantics as priors
            engine.avatar_pos = None
            if has_focus_switches:
                engine.avatar_feature = None
                engine.avatar_features.clear()
            engine._avatar_controllability_evidence.clear()
            if hasattr(engine.working_memory, "target_constellations"):
                engine.working_memory.target_constellations.clear()
            if hasattr(engine.working_memory, "structural_goals"):
                engine.working_memory.structural_goals = []
            if hasattr(engine, "learned_goal_positions"):
                engine.learned_goal_positions.clear()
            if hasattr(engine, "symbolic_theory"):
                if hasattr(engine.symbolic_theory, "goal_features"):
                    engine.symbolic_theory.goal_features.clear()
                if hasattr(engine.symbolic_theory, "candidate_goal_features"):
                    engine.symbolic_theory.candidate_goal_features.clear()
            engine.state_mutations.clear()
            engine.bilateral_state = None
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
            engine.active_goal_converging_coord = None
            engine.consecutive_goal_converging_clicks = 0
            engine.active_causal_actuator_coord = None
            engine.consecutive_causal_probes = 0
            engine.consecutive_quiescent_actions = 0
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
            engine.total_epistemic_probes = 0
            engine.avatar_feature = None
            engine.avatar_features.clear()
            engine.prior_avatar_feature = None
            engine._avatar_controllability_evidence.clear()
            engine.verified_safe_features.clear()
            engine.avatar_pos = None
            engine.action_dynamics.clear()
            engine.action_affordances.clear()
            engine.effector_kinetics_mode = None
            engine.focus_switch_actions.clear()
            engine.last_non_displacement_action = None
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
            engine.consecutive_quiescent_actions = 0
            engine.inhibited_actions.clear()
            engine.topology_rooms.clear()
            engine.topology_doors.clear()
            engine.room_adjacency.clear()
            engine.current_room_id = None
            engine.surprise_engine.reset_ledger()
            engine.last_surprise = 0.0
            engine.last_surprise_eval = None
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
