"""Hippocampal episodic cortex, sharp-wave ripple backward replay, and credit assignment."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_perception import EpistemicObservationDiff
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.episodic.attractor import (
    HippocampalAttractorNetwork,
)
from hbllm.hcir.world.episodic.models import (
    EpisodicTransition,
    ProceduralMacro,
    ReplayCreditAssignment,
    TDReplaySummary,
    compress_action_trajectory,
)
from hbllm.hcir.world.episodic.replay import (
    HippocampalReplayConsolidator,
)
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.percept import PlaceLocus, TerminalSignal, locus_token

logger = logging.getLogger(__name__)


class HippocampalEpisodicCortex:
    """Hippocampal-Habenular episodic memory system for trial-and-error learning."""

    def __init__(
        self,
        habenular_ior: HabenularEpisodicIOR | None = None,
        max_trace_len: int = 512,
        replay_horizon: int = 12,
        discount_factor: float = 0.85,
        max_episodes: int = 64,
    ) -> None:
        self.habenular_ior = habenular_ior or HabenularEpisodicIOR()
        self.max_trace_len = max_trace_len
        self.replay_horizon = replay_horizon
        self.discount_factor = discount_factor
        self.max_episodes = max(1, int(max_episodes))

        # Active episode buffer (Hippocampus CA3 / CA1)
        self.current_episode: deque[EpisodicTransition] = deque(maxlen=max_trace_len)

        # Multi-episode history archives (bounded by max_episodes)
        self.past_episodes: list[list[EpisodicTransition]] = []
        self.successful_trajectories: list[list[EpisodicTransition]] = []
        self.consolidated_trajectory_count: int = 0
        self.failed_trajectories: list[list[EpisodicTransition]] = []
        self.catastrophic_replays: list[ReplayCreditAssignment] = []

        # Grounded lethal transitions: set of (avatar_pos, action)
        self.grounded_lethal_transitions: set[tuple[PlaceLocus, Any]] = set()

        # Lethal candidate features identified across trials
        self.discovered_lethal_features: set[int] = set()

        # State-Action Value Table (offline TD consolidation)
        self.q_table: dict[tuple[str, Any], float] = {}
        self.transition_counts: dict[tuple[str, Any, str], int] = {}

        # ── Procedural Consolidation Store ──────────────────────────────────
        # Winning trajectories compiled into replayable action macros.
        # Key: (context_id, context_epoch) or auto-generated signature.
        # Value: ProceduralMacro containing proven action sequence + context.
        self.consolidated_macros: dict[str, ProceduralMacro] = {}
        self.relational_macros: dict[str, Any] = {}
        self.replay_consolidator: HippocampalReplayConsolidator = HippocampalReplayConsolidator(
            gamma=self.discount_factor
        )
        self.attractor_network: HippocampalAttractorNetwork = HippocampalAttractorNetwork()

    def replay_forward_and_reverse(
        self,
        trajectory: list[EpisodicTransition],
        terminal_reward: float = 1.0,
    ) -> dict[str, Any]:
        """Perform forward and reverse replay through the hippocampal replay consolidator."""
        return self.replay_consolidator.replay_forward_and_reverse(
            trajectory, terminal_reward=terminal_reward
        )

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
    ) -> None:
        """Record an experiential transition into the hippocampal episodic buffer."""
        is_success = is_success or terminal == TerminalSignal.SUCCESS
        is_failure = is_failure or terminal in (
            TerminalSignal.FAILURE,
            TerminalSignal.TIMEOUT,
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
    ) -> ReplayCreditAssignment | None:
        """Execute Sharp-Wave Ripple (SWR) replay on episode termination.

        On catastrophe (is_failure=True):
        - Backpropagates negative prediction error along the fatal prefix.
        - Suppresses the fatal state-action transition in Habenular IOR.
        - Deduces candidate lethal hazard features in the landing zone and line-of-sight.
        """
        is_success = is_success or terminal == TerminalSignal.SUCCESS
        is_failure = is_failure or terminal in (
            TerminalSignal.FAILURE,
            TerminalSignal.TIMEOUT,
        )
        if not self.current_episode:
            return None

        final_transition = self.current_episode[-1]
        av_feats = avatar_features or set()

        if is_success:
            # Consolidate successful trajectory
            self.successful_trajectories.append(list(self.current_episode))
            if len(self.successful_trajectories) > self.max_episodes:
                drop = len(self.successful_trajectories) - self.max_episodes
                del self.successful_trajectories[:drop]
                self.consolidated_trajectory_count = max(
                    0, self.consolidated_trajectory_count - drop
                )
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
        if (
            prev_grid is not None
            and isinstance(fatal_pos, tuple)
            and len(fatal_pos) >= 2
            and isinstance(fatal_pos[0], (int, float))
            and isinstance(fatal_pos[1], (int, float))
        ):
            H, W = prev_grid.shape
            pr, pc = int(fatal_pos[0]), int(fatal_pos[1])

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

        # 4. Extract inhibited state-actions from SWR backward horizon
        inhibited: list[tuple[PlaceLocus | None, Any]] = []
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
        if len(self.catastrophic_replays) > self.max_episodes:
            del self.catastrophic_replays[: len(self.catastrophic_replays) - self.max_episodes]
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
                    f"pos_{locus_token(trans.avatar_pos)}"
                    if trans.avatar_pos is not None
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
                        f"pos_{locus_token(next_trans.avatar_pos)}"
                        if next_trans.avatar_pos is not None
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
        """Archive current episode to bounded long-term memory and clear active buffer."""
        if self.current_episode:
            self.past_episodes.append(list(self.current_episode))
            if len(self.past_episodes) > self.max_episodes:
                del self.past_episodes[: len(self.past_episodes) - self.max_episodes]
            self.current_episode.clear()

    def on_boundary(self, boundary) -> None:
        """Episode lifecycle boundary handler (WP0.5). Delegates to the
        deprecated reset_episode for backward-compatible semantics."""
        self.reset_episode()

    @property
    def boundary_retention(self):
        """Declared retention policy: default episode-local semantics."""
        from hbllm.hcir.world.percept import RetentionPolicy

        return RetentionPolicy.EPISODE_LOCAL

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
        self.consolidated_trajectory_count = 0
        self.catastrophic_replays.clear()
        self.grounded_lethal_transitions.clear()
        self.discovered_lethal_features.clear()
        self.habenular_ior.reset()
        # NOTE: consolidated_macros are intentionally NOT cleared here.
        # Learned procedural knowledge persists across contexts within an environment.
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
        min_match_score: float = 0.0,
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
            if score > 0.0 and score >= min_match_score:
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
        view: Any,
        delta: Any,
        retain_dynamics: bool = True,
        is_new_context: bool | None = None,
        context_epoch: int | None = None,
        *,
        is_new_level: bool | None = None,
        level: int | None = None,
    ) -> None:
        """Consolidate episodic schemas and coordinate multi-faculty memory reset.
        Handles three distinct cognitive transitions:
        1. Same-context retry on catastrophe (retain dynamics, barriers, lethal features).
        2. Cross-context transition (transfer semantic & motor priors, point to new context map).
        3. Cross-environment reset (strict zero-leakage cognitive isolation).
        """
        new_context = (
            is_new_context
            if is_new_context is not None
            else (is_new_level if is_new_level is not None else False)
        )
        target_epoch = context_epoch if context_epoch is not None else level
        # 1. Offline Hippocampal TD Experience Replay & Q-value consolidation
        if self.current_episode:
            td_summary = self.consolidate_episodic_replay_td(
                learning_rate=0.2,
                discount=0.9,
                prioritized=True,
                max_batch_size=32,
            )
            # Sync consolidated causal predicates into Symbolic World Theory
            if hasattr(view, "symbolic_theory") and view.symbolic_theory is not None:
                for pred in td_summary.consolidated_predicates:
                    view.symbolic_theory.register_predicate(pred)

        # 2. SWR Trajectory Consolidation: Archive winning or catastrophic sequences
        if self.current_episode and any(t.is_success for t in self.current_episode):
            winning_trajectory = list(self.current_episode)
            self.successful_trajectories.append(winning_trajectory)
            if len(self.successful_trajectories) > self.max_episodes:
                drop = len(self.successful_trajectories) - self.max_episodes
                del self.successful_trajectories[:drop]
                self.consolidated_trajectory_count = max(
                    0, self.consolidated_trajectory_count - drop
                )

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
                            logger.debug("Non-integer macro action_id ignored: %r", t.action_id)

                context_id = (
                    get_engine_state_attr(view, "context_id", "")
                    or get_engine_state_attr(view, "current_context_id", "")
                    or ""
                )
                current_epoch = get_engine_state_attr(view, "context_epoch", target_epoch or 0)
                macro_id = f"macro_{context_id}_ctx{current_epoch}_{len(action_seq)}steps"

                # Extract relational waypoints for coordinate-invariant lifting
                from hbllm.hcir.world.autonomous_learning import (
                    RelationalSubgoalMacro,
                    RelationalWaypoint,
                )

                extracted_waypoints: list[RelationalWaypoint] = []
                seen_features: set[int] = set()
                bg_val = get_engine_state_attr(view, "bg_feature", 0)
                wm_ref = get_engine_state_attr(view, "working_memory", None)
                sym_ref = get_engine_state_attr(view, "symbolic_theory", None)
                av_feats_ref = set(get_engine_state_attr(view, "avatar_features", ()) or ())
                av_single = get_engine_state_attr(view, "avatar_feature", None)
                if av_single is not None:
                    av_feats_ref.add(av_single)
                walkable_ref = sym_ref.walkable_features if sym_ref is not None else set()
                barrier_ref = sym_ref.barrier_features if sym_ref is not None else set()
                for idx, step_t in enumerate(winning_trajectory):
                    ref_grid = (
                        winning_trajectory[idx - 1].grid_snapshot
                        if idx > 0 and winning_trajectory[idx - 1].grid_snapshot is not None
                        else step_t.grid_snapshot
                    )
                    if (
                        ref_grid is not None
                        and isinstance(step_t.avatar_pos, tuple)
                        and len(step_t.avatar_pos) >= 2
                        and isinstance(step_t.avatar_pos[0], (int, float))
                        and isinstance(step_t.avatar_pos[1], (int, float))
                    ):
                        ar, ac = int(step_t.avatar_pos[0]), int(step_t.avatar_pos[1])
                        H_s, W_s = ref_grid.shape[:2]
                        if 0 <= ar < H_s and 0 <= ac < W_s:
                            f_val = int(ref_grid[ar, ac])
                            is_exog = (
                                wm_ref.is_exogenous_locus(
                                    (ar, ac), lattice=ref_grid, bg=bg_val, include_outer_edge=True
                                )
                                if wm_ref is not None
                                else False
                            )
                            if (
                                f_val != bg_val
                                and f_val not in seen_features
                                and f_val not in av_feats_ref
                                and f_val not in walkable_ref
                                and f_val not in barrier_ref
                                and not is_exog
                                and int(np.count_nonzero(ref_grid == f_val))
                                <= max(4, int(H_s * W_s * 0.08))
                            ):
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
                    context_epoch=current_epoch,
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
                        source_context_epoch=current_epoch,
                    )

                logger.info(
                    "Procedural Consolidation: Compiled macro '%s' (%d raw actions -> %d compressed, %d relational waypoints) from winning trajectory",
                    macro_id,
                    len(action_seq),
                    len(comp_actions),
                    len(extracted_waypoints),
                )

                # Also register in working memory procedural library if view has it
                if hasattr(view, "working_memory") and view.working_memory is not None:
                    try:
                        skill = macro.to_declarative_skill()
                        view.working_memory.register_procedural_skill(macro_id, skill)
                    except Exception as e:
                        logger.debug("Could not register macro as procedural skill: %s", e)
        elif self.current_episode and any(t.is_failure for t in self.current_episode):
            self.failed_trajectories.append(list(self.current_episode))
            if len(self.failed_trajectories) > self.max_episodes:
                del self.failed_trajectories[: len(self.failed_trajectories) - self.max_episodes]
        self.current_episode.clear()

        if target_epoch is not None:
            delta.set("_context_epoch", target_epoch)
        delta.set("prev_grid", None)
        delta.set("_prev_oriented_threats", [])
        delta.set("avatar_pos", None)
        delta.set("last_action", None)
        delta.set("last_action_data", None)
        delta.set("last_predicted_pos", None)
        delta.set("bilateral_state", None)
        delta.set("pending_phase_wait_steps", 0)
        delta.set("consecutive_pacing_steps", 0)
        delta.set("consecutive_quiescent_actions", 0)

        delta.set("consecutive_stuck_steps", 0)
        if new_context or not retain_dynamics:
            delta.set("entity_action_failed_positions", {})
            delta.set("immobile_entity_actions", set())
        else:
            # On same-context retry after catastrophe: preserve learned spatial negative affordances,
            # barriers, and quiescent inert coordinates so inert scenery/borders are not re-tested,
            # while decaying entity visit counts so interactive objects can be re-sampled with fresh hypotheses.
            _evc = delta.mutate("entity_visit_counts", view.entity_visit_counts)
            for k in list(_evc.keys()):
                _evc[k] = _evc[k] // 2
        delta.set("last_effective_actuation_coord", None)
        delta.set("consecutive_effective_actuations", 0)
        delta.mutate("mental_plan", view.mental_plan).clear()
        delta.set("active_hypothesis", None)
        delta.set("active_probe_target", None)
        delta.set("active_probe_id", None)
        delta.set("probe_target_steps", 0)
        delta.mutate("recent_positions", view.recent_positions).clear()
        delta.mutate("recent_actions", view.recent_actions).clear()
        delta.set("context_epistemic_probes", 0)
        is_same_context_retry = not new_context and retain_dynamics
        if is_same_context_retry:
            # Same level retry on death: trigger Habenular catastrophe backpropagation FIRST
            # before transient episode buffers are reset
            view.working_memory.habenular_ior.record_catastrophe(
                final_step=view.step_counter,
                is_fatal=True,
            )

        view.working_memory.reset_episode(
            retain_long_term=retain_dynamics,
            retain_inhibitions=is_same_context_retry,
        )
        delta.set("consecutive_plan_failures", 0)
        delta.set("exploration_cooldown", 0)
        delta.set("consecutive_simulation_failures", 0)
        delta.set("simulation_cooldown", 0)
        view.working_memory.reset_salience_baseline()
        delta.set("current_simulated_goal", None)
        delta.set("_feedback_assimilated", False)
        view.symbolic_theory.clear_explored_loci(
            clear_features=(new_context or not retain_dynamics)
        )
        if new_context or not retain_dynamics:
            view.symbolic_theory.clear_candidate_goals()
            if new_context:
                view.symbolic_theory.clear_barriers()

        _epoch = (
            ((view.context_epoch or 0) + 1 if new_context else (view.context_epoch or 0))
            if target_epoch is None
            else target_epoch
        )
        if not retain_dynamics:
            delta.mutate("context_failed_transitions", view.context_failed_transitions).clear()
            delta.mutate("context_barriers", view.context_barriers).clear()
            delta.mutate("context_hazards", view.context_hazards).clear()
            delta.set("_context_epoch", _epoch)
            _lft = delta.mutate("context_failed_transitions", view.context_failed_transitions)
            delta.set("failed_transitions", _lft.setdefault(_epoch, set()))
            _llb = delta.mutate("context_barriers", view.context_barriers)
            delta.set("learned_barriers", _llb.setdefault(_epoch, set()))
            view.hazard_tracker.clear_lethal_positions()
            view.hazard_tracker.reset_episode()
            delta.mutate("exhausted_candidate_goals", view.exhausted_candidate_goals).clear()
            delta.mutate("tested_action_positions", view.tested_action_positions).clear()
            delta.mutate("learned_goal_positions", view.learned_goal_positions).clear()
            delta.mutate("learned_receptacle_positions", view.learned_receptacle_positions).clear()
            delta.mutate("topology_rooms", view.topology_rooms).clear()
            delta.mutate("topology_doors", view.topology_doors).clear()
            delta.mutate("room_adjacency", view.room_adjacency).clear()
            delta.set("current_room_id", None)
            delta.set("last_surprise", 0.0)
            delta.set("last_surprise_eval", None)
        elif new_context:
            # Level transition: advance/point episodic spatial memory to the target level epoch
            view.symbolic_theory.begin_context_epoch()
            delta.set("_context_epoch", _epoch)
            _lft = delta.mutate("context_failed_transitions", view.context_failed_transitions)
            delta.set("failed_transitions", _lft.setdefault(_epoch, set()))
            _llb = delta.mutate("context_barriers", view.context_barriers)
            delta.set("learned_barriers", _llb.setdefault(_epoch, set()))
            _lhz = delta.mutate("context_hazards", view.context_hazards)
            view.hazard_tracker.static_lethal_positions = _lhz.setdefault(_epoch, set())
            view.hazard_tracker.reset_episode()
            delta.mutate("exhausted_candidate_goals", view.exhausted_candidate_goals).clear()
            delta.mutate("tested_action_positions", view.tested_action_positions).clear()
            delta.mutate("learned_goal_positions", view.learned_goal_positions).clear()
            delta.mutate("learned_receptacle_positions", view.learned_receptacle_positions).clear()
            delta.mutate("topology_rooms", view.topology_rooms).clear()
            delta.mutate("topology_doors", view.topology_doors).clear()
            delta.mutate("room_adjacency", view.room_adjacency).clear()
            delta.set("current_room_id", None)
            delta.set("last_surprise", 0.0)
            delta.set("last_surprise_eval", None)
            delta.set("_prev_oriented_threats", [])
            delta.set("prev_grid", None)
            delta.set("last_action", None)
            delta.set("last_predicted_pos", None)
            delta.mutate("mental_plan", view.mental_plan).clear()
            delta.set("consecutive_simulation_failures", 0)
            delta.set("simulation_cooldown", 0)
            view.state_mutations.clear()
            if hasattr(view.working_memory, "active_schema"):
                view.working_memory.active_schema = None
        else:
            view.hazard_tracker.grid_history.clear()
            view.hazard_tracker.step_history.clear()
            delta.mutate("tested_action_positions", view.tested_action_positions).clear()
            delta.set("last_surprise", 0.0)
            delta.set("last_surprise_eval", None)
            delta.set("prev_grid", None)
            delta.set("last_action", None)
            delta.set("last_predicted_pos", None)
            delta.mutate("mental_plan", view.mental_plan).clear()
            delta.set("_prev_oriented_threats", [])
            delta.mutate("recent_positions", view.recent_positions).clear()
            if hasattr(view, "acc_conflict_monitor"):
                view.acc_conflict_monitor.reset()

        if new_context and retain_dynamics:
            # ── Role Abstraction Transfer ────────────────────────────────────
            # Snapshot feature-level conditional affordances as role-level templates
            # before clearing layout-specific bindings.  Role priors and templates
            # survive the transition; when the new level's perception induces goals,
            # barriers, etc. via ``induce_*``, ``bind_feature_to_role`` lazily
            # re-instantiates any matching templates (e.g. key→door).
            view.symbolic_theory.snapshot_conditional_affordance_templates()
            view.symbolic_theory.clear_feature_bindings()
            acc_mon = getattr(view.working_memory, "acc_conflict", None)
            if acc_mon is not None and hasattr(acc_mon, "reset_episode"):
                acc_mon.reset_episode(retain_long_term=False)
            if view.symbolic_theory.labels_non_stationary:
                from hbllm.hcir.world.cortex_causal import invalidate_stale_context_ontology

                invalidate_stale_context_ontology(
                    view.symbolic_theory,
                    hazard_tracker=view.hazard_tracker,
                    episodic_cortex=self,
                    working_memory=view.working_memory,
                    engine_or_view=view,
                    bg_feature=getattr(view, "bg_feature", 0),
                    preserve_current_context=False,
                    reason="non_stationary_new_level",
                )
                delta.mutate("mobile_threat_features", view.mobile_threat_features).clear()

            # Carry forward hypotheses & semantics as priors
            retained_av_feats = set(view.avatar_features or ())
            if view.avatar_feature is not None:
                delta.set("prior_avatar_feature", view.avatar_feature)
                retained_av_feats.add(view.avatar_feature)
            if retained_av_feats:
                delta.set("avatar_features", retained_av_feats)
            else:
                view.avatar_features.clear()
            view._avatar_controllability_evidence.clear()
            view.state_mutations.clear()
            delta.set("bilateral_state", None)

            view.entity_visit_counts.clear()
            view.position_visit_counts.clear()
            view.probed_entity_ids.clear()
            view.quiescent_actuation_targets.clear()
            view.effective_actuation_targets.clear()
            view.actuation_affordances.clear()
            view.quiescent_features.clear()
            if hasattr(view.working_memory, "remote_causal"):
                # Coordinate-keyed distal links are layout-specific; the feature-level
                # law survives in symbolic_theory.conditional_affordances.
                view.working_memory.remote_causal.reset_episode(retain_long_term=False)
            if hasattr(view, "feature_quiescent_counts"):
                view.feature_quiescent_counts.clear()
            if hasattr(view, "_feature_quiescent_counts"):
                view._feature_quiescent_counts.clear()
            view.last_effector_target_step.clear()
            delta.set("active_goal_converging_coord", None)
            delta.set("consecutive_goal_converging_actuations", 0)
            delta.set("active_causal_actuator_coord", None)
            delta.set("consecutive_causal_probes", 0)
            delta.set("consecutive_quiescent_actions", 0)
            view.inhibited_actions.clear()

            # Affordance priors re-evaluation on new level transitions:
            # Physical terrain affordances (e.g. tracks, bridges, road surfaces) persist across
            # rooms of the same environment unless counterfactual evidence invalidates them.
            # Local coordinate barriers reset, but empirical walkable affordances are retained.
            view.learned_barriers.clear()
            for f in view.symbolic_theory.walkable_features:
                if not any(p.value == f for p in view.symbolic_theory.walkable_predicates):
                    view.symbolic_theory.walkable_predicates.append(
                        CausalPredicate(variable="feature_id", operator="==", value=f)
                    )

            view.transition_to_phase(
                EpistemicPhase.MENTAL_SIMULATION
                if view.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING,
                reason="episode_reset_context_transition",
            )

        elif not retain_dynamics:
            # Full Cross-Game Isolation: Zero cross-game leakage
            delta.set("total_epistemic_probes", 0)
            delta.set("avatar_feature", None)
            view.avatar_features.clear()
            delta.set("prior_avatar_feature", None)
            view._avatar_controllability_evidence.clear()
            delta.set("avatar_pos", None)
            view.action_dynamics.clear()
            view.action_affordances.clear()
            delta.set("effector_kinetics_mode", None)
            view.focus_switch_actions.clear()
            delta.set("last_non_displacement_action", None)
            view.tested_actions.clear()
            view.symbolic_theory.clear()
            view.state_mutations.clear()
            view.entity_visit_counts.clear()
            view.position_visit_counts.clear()
            view.probed_entity_ids.clear()
            view.quiescent_actuation_targets.clear()
            view.effective_actuation_targets.clear()
            view.actuation_affordances.clear()
            view.quiescent_features.clear()
            if hasattr(view, "feature_quiescent_counts"):
                view.feature_quiescent_counts.clear()
            if hasattr(view, "_feature_quiescent_counts"):
                view._feature_quiescent_counts.clear()
            view.effective_features.clear()
            delta.set("consecutive_quiescent_actions", 0)
            view.inhibited_actions.clear()
            delta.mutate("topology_rooms", view.topology_rooms).clear()
            delta.mutate("topology_doors", view.topology_doors).clear()
            delta.mutate("room_adjacency", view.room_adjacency).clear()
            delta.set("current_room_id", None)
            delta.set("last_surprise", 0.0)
            delta.set("last_surprise_eval", None)
            view.hazard_tracker.known_lethal_features.clear()
            view.hazard_tracker.periodic_cells.clear()
            view.mobile_threat_features.clear()
            view.transition_to_phase(
                EpistemicPhase.MOTOR_GROUNDING,
                reason="episode_reset_cross_game",
            )
        else:
            view.transition_to_phase(
                EpistemicPhase.MENTAL_SIMULATION
                if view.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING,
                reason="episode_reset_retained_dynamics",
            )

        view.object_planner.reset_episode(is_new_context=new_context)
        view.subgoal_stack.reset_episode()

        # Cortical faculties reset
        if not retain_dynamics:
            self.reset()
        else:
            self.reset_episode(
                retain_dynamics=True,
                retain_inhibitions=is_same_context_retry,
            )

        view.acc_conflict_monitor.reset()

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
        """Export all internal episodic traces and SWR replays as causal predicates."""
        predicates = [self.to_causal_predicate()]
        predicates.extend(t.to_causal_predicate() for t in self.current_episode)
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


__all__ = [
    "HippocampalEpisodicCortex",
]
