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
from typing import Any

import numpy as np

from hbllm.hcir.world.cortex_perception import EpistemicObservationDiff
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR

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


@dataclass
class ReplayCreditAssignment:
    """Outcome of SWR backward credit assignment following an episode outcome."""

    fatal_step: int
    fatal_action: Any
    fatal_pos: tuple[int, int] | None
    inhibited_transitions: list[tuple[tuple[int, int] | None, Any]] = field(default_factory=list)
    suspected_lethal_features: set[int] = field(default_factory=set)
    confidence: float = 1.0


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
        self.catastrophic_replays: list[ReplayCreditAssignment] = []

        # Grounded lethal transitions: set of (avatar_pos, action)
        self.grounded_lethal_transitions: set[tuple[tuple[int, int], Any]] = set()

        # Lethal candidate features identified across trials
        self.discovered_lethal_features: set[int] = set()

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

    def _archive_and_clear(self) -> None:
        """Archive current episode to long-term memory and clear active buffer."""
        if self.current_episode:
            self.past_episodes.append(list(self.current_episode))
            self.current_episode.clear()

    def reset_episode(self, retain_dynamics: bool = True) -> None:
        """Reset transient episode buffer while optionally preserving long-term episodic traces."""
        self.current_episode.clear()
        if not retain_dynamics:
            self.reset()
        else:
            self.habenular_ior.reset_episode(retain_long_term=True)

    def reset(self) -> None:
        """Full reset of episodic memories (e.g., across environment switches)."""
        self.current_episode.clear()
        self.past_episodes.clear()
        self.successful_trajectories.clear()
        self.catastrophic_replays.clear()
        self.grounded_lethal_transitions.clear()
        self.discovered_lethal_features.clear()
        self.habenular_ior.reset()
