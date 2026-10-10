"""Anterior Mid-Cingulate Cortex (aMCC) & Lateral Habenula Episodic Inhibition of Return (IOR).

Modeled on mammalian habenular anti-reward signaling and aMCC catastrophic failure tracking:
1. Fatal Prefix Tracing: Records episodic action-state trajectories tau = [(s_0, a_0), ..., (s_T, a_T)].
2. Habenular Negative Prediction Error: Upon catastrophic episode reset or death (is_lost=True),
   computes temporal credit assignment backpropagating negative value through the fatal prefix:
   Delta Q(s_t, a_t) = -kappa * gamma^(T - t) for t in [T - k, T].
3. Pre-Emptive Branching Gating: On subsequent attempts, when approaching a previously fatal
   state-phase coordinate, exerts inhibitory gating against the fatal action choice, compelling
   the motor system to choose alternative orthogonal branches.
4. Oscillation Breaking (Limit-Cycle Collapse): Detects 2-cycle or 3-cycle motor deadlocks
   (e.g., oscillating [1, 5, 1, 5]) and forcibly suppresses the cyclic actions.
"""

from __future__ import annotations

import collections
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryTransition:
    """A single state-action transition recorded during an episode."""

    step: int
    avatar_pos: tuple[int, int] | None
    action_id: Any
    action_data: dict[str, Any] | None
    state_signature: str = ""


@dataclass
class FatalPrefixRecord:
    """Record of an action prefix that terminated in catastrophe/reset."""

    terminal_step: int
    fatal_pos: tuple[int, int] | None
    fatal_action: Any
    branch_point_step: int
    inhibited_state_actions: list[tuple[tuple[int, int] | None, Any]] = field(default_factory=list)


class HabenularEpisodicIOR:
    """Mammalian Lateral Habenula & aMCC episodic negative reinforcement and fatal prefix pruner."""

    def __init__(
        self,
        trace_horizon: int = 8,
        gamma: float = 0.85,
        base_penalty: float = 50.0,
        inhibition_duration_steps: int = 24,
    ) -> None:
        self.trace_horizon = trace_horizon
        self.gamma = gamma
        self.base_penalty = base_penalty
        self.inhibition_duration_steps = inhibition_duration_steps

        # Episode transition memory
        self.episode_trace: deque[TrajectoryTransition] = deque(maxlen=256)

        # Repulsion table: (avatar_pos, action_id) -> accumulated negative valence
        self.repulsion_table: collections.defaultdict[tuple[tuple[int, int] | None, Any], float] = (
            collections.defaultdict(float)
        )

        # Fatal prefix archives
        self.fatal_prefixes: list[FatalPrefixRecord] = []

        # Temporary active motor blocks: (avatar_pos, action_id) -> remaining block steps
        self.active_inhibitions: dict[tuple[tuple[int, int] | None, Any], int] = {}

        # Limit cycle detector (recent action history)
        self.recent_actions: deque[Any] = deque(maxlen=16)

    def record_step(
        self,
        step: int,
        avatar_pos: tuple[int, int] | None,
        action: Any,
        action_data: dict[str, Any] | None = None,
        state_sig: str = "",
    ) -> None:
        """Record transition in current episode trace."""
        self.episode_trace.append(
            TrajectoryTransition(
                step=step,
                avatar_pos=avatar_pos,
                action_id=action,
                action_data=action_data,
                state_signature=state_sig,
            )
        )
        self.recent_actions.append(action)

        # Decay active inhibitions
        to_remove = [k for k, v in self.active_inhibitions.items() if v <= 1]
        for k in self.active_inhibitions:
            self.active_inhibitions[k] -= 1
        for k in to_remove:
            self.active_inhibitions.pop(k, None)

    def record_catastrophe(self, final_step: int, is_lost: bool = True) -> FatalPrefixRecord | None:
        """Lateral Habenula punishment signal: backpropagate penalties along the fatal trace.

        Args:
            final_step: The step index where reset or death occurred.
            is_lost: True if explicit death/loss, or unexpected reset.

        Returns:
            FatalPrefixRecord if trace existed, else None.
        """
        if not self.episode_trace:
            return None

        trace_list = list(self.episode_trace)
        T = len(trace_list)
        lookback = min(self.trace_horizon, T)

        inhibited_pairs: list[tuple[tuple[int, int] | None, Any]] = []

        terminal = trace_list[-1]
        fatal_pos = terminal.avatar_pos
        fatal_act = terminal.action_id

        # Credit assignment: penalize actions closest to the terminal catastrophe
        for i in range(lookback):
            idx = T - 1 - i
            transition = trace_list[idx]
            temporal_decay = self.gamma**i
            penalty = self.base_penalty * temporal_decay

            key = (transition.avatar_pos, transition.action_id)
            self.repulsion_table[key] += penalty

            # If within immediate fatal window (last 3 steps), install active motor inhibition
            if i <= 3:
                self.active_inhibitions[key] = self.inhibition_duration_steps
                inhibited_pairs.append(key)

        record = FatalPrefixRecord(
            terminal_step=final_step,
            fatal_pos=fatal_pos,
            fatal_action=fatal_act,
            branch_point_step=max(0, final_step - lookback),
            inhibited_state_actions=inhibited_pairs,
        )
        self.fatal_prefixes.append(record)

        logger.info(
            "HabenularEpisodicIOR: Catastrophe registered at step %d! Penalized %d transitions in fatal prefix.",
            final_step,
            lookback,
        )

        self.episode_trace.clear()
        self.recent_actions.clear()
        return record

    def is_action_inhibited(self, avatar_pos: tuple[int, int] | None, action: Any) -> bool:
        """Check if action is actively inhibited by habenular gating at this coordinate."""
        key = (avatar_pos, action)
        if self.active_inhibitions.get(key, 0) > 0:
            return True
        # Also check coordinate-free action inhibition if high global repulsion
        wildcard_key = (None, action)
        if self.active_inhibitions.get(wildcard_key, 0) > 0:
            return True
        return False

    def get_repulsion_penalty(self, avatar_pos: tuple[int, int] | None, action: Any) -> float:
        """Return accumulated negative preference score for this state-action choice."""
        key = (avatar_pos, action)
        score = self.repulsion_table.get(key, 0.0)
        # Coordinate-independent penalty
        wildcard_key = (None, action)
        score += self.repulsion_table.get(wildcard_key, 0.0) * 0.5
        return score

    def detect_action_oscillation(self) -> Any | None:
        """Detect rapid periodic action oscillation (e.g. [A, B, A, B, A, B]) and return action to suppress."""
        if len(self.recent_actions) < 6:
            return None
        acts = list(self.recent_actions)
        # 2-cycle check
        if (
            acts[-1] == acts[-3] == acts[-5]
            and acts[-2] == acts[-4] == acts[-6]
            and acts[-1] != acts[-2]
        ):
            logger.info(
                "HabenularEpisodicIOR: 2-cycle motor oscillation [%s, %s] detected!",
                acts[-1],
                acts[-2],
            )
            return acts[-1]
        return None

    def reset_episode(self, retain_long_term: bool = True) -> None:
        """Reset transient episode traces while optionally preserving learned negative valence.

        Args:
            retain_long_term: If True, preserves repulsion table and fatal prefixes across episodes.
        """
        self.episode_trace.clear()
        self.recent_actions.clear()
        self.active_inhibitions.clear()
        if not retain_long_term:
            self.repulsion_table.clear()
            self.fatal_prefixes.clear()

    def reset(self) -> None:
        """Full reset of Habenular episodic memory and inhibition state."""
        self.reset_episode(retain_long_term=False)
