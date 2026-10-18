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

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export fatal prefix as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"fatal_prefix:{self.terminal_step}",
            operator="==",
            value=str(self.fatal_action),
            relation=f"branch_point:{self.branch_point_step}",
            target_entity=f"pos:{self.fatal_pos}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording fatal prefix."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"fatal_prefix_{self.terminal_step}",
                node_data={
                    "terminal_step": self.terminal_step,
                    "fatal_pos": self.fatal_pos,
                    "fatal_action": str(self.fatal_action),
                    "branch_point_step": self.branch_point_step,
                    "inhibited_count": len(self.inhibited_state_actions),
                },
            )
        ]


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
        self.recent_actions: deque[Any] = deque(maxlen=64)

        # Negative transition archives (Preserved per-level across attempts)
        self.current_level: int = 0
        self.level_failed_transitions: dict[int, set[tuple[tuple[int, int], int]]] = {}

    @property
    def failed_transitions(self) -> set[tuple[tuple[int, int], int]]:
        """Failed (state, action) transitions recorded on current level."""
        return self.level_failed_transitions.setdefault(self.current_level, set())

    @failed_transitions.setter
    def failed_transitions(self, val: set[tuple[tuple[int, int], int]]) -> None:
        self.level_failed_transitions[self.current_level] = val

    @staticmethod
    def _make_action_token(action: Any, action_data: dict[str, Any] | None = None) -> Any:
        if (
            action_data
            and isinstance(action_data, dict)
            and "x" in action_data
            and "y" in action_data
        ):
            return (action, int(action_data["y"]), int(action_data["x"]))
        return action

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

        # Root counterfactual credit assignment:
        # In a failed attempt that died without level progress, the root decision branch at step 0
        # was insufficient. Inhibit that root choice so the subsequent life explores alternative branches.
        if trace_list:
            root = trace_list[0]
            root_tok = self._make_action_token(root.action_id, root.action_data)
            root_key = (root.avatar_pos, root_tok)
            inh_dur = max(self.inhibition_duration_steps * 2, 48)
            root_penalty = self.base_penalty * 0.4
            if root.avatar_pos is not None:
                self.active_inhibitions[root_key] = inh_dur
                self.active_inhibitions[(root.avatar_pos, root.action_id)] = inh_dur
                self.repulsion_table[root_key] = (
                    self.repulsion_table.get(root_key, 0.0) + root_penalty
                )
            else:
                self.active_inhibitions[(None, root_tok)] = inh_dur
                self.active_inhibitions[(None, root.action_id)] = inh_dur
                self.repulsion_table[(None, root_tok)] = (
                    self.repulsion_table.get((None, root_tok), 0.0) + root_penalty * 0.5
                )
            logger.info(
                "HabenularEpisodicIOR: Root action %s repulsed (key=%s, duration=%d steps, penalty=%.1f) for next life branch",
                root.action_id,
                root_key,
                inh_dur,
                root_penalty,
            )

        # Credit assignment: penalize actions closest to the terminal catastrophe
        for i in range(lookback):
            idx = T - 1 - i
            transition = trace_list[idx]
            temporal_decay = self.gamma**i
            penalty = self.base_penalty * temporal_decay

            act_tok = self._make_action_token(transition.action_id, transition.action_data)
            key = (transition.avatar_pos, act_tok)
            self.repulsion_table[key] += penalty

            # If within immediate fatal window (last 3 steps), install active motor inhibition
            if i <= 3:
                self.active_inhibitions[key] = self.inhibition_duration_steps
                inhibited_pairs.append(key)
                if transition.avatar_pos is None:
                    self.active_inhibitions[(None, act_tok)] = self.inhibition_duration_steps

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

    def record_ineffectual_action(
        self,
        avatar_pos: tuple[int, int] | None,
        action: Any,
        action_data: dict[str, Any] | None = None,
        streak: int = 1,
    ) -> None:
        """Habituation & action inhibition for ineffectual actions (e.g. wall bumps, zero-diff probes).

        Applies exponential backoff to action inhibition duration and increases repulsion penalty.
        """
        act_token = self._make_action_token(action, action_data)
        key = (avatar_pos, act_token)
        duration = min(64, 4 * (2 ** min(max(0, streak - 1), 4)))
        self.active_inhibitions[key] = duration
        self.repulsion_table[key] += self.base_penalty * 0.3 * max(1, streak)
        if avatar_pos is not None:
            self.active_inhibitions[(avatar_pos, action)] = duration
            self.repulsion_table[(avatar_pos, action)] += self.base_penalty * 0.3 * max(1, streak)
        else:
            self.active_inhibitions[(None, act_token)] = duration
            self.repulsion_table[(None, act_token)] += self.base_penalty * 0.3 * max(1, streak)

    def is_action_inhibited(
        self,
        avatar_pos: tuple[int, int] | None,
        action: Any,
        action_data: dict[str, Any] | None = None,
    ) -> bool:
        """Check if action is actively inhibited by habenular gating at this coordinate."""
        act_token = self._make_action_token(action, action_data)
        if self.active_inhibitions.get((avatar_pos, act_token), 0) > 0:
            return True
        if self.active_inhibitions.get((None, act_token), 0) > 0:
            return True
        if act_token != action:
            if self.active_inhibitions.get((avatar_pos, action), 0) > 0:
                return True
            if self.active_inhibitions.get((None, action), 0) > 0:
                return True
        # Check high persistent repulsion threshold from prior fatal episode traces
        if self.get_repulsion_penalty(avatar_pos, action, action_data) >= self.base_penalty * 0.9:
            return True
        return False

    def get_repulsion_penalty(
        self,
        avatar_pos: tuple[int, int] | None,
        action: Any,
        action_data: dict[str, Any] | None = None,
    ) -> float:
        """Return accumulated negative preference score for this state-action choice."""
        act_token = self._make_action_token(action, action_data)
        key = (avatar_pos, act_token)
        score = self.repulsion_table.get(key, 0.0)
        wildcard_key = (None, act_token)
        score += self.repulsion_table.get(wildcard_key, 0.0) * 0.5
        return score

    def detect_action_oscillation(self) -> Any | None:
        """Detect rapid periodic action oscillation (e.g. [A, B, A, B, A, B]) and return action to suppress."""
        cycle = self.detect_cyclic_actions()
        return cycle[0] if cycle else None

    def detect_cyclic_actions(self) -> list[Any]:
        """Detect 2-cycle or 3-cycle motor oscillations and return all cyclic actions involved."""
        if len(self.recent_actions) < 4:
            return []
        acts = list(self.recent_actions)
        cyclic_acts: list[Any] = []
        cycle_len = 0
        # 2-cycle check: [A, B, A, B] or longer
        if (
            len(acts) >= 4
            and acts[-1] == acts[-3]
            and acts[-2] == acts[-4]
            and acts[-1] != acts[-2]
        ):
            cyclic_acts = [acts[-1], acts[-2]]
            cycle_len = 2
        # 3-cycle check: [A, B, C, A, B, C]
        elif (
            len(acts) >= 6
            and acts[-1] == acts[-4]
            and acts[-2] == acts[-5]
            and acts[-3] == acts[-6]
            and len({acts[-1], acts[-2], acts[-3]}) > 1
        ):
            cyclic_acts = [acts[-1], acts[-2], acts[-3]]
            cycle_len = 3
        # 4-repetition check: [A, A, A, A]
        elif len(acts) >= 4 and len(set(acts[-4:])) == 1:
            cyclic_acts = [acts[-1]]
            cycle_len = 4

        if cyclic_acts and self.episode_trace:
            traces = list(self.episode_trace)
            # In embodied spatial navigation, an action repetition is only an oscillation
            # or stagnant loop if the organism returns to the same location or fails to move.
            # If there is net spatial progress, it is corridor or staircase traversal.
            if cycle_len == 2 and len(traces) >= 4:
                p_curr, p_prev = traces[-1].avatar_pos, traces[-3].avatar_pos
                if p_curr is not None and p_prev is not None and p_curr != p_prev:
                    return []
            elif cycle_len == 3 and len(traces) >= 6:
                p_curr, p_prev = traces[-1].avatar_pos, traces[-4].avatar_pos
                if p_curr is not None and p_prev is not None and p_curr != p_prev:
                    return []
            elif cycle_len == 4 and len(traces) >= 4:
                p_curr, p_prev = traces[-1].avatar_pos, traces[-4].avatar_pos
                if p_curr is not None and p_prev is not None and p_curr != p_prev:
                    return []

        if cyclic_acts:
            if cycle_len == 2:
                logger.info(
                    "HabenularEpisodicIOR: 2-cycle motor oscillation [%s, %s] detected!",
                    cyclic_acts[0],
                    cyclic_acts[1],
                )
            elif cycle_len == 3:
                logger.info(
                    "HabenularEpisodicIOR: 3-cycle motor oscillation [%s, %s, %s] detected!",
                    cyclic_acts[0],
                    cyclic_acts[1],
                    cyclic_acts[2],
                )
            elif cycle_len == 4:
                logger.info(
                    "HabenularEpisodicIOR: Stagnant repeated action %s detected!",
                    cyclic_acts[0],
                )
        return cyclic_acts

    def reset_episode(
        self, retain_long_term: bool = True, retain_inhibitions: bool = False
    ) -> None:
        """Reset transient episode traces while optionally preserving learned negative valence.

        Args:
            retain_long_term: If True, preserves repulsion table and fatal prefixes across episodes.
            retain_inhibitions: If True, preserves active inhibitions across retries for tree search.
        """
        self.episode_trace.clear()
        self.recent_actions.clear()
        if not retain_inhibitions:
            self.active_inhibitions.clear()
        if not retain_long_term:
            self.repulsion_table.clear()
            self.fatal_prefixes.clear()
            self.level_failed_transitions.clear()

    def reset(self) -> None:
        """Full reset of Habenular episodic memory and inhibition state."""
        self.reset_episode(retain_long_term=False)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export Habenular IOR state as a declarative causal predicate."""
        return CausalPredicate(
            variable="habenular_ior",
            operator="==",
            value=len(self.active_inhibitions),
            relation=f"fatal_prefixes:{len(self.fatal_prefixes)}",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all Habenular IOR state and fatal prefix causal predicates."""
        preds = [self.to_causal_predicate()]
        for rec in self.fatal_prefixes:
            preds.append(rec.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting Habenular IOR state."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="habenular_ior_state",
                node_data={
                    "active_inhibition_count": len(self.active_inhibitions),
                    "fatal_prefix_count": len(self.fatal_prefixes),
                    "trace_horizon": self.trace_horizon,
                    "gamma": self.gamma,
                },
            )
        ]
        for rec in self.fatal_prefixes:
            ops.extend(rec.to_transaction_operations())
        return ops
