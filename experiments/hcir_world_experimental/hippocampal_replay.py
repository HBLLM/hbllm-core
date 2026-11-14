"""Hippocampal-Entorhinal Prioritized Experience Replay Buffer.

Biologically grounded in mammalian hippocampal sharp-wave ripple (SWR) replay
dynamics (Foster & Wilson, 2006) and temporal-difference prioritized experience
replay (Schaul et al., 2015):
1. Prioritized Experience Retention:
   - Evaluates transition priority based on both reinforcement TD error (|delta_TD|)
     and Friston active inference predictive surprise (S_surp):
     p_i = (|delta_TD,i| + omega * S_surp,i + epsilon)^alpha.
2. Stochastic Prioritized Sampling & Importance-Sampling Weighting:
   - Proportional priority sampling: P(i) = p_i / sum_k p_k.
   - Computes non-parametric importance sampling weights:
     w_i = (N * P(i))^(-beta) / max_k w_k, with beta annealed toward 1.0.
3. Bidirectional Replay Sequences:
   - Backward SWR Replay: replays preceding transitions from a salient terminal
     event (reward/catastrophe) in reverse order for causal credit assignment.
   - Forward SWR Preplay: replays prospective rollout sequences from a cue state.
4. Schema Distillation:
   - Consolidates recurring, high-priority state-action transitions into schema
     candidates for persistent procedural memory.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReplayTransition:
    """An episodic state-action transition stored in hippocampal replay memory."""

    transition_id: str
    state: dict[str, Any]
    action: Any
    reward: float
    next_state: dict[str, Any]
    done: bool
    td_error: float = 0.0
    surprise: float = 0.0
    priority: float = 1.0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert transition to a JSON-serializable dictionary."""
        return {
            "transition_id": self.transition_id,
            "state": copy.deepcopy(self.state),
            "action": self.action,
            "reward": float(self.reward),
            "next_state": copy.deepcopy(self.next_state),
            "done": self.done,
            "td_error": float(self.td_error),
            "surprise": float(self.surprise),
            "priority": float(self.priority),
            "timestamp": float(self.timestamp),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ReplayBatch:
    """A sampled batch of transitions with normalized importance-sampling weights."""

    transitions: list[ReplayTransition]
    indices: list[int]
    weights: list[float]  # w_i in (0, 1] for bias correction

    def __len__(self) -> int:
        return len(self.transitions)


class HippocampalPrioritizedReplayBuffer:
    """Prioritized experience replay buffer modeling hippocampal SWR replay.

    Prioritizes transitions by surprise and TD-error with importance-sampling
    bias correction and bidirectional replay synthesis.
    """

    def __init__(
        self,
        capacity: int = 2000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_annealing_rate: float = 0.001,
        surprise_weight: float = 0.5,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize hippocampal prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to retain.
            alpha: Exponent determining priority magnitude (0 = uniform, 1 = full).
            beta_start: Initial importance-sampling weight exponent.
            beta_annealing_rate: Increment added to beta on each sampling step.
            surprise_weight: Relative weight of Friston surprise vs TD-error.
            epsilon: Small positive constant preventing zero priority.
        """
        self.capacity = max(10, capacity)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.beta = float(np.clip(beta_start, 0.0, 1.0))
        self.beta_annealing_rate = max(0.0, beta_annealing_rate)
        self.surprise_weight = max(0.0, surprise_weight)
        self.epsilon = max(1e-9, epsilon)

        self._storage: list[ReplayTransition] = []
        self._priorities: list[float] = []
        self._id_to_idx: dict[str, int] = {}
        self._max_priority: float = 1.0
        self._sample_count: int = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def is_empty(self) -> bool:
        """Return True if the buffer contains no transitions."""
        return len(self._storage) == 0

    def compute_priority(self, td_error: float, surprise: float) -> float:
        """Calculate scaled composite priority from TD-error and surprise.

        Formula: p = (|td_error| + surprise_weight * |surprise| + epsilon)^alpha
        """
        raw_error = abs(td_error) + self.surprise_weight * abs(surprise) + self.epsilon
        return float(raw_error**self.alpha)

    def add(
        self,
        state: dict[str, Any],
        action: Any,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
        td_error: float = 0.0,
        surprise: float = 0.0,
        timestamp: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ReplayTransition:
        """Add a state-action transition to the buffer.

        If the buffer is full, the transition with the lowest priority is evicted.
        """
        tid = f"trans_{uuid.uuid4().hex[:10]}"
        priority = self.compute_priority(td_error, surprise)
        self._max_priority = max(self._max_priority, priority)

        transition = ReplayTransition(
            transition_id=tid,
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=done,
            td_error=float(td_error),
            surprise=float(surprise),
            priority=priority,
            timestamp=float(timestamp),
            metadata=metadata or {},
        )

        if len(self._storage) < self.capacity:
            idx = len(self._storage)
            self._storage.append(transition)
            self._priorities.append(priority)
            self._id_to_idx[tid] = idx
        else:
            # Evict lowest-priority transition to preserve high-salience experiences
            min_idx = int(np.argmin(self._priorities))
            evicted_id = self._storage[min_idx].transition_id
            self._id_to_idx.pop(evicted_id, None)

            self._storage[min_idx] = transition
            self._priorities[min_idx] = priority
            self._id_to_idx[tid] = min_idx

        return transition

    def sample(self, batch_size: int, beta: float | None = None) -> ReplayBatch:
        """Sample a batch of transitions proportional to their priority.

        Args:
            batch_size: Number of transitions to sample.
            beta: Optional override for importance sampling exponent.

        Returns:
            ReplayBatch containing sampled transitions, indices, and normalized weights.
        """
        if self.is_empty:
            return ReplayBatch(transitions=[], indices=[], weights=[])

        n = len(self._storage)
        k = min(batch_size, n)

        if beta is None:
            eff_beta = min(1.0, self.beta + self._sample_count * self.beta_annealing_rate)
            self._sample_count += 1
        else:
            eff_beta = float(np.clip(beta, 0.0, 1.0))

        # Calculate sampling probabilities P(i) = p_i / sum(p)
        priorities_arr = np.array(self._priorities, dtype=np.float64)
        total_p = np.sum(priorities_arr)
        if total_p <= 0.0 or not np.isfinite(total_p):
            probs = np.full(n, 1.0 / n)
        else:
            probs = priorities_arr / total_p

        sampled_indices = np.random.choice(n, size=k, replace=False, p=probs)

        # Compute importance sampling weights: w_i = (N * P(i))^(-beta)
        sampled_probs = probs[sampled_indices]
        weights = (n * sampled_probs) ** (-eff_beta)
        max_w = np.max(weights) if len(weights) > 0 and np.max(weights) > 0 else 1.0
        normalized_weights = (weights / max_w).tolist()

        sampled_transitions = [self._storage[idx] for idx in sampled_indices]

        return ReplayBatch(
            transitions=sampled_transitions,
            indices=sampled_indices.tolist(),
            weights=normalized_weights,
        )

    def update_priorities(
        self,
        transition_ids: list[str],
        new_td_errors: list[float],
        new_surprises: list[float] | None = None,
    ) -> None:
        """Update priorities for sampled transitions following learning/simulation."""
        for i, tid in enumerate(transition_ids):
            if tid not in self._id_to_idx:
                continue
            idx = self._id_to_idx[tid]
            td = new_td_errors[i] if i < len(new_td_errors) else 0.0
            surp = (
                new_surprises[i]
                if (new_surprises is not None and i < len(new_surprises))
                else self._storage[idx].surprise
            )

            p = self.compute_priority(td, surp)
            self._priorities[idx] = p
            self._storage[idx].td_error = float(td)
            self._storage[idx].surprise = float(surp)
            self._storage[idx].priority = p
            self._max_priority = max(self._max_priority, p)

    def generate_reverse_replay(
        self,
        terminal_id: str,
        max_depth: int = 10,
    ) -> list[ReplayTransition]:
        """Generate reverse hippocampal replay sequence leading to a terminal transition.

        Traces backward through temporal contiguous transitions ending at terminal_id.
        """
        if terminal_id not in self._id_to_idx:
            return []

        end_idx = self._id_to_idx[terminal_id]
        start_idx = max(0, end_idx - max_depth + 1)

        # Reverse slice from end_idx down to start_idx
        slice_seq = self._storage[start_idx : end_idx + 1]
        return list(reversed(slice_seq))

    def generate_forward_preplay(
        self,
        start_id: str,
        max_depth: int = 10,
    ) -> list[ReplayTransition]:
        """Generate forward prospective rollout preplay from a starting transition."""
        if start_id not in self._id_to_idx:
            return []

        start_idx = self._id_to_idx[start_id]
        end_idx = min(len(self._storage), start_idx + max_depth)
        return list(self._storage[start_idx:end_idx])

    def distill_high_priority_schemas(
        self,
        min_priority_percentile: float = 75.0,
    ) -> list[dict[str, Any]]:
        """Identify recurrent state-action motifs among high-priority transitions.

        Returns condensed schemas suitable for procedural skill induction.
        """
        if not self._storage:
            return []

        threshold = float(np.percentile(self._priorities, min_priority_percentile))
        high_salience = [t for t in self._storage if t.priority >= threshold]

        schemas: list[dict[str, Any]] = []
        action_counts: dict[Any, int] = {}
        action_rewards: dict[Any, float] = {}

        for t in high_salience:
            act = t.action
            action_counts[act] = action_counts.get(act, 0) + 1
            action_rewards[act] = action_rewards.get(act, 0.0) + t.reward

        for act, count in action_counts.items():
            if count >= 2:
                avg_r = action_rewards[act] / count
                schemas.append(
                    {
                        "action": act,
                        "support": count,
                        "average_reward": float(avg_r),
                        "salience_rank": float(count * avg_r),
                    }
                )

        schemas.sort(key=lambda s: s["salience_rank"], reverse=True)
        return schemas

    def clear(self) -> None:
        """Clear all transitions and reset priority indices."""
        self._storage.clear()
        self._priorities.clear()
        self._id_to_idx.clear()
        self._max_priority = 1.0
        self._sample_count = 0
