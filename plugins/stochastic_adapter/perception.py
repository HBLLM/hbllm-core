"""
Stochastic Perception Adapter.

Provides epistemic belief maintenance, object permanence filtering under sensory
dropout, and sensorimotor surprise detection for HBLLM CognitiveGraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import EpistemicEntityBelief, StochasticObservation

logger = logging.getLogger(__name__)


class StochasticPerceptionAdapter:
    """Maintains epistemic state tracking and surprise detection under noise."""

    def __init__(self) -> None:
        self.target_belief: EpistemicEntityBelief | None = None
        self.expected_player_pos: tuple[int, int] | None = None
        self.known_obstacles: set[tuple[int, int]] = set()

    def reset(self) -> None:
        """Reset belief tracking."""
        self.target_belief = None
        self.expected_player_pos = None
        self.known_obstacles.clear()

    def register_expected_transition(self, expected_pos: tuple[int, int]) -> None:
        """Register forward model expectation for next step."""
        self.expected_player_pos = expected_pos

    def process_observation(self, obs: StochasticObservation) -> dict[str, Any]:
        """Ingest noisy observation and perform epistemic belief update."""
        surprise_detected = False

        # Sensorimotor surprise check: Did agent land where predicted?
        if self.expected_player_pos is not None:
            if obs.player_pos != self.expected_player_pos:
                surprise_detected = True

        # Accumulate obstacle knowledge
        for obs_pos in obs.obstacles:
            self.known_obstacles.add(obs_pos)

        # Epistemic object permanence filtering for target
        if obs.target_pos is not None:
            self.target_belief = EpistemicEntityBelief(
                entity_id="target_goal",
                estimated_pos=obs.target_pos,
                confidence=1.0,
                last_seen_step=obs.step_count,
                occluded=False,
            )
        elif self.target_belief is not None:
            # Maintain object permanence with decaying confidence
            steps_since = obs.step_count - self.target_belief.last_seen_step
            decayed_conf = max(0.2, 1.0 - (0.05 * steps_since))
            self.target_belief.confidence = decayed_conf
            self.target_belief.occluded = True

        return {
            "player_pos": obs.player_pos,
            "target_belief": self.target_belief,
            "target_pos": self.target_belief.estimated_pos if self.target_belief else None,
            "obstacles": set(self.known_obstacles),
            "surprise_detected": surprise_detected,
            "was_slipped": obs.was_slipped,
            "was_occluded": obs.was_occluded,
            "grid": obs.grid,
            "step_count": obs.step_count,
        }
