"""
Stochastic Environment Wrapper and Simulator.

Injects controlled actuator slips, sensory dropouts, and dynamic perturbations
to stress-test epistemic belief maintenance and surprise recovery.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .types import StochasticAction, StochasticObservation, StochasticTier

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = {
    StochasticAction.UP: (-1, 0),
    StochasticAction.DOWN: (1, 0),
    StochasticAction.LEFT: (0, -1),
    StochasticAction.RIGHT: (0, 1),
    StochasticAction.NOOP: (0, 0),
}


class StandaloneStochasticEnv:
    """Configurable grid environment with sensorimotor noise."""

    def __init__(
        self,
        tier: StochasticTier | str = StochasticTier.TIER_1_DETERMINISTIC_BASELINE,
        seed: int = 42,
        max_steps: int = 100,
    ) -> None:
        self.tier = StochasticTier(tier)
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.step_count = 0

        self.width = 9
        self.height = 9
        self.true_player_pos = (1, 1)
        self.true_target_pos = (7, 7)
        self.obstacles: set[tuple[int, int]] = set()

        self._configure_tier_parameters()
        self.reset(seed=seed)

    def _configure_tier_parameters(self) -> None:
        """Set slip and dropout probabilities based on tier."""
        if self.tier == StochasticTier.TIER_1_DETERMINISTIC_BASELINE:
            self.slip_prob = 0.0
            self.dropout_prob = 0.0
        elif self.tier == StochasticTier.TIER_2_ACTUATOR_SLIP:
            self.slip_prob = 0.15
            self.dropout_prob = 0.0
        elif self.tier == StochasticTier.TIER_3_SENSORY_DROPOUT:
            self.slip_prob = 0.0
            self.dropout_prob = 0.25
        elif self.tier == StochasticTier.TIER_4_COMPOUND_PERTURBATION:
            self.slip_prob = 0.20
            self.dropout_prob = 0.20
        elif self.tier == StochasticTier.TIER_5_DYNAMIC_DRIFT:
            self.slip_prob = 0.25
            self.dropout_prob = 0.20

    def reset(self, seed: int | None = None) -> StochasticObservation:
        """Reset environment to initial state."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        self.step_count = 0
        self.true_player_pos = (1, 1)
        self.true_target_pos = (7, 7)

        # Place central obstacles
        self.obstacles = {
            (3, 3),
            (3, 4),
            (3, 5),
            (5, 3),
            (5, 4),
            (5, 5),
        }

        return self._get_obs(was_slipped=False, was_occluded=False)

    def step(
        self,
        action: StochasticAction | int,
    ) -> tuple[StochasticObservation, float, bool, dict[str, Any]]:
        """Apply action with stochastic slip dynamics."""
        self.step_count += 1
        act = StochasticAction(action)

        was_slipped = False
        if self.rng.random() < self.slip_prob:
            was_slipped = True
            # Slip: 50% chance action drops to NOOP, 50% chance slight orthogonal deflection
            if self.rng.random() < 0.5:
                act = StochasticAction.NOOP
            else:
                act = self.rng.choice(
                    [
                        StochasticAction.UP,
                        StochasticAction.DOWN,
                        StochasticAction.LEFT,
                        StochasticAction.RIGHT,
                    ]
                )

        dr, dc = DIRECTION_DELTAS[act]
        pr, pc = self.true_player_pos
        nr, nc = pr + dr, pc + dc

        # Check boundaries and obstacles
        if 0 <= nr < self.height and 0 <= nc < self.width:
            if (nr, nc) not in self.obstacles:
                self.true_player_pos = (nr, nc)

        # Dynamic drift logic for Tier 5: target moves slightly every 15 steps if unreached
        if self.tier == StochasticTier.TIER_5_DYNAMIC_DRIFT and self.step_count % 15 == 0:
            tr, tc = self.true_target_pos
            candidates = [
                (tr + dr, tc + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 1 <= tr + dr < self.height - 1
                and 1 <= tc + dc < self.width - 1
                and (tr + dr, tc + dc) not in self.obstacles
            ]
            if candidates:
                self.true_target_pos = self.rng.choice(candidates)

        won = self.true_player_pos == self.true_target_pos
        done = won or self.step_count >= self.max_steps
        reward = 10.0 if won else -0.01

        was_occluded = self.rng.random() < self.dropout_prob

        info = {
            "won": won,
            "was_slipped": was_slipped,
            "was_occluded": was_occluded,
            "steps": self.step_count,
        }

        obs = self._get_obs(
            done=done, won=won, was_slipped=was_slipped, was_occluded=was_occluded, info=info
        )
        return obs, reward, done, info

    def _get_obs(
        self,
        done: bool = False,
        won: bool = False,
        was_slipped: bool = False,
        was_occluded: bool = False,
        info: dict[str, Any] | None = None,
    ) -> StochasticObservation:
        """Construct observation, simulating perceptual occlusion when active."""
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for r, c in self.obstacles:
            grid[r][c] = 1

        observed_target: tuple[int, int] | None = self.true_target_pos
        if was_occluded:
            # Target disappears from sensor frame this step
            observed_target = None
        else:
            tr, tc = self.true_target_pos
            grid[tr][tc] = 2

        pr, pc = self.true_player_pos
        grid[pr][pc] = 3

        return StochasticObservation(
            grid=grid,
            player_pos=self.true_player_pos,
            target_pos=observed_target,
            obstacles=sorted(list(self.obstacles)),
            step_count=self.step_count,
            max_steps=self.max_steps,
            done=done,
            won=won,
            was_slipped=was_slipped,
            was_occluded=was_occluded,
            info=info or {},
        )


def make_stochastic_env(
    tier: StochasticTier | str = StochasticTier.TIER_1_DETERMINISTIC_BASELINE,
    seed: int = 42,
    max_steps: int = 100,
) -> StandaloneStochasticEnv:
    """Factory creating stochastic test environments."""
    return StandaloneStochasticEnv(tier=tier, seed=seed, max_steps=max_steps)
