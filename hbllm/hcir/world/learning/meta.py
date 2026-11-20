"""Meta-Learning Strategy Governor & Feature Signature Matching."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.causal.predicates import CausalPredicate

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GameFeatureSignature:
    """Feature signature of a game environment for strategy prediction."""

    action_space: frozenset[int] = field(default_factory=frozenset)
    grid_shape: tuple[int, int] = (0, 0)
    num_distinct_colors: int = 0
    has_spatial_effector_actions: bool = False
    has_special_actions: bool = False
    has_movement_actions: bool = False
    grid_density: float = 0.0

    @classmethod
    def from_observation(
        cls,
        grid: np.ndarray,
        available_actions: list[int] | set[int],
        bg_feature: int = 0,
        is_spatial_effector_fn: Any = None,
        is_displacement_fn: Any = None,
    ) -> GameFeatureSignature:
        """Extract feature signature from first observation in a domain-agnostic manner."""
        action_set = set(available_actions)
        distinct = len(set(int(x) for x in grid.flatten()))
        total = grid.size
        non_bg = int(np.count_nonzero(grid != bg_feature)) if total > 0 else 0
        g_shape = (int(grid.shape[0]), int(grid.shape[1])) if grid.ndim >= 2 else (0, 0)

        # Domain-agnostic detection of effector and movement capabilities
        has_effector = False
        has_movement = False
        if is_spatial_effector_fn is not None:
            has_effector = any(is_spatial_effector_fn(a) for a in action_set)
        if is_displacement_fn is not None:
            has_movement = any(is_displacement_fn(a) for a in action_set)
        else:
            has_movement = len(action_set) >= 4

        return cls(
            action_space=frozenset(action_set),
            grid_shape=g_shape,
            num_distinct_colors=distinct,
            has_spatial_effector_actions=has_effector,
            has_special_actions=len(action_set) > 4,
            has_movement_actions=has_movement,
            grid_density=non_bg / total if total > 0 else 0.0,
        )

    def similarity(self, other: GameFeatureSignature) -> float:
        """Compute similarity to another game's feature signature."""
        score = 0.0
        if self.action_space or other.action_space:
            intersection = len(self.action_space & other.action_space)
            union = len(self.action_space | other.action_space)
            score += 0.4 * (intersection / union if union > 0 else 0.0)
        if self.grid_shape == other.grid_shape:
            score += 0.2
        elif self.grid_shape[0] > 0 and other.grid_shape[0] > 0:
            ratio = min(self.grid_shape[0], other.grid_shape[0]) / max(
                self.grid_shape[0], other.grid_shape[0]
            )
            score += 0.2 * ratio
        if self.num_distinct_colors > 0 and other.num_distinct_colors > 0:
            ratio = min(self.num_distinct_colors, other.num_distinct_colors) / max(
                self.num_distinct_colors, other.num_distinct_colors
            )
            score += 0.2 * ratio
        if self.has_spatial_effector_actions == other.has_spatial_effector_actions:
            score += 0.1
        if self.has_movement_actions == other.has_movement_actions:
            score += 0.1
        return score


@dataclass
class StrategyOutcome:
    """Record of a strategy's performance on a specific context."""

    context_id: str
    features: GameFeatureSignature
    strategy_name: str
    levels_won: int
    total_levels: int
    time_seconds: float
    success_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.total_levels > 0:
            self.success_rate = self.levels_won / self.total_levels


class MetaLearningGovernor:
    """Metacognitive strategy governor that learns which strategies work best.

    After each context, records (features, strategy, outcome) in the meta tier.
    On new context start, predicts the best strategy allocation based on
    similarity to previously seen contexts.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.outcome_history: list[StrategyOutcome] = []
        self.strategy_success_rates: dict[str, list[float]] = defaultdict(list)
        self.storage_dir: Path = Path(storage_dir or "./knowledge_store")
        self.load_from_disk()

    def save_to_disk(self, storage_dir: str | Path | None = None) -> bool:
        """Atomically persist strategy outcome history to disk."""
        target_dir = Path(storage_dir or self.storage_dir)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / "meta_strategy_history.json"
            records = [
                {
                    "context_id": o.context_id,
                    "strategy_name": o.strategy_name,
                    "levels_won": o.levels_won,
                    "total_levels": o.total_levels,
                    "time_seconds": o.time_seconds,
                    "timestamp": o.timestamp,
                }
                for o in self.outcome_history
            ]
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            temp_path.replace(file_path)
            return True
        except Exception as e:
            logger.warning("Could not persist meta strategy outcomes: %s", e)
            return False

    def load_from_disk(self, storage_dir: str | Path | None = None) -> int:
        """Load previously recorded strategy outcomes from disk."""
        target_dir = Path(storage_dir or self.storage_dir)
        file_path = target_dir / "meta_strategy_history.json"
        if not file_path.is_file():
            return 0
        try:
            with open(file_path, encoding="utf-8") as f:
                records = json.load(f)
            loaded = 0
            for r in records:
                outcome = StrategyOutcome(
                    context_id=r.get("context_id", ""),
                    features=GameFeatureSignature(action_space=frozenset(), grid_shape=(0, 0)),
                    strategy_name=r["strategy_name"],
                    levels_won=r["levels_won"],
                    total_levels=r["total_levels"],
                    time_seconds=float(r.get("time_seconds", 0.0)),
                    timestamp=float(r.get("timestamp", 0.0)),
                )
                self.outcome_history.append(outcome)
                rate = (
                    outcome.levels_won / outcome.total_levels if outcome.total_levels > 0 else 0.0
                )
                self.strategy_success_rates[outcome.strategy_name].append(rate)
                loaded += 1
            return loaded
        except Exception as e:
            logger.warning("Could not load meta strategy outcomes: %s", e)
            return 0

    def record_outcome(self, outcome: StrategyOutcome) -> None:
        """Record a strategy outcome for future reference and persist to disk."""
        self.outcome_history.append(outcome)
        self.strategy_success_rates[outcome.strategy_name].append(outcome.success_rate)
        self.save_to_disk()
        logger.info(
            "MetaLearning: Recorded %s on %s: %.1f%% success (%d/%d levels)",
            outcome.strategy_name,
            outcome.context_id,
            outcome.success_rate * 100,
            outcome.levels_won,
            outcome.total_levels,
        )

    def predict_strategy_ranking(
        self,
        features: GameFeatureSignature,
        available_strategies: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Predict strategy effectiveness for a new game based on similarity to past games."""
        if not self.outcome_history:
            return []

        strategy_scores: dict[str, float] = defaultdict(float)
        strategy_weights: dict[str, float] = defaultdict(float)

        for outcome in self.outcome_history:
            sim = features.similarity(outcome.features)
            if sim < 0.1:
                continue
            weight = sim * sim
            strategy_scores[outcome.strategy_name] += weight * outcome.success_rate
            strategy_weights[outcome.strategy_name] += weight

        predictions: list[tuple[str, float]] = []
        for name, total_score in strategy_scores.items():
            weight = strategy_weights[name]
            predicted_rate = total_score / weight if weight > 0 else 0.0
            if available_strategies is None or name in available_strategies:
                predictions.append((name, predicted_rate))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions

    def allocate_time_budget(
        self,
        features: GameFeatureSignature,
        total_budget_seconds: float = 600.0,
        available_strategies: list[str] | None = None,
    ) -> dict[str, float]:
        """Allocate time budget across strategies proportional to predicted success."""
        ranking = self.predict_strategy_ranking(features, available_strategies)
        if not ranking:
            if available_strategies:
                equal = total_budget_seconds / len(available_strategies)
                return {s: equal for s in available_strategies}
            return {}

        total_score = sum(max(0.01, score) for _, score in ranking)
        return {
            name: total_budget_seconds * max(0.01, score) / total_score for name, score in ranking
        }

    def get_global_stats(self) -> dict[str, Any]:
        """Return aggregate statistics across all recorded outcomes."""
        if not self.outcome_history:
            return {"total_games": 0}
        total_levels_won = sum(o.levels_won for o in self.outcome_history)
        total_levels = sum(o.total_levels for o in self.outcome_history)
        return {
            "total_games": len(self.outcome_history),
            "total_levels_won": total_levels_won,
            "total_levels": total_levels,
            "overall_success_rate": total_levels_won / total_levels if total_levels > 0 else 0.0,
            "strategies_known": sorted(set(o.strategy_name for o in self.outcome_history)),
            "per_strategy_avg": {
                name: sum(rates) / len(rates) if rates else 0.0
                for name, rates in self.strategy_success_rates.items()
            },
        }

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export meta-learning state as formal causal predicates."""
        return [
            CausalPredicate(
                variable=f"meta_strategy:{name}",
                operator="==",
                value=f"avg_success:{sum(rates) / len(rates):.3f}:n_games:{len(rates)}",
                relation="meta_learning_governor",
                target_entity=name,
            )
            for name, rates in self.strategy_success_rates.items()
            if rates
        ]
