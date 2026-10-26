"""Autonomous Learning Integration — Phases B, C, D.

Connects the existing HCIR cognitive faculties into a closed self-improving loop:

Phase B: Causal Theory → Active Planning
    - CausalTheoryPlanner: Converts discovered CausalPredicates and action dynamics
      into dynamic A* planning heuristics (no hand-coded solvers needed).

Phase C: Cross-Game Knowledge Transfer
    - CrossGameKnowledgeTransfer: Promotes universal causal invariants to
      WorkspaceTier.PERSISTENT on game end; seeds new games with prior knowledge.

Phase D: Meta-Learning Strategy Governor
    - MetaLearningGovernor: Tracks (game_features, strategy, outcome) tuples in
      WorkspaceTier.META; predicts best strategy for new games.

Conforms to HBLLM invariants: LLM-free, deterministic, state via transactions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Causal Theory → Active Planning
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CausalPlanningContext:
    """Frozen planning context derived from the symbolic world theory.

    Computed once per planning cycle, consumed by the A* heuristic
    without re-querying the live causal theory mid-search.
    """

    goal_positions: set[tuple[int, int]]
    barrier_positions: set[tuple[int, int]]
    lethal_features: set[int]
    goal_features: set[int]
    barrier_features: set[int]
    walkable_features: set[int]
    action_displacements: dict[int, tuple[int, int]]  # action_id -> (dr, dc)
    grid_shape: tuple[int, int] = (0, 0)


class CausalTheoryPlanner:
    """Constructs dynamic A* heuristics from discovered CausalPredicates.

    Instead of hand-coding which features are goals/barriers/hazards, this
    planner reads the symbolic world theory and action dynamics to construct
    a grid-aware planning heuristic dynamically.
    """

    @staticmethod
    def extract_planning_context(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> CausalPlanningContext:
        """Extract a frozen planning context from the engine's current causal theory."""
        theory = engine.symbolic_theory

        # Extract goal positions from grid using discovered goal features
        goal_positions: set[tuple[int, int]] = set(engine.learned_goal_positions)
        barrier_positions: set[tuple[int, int]] = set(engine.learned_barriers)

        # Scan grid for goal and barrier cells
        for r in range(curr_grid.shape[0]):
            for c in range(curr_grid.shape[1]):
                feat = int(curr_grid[r, c])
                if feat in theory.goal_features:
                    goal_positions.add((r, c))
                if feat in theory.barrier_features:
                    barrier_positions.add((r, c))

        # Extract action displacements from empirical action dynamics
        action_displacements: dict[int, tuple[int, int]] = {}
        for act_id, dyn in engine.action_dynamics.items():
            if dyn.is_displacement_action():
                action_displacements[act_id] = (dyn.delta_r, dyn.delta_c)

        # Extract lethal features from episodic memory
        lethal_features = set(engine.episodic_cortex.discovered_lethal_features)

        return CausalPlanningContext(
            goal_positions=goal_positions,
            barrier_positions=barrier_positions,
            lethal_features=lethal_features,
            goal_features=set(theory.goal_features),
            barrier_features=set(theory.barrier_features),
            walkable_features=set(theory.walkable_features),
            action_displacements=action_displacements,
            grid_shape=(int(curr_grid.shape[0]), int(curr_grid.shape[1])),
        )

    @staticmethod
    def compute_causal_heuristic(
        pos: tuple[int, int],
        context: CausalPlanningContext,
    ) -> float:
        """Manhattan distance heuristic to nearest goal, with barrier penalty.

        Uses discovered causal knowledge rather than hard-coded features.
        """
        if not context.goal_positions:
            return 0.0
        min_dist = min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in context.goal_positions)
        # Penalty if position is adjacent to lethal/barrier cells
        penalty = 0.0
        if pos in context.barrier_positions:
            penalty += 100.0
        return float(min_dist) + penalty

    @staticmethod
    def is_position_safe(
        pos: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
    ) -> bool:
        """Check if a position is safe to traverse based on causal theory."""
        r, c = pos
        if r < 0 or c < 0 or r >= context.grid_shape[0] or c >= context.grid_shape[1]:
            return False
        feat = int(grid[r, c])
        if feat in context.lethal_features:
            return False
        if pos in context.barrier_positions:
            return False
        if feat in context.barrier_features:
            return False
        return True

    @staticmethod
    def plan_path(
        start: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
        max_expansions: int = 500,
    ) -> list[int] | None:
        """A* pathfinding using causally-discovered action dynamics and hazards.

        Returns a list of action_ids forming a path from start to nearest goal,
        or None if no path found.
        """
        if not context.goal_positions or not context.action_displacements:
            return None

        import heapq

        # Priority queue: (f_score, g_score, position, action_path)
        start_h = CausalTheoryPlanner.compute_causal_heuristic(start, context)
        open_set: list[tuple[float, int, tuple[int, int], list[int]]] = [(start_h, 0, start, [])]
        visited: set[tuple[int, int]] = set()
        expansions = 0

        while open_set and expansions < max_expansions:
            f, g, pos, path = heapq.heappop(open_set)
            if pos in visited:
                continue
            visited.add(pos)
            expansions += 1

            # Check if we reached a goal
            if pos in context.goal_positions:
                logger.info(
                    "CausalTheoryPlanner: Found path of %d actions (%d nodes expanded)",
                    len(path),
                    expansions,
                )
                return path

            # Expand neighbors using discovered action dynamics
            for act_id, (dr, dc) in context.action_displacements.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                next_pos = (nr, nc)
                if next_pos in visited:
                    continue
                if not CausalTheoryPlanner.is_position_safe(next_pos, grid, context):
                    continue
                new_g = g + 1
                new_h = CausalTheoryPlanner.compute_causal_heuristic(next_pos, context)
                new_f = new_g + new_h
                new_path = path + [act_id]
                heapq.heappush(open_set, (new_f, new_g, next_pos, new_path))

        return None


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: Cross-Game Knowledge Transfer
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TransferableInvariant:
    """A causal invariant abstracted for cross-game transfer.

    Stripped of position-specific data, retaining only the structural
    pattern that might generalize across environments.
    """

    invariant_id: str
    category: str  # e.g., "goal_feature", "barrier_feature", "action_dynamics", "hazard"
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    source_game: str = ""
    confidence: float = 1.0
    transfer_successes: int = 0
    transfer_failures: int = 0

    def decay_confidence(self) -> None:
        """Decay confidence when transfer fails."""
        self.transfer_failures += 1
        self.confidence = max(0.1, self.confidence * 0.7)

    def boost_confidence(self) -> None:
        """Boost confidence when transfer succeeds."""
        self.transfer_successes += 1
        self.confidence = min(1.0, self.confidence * 1.1 + 0.05)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable=f"transfer_invariant:{self.invariant_id}",
            operator="==",
            value=f"cat:{self.category}:conf:{self.confidence:.2f}",
            relation=f"source:{self.source_game}",
            target_entity=self.invariant_id,
        )


class CrossGameKnowledgeTransfer:
    """Manages extraction, promotion, and retrieval of cross-game knowledge.

    On game completion:
        1. Extracts universal causal invariants from the symbolic theory
        2. Abstracts position-specific data into transferable patterns
        3. Stores in persistent knowledge base

    On new game start:
        1. Queries persistent knowledge for applicable priors
        2. Seeds the engine's symbolic theory with transferable invariants
        3. Tracks transfer success/failure for confidence updating
    """

    def __init__(self) -> None:
        self.persistent_invariants: dict[str, TransferableInvariant] = {}
        self.active_transfers: dict[str, TransferableInvariant] = {}

    def extract_invariants(
        self,
        engine: AutonomousEpistemicEngine,
        game_id: str = "",
    ) -> list[TransferableInvariant]:
        """Extract universal causal invariants from current game's learned knowledge."""
        invariants: list[TransferableInvariant] = []
        theory = engine.symbolic_theory

        # Extract action dynamics (these often transfer across games with same action space)
        for act_id, dyn in engine.action_dynamics.items():
            if dyn.is_displacement_action() and dyn.probes_tested >= 3:
                inv_id = f"action_dynamics_{act_id}_{dyn.delta_r}_{dyn.delta_c}"
                invariants.append(
                    TransferableInvariant(
                        invariant_id=inv_id,
                        category="action_dynamics",
                        description=f"Action {act_id} displaces avatar by ({dyn.delta_r}, {dyn.delta_c})",
                        data={"action_id": act_id, "delta_r": dyn.delta_r, "delta_c": dyn.delta_c},
                        source_game=game_id,
                    )
                )

        # Extract lethal features (contact hazards)
        for feat in engine.episodic_cortex.discovered_lethal_features:
            inv_id = f"lethal_feature_{feat}"
            invariants.append(
                TransferableInvariant(
                    invariant_id=inv_id,
                    category="hazard",
                    description=f"Feature {feat} is lethal upon contact",
                    data={"feature": feat},
                    source_game=game_id,
                )
            )

        # Extract tool-barrier affinities (key-door relationships)
        for tool_feat, barrier_feats in theory.tool_barrier_affinities.items():
            for barrier_feat in barrier_feats:
                inv_id = f"tool_barrier_{tool_feat}_{barrier_feat}"
                invariants.append(
                    TransferableInvariant(
                        invariant_id=inv_id,
                        category="tool_barrier_affinity",
                        description=f"Feature {tool_feat} clears barrier feature {barrier_feat}",
                        data={"tool_feature": tool_feat, "barrier_feature": barrier_feat},
                        source_game=game_id,
                    )
                )

        return invariants

    def promote_to_persistent(self, invariants: list[TransferableInvariant]) -> int:
        """Promote extracted invariants to persistent cross-game knowledge store."""
        new_count = 0
        for inv in invariants:
            if inv.invariant_id in self.persistent_invariants:
                self.persistent_invariants[inv.invariant_id].boost_confidence()
            else:
                self.persistent_invariants[inv.invariant_id] = inv
                new_count += 1
        logger.info(
            "CrossGameTransfer: Promoted %d invariants (%d new, %d boosted)",
            len(invariants),
            new_count,
            len(invariants) - new_count,
        )
        return new_count

    def query_applicable_priors(
        self,
        available_actions: set[int],
        min_confidence: float = 0.3,
    ) -> list[TransferableInvariant]:
        """Query persistent knowledge for invariants applicable to a new game."""
        applicable: list[TransferableInvariant] = []
        for inv in self.persistent_invariants.values():
            if inv.confidence < min_confidence:
                continue
            if inv.category == "action_dynamics":
                act_id = inv.data.get("action_id")
                if act_id is not None and act_id not in available_actions:
                    continue
            applicable.append(inv)
        applicable.sort(key=lambda x: x.confidence, reverse=True)
        return applicable

    def seed_engine_with_priors(
        self,
        engine: AutonomousEpistemicEngine,
        available_actions: set[int],
    ) -> int:
        """Seed a new game's engine with applicable prior knowledge."""
        priors = self.query_applicable_priors(available_actions)
        applied = 0
        self.active_transfers.clear()

        for inv in priors:
            if inv.category == "action_dynamics":
                act_id = inv.data.get("action_id")
                if act_id is not None and act_id in available_actions:
                    if act_id not in engine.action_dynamics:
                        from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

                        engine.action_dynamics[act_id] = ActionDynamicsModel(
                            action_id=act_id,
                            delta_r=int(inv.data.get("delta_r", 0)),
                            delta_c=int(inv.data.get("delta_c", 0)),
                            confidence=inv.confidence,
                            probes_tested=1,
                        )
                        self.active_transfers[inv.invariant_id] = inv
                        applied += 1
            elif inv.category == "hazard":
                feat = inv.data.get("feature")
                if feat is not None:
                    engine.episodic_cortex.discovered_lethal_features.add(feat)
                    self.active_transfers[inv.invariant_id] = inv
                    applied += 1
            elif inv.category == "tool_barrier_affinity":
                tool_feat = inv.data.get("tool_feature")
                barrier_feat = inv.data.get("barrier_feature")
                if tool_feat is not None and barrier_feat is not None:
                    engine.symbolic_theory.record_tool_barrier_affinity(tool_feat, barrier_feat)
                    self.active_transfers[inv.invariant_id] = inv
                    applied += 1

        if applied > 0:
            logger.info("CrossGameTransfer: Seeded engine with %d prior invariants", applied)
        return applied

    def record_transfer_outcome(self, game_won_any_levels: bool) -> None:
        """Update confidence of actively transferred invariants based on game outcome."""
        for inv in self.active_transfers.values():
            if game_won_any_levels:
                inv.boost_confidence()
            else:
                inv.decay_confidence()
        self.active_transfers.clear()

    def on_game_complete(
        self,
        engine: AutonomousEpistemicEngine,
        game_id: str,
        levels_won: int,
    ) -> None:
        """Full lifecycle hook: extract, promote, and record outcome."""
        invariants = self.extract_invariants(engine, game_id=game_id)
        self.promote_to_persistent(invariants)
        self.record_transfer_outcome(levels_won > 0)

    def on_game_start(
        self,
        engine: AutonomousEpistemicEngine,
        available_actions: set[int],
    ) -> int:
        """Full lifecycle hook: query priors and seed engine."""
        return self.seed_engine_with_priors(engine, available_actions)


# ═══════════════════════════════════════════════════════════════════════════
# Phase D: Meta-Learning Strategy Governor
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GameFeatureSignature:
    """Feature signature of a game environment for strategy prediction."""

    action_space: frozenset[int] = field(default_factory=frozenset)
    grid_shape: tuple[int, int] = (0, 0)
    num_distinct_colors: int = 0
    has_click_action: bool = False
    has_special_actions: bool = False
    has_movement_actions: bool = False
    grid_density: float = 0.0

    @classmethod
    def from_observation(
        cls,
        grid: np.ndarray,
        available_actions: list[int] | set[int],
        bg_feature: int = 0,
    ) -> GameFeatureSignature:
        """Extract feature signature from first observation."""
        action_set = set(available_actions)
        distinct = len(set(int(x) for x in grid.flatten()))
        total = grid.size
        non_bg = int(np.count_nonzero(grid != bg_feature)) if total > 0 else 0
        g_shape = (int(grid.shape[0]), int(grid.shape[1])) if grid.ndim >= 2 else (0, 0)
        return cls(
            action_space=frozenset(action_set),
            grid_shape=g_shape,
            num_distinct_colors=distinct,
            has_click_action=6 in action_set,
            has_special_actions=any(a >= 5 for a in action_set),
            has_movement_actions=bool(action_set & {1, 2, 3, 4}),
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
        if self.has_click_action == other.has_click_action:
            score += 0.1
        if self.has_movement_actions == other.has_movement_actions:
            score += 0.1
        return score


@dataclass
class StrategyOutcome:
    """Record of a strategy's performance on a specific game."""

    game_id: str
    features: GameFeatureSignature
    strategy_name: str
    levels_won: int
    total_levels: int
    time_seconds: float
    success_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.total_levels > 0:
            self.success_rate = self.levels_won / self.total_levels


class MetaLearningGovernor:
    """Metacognitive strategy governor that learns which strategies work best.

    After each game, records (features, strategy, outcome) in the meta tier.
    On new game start, predicts the best strategy allocation based on
    similarity to previously seen games.
    """

    def __init__(self) -> None:
        self.outcome_history: list[StrategyOutcome] = []
        self.strategy_success_rates: dict[str, list[float]] = defaultdict(list)

    def record_outcome(self, outcome: StrategyOutcome) -> None:
        """Record a strategy outcome for future reference."""
        self.outcome_history.append(outcome)
        self.strategy_success_rates[outcome.strategy_name].append(outcome.success_rate)
        logger.info(
            "MetaLearning: Recorded %s on %s: %.1f%% success (%d/%d levels)",
            outcome.strategy_name,
            outcome.game_id,
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
