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


def get_sub_attr(target: Any, attr: str, default: Any = None) -> Any:
    return getattr(target, attr, default) if target is not None else default


import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class RelationalWaypoint:
    """A semantic milestone entity along a winning trajectory."""

    role: str  # "tool", "key", "barrier", "goal", "waypoint"
    feature: int
    target_pos: tuple[int, int] = (0, 0)
    relative_step: int = 0


@dataclass
class RelationalSubgoalMacro:
    """A coordinate-invariant relational macro composed of waypoint subgoals."""

    macro_id: str
    waypoints: list[RelationalWaypoint]
    action_space: frozenset[int]
    confidence: float = 1.0
    source_level: int = 0
    replay_successes: int = 0
    replay_failures: int = 0

    def matches_grid_features(self, grid: np.ndarray) -> bool:
        """Check if all required waypoint features are present in the target grid."""
        present = set(int(x) for x in grid.flat)
        return all(wp.feature in present for wp in self.waypoints)


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

        # Fallback to candidate goal features if no concrete goal cells located
        if not goal_positions and theory.candidate_goal_features:
            for r in range(curr_grid.shape[0]):
                for c in range(curr_grid.shape[1]):
                    feat = int(curr_grid[r, c])
                    if feat in theory.candidate_goal_features:
                        goal_positions.add((r, c))

        # Fallback to structural perception goals if still empty
        if not goal_positions and hasattr(engine, "detect_structural_goals"):
            try:
                struct_goals = engine.detect_structural_goals(curr_grid)
                for sg in struct_goals:
                    if sg.get("cells"):
                        for cp in sg["cells"]:
                            if cp != engine.avatar_pos:
                                goal_positions.add(cp)
                    elif sg.get("position") and sg["position"] != engine.avatar_pos:
                        goal_positions.add(sg["position"])
                    if sg.get("feature") is not None:
                        theory.candidate_goal_features.add(int(sg["feature"]))
            except Exception:
                pass

        # Discard avatar position and exhausted goals
        if engine.avatar_pos is not None:
            goal_positions.discard(engine.avatar_pos)
        exhausted = get_sub_attr(engine.working_memory, "exhausted_candidate_goals", set())
        if exhausted and len(goal_positions - exhausted) > 0:
            goal_positions.difference_update(exhausted)

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
        if pos in context.barrier_positions and pos not in context.goal_positions:
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
        if pos not in context.goal_positions:
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

    @staticmethod
    def plan_chained_path(
        start: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
        tool_barrier_affinities: dict[int, set[int]],
        max_expansions: int = 500,
    ) -> list[int] | None:
        """Synthesize a multi-stage prerequisite plan (e.g. acquire tool -> unlock barrier -> reach goal).

        Recursively backchains when direct path is blocked by barriers.
        """
        direct = CausalTheoryPlanner.plan_path(start, grid, context, max_expansions=max_expansions)
        if direct is not None:
            return direct

        if not tool_barrier_affinities or not context.goal_positions:
            return None

        H, W = context.grid_shape
        if H <= 0 or W <= 0:
            H, W = grid.shape[:2]

        for tool_feat, barrier_feats in tool_barrier_affinities.items():
            relevant_barriers = barrier_feats & context.barrier_features
            if not relevant_barriers and not any(
                0 <= r < H and 0 <= c < W and int(grid[r, c]) in barrier_feats
                for r, c in context.barrier_positions
            ):
                continue

            tool_positions: list[tuple[int, int]] = []
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) == tool_feat and (r, c) != start:
                        tool_positions.append((r, c))

            if not tool_positions:
                continue

            # Stage 1: Plan path from start to tool
            stage1_context = CausalPlanningContext(
                goal_positions=set(tool_positions),
                barrier_positions=context.barrier_positions,
                lethal_features=context.lethal_features,
                goal_features={tool_feat},
                barrier_features=context.barrier_features,
                walkable_features=context.walkable_features,
                action_displacements=context.action_displacements,
                grid_shape=context.grid_shape,
            )
            path_to_tool = CausalTheoryPlanner.plan_path(
                start, grid, stage1_context, max_expansions=max_expansions
            )
            if not path_to_tool:
                continue

            curr_pos = start
            for act_id in path_to_tool:
                dr, dc = context.action_displacements.get(act_id, (0, 0))
                curr_pos = (curr_pos[0] + dr, curr_pos[1] + dc)

            # Stage 2: Barriers of this type become walkable
            unlocked_barriers = {
                pos
                for pos in context.barrier_positions
                if not (
                    0 <= pos[0] < H
                    and 0 <= pos[1] < W
                    and int(grid[pos[0], pos[1]]) in barrier_feats
                )
            }
            stage2_barrier_features = context.barrier_features - barrier_feats
            stage2_context = CausalPlanningContext(
                goal_positions=context.goal_positions,
                barrier_positions=unlocked_barriers,
                lethal_features=context.lethal_features,
                goal_features=context.goal_features,
                barrier_features=stage2_barrier_features,
                walkable_features=context.walkable_features | barrier_feats,
                action_displacements=context.action_displacements,
                grid_shape=context.grid_shape,
            )
            path_to_goal = CausalTheoryPlanner.plan_path(
                curr_pos, grid, stage2_context, max_expansions=max_expansions
            )
            if path_to_goal:
                logger.info(
                    "CausalTheoryPlanner: Synthesized multi-stage chained plan "
                    "(Tool %d: %d steps -> Goal: %d steps)",
                    tool_feat,
                    len(path_to_tool),
                    len(path_to_goal),
                )
                return path_to_tool + path_to_goal

        return None

    @staticmethod
    def plan_relational_macro(
        start: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
        macro: RelationalSubgoalMacro,
        max_expansions_per_segment: int = 300,
    ) -> list[int] | None:
        """Plan composite path dynamically through a sequence of relational waypoints."""
        if not macro.waypoints:
            return None

        H, W = context.grid_shape
        if H <= 0 or W <= 0:
            H, W = grid.shape[:2]
        curr_start = start
        total_path: list[int] = []

        for wp in macro.waypoints:
            target_positions: list[tuple[int, int]] = []
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) == wp.feature and (r, c) != curr_start:
                        target_positions.append((r, c))

            if not target_positions:
                return None

            segment_context = CausalPlanningContext(
                goal_positions=set(target_positions),
                barrier_positions=context.barrier_positions,
                lethal_features=context.lethal_features,
                goal_features={wp.feature},
                barrier_features=context.barrier_features,
                walkable_features=context.walkable_features,
                action_displacements=context.action_displacements,
                grid_shape=context.grid_shape,
            )
            segment_path = CausalTheoryPlanner.plan_path(
                curr_start, grid, segment_context, max_expansions=max_expansions_per_segment
            )
            if not segment_path:
                return None

            total_path.extend(segment_path)
            for act_id in segment_path:
                dr, dc = context.action_displacements.get(act_id, (0, 0))
                curr_start = (curr_start[0] + dr, curr_start[1] + dc)

        return total_path if total_path else None


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


class CrossContextKnowledgeTransfer:
    """Manages extraction, promotion, and retrieval of cross-context knowledge.

    On context completion:
        1. Extracts universal causal invariants from the symbolic theory
        2. Abstracts position-specific data into transferable patterns
        3. Stores in persistent knowledge base

    On new context start:
        1. Queries persistent knowledge for applicable priors
        2. Seeds the engine's symbolic theory with transferable invariants
        3. Tracks transfer success/failure for confidence updating
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.persistent_invariants: dict[str, TransferableInvariant] = {}
        self.active_transfers: dict[str, TransferableInvariant] = {}
        self.storage_dir: Path = Path(storage_dir or "./knowledge_store")
        self.load_from_disk()

    def save_to_disk(self, storage_dir: str | Path | None = None) -> bool:
        """Atomically persist learned invariants to disk."""
        target_dir = Path(storage_dir or self.storage_dir)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / "persistent_invariants.json"
            records = [
                {
                    "invariant_id": inv.invariant_id,
                    "category": inv.category,
                    "description": inv.description,
                    "data": inv.data,
                    "source_game": inv.source_game,
                    "confidence": inv.confidence,
                }
                for inv in self.persistent_invariants.values()
            ]
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            temp_path.replace(file_path)
            logger.info("Saved %d persistent invariants to %s", len(records), file_path)
            return True
        except Exception as e:
            logger.warning("Could not persist invariants to disk: %s", e)
            return False

    def load_from_disk(self, storage_dir: str | Path | None = None) -> int:
        """Load previously learned invariants from disk."""
        target_dir = Path(storage_dir or self.storage_dir)
        file_path = target_dir / "persistent_invariants.json"
        if not file_path.is_file():
            return 0
        try:
            with open(file_path, encoding="utf-8") as f:
                records = json.load(f)
            loaded = 0
            for r in records:
                inv = TransferableInvariant(
                    invariant_id=r["invariant_id"],
                    category=r["category"],
                    description=r["description"],
                    data=r.get("data", {}),
                    source_game=r.get("source_game", ""),
                    confidence=float(r.get("confidence", 1.0)),
                )
                self.persistent_invariants[inv.invariant_id] = inv
                loaded += 1
            logger.info("Loaded %d persistent invariants from %s", loaded, file_path)
            return loaded
        except Exception as e:
            logger.warning("Could not load invariants from disk: %s", e)
            return 0

    def extract_invariants(
        self,
        engine: AutonomousEpistemicEngine,
        context_id: str = "",
    ) -> list[TransferableInvariant]:
        """Extract universal causal invariants from current context's learned knowledge."""
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
                        source_game=context_id,
                    )
                )

        # Extract lethal features (contact hazards)
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        for feat in engine.episodic_cortex.discovered_lethal_features:
            if feat in av_feats:
                continue
            inv_id = f"lethal_feature_{feat}"
            invariants.append(
                TransferableInvariant(
                    invariant_id=inv_id,
                    category="hazard",
                    description=f"Feature {feat} is lethal upon contact",
                    data={"feature": feat},
                    source_game=context_id,
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
                        source_game=context_id,
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
                av_f = engine.avatar_features or (
                    {engine.avatar_feature} if engine.avatar_feature is not None else set()
                )
                if feat is not None and feat not in av_f:
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
        context_id: str,
        levels_won: int,
    ) -> None:
        """Full lifecycle hook: extract, promote, record outcome, and save to disk."""
        invariants = self.extract_invariants(engine, context_id=context_id)
        self.promote_to_persistent(invariants)
        self.record_transfer_outcome(levels_won > 0)
        self.save_to_disk()

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


# ═══════════════════════════════════════════════════════════════════════════
# Phase G & I: Point Actuation Affordances & Metacognitive Fail-Fast
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PointActuationObservation:
    """Outcome of an empirical point actuation intervention at (row, col)."""

    pos: tuple[int, int]
    color_before: int
    color_after: int
    effect_type: str  # "toggle", "cycle", "trigger", "noop"
    timestamp: float = field(default_factory=time.time)


class PointActuationAffordanceLearner:
    """Empirical affordance learner for coordinate-targeted actuation."""

    def __init__(self) -> None:
        self.actuation_history: list[PointActuationObservation] = []
        self.interactive_features: set[int] = set()
        self.quiescent_cells: set[tuple[int, int]] = set()
        self.feature_cycle_maps: dict[int, int] = {}

    def observe_point_actuation(
        self,
        pos: tuple[int, int],
        grid_before: np.ndarray,
        grid_after: np.ndarray,
    ) -> PointActuationObservation | None:
        """Observe the delta caused by a targeted actuation intervention."""
        r, c = pos
        H, W = grid_before.shape[:2]
        if not (0 <= r < H and 0 <= c < W):
            return None
        c_before = int(grid_before[r, c])
        c_after = int(grid_after[r, c])
        diff_mask = grid_before != grid_after
        diff_count = int(np.count_nonzero(diff_mask))

        if diff_count == 0:
            self.quiescent_cells.add(pos)
            effect = "noop"
        elif diff_count == 1 and diff_mask[r, c]:
            effect = "cycle" if c_before in self.feature_cycle_maps else "toggle"
            self.feature_cycle_maps[c_before] = c_after
            self.interactive_features.add(c_before)
            self.interactive_features.add(c_after)
        else:
            effect = "trigger"
            self.interactive_features.add(c_before)

        obs = PointActuationObservation(
            pos=pos, color_before=c_before, color_after=c_after, effect_type=effect
        )
        self.actuation_history.append(obs)
        return obs

    def pick_exploratory_actuation(
        self,
        grid: np.ndarray,
        bg_feature: int = 0,
        attempted_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """Select high-EIG actuation target (interactive feature or unprobed non-bg cell)."""
        attempted = (attempted_cells or set()) | self.quiescent_cells
        H, W = grid.shape[:2]
        # 1. Prioritize known interactive features
        for r in range(H):
            for c in range(W):
                if (r, c) not in attempted and int(grid[r, c]) in self.interactive_features:
                    return (r, c)
        # 2. Prioritize non-background cells
        for r in range(H):
            for c in range(W):
                if (r, c) not in attempted and int(grid[r, c]) != bg_feature:
                    return (r, c)
        return None


@dataclass
class DeliberationEntropyMonitor:
    """Monitors heuristic stagnation and triggers fail-fast bailouts.

    Prevents runaway simulation loops and timeouts when forward search
    or planning cannot find a path.
    """

    stagnation_counter: int = 0
    stagnation_threshold: int = 35
    min_heuristic_seen: float = float("inf")
    visited_window: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=24))

    def reset_for_level(self) -> None:
        self.stagnation_counter = 0
        self.min_heuristic_seen = float("inf")
        self.visited_window.clear()

    def update(
        self,
        current_pos: tuple[int, int] | None,
        current_heuristic: float,
    ) -> bool:
        """Update monitor with current step. Returns True if stuck."""
        if current_pos is not None:
            self.visited_window.append(current_pos)

        if current_heuristic < self.min_heuristic_seen - 0.01:
            self.min_heuristic_seen = current_heuristic
            self.stagnation_counter = 0
            return False

        self.stagnation_counter += 1
        return self.stagnation_counter >= self.stagnation_threshold

    def trigger_fail_fast(self, engine: AutonomousEpistemicEngine) -> None:
        """Abort stalled mental plan and force exploratory state reset."""
        logger.info(
            "DeliberationEntropyMonitor: Fail-fast bailout triggered "
            "(%d stagnant steps). Invalidating dead-end plan.",
            self.stagnation_counter,
        )
        engine.mental_plan.clear()
        if hasattr(engine.working_memory, "macro_plan") and engine.working_memory.macro_plan:
            engine.working_memory.macro_plan.clear()
        engine.working_memory.active_macro_goal = None
        if engine.avatar_pos is not None:
            engine.working_memory.exhausted_candidate_goals.add(engine.avatar_pos)
        engine.simulation_cooldown = 0
        self.stagnation_counter = 0


# Backwards compatibility aliases (WP3)
CrossGameKnowledgeTransfer = CrossContextKnowledgeTransfer
