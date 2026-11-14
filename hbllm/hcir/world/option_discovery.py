"""Temporal Abstraction & Option Discovery Engine — Sutton, Precup & Singh (AGI Gap 11).

Autonomous discovery of temporal options, macro-actions, and bottleneck states:
1. Bottleneck State Discovery (McGovern & Barto, Şimşek & Barto):
   - Computes articulation points (cut vertices) and betweenness centrality over
     topological manifolds and empirical state transition graphs.
   - Identifies critical choke points (doors, keys, switches, passage corridors)
     whose traversal is necessary to partition state space.
2. Option Induction <I, pi, beta> (Sutton, Precup, Singh 1999):
   - Initiation set I: Spatial/causal precondition under which the option is viable.
   - Policy / Subgoal sequence pi: Abstract action sequence driving agent to the bottleneck.
   - Termination condition beta: Deterministic termination upon reaching the bottleneck state.
3. Hierarchy Formation & Option Reuse Tracking:
   - Tracks option invocations across episodes and contexts.
   - Calculates option reuse rate: R_option = reused_steps / total_steps.
   - High-reuse options are consolidated into DeclarativeNeuroSymbolicSkill instances
     and registered into the LibraryLearner.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation

if TYPE_CHECKING:
    from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill
    from hbllm.hcir.world.library_learning import LibraryLearner

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredOption:
    """Formal temporal option <I, pi, beta> discovered from topological bottlenecks."""

    option_id: str
    name: str
    bottleneck_state: tuple[int, int]
    initiation_region: list[tuple[int, int]] = field(default_factory=list)
    subgoal_sequence: list[str] = field(default_factory=list)
    termination_condition: str = "at_bottleneck"
    reuse_count: int = 0
    success_count: int = 0
    total_steps_executed: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Empirical success rate of this option."""
        return (self.success_count / self.reuse_count) if self.reuse_count > 0 else 0.0

    def to_declarative_skill(self) -> DeclarativeNeuroSymbolicSkill:
        """Convert discovered option into a DeclarativeNeuroSymbolicSkill."""
        from hbllm.hcir.skills.declarative import (
            ActionAffordancePredicate,
            DeclarativeNeuroSymbolicSkill,
            SpatialActionIntent,
            SubgoalSequence,
            SymbolicSubgoal,
        )

        subgoals = []
        for s in self.subgoal_sequence:
            intent = getattr(SpatialActionIntent, s, SpatialActionIntent.NAVIGATE_TO)
            subgoals.append(
                SymbolicSubgoal(
                    intent=intent,
                    target_query={"target": self.bottleneck_state},
                )
            )

        if not subgoals:
            subgoals.append(
                SymbolicSubgoal(
                    intent=SpatialActionIntent.NAVIGATE_TO,
                    target_query={"target": self.bottleneck_state},
                )
            )

        class AutonomousOptionSkill(DeclarativeNeuroSymbolicSkill):
            name = self.name
            description = f"Discovered option targeting bottleneck state {self.bottleneck_state}"
            signature = ActionAffordancePredicate(required={1, 2, 3, 4})
            program = SubgoalSequence(*subgoals)

        return AutonomousOptionSkill()

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting discovered option beliefs."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"discovered_option_{self.option_id}",
                node_data={
                    "option_id": self.option_id,
                    "name": self.name,
                    "bottleneck_state": self.bottleneck_state,
                    "subgoal_sequence": self.subgoal_sequence,
                    "reuse_count": self.reuse_count,
                    "success_count": self.success_count,
                    "success_rate": self.success_rate,
                },
            )
        ]


class OptionDiscoveryEngine:
    """Discovers temporal abstractions, graph bottlenecks, and reusable options (G11)."""

    def __init__(
        self,
        min_reuse_for_consolidation: int = 2,
        betweenness_threshold: float = 0.15,
    ) -> None:
        self.min_reuse_for_consolidation = min_reuse_for_consolidation
        self.betweenness_threshold = betweenness_threshold
        self.discovered_options: dict[str, DiscoveredOption] = {}
        self.total_episode_steps: int = 0
        self.option_executed_steps: int = 0
        self.last_scanned_grid_sig: tuple[int, int, bytes] | None = None

    def on_boundary(self, boundary: Any) -> None:
        """Reset layout-local bottleneck cache on cross-layout episode boundaries."""
        if not getattr(boundary, "same_level", False):
            self.discovered_options.clear()
            self.last_scanned_grid_sig = None

    def find_articulation_points(
        self,
        adjacency: dict[tuple[int, int], set[tuple[int, int]]],
    ) -> list[tuple[int, int]]:
        """Identify cut-vertices (articulation points) in undirected spatial connectivity graph.

        Uses Hopcroft-Tarjan DFS traversal: a vertex whose removal increases disconnected
        components represents a topological bottleneck (e.g. doorways, choke points).
        """
        if not adjacency:
            return []

        tin: dict[tuple[int, int], int] = {}
        low: dict[tuple[int, int], int] = {}
        visited: set[tuple[int, int]] = set()
        articulation_points: set[tuple[int, int]] = set()
        timer = 0

        def dfs(u: tuple[int, int], p: tuple[int, int] | None = None) -> None:
            nonlocal timer
            visited.add(u)
            tin[u] = low[u] = timer
            timer += 1
            children = 0

            for v in adjacency.get(u, set()):
                if v == p:
                    continue
                if v in visited:
                    low[u] = min(low[u], tin[v])
                else:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] >= tin[u] and p is not None:
                        articulation_points.add(u)
                    children += 1

            if p is None and children > 1:
                articulation_points.add(u)

        for node in list(adjacency.keys()):
            if node not in visited:
                dfs(node)

        return sorted(list(articulation_points))

    def compute_betweenness_centrality(
        self,
        adjacency: dict[tuple[int, int], set[tuple[int, int]]],
    ) -> dict[tuple[int, int], float]:
        """Compute Brandes betweenness centrality over spatial connectivity manifold."""
        nodes = list(adjacency.keys())
        n = len(nodes)
        if n <= 2:
            return {node: 0.0 for node in nodes}

        cb: dict[tuple[int, int], float] = defaultdict(float)

        for s in nodes:
            stack: list[tuple[int, int]] = []
            pred: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
            sigma: dict[tuple[int, int], float] = defaultdict(float)
            sigma[s] = 1.0
            dist: dict[tuple[int, int], int] = {s: 0}
            queue: deque[tuple[int, int]] = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                dv = dist[v]
                for w in adjacency.get(v, set()):
                    if w not in dist:
                        dist[w] = dv + 1
                        queue.append(w)
                    if dist[w] == dv + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta: dict[tuple[int, int], float] = defaultdict(float)
            while stack:
                w = stack.pop()
                coeff = (1.0 + delta[w]) / sigma[w]
                for v in pred[w]:
                    delta[v] += sigma[v] * coeff
                if w != s:
                    cb[w] += delta[w]

        # Normalization for undirected graph: 2 / ((n-1)(n-2))
        scale = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
        return {node: cb[node] * scale for node in nodes}

    def grid_to_adjacency(
        self,
        grid: np.ndarray,
        walkable_val: int = 0,
    ) -> dict[tuple[int, int], set[tuple[int, int]]]:
        """Convert 2D lattice into an undirected 4-connected spatial adjacency graph."""
        H, W = grid.shape
        adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for r in range(H):
            for c in range(W):
                if grid[r, c] != walkable_val:
                    continue
                u = (r, c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == walkable_val:
                        adjacency[u].add((nr, nc))
        return adjacency

    def discover_bottlenecks_from_grid(
        self,
        grid: np.ndarray,
        walkable_val: int = 0,
    ) -> list[tuple[int, int]]:
        """Discover spatial bottleneck states in grid using articulation cuts and centrality."""
        adjacency = self.grid_to_adjacency(grid, walkable_val=walkable_val)
        cuts = set(self.find_articulation_points(adjacency))
        centralities = self.compute_betweenness_centrality(adjacency)

        high_centrality = {
            pos for pos, score in centralities.items() if score >= self.betweenness_threshold
        }

        bottlenecks = cuts.union(high_centrality)
        return sorted(list(bottlenecks))

    def induce_option(
        self,
        bottleneck: tuple[int, int],
        initiation_region: list[tuple[int, int]] | None = None,
        subgoal_sequence: list[str] | None = None,
    ) -> DiscoveredOption:
        """Construct and register a discovered option targeting a bottleneck state."""
        opt_id = f"opt_bottleneck_{bottleneck[0]}_{bottleneck[1]}"
        if opt_id in self.discovered_options:
            return self.discovered_options[opt_id]

        opt = DiscoveredOption(
            option_id=opt_id,
            name=f"ReachBottleneck_{bottleneck[0]}_{bottleneck[1]}",
            bottleneck_state=bottleneck,
            initiation_region=initiation_region or [],
            subgoal_sequence=subgoal_sequence or ["NAVIGATE_TO"],
        )
        self.discovered_options[opt_id] = opt
        logger.info("Discovered option %s targeting bottleneck %s", opt_id, bottleneck)
        return opt

    def record_option_execution(
        self,
        option_id: str,
        steps: int,
        success: bool,
    ) -> None:
        """Record an option execution to update reuse statistics and reuse rate."""
        self.total_episode_steps += max(1, steps)
        self.option_executed_steps += max(1, steps)

        if option_id in self.discovered_options:
            opt = self.discovered_options[option_id]
            opt.reuse_count += 1
            opt.total_steps_executed += steps
            if success:
                opt.success_count += 1

    def record_primitive_execution(self, steps: int = 1) -> None:
        """Record primitive single-step execution to benchmark option reuse rate."""
        self.total_episode_steps += max(1, steps)

    def get_option_reuse_rate(self) -> float:
        """Calculate the option reuse rate R_option = option_steps / total_steps."""
        if self.total_episode_steps == 0:
            return 0.0
        return min(1.0, float(self.option_executed_steps) / float(self.total_episode_steps))

    def consolidate_options_to_library(
        self,
        library_learner: LibraryLearner,
    ) -> list[str]:
        """Consolidate high-reuse options into the master LibraryLearner macro skills."""
        consolidated = []
        for opt_id, opt in self.discovered_options.items():
            if opt.reuse_count >= self.min_reuse_for_consolidation and opt.success_rate >= 0.5:
                skill = opt.to_declarative_skill()
                library_learner.macro_skills[opt.name] = skill
                consolidated.append(opt.name)
                logger.info(
                    "Option %s consolidated into LibraryLearner (reuse=%d, success=%.2f)",
                    opt.name,
                    opt.reuse_count,
                    opt.success_rate,
                )
        return consolidated
