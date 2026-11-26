"""Temporal Abstraction & Option Discovery Engine — Sutton, Precup & Singh (AGI Gap 11)."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import NodeLifecycle, SkillNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope

if TYPE_CHECKING:
    from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill

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

        subgoals = [
            SymbolicSubgoal(
                intent=getattr(SpatialActionIntent, s, SpatialActionIntent.NAVIGATE_TO),
                target_query={"target": self.bottleneck_state},
            )
            for s in (self.subgoal_sequence or ["NAVIGATE_TO"])
        ]
        skill = DeclarativeNeuroSymbolicSkill()
        skill.name = skill.skill_name = self.name
        skill.description = f"Discovered option targeting bottleneck state {self.bottleneck_state}"
        skill.signature = ActionAffordancePredicate(required={1, 2, 3, 4})
        skill.program = SubgoalSequence(*subgoals)
        return skill

    def to_skill_node(self, tenant_id: str = "default") -> SkillNode:
        """Serialize discovered option into a parameterized homoiconic `SkillNode`."""
        from hbllm.hcir.skills.declarative import ActionAffordancePredicate

        def _ins(op: Opcode, cost: int, label: str | None = None, **p: Any) -> Instruction:
            return Instruction(opcode=op, params=p, cost_estimate=cost, label=label)

        stream = InstructionStream(
            author="option_discovery",
            description=f"Option {self.name} reaching bottleneck {self.bottleneck_state}",
            instructions=[
                _ins(
                    Opcode.BRANCH,
                    1,
                    condition="if_not_empty",
                    register="$r0",
                    target_label="act_nav",
                ),
                _ins(Opcode.UNIFY, 10, pattern={"role": "portal", "id": "?bottleneck"}, bind="$r0"),
                _ins(
                    Opcode.EXECUTE,
                    20,
                    "act_nav",
                    capability="motor.navigate",
                    params={"target": "$r0"},
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            ],
        )
        return SkillNode(
            id=f"discovered_option_{self.option_id}",
            skill_name=self.name,
            description=f"Discovered option targeting bottleneck state {self.bottleneck_state}",
            parameters=[
                {
                    "name": "bottleneck",
                    "role": "portal",
                    "required": False,
                    "default": list(self.bottleneck_state),
                }
            ],
            precondition_pattern={
                "symbolic_expr": ActionAffordancePredicate(required={1, 2, 3, 4}).to_dict()
            },
            postcondition_pattern={"intent": "reach_bottleneck", "target": "?bottleneck"},
            bytecode_stream=stream.model_dump(),
            alpha_success=1.0 + float(self.success_count),
            beta_failure=1.0 + float(max(0, self.reuse_count - self.success_count)),
            success_rate=self.success_rate,
            invocation_count=self.reuse_count,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by="option_discovery"),
            scope=Scope(tenant_id=tenant_id),
            tags=["discovered_option", "portal"],
            properties={
                "option_id": self.option_id,
                "bottleneck_state": list(self.bottleneck_state),
                "subgoal_sequence": list(self.subgoal_sequence),
                "reuse_count": self.reuse_count,
                "success_count": self.success_count,
                "success_rate": self.success_rate,
            },
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting the homoiconic option `SkillNode`."""
        node = self.to_skill_node()
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=node.id, node_data=node.model_dump()
            )
        ]


class OptionDiscoveryEngine:
    """Discovers temporal abstractions, graph bottlenecks, and reusable options (G11)."""

    def __init__(
        self,
        min_reuse_for_consolidation: int = 2,
        betweenness_threshold: float = 0.15,
        live_pipeline_enabled: bool = False,
    ) -> None:
        self.min_reuse_for_consolidation = min_reuse_for_consolidation
        self.betweenness_threshold = betweenness_threshold
        self.live_pipeline_enabled: bool = live_pipeline_enabled
        self.discovered_options: dict[str, DiscoveredOption] = {}
        self.total_episode_steps: int = 0
        self.option_executed_steps: int = 0
        self.last_scanned_grid_sig: tuple[int, int, bytes] | None = None
        self._topology_cache: dict[tuple[int, int, int, bytes], list[tuple[int, int]]] = {}

    def on_boundary(self, boundary: Any) -> None:
        """Reset layout-local bottleneck cache on cross-context episode boundaries."""
        if not getattr(boundary, "is_same_context", getattr(boundary, "same_level", False)):
            self.discovered_options.clear()
            self.last_scanned_grid_sig = None
            self._topology_cache.clear()

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
        max_landmarks: int = 128,
    ) -> dict[tuple[int, int], float]:
        """Compute integer-indexed Brandes betweenness centrality (exact for <= 128 nodes)."""
        nodes = list(adjacency.keys())
        n = len(nodes)
        if n <= 2:
            return {node: 0.0 for node in nodes}

        idx_of = {u: i for i, u in enumerate(nodes)}
        adj_idx: list[list[int]] = [[idx_of[w] for w in adjacency[u] if w in idx_of] for u in nodes]
        cb = [0.0] * n

        if n <= max_landmarks:
            sources = range(n)
        else:
            stride = n / max_landmarks
            sources = [int(i * stride) for i in range(max_landmarks)]

        for s in sources:
            stack: list[int] = []
            pred: list[list[int]] = [[] for _ in range(n)]
            sigma = [0.0] * n
            sigma[s] = 1.0
            dist = [-1] * n
            dist[s] = 0
            queue: deque[int] = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                dv = dist[v]
                nd = dv + 1
                sv = sigma[v]
                for w in adj_idx[v]:
                    dw = dist[w]
                    if dw < 0:
                        dist[w] = nd
                        queue.append(w)
                        sigma[w] += sv
                        pred[w].append(v)
                    elif dw == nd:
                        sigma[w] += sv
                        pred[w].append(v)

            delta = [0.0] * n
            while stack:
                w = stack.pop()
                coeff = (1.0 + delta[w]) / sigma[w]
                for v in pred[w]:
                    delta[v] += sigma[v] * coeff
                if w != s:
                    cb[w] += delta[w]

        source_cnt = len(sources)
        scale = (2.0 * n) / (source_cnt * (n - 1) * (n - 2))
        return {node: cb[i] * scale for i, node in enumerate(nodes)}

    def grid_to_adjacency(
        self, grid: np.ndarray, walkable_val: int = 0
    ) -> dict[tuple[int, int], set[tuple[int, int]]]:
        """Convert 2D lattice into an undirected 4-connected spatial adjacency graph."""
        H, W = grid.shape
        mask = grid == walkable_val
        rows, cols = np.nonzero(mask)
        adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for r, c in zip(rows.tolist(), cols.tolist()):
            nbs = {
                (nr, nc)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]
            }
            if nbs:
                adjacency[(r, c)] = nbs
        return adjacency

    def discover_bottlenecks_from_grid(
        self,
        grid: np.ndarray,
        walkable_val: int = 0,
    ) -> list[tuple[int, int]]:
        """Discover spatial bottleneck states in grid using articulation cuts and centrality."""
        mask = grid == walkable_val
        key = (int(grid.shape[0]), int(grid.shape[1]), int(walkable_val), mask.tobytes())
        cached = self._topology_cache.get(key)
        if cached is not None:
            return list(cached)

        adjacency = self.grid_to_adjacency(grid, walkable_val=walkable_val)
        cuts = set(self.find_articulation_points(adjacency))
        centralities = self.compute_betweenness_centrality(adjacency)

        high_centrality = {
            pos for pos, score in centralities.items() if score >= self.betweenness_threshold
        }

        bottlenecks = sorted(cuts.union(high_centrality))
        if len(self._topology_cache) >= 32:
            self._topology_cache.pop(next(iter(self._topology_cache)))
        self._topology_cache[key] = list(bottlenecks)
        return bottlenecks

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
