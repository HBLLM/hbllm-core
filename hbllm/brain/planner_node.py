"""
Graph-of-Thoughts (GoT) Task Planner Node.

Implements a DAG-based reasoning framework where thoughts are nodes in a 
directed acyclic graph. Supports branching (generating alternatives), 
merging (combining complementary thoughts), and scoring (evaluating quality).

Reference: "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
           (Besta et al., 2023)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType, QueryPayload
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """A single node in the Graph-of-Thoughts DAG."""
    
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    content: str = ""
    score: float = 0.0
    depth: int = 0
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)
    is_merged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0


class ThoughtGraph:
    """
    Directed Acyclic Graph of thoughts.
    
    Supports three core operations:
    - Branch: Generate N child thoughts from a parent (fan-out)
    - Score: Evaluate thought quality (0.0 to 1.0) 
    - Merge: Combine complementary sibling thoughts into a synthesis (fan-in)
    """
    
    def __init__(self):
        self.nodes: dict[str, ThoughtNode] = {}
        self.root_ids: list[str] = []
    
    def add_root(self, content: str, score: float = 0.0) -> ThoughtNode:
        """Add a root thought (no parents)."""
        node = ThoughtNode(content=content, score=score, depth=0)
        self.nodes[node.id] = node
        self.root_ids.append(node.id)
        return node
    
    def branch(self, parent_id: str, content: str, score: float = 0.0) -> ThoughtNode:
        """Create a child thought branching from a parent."""
        parent = self.nodes[parent_id]
        child = ThoughtNode(
            content=content,
            score=score,
            depth=parent.depth + 1,
            parent_ids=[parent_id],
        )
        parent.children_ids.append(child.id)
        self.nodes[child.id] = child
        return child
    
    def merge(self, parent_ids: list[str], content: str, score: float = 0.0) -> ThoughtNode:
        """Merge multiple parent thoughts into a synthesis node."""
        max_depth = max(self.nodes[pid].depth for pid in parent_ids)
        merged = ThoughtNode(
            content=content,
            score=score,
            depth=max_depth + 1,
            parent_ids=parent_ids,
            is_merged=True,
        )
        for pid in parent_ids:
            self.nodes[pid].children_ids.append(merged.id)
        self.nodes[merged.id] = merged
        return merged
    
    def best_path(self) -> list[ThoughtNode]:
        """Find the highest-scoring path from any root to any leaf."""
        leaves = [n for n in self.nodes.values() if n.is_leaf]
        if not leaves:
            return []
        
        best_leaf = max(leaves, key=lambda n: n.score)
        
        # Trace back to root
        path = [best_leaf]
        current = best_leaf
        while current.parent_ids:
            # Pick the highest-scoring parent
            best_parent = max(
                (self.nodes[pid] for pid in current.parent_ids),
                key=lambda n: n.score
            )
            path.append(best_parent)
            current = best_parent
        
        path.reverse()
        return path
    
    def prune(self, min_score: float = 0.3) -> int:
        """Remove leaf nodes below the score threshold. Returns count removed."""
        pruned = 0
        leaves = [n for n in self.nodes.values() if n.is_leaf]
        for leaf in leaves:
            if leaf.score < min_score and leaf.id not in self.root_ids:
                # Remove from parent's children list
                for pid in leaf.parent_ids:
                    if pid in self.nodes:
                        self.nodes[pid].children_ids = [
                            cid for cid in self.nodes[pid].children_ids if cid != leaf.id
                        ]
                del self.nodes[leaf.id]
                pruned += 1
        return pruned
    
    def summary(self) -> dict[str, Any]:
        """Return a serializable summary of the graph."""
        return {
            "total_nodes": len(self.nodes),
            "root_count": len(self.root_ids),
            "leaf_count": sum(1 for n in self.nodes.values() if n.is_leaf),
            "max_depth": max((n.depth for n in self.nodes.values()), default=0),
            "merged_count": sum(1 for n in self.nodes.values() if n.is_merged),
            "avg_score": (
                sum(n.score for n in self.nodes.values()) / len(self.nodes)
                if self.nodes else 0.0
            ),
        }


class PlannerNode(Node):
    """
    Graph-of-Thoughts planner that decomposes complex queries into
    a DAG of reasoning steps with branching, scoring, merging, and pruning.
    """

    def __init__(self, node_id: str, branch_factor: int = 3, max_depth: int = 2):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PLANNER,
            capabilities=["task_decomposition", "graph_of_thoughts", "aggregation"],
        )
        self.branch_factor = branch_factor
        self.max_depth = max_depth

    async def on_start(self) -> None:
        """Subscribe to planning requests."""
        logger.info("Starting PlannerNode (Graph-of-Thoughts)")
        await self.bus.subscribe("planner.decompose", self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping PlannerNode")

    async def handle_message(self, message: Message) -> Message | None:
        """
        Execute a Graph-of-Thoughts reasoning process:
        1. Branch: Generate N diverse initial thoughts
        2. Score: Evaluate each thought
        3. Prune: Remove low-quality thoughts  
        4. Refine: Branch from surviving thoughts
        5. Merge: Combine complementary refined thoughts
        6. Select: Return the best path through the graph
        """
        if message.type != MessageType.QUERY:
            return None

        text = message.payload.get("text", "")
        logger.info("[GoT] Starting Graph-of-Thoughts for: '%s...'", text[:40])

        graph = ThoughtGraph()

        # ── Step 1: Branch — Generate initial diverse thoughts ────────────
        initial_thoughts = await asyncio.gather(
            *[self._generate_thought(text, i + 1) for i in range(self.branch_factor)]
        )
        
        root_nodes = []
        for i, thought_text in enumerate(initial_thoughts):
            if thought_text:
                node = graph.add_root(thought_text)
                root_nodes.append(node)

        if not root_nodes:
            return message.create_error("GoT: All initial branches failed.")

        # ── Step 2: Score — Evaluate initial thoughts ─────────────────────
        await asyncio.gather(
            *[self._score_thought(graph, node) for node in root_nodes]
        )

        # ── Step 3: Prune — Remove weak initial thoughts ──────────────────
        pruned = graph.prune(min_score=0.3)
        logger.info("[GoT] Pruned %d weak initial thoughts", pruned)

        # ── Step 4: Refine — Branch from surviving thoughts ───────────────
        survivors = [n for n in graph.nodes.values() if n.is_leaf and n.depth < self.max_depth]
        
        refinement_tasks = []
        for parent in survivors:
            for j in range(2):  # 2 refinements per surviving thought
                refinement_tasks.append(
                    self._refine_thought(graph, parent, text, j + 1)
                )
        
        if refinement_tasks:
            await asyncio.gather(*refinement_tasks)

        # ── Step 5: Score refinements ─────────────────────────────────────
        new_leaves = [n for n in graph.nodes.values() if n.is_leaf and n.score == 0.0]
        if new_leaves:
            await asyncio.gather(
                *[self._score_thought(graph, node) for node in new_leaves]
            )

        # ── Step 6: Merge — Combine complementary top thoughts ────────────
        top_leaves = sorted(
            [n for n in graph.nodes.values() if n.is_leaf],
            key=lambda n: n.score,
            reverse=True
        )[:2]
        
        if len(top_leaves) >= 2:
            merged_content = await self._merge_thoughts(
                text, top_leaves[0].content, top_leaves[1].content
            )
            if merged_content:
                merged_node = graph.merge(
                    [n.id for n in top_leaves], merged_content
                )
                await self._score_thought(graph, merged_node)

        # ── Step 7: Select best path ──────────────────────────────────────
        best_path = graph.best_path()
        graph_stats = graph.summary()
        
        if not best_path:
            return message.create_error("GoT: No valid reasoning path found.")

        final_node = best_path[-1]
        
        path_desc = " → ".join(
            f"[D{n.depth}:{n.score:.2f}]" for n in best_path
        )
        
        final_text = (
            f"[GoT Planner] Explored {graph_stats['total_nodes']} thoughts "
            f"across {graph_stats['max_depth']+1} depths "
            f"({graph_stats['merged_count']} merged, "
            f"avg score: {graph_stats['avg_score']:.2f}).\n"
            f"Best path: {path_desc}\n\n"
            f"{final_node.content}"
        )
        
        logger.info("[GoT] Selected path: %s (score=%.2f)", path_desc, final_node.score)
        return message.create_response({"text": final_text, "domain": "planner"})

    # ── Private helpers ───────────────────────────────────────────────────

    async def _generate_thought(self, query: str, branch_id: int) -> str:
        """Generate one diverse initial thought."""
        prompt = f"Exploring Approach {branch_id} to solve: {query}"
        req = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="domain.general.query",
            payload={"text": prompt},
        )
        try:
            resp = await self.request("domain.general.query", req, timeout=30.0)
            return resp.payload.get("text", "")
        except Exception as e:
            logger.warning("[GoT] Branch %d generation failed: %s", branch_id, e)
            return ""

    async def _score_thought(self, graph: ThoughtGraph, node: ThoughtNode) -> None:
        """Score a thought node by querying the evaluation domain."""
        prompt = f"Rate the quality and correctness of this reasoning approach on a 0-1 scale: {node.content[:300]}"
        req = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="domain.general.query",
            payload={"text": prompt},
        )
        try:
            resp = await self.request("domain.general.query", req, timeout=30.0)
            # Parse score from response or use heuristic
            resp_text = resp.payload.get("text", "")
            import re
            match = re.search(r"(\d+\.?\d*)", resp_text)
            if match:
                raw = float(match.group(1))
                node.score = min(1.0, raw if raw <= 1.0 else raw / 10.0)
            else:
                node.score = 0.5  # Default if unparseable
        except Exception:
            node.score = 0.3

    async def _refine_thought(
        self, graph: ThoughtGraph, parent: ThoughtNode, query: str, refinement_id: int
    ) -> None:
        """Refine a thought by branching deeper."""
        prompt = (
            f"Refine and improve this reasoning (attempt {refinement_id}):\n"
            f"Original query: {query}\n"
            f"Current approach: {parent.content[:300]}\n"
            f"Provide a more detailed and accurate solution."
        )
        req = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="domain.general.query",
            payload={"text": prompt},
        )
        try:
            resp = await self.request("domain.general.query", req, timeout=30.0)
            content = resp.payload.get("text", "")
            if content:
                graph.branch(parent.id, content)
        except Exception as e:
            logger.warning("[GoT] Refinement failed: %s", e)

    async def _merge_thoughts(self, query: str, thought_a: str, thought_b: str) -> str:
        """Merge two complementary thoughts into a synthesis."""
        prompt = (
            f"Synthesize these two approaches into a single, comprehensive solution:\n"
            f"Query: {query}\n"
            f"Approach A: {thought_a[:200]}\n"
            f"Approach B: {thought_b[:200]}\n"
            f"Provide a unified, improved answer."
        )
        req = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="domain.general.query",
            payload={"text": prompt},
        )
        try:
            resp = await self.request("domain.general.query", req, timeout=30.0)
            return resp.payload.get("text", "")
        except Exception as e:
            logger.warning("[GoT] Merge failed: %s", e)
            return ""
