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
import hashlib
import logging
import re
import uuid
from collections import OrderedDict
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
    
    # MCTS Tracking
    visits: int = 0
    cumulative_reward: float = 0.0
    trajectory_history: list[str] = field(default_factory=list)
    
    @property
    def q_value(self) -> float:
        """Average reward (exploitation term)"""
        return self.cumulative_reward / self.visits if self.visits > 0 else 0.0
    
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
    
    def branch(self, parent_id: str, content: str, score: float = 0.0, is_observation: bool = False) -> ThoughtNode:
        """Create a child thought branching from a parent."""
        parent = self.nodes[parent_id]
        child = ThoughtNode(
            content=content,
            score=score,
            depth=parent.depth + 1,
            parent_ids=[parent_id],
            trajectory_history=parent.trajectory_history + [parent.content]
        )
        if is_observation:
            child.metadata["is_observation"] = True
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
        """Find the path from root to a leaf with the highest visit count / Q-value."""
        leaves = [n for n in self.nodes.values() if n.is_leaf]
        if not leaves:
            return []
        
        # In MCTS, the best action after the budget is spent is typically
        # the child with the highest visit count or highest Q-value.
        # We'll use visits as primary, Q-value as secondary tie-breaker.
        best_leaf = max(leaves, key=lambda n: (n.visits, n.q_value))
        
        # Trace back to root
        path = [best_leaf]
        current = best_leaf
        while current.parent_ids:
            # Pick the highest-scoring parent (using Q-value/visits)
            best_parent = max(
                (self.nodes[pid] for pid in current.parent_ids),
                key=lambda n: (n.visits, n.q_value)
            )
            path.append(best_parent)
            current = best_parent
        
        path.reverse()
        return path
    
    def select_leaf_uct(self, root_id: str, c_param: float = 1.414) -> ThoughtNode:
        import math
        """Select a leaf node to expand using the UCT formula."""
        current = self.nodes[root_id]
        
        while not current.is_leaf:
            # If any child is completely unexplored, return it instantly
            unvisited = [self.nodes[cid] for cid in current.children_ids if self.nodes[cid].visits == 0]
            if unvisited:
                return unvisited[0]
            
            # UCT Selection
            best_score = -float('inf')
            best_child = None
            parent_visits = current.visits
            
            for child_id in current.children_ids:
                child = self.nodes[child_id]
                exploitation = child.q_value
                # UCT term: Q(s,a) + c * sqrt(ln(N) / n)
                exploration = c_param * math.sqrt(math.log(parent_visits) / child.visits)
                score = exploitation + exploration
                
                if score > best_score:
                    best_score = score
                    best_child = child
                    
            if not best_child:
                break
            current = best_child
            
        return current

    def backpropagate(self, leaf_id: str, reward: float) -> None:
        """Propagate reward up the tree to the root."""
        current = self.nodes[leaf_id]
        while True:
            current.visits += 1
            current.cumulative_reward += reward
            # Backprop up the first primary parent chain (classic MCTS treats state as a tree)
            if not current.parent_ids:
                break
            current = self.nodes[current.parent_ids[0]]
    
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

    # Max cached prompt/response pairs
    MAX_CACHE_SIZE = 200

    def __init__(self, node_id: str, branch_factor: int = 3, max_depth: int = 2, policy_engine=None):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PLANNER,
            capabilities=["task_decomposition", "graph_of_thoughts", "aggregation"],
        )
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.policy_engine = policy_engine  # PolicyEngine for plan validation
        # LRU cache: hash(prompt) → response text
        self._response_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    async def on_start(self) -> None:
        """Subscribe to planning requests and workspace updates."""
        logger.info("Starting PlannerNode (Graph-of-Thoughts)")
        await self.bus.subscribe("planner.decompose", self.handle_message)
        # Also participate as a workspace thought contributor
        await self.bus.subscribe("workspace.update", self._contribute_to_workspace)

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
        
        # Check prompt cache for exact matches
        cache_key = self._cache_key(text)
        if cache_key in self._response_cache:
            self._cache_hits += 1
            cached = self._response_cache[cache_key]
            # Move to end (most recently used)
            self._response_cache.move_to_end(cache_key)
            logger.info("[GoT] Cache HIT for query (hits=%d, misses=%d)", self._cache_hits, self._cache_misses)
            return message.create_response({"text": cached, "domain": "planner", "cached": True})
        
        self._cache_misses += 1
        logger.info("[GoT] Starting Graph-of-Thoughts for: '%s...'", text[:40])

        graph = ThoughtGraph()

        # ── Step 1: Root Node & Initial Plausible Branches ────────────────
        import time
        root_node = graph.add_root("Root Query: " + text[:50], score=0.5)
        
        initial_thoughts = await asyncio.gather(
            *[self._generate_thought(text, i + 1) for i in range(self.branch_factor)]
        )
        
        branch_nodes = []
        for thought_text in initial_thoughts:
            if thought_text:
                child = graph.branch(root_node.id, thought_text)
                branch_nodes.append(child)

        if not branch_nodes:
            return message.create_error("MCTS: All initial branches failed.")

        # ── Step 2: Policy gate ── validate initial thoughts against gov ──
        if self.policy_engine:
            tenant_id = message.tenant_id or "default"
            for node in branch_nodes:
                result = self.policy_engine.evaluate(
                    text=node.content,
                    tenant_id=tenant_id,
                    context=message.payload.get("context", {}),
                )
                if not result.passed:
                    node.score = 0.0
                    node.metadata["policy_blocked"] = True
                    node.metadata["violations"] = result.violations
                    node.content = (
                        f"[BLOCKED by policy: {'; '.join(result.violations)}] "
                        f"Original: {node.content[:200]}"
                    )
                    logger.info("[MCTS] Thought blocked by policy: %s", result.violations)

        # Score the initial unblocked roots
        unblocked_branches = [n for n in branch_nodes if not n.metadata.get("policy_blocked")]
        if unblocked_branches:
            await asyncio.gather(
                *[self._score_thought(graph, node) for node in unblocked_branches]
            )
            # Backpropagate initial scores
            for node in unblocked_branches:
                graph.backpropagate(node.id, node.score)

        # ── Step 3: MCTS Compute Loop (Infinite Test-Time Compute) ────────
        # Run until depth budget or time budget exhausted
        import time
        deadline = time.time() + 15.0  # 15 seconds thinking budget per query
        max_iterations = 20
        iteration = 0
        
        while time.time() < deadline and iteration < max_iterations:
            # 1. Selection
            leaf = graph.select_leaf_uct(root_node.id, c_param=1.414)
            
            # 2. Expansion (only expand if we've already scored it and haven't hit max depth)
            if leaf.visits > 0 and leaf.depth < self.max_depth:
                await self._refine_thought(graph, leaf, text, iteration)
                
                # Pick one of the newly expanded children to simulate
                unvisited = [graph.nodes[cid] for cid in leaf.children_ids if graph.nodes[cid].visits == 0]
                if unvisited:
                    leaf = unvisited[0]
            
            # 3. Simulation & Scoring
            if leaf.visits == 0:
                await self._score_thought(graph, leaf)
                reward = leaf.score
            else:
                # If we selected a terminal node that can't expand, re-simulate or just use existing score
                reward = leaf.score
                
            # 4. Backpropagation
            graph.backpropagate(leaf.id, reward)
            
            iteration += 1

        logger.info("[MCTS] Completed %d iterations before budget exhaustion.", iteration)

        # ── Step 4: Optional Merge — Combine complementary top thoughts ───
        top_leaves = sorted(
            [n for n in graph.nodes.values() if n.is_leaf and n.visits > 0 and not n.metadata.get("policy_blocked")],
            key=lambda n: n.q_value,
            reverse=True
        )[:2]
        
        if len(top_leaves) >= 2 and top_leaves[0].q_value > 0.5:
            merged_content = await self._merge_thoughts(
                text, top_leaves[0].content, top_leaves[1].content
            )
            if merged_content:
                merged_node = graph.merge(
                    [n.id for n in top_leaves], merged_content
                )
                await self._score_thought(graph, merged_node)
                graph.backpropagate(merged_node.id, merged_node.score)

        # ── Step 7: Select best path ──────────────────────────────────────
        best_path = graph.best_path()
        graph_stats = graph.summary()
        
        if not best_path:
            return message.create_error("GoT: No valid reasoning path found.")

        final_node = best_path[-1]
        
        path_desc = " → ".join(
            f"[D{n.depth}: Q={n.q_value:.2f}, N={n.visits}]" for n in best_path
        )
        
        final_text = (
            f"[MCTS Planner] Explored {graph_stats['total_nodes']} thoughts "
            f"across {graph_stats['max_depth']+1} depths "
            f"({graph_stats['merged_count']} merged, "
            f"avg score: {graph_stats['avg_score']:.2f}).\n"
            f"Best path: {path_desc}\n\n"
            f"{final_node.content}"
        )
        
        logger.info("[MCTS] Selected path: %s (Q=%.2f, N=%d)", path_desc, final_node.q_value, final_node.visits)
        
        # Cache the result
        self._cache_response(cache_key, final_text)
        
        return message.create_response({"text": final_text, "domain": "planner"})

    # ── Cache helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key from prompt text."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def _cache_response(self, key: str, response: str) -> None:
        """Store a response in the LRU cache, evicting oldest if full."""
        self._response_cache[key] = response
        while len(self._response_cache) > self.MAX_CACHE_SIZE:
            self._response_cache.popitem(last=False)

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
        """Score a thought node by querying the evaluation domain or ExecutionNode/ToolRouter."""

        if node.metadata.get("is_observation"):
            node.score = 1.0 # Observations are always factual steps
            return

        # 1. Deterministic verification for Python code
        match = re.search(r"```python\n(.*?)```", node.content, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="action.execute_code",
                payload={"code": code},
            )
            try:
                resp = await self.request("action.execute_code", req, timeout=5.0)
                status = resp.payload.get("status")
                if status == "SUCCESS":
                    node.score = 1.0
                    node.metadata["execution_output"] = resp.payload.get("output", "")
                else:
                    node.score = 0.1
                    err = resp.payload.get("error", "Unknown execution error")
                    node.metadata["execution_error"] = err
                    node.content += f"\n\n[EXECUTION FAILED]\n{err}"
                return
            except Exception as e:
                logger.warning("[GoT] Execution verification failed/timed out: %s", e)
                # Fall through to LLM scoring if execution bus falls over

        # 1.5. Tool Call Interception
        tool_match = re.search(r"<tool_call\s+name=[\"'](.*?)[\"']>(.*?)</tool_call>", node.content, re.DOTALL | re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_args_str = tool_match.group(2).strip()
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="action.tool_call",
                payload={"tool_name": tool_name, "arguments": tool_args_str},
            )
            try:
                resp = await self.request("action.tool_call", req, timeout=30.0)
                if resp.type == MessageType.ERROR:
                    err = resp.payload.get("error", "Unknown tool error")
                    obs_content = f"Observation: Error: {err}"
                else:
                    obs_content = f"Observation:\n{resp.payload}"
                
                # Expand the tree deterministically with the observation
                graph.branch(node.id, obs_content, score=1.0, is_observation=True)
                node.score = 1.0 # Reward emitting a valid tool call
                return
            except Exception as e:
                logger.warning("[GoT] Tool execution failed/timed out: %s", e)
                obs_content = f"Observation: Execution failed: {str(e)}"
                graph.branch(node.id, obs_content, score=0.1, is_observation=True)
                node.score = 0.5
                return

        # 2. Use Process Reward Model (PRM) network via event bus
        req = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="action.score_thought",
            payload={"content": node.content},
        )
        try:
            # PRM scoring is extremely fast (single forward pass)
            resp = await self.request("action.score_thought", req, timeout=5.0)
            score = resp.payload.get("score", 0.5)
            node.score = min(max(float(score), 0.0), 1.0)
        except Exception as e:
            logger.warning("[GoT] PRM evaluation failed/timed out, defaulting to 0.5: %s", e)
            node.score = 0.5

    def _compress_text(self, text: str, max_chars: int = 2000) -> str:
        """Middle-out truncation for excessively long strings."""
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        head = text[:half]
        tail = text[-half:]
        omitted = len(text) - max_chars
        return f"{head}\n\n[... {omitted} characters dynamically omitted to preserve context bounds ...]\n\n{tail}"

    async def _refine_thought(
        self, graph: ThoughtGraph, parent: ThoughtNode, query: str, refinement_id: int
    ) -> None:
        """Refine a thought by branching deeper with an adaptive context window."""
        trajectory = parent.trajectory_history + [parent.content]
        compressed_traj = [self._compress_text(txt) for txt in trajectory]
        
        MAX_CONTEXT_STEPS = 6
        if len(compressed_traj) > MAX_CONTEXT_STEPS:
            omitted_count = len(compressed_traj) - MAX_CONTEXT_STEPS
            head_step = f"Step 1: {compressed_traj[0]}"
            
            tail_steps = compressed_traj[-(MAX_CONTEXT_STEPS - 1):]
            start_num = len(compressed_traj) - (MAX_CONTEXT_STEPS - 1) + 1
            
            tail_text = "\n\n".join([f"Step {start_num + i}: {txt}" for i, txt in enumerate(tail_steps)])
            
            history_text = (
                f"{head_step}\n\n"
                f"[... {omitted_count} intermediate logic steps dynamically omitted to preserve context bounds ...]\n\n"
                f"{tail_text}"
            )
        else:
            history_text = "\n\n".join([f"Step {i+1}: {txt}" for i, txt in enumerate(compressed_traj)])
        
        prompt = (
            f"Solve the original query based on the following trajectory of thoughts and tool observations.\n"
            f"Original query: {query}\n"
            f"Trajectory:\n{history_text}\n\n"
            f"If you need to use a tool to continue reasoning, output exactly <tool_call name=\"tool_name\">{{\"arg\":\"val\"}}</tool_call>.\n"
            f"If the trajectory contains the final answer, provide a conclusive explanation without tool calls."
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

    async def _contribute_to_workspace(self, message: Message) -> Message | None:
        """
        Participate in workspace deliberation by generating GoT thoughts.
        
        When a workspace.update arrives, runs a lightweight GoT process
        and posts the best-path result as a workspace.thought.
        """
        text = message.payload.get("text", "")
        if not text or len(text) < 10:
            return None

        # Check cache first
        cache_key = self._cache_key(text)
        if cache_key in self._response_cache:
            cached_content = self._response_cache[cache_key]
        else:
            # Create a synthetic planning request
            plan_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="planner.decompose",
                payload={"text": text},
                correlation_id=message.correlation_id or message.id,
            )
            
            result = await self.handle_message(plan_msg)
            if result is None:
                return None
            cached_content = result.payload.get("text", "")
        
        if not cached_content:
            return None

        # Post as a workspace thought
        thought_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.thought",
            payload={
                "type": "graph_of_thoughts",
                "confidence": 0.8,  # GoT thoughts get a confidence boost
                "content": cached_content,
                "metadata": {
                    "source": "got_planner",
                    "branch_factor": self.branch_factor,
                    "max_depth": self.max_depth,
                },
            },
            correlation_id=message.correlation_id or message.id,
        )
        await self.bus.publish("workspace.thought", thought_msg)
        logger.debug("[GoT] Posted workspace thought for: %s", text[:60])
        return None

