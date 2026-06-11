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
from typing import TYPE_CHECKING, Any

from hbllm.brain.utility_engine import CognitiveUtilityEngine, ThoughtBudget, UtilityBreakdown
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.policy_engine import PolicyEngine
    from hbllm.brain.provider_adapter import ProviderLLM

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

    def __init__(self) -> None:
        self.nodes: dict[str, ThoughtNode] = {}
        self.root_ids: list[str] = []

    def add_root(self, content: str, score: float = 0.0) -> ThoughtNode:
        """Add a root thought (no parents)."""
        node = ThoughtNode(content=content, score=score, depth=0)
        self.nodes[node.id] = node
        self.root_ids.append(node.id)
        return node

    def branch(
        self, parent_id: str, content: str, score: float = 0.0, is_observation: bool = False
    ) -> ThoughtNode:
        """Create a child thought branching from a parent."""
        parent = self.nodes[parent_id]
        child = ThoughtNode(
            content=content,
            score=score,
            depth=parent.depth + 1,
            parent_ids=[parent_id],
            trajectory_history=parent.trajectory_history + [parent.content],
        )
        if is_observation:
            child.metadata["is_observation"] = True
        parent.children_ids.append(child.id)
        self.nodes[child.id] = child
        return child

    def merge(self, parent_ids: list[str], content: str, score: float = 0.0) -> ThoughtNode:
        """Merge multiple parent thoughts into a synthesis node.

        Raises ValueError if the merge would create a cycle in the DAG.
        """
        # Cycle detection: ensure no parent is a descendant of another parent
        # (which would create a cycle when they share a child)
        for pid in parent_ids:
            ancestors = self._get_ancestors(pid)
            for other_pid in parent_ids:
                if other_pid != pid and other_pid in ancestors:
                    raise ValueError(
                        f"Merge would create cycle: '{other_pid}' is an ancestor of '{pid}'"
                    )

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

    def _get_ancestors(self, node_id: str) -> set[str]:
        """Get all ancestor node IDs for cycle detection."""
        ancestors: set[str] = set()
        stack = list(self.nodes[node_id].parent_ids)
        while stack:
            current = stack.pop()
            if current not in ancestors:
                ancestors.add(current)
                stack.extend(self.nodes[current].parent_ids)
        return ancestors

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
                (self.nodes[pid] for pid in current.parent_ids), key=lambda n: (n.visits, n.q_value)
            )
            path.append(best_parent)
            current = best_parent

        path.reverse()
        return path

    def select_leaf_uct(
        self, root_id: str, c_param: float = 1.414, prm_weight: float = 0.5
    ) -> ThoughtNode:
        import math

        """Select a leaf node to expand using the UCT formula, blended with PRM scores."""
        current = self.nodes[root_id]

        while not current.is_leaf:
            # If any child is completely unexplored, return it instantly
            unvisited = [
                self.nodes[cid] for cid in current.children_ids if self.nodes[cid].visits == 0
            ]
            if unvisited:
                return unvisited[0]

            # UCT Selection (blending standard MCTS Q-value with immediate PRM score)
            best_score = -float("inf")
            best_child = None
            parent_visits = current.visits

            for child_id in current.children_ids:
                child = self.nodes[child_id]

                # Blend the long-term Q-value with the immediate step score (PRM)
                blended_q = (1.0 - prm_weight) * child.q_value + prm_weight * child.score

                # Exploration term: c * sqrt(ln(N) / n)
                exploration = c_param * math.sqrt(math.log(parent_visits) / child.visits)

                score = blended_q + exploration

                if score > best_score:
                    best_score = score
                    best_child = child

            if not best_child:
                break
            current = best_child

        return current

    def backpropagate(self, leaf_id: str, reward: float, decay: float = 0.95) -> None:
        """Propagate reward up the tree to the root with optional decay."""
        current = self.nodes[leaf_id]
        current_reward = reward
        while True:
            current.visits += 1
            current.cumulative_reward += current_reward
            # Backprop up the first primary parent chain
            if not current.parent_ids:
                break
            current = self.nodes[current.parent_ids[0]]
            current_reward *= decay  # Discount future rewards

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
                sum(n.score for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0.0
            ),
        }


class PlannerNode(Node):
    """
    Graph-of-Thoughts planner that decomposes complex queries into
    a DAG of reasoning steps with branching, scoring, merging, and pruning.
    """

    # Max cached prompt/response pairs
    MAX_CACHE_SIZE = 200

    def __init__(
        self,
        node_id: str,
        branch_factor: int = 3,
        max_depth: int = 2,
        policy_engine: PolicyEngine | None = None,
        cache_ttl_seconds: float = 3600.0,
        llm: ProviderLLM | None = None,
        utility_engine: CognitiveUtilityEngine | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PLANNER,
            capabilities=["task_decomposition", "graph_of_thoughts", "aggregation"],
        )
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.policy_engine = policy_engine  # PolicyEngine for plan validation
        self.cache_ttl_seconds = cache_ttl_seconds
        # LRU cache: hash(prompt) → (response text, timestamp)
        self._response_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.llm = llm
        self.utility_engine = utility_engine or CognitiveUtilityEngine()

    def _count_tokens(self, text: str) -> int:
        """Estimate or count the number of tokens in text."""
        if self.llm and hasattr(self.llm, "tokenizer") and self.llm.tokenizer:
            try:
                enc = self.llm.tokenizer.encode(text)
                if hasattr(enc, "ids"):
                    return len(enc.ids)
                elif isinstance(enc, list):
                    return len(enc)
            except Exception:
                pass
        # Fallback to rough character-based token estimation (4 chars per token)
        return max(1, len(text) // 4)

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
        domain_hint = message.payload.get("domain_hint", "general")

        # Check prompt cache for exact matches
        cache_key = self._cache_key(text)
        if cache_key in self._response_cache:
            import time

            cached, timestamp = self._response_cache[cache_key]
            if time.time() - timestamp <= self.cache_ttl_seconds:
                self._cache_hits += 1
                # Move to end (most recently used)
                self._response_cache.move_to_end(cache_key)
                logger.info(
                    "[GoT] Cache HIT for query (hits=%d, misses=%d)",
                    self._cache_hits,
                    self._cache_misses,
                )
                return message.create_response(
                    {"text": cached, "domain": "planner", "cached": True}
                )
            else:
                # Evict expired entry
                self._response_cache.pop(cache_key, None)
                logger.debug("[GoT] Cache EXPIRED for query")

        # Instantiate ThoughtBudget
        budget_payload = message.payload.get("thought_budget", {})
        max_tokens = budget_payload.get("max_tokens", 8192)
        max_time_ms = budget_payload.get("max_time_ms", 15000.0)
        max_branches = budget_payload.get("max_branches", 20)
        budget = ThoughtBudget(
            max_tokens=max_tokens, max_time_ms=max_time_ms, max_branches=max_branches
        )

        self._cache_misses += 1
        logger.info("[GoT] Starting Graph-of-Thoughts for: '%s...'", text[:40])

        graph = ThoughtGraph()

        # ── Step 1: Root Node & Initial Plausible Branches ────────────────
        root_node = graph.add_root("Root Query: " + text[:50], score=0.5)

        initial_thoughts = await asyncio.gather(
            *[
                self._generate_thought(text, i + 1, domain_hint, budget)
                for i in range(self.branch_factor)
            ]
        )

        branch_nodes = []
        for i, res_tuple in enumerate(initial_thoughts):
            if res_tuple:
                thought_text, tokens, latency = res_tuple
                if thought_text:
                    child = graph.branch(root_node.id, thought_text)
                    child.metadata["tokens_used"] = tokens
                    child.metadata["latency_ms"] = latency
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
                *[self._score_thought(graph, node, budget) for node in unblocked_branches]
            )
            # Backpropagate initial scores
            for node in unblocked_branches:
                graph.backpropagate(node.id, node.score)

        # Compute utility for policy-blocked initial nodes
        blocked_branches = [n for n in branch_nodes if n.metadata.get("policy_blocked")]
        for node in blocked_branches:
            tokens_used = node.metadata.get("tokens_used", 0)
            latency_ms = node.metadata.get("latency_ms", 0.0)
            breakdown = self.utility_engine.calculate_utility(
                progress_score=node.score,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                risk_score=1.0,
            )
            node.metadata["utility_breakdown"] = breakdown.to_dict()
            node.metadata["utility_score"] = breakdown.utility

        # ── Step 3: MCTS Compute Loop (Infinite Test-Time Compute) ────────
        # Run until depth budget or time budget exhausted
        iteration = 0

        while not budget.is_exhausted() and iteration < budget.max_branches:
            # Marginal Utility Gating: check if we should think more
            scored_nodes = [n for n in graph.nodes.values() if "utility_breakdown" in n.metadata]
            if len(scored_nodes) >= 2:
                avg_token_cost = sum(
                    n.metadata["utility_breakdown"]["token_cost"] for n in scored_nodes
                ) / len(scored_nodes)
                avg_latency_cost = sum(
                    n.metadata["utility_breakdown"]["latency_cost"] for n in scored_nodes
                ) / len(scored_nodes)
                expected_step_cost = avg_token_cost + avg_latency_cost

                best_utility = max(n.metadata["utility_score"] for n in scored_nodes)
                best_score = max(n.score for n in scored_nodes)

                expected_new_progress = min(1.0, best_score + 0.05)
                expected_next_utility = expected_new_progress - expected_step_cost
                marginal_utility = expected_next_utility - best_utility

                if marginal_utility < 0:
                    logger.info(
                        "[MCTS] Terminating search early due to negative marginal utility (%.3f). Best Utility: %.3f",
                        marginal_utility,
                        best_utility,
                    )
                    break

            # 1. Selection
            leaf = graph.select_leaf_uct(root_node.id, c_param=1.414)

            # 2. Expansion (only expand if we've already scored it and haven't hit max depth)
            if leaf.visits > 0 and leaf.depth < self.max_depth:
                await self._refine_thought(graph, leaf, text, iteration, domain_hint, budget)

                # Pick one of the newly expanded children to simulate
                unvisited = [
                    graph.nodes[cid] for cid in leaf.children_ids if graph.nodes[cid].visits == 0
                ]
                if unvisited:
                    leaf = unvisited[0]

            # 3. Simulation & Scoring
            if leaf.visits == 0:
                await self._score_thought(graph, leaf, budget)
                reward = leaf.score
            else:
                # If we selected a terminal node that can't expand, re-simulate or just use existing score
                reward = leaf.score

            # EARLY CONVERGENCE EXIT
            if reward > 0.90:
                logger.info(
                    "[MCTS] Early convergence reached at iteration %d (Score: %.2f)",
                    iteration,
                    reward,
                )
                graph.backpropagate(leaf.id, reward)
                break

            # 4. Backpropagation
            graph.backpropagate(leaf.id, reward)

            iteration += 1

        logger.info("[MCTS] Completed %d iterations before budget exhaustion.", iteration)

        # ── Step 4: Optional Merge — Combine complementary top thoughts ───
        top_leaves = sorted(
            [
                n
                for n in graph.nodes.values()
                if n.is_leaf and n.visits > 0 and not n.metadata.get("policy_blocked")
            ],
            key=lambda n: n.q_value,
            reverse=True,
        )[:2]

        if len(top_leaves) >= 2 and top_leaves[0].q_value > 0.5:
            if not budget.is_exhausted():
                merged_content, merge_tokens, merge_latency = await self._merge_thoughts(
                    text, top_leaves[0].content, top_leaves[1].content, domain_hint, budget
                )
                if merged_content:
                    merged_node = graph.merge([n.id for n in top_leaves], merged_content)
                    merged_node.metadata["tokens_used"] = merge_tokens
                    merged_node.metadata["latency_ms"] = merge_latency
                    await self._score_thought(graph, merged_node, budget)
                    graph.backpropagate(merged_node.id, merged_node.score)

        # ── Step 7: Select best path ──────────────────────────────────────
        best_path = graph.best_path()
        graph_stats = graph.summary()

        if not best_path:
            return message.create_error("GoT: No valid reasoning path found.")

        final_node = best_path[-1]

        path_desc = " → ".join(f"[D{n.depth}: Q={n.q_value:.2f}, N={n.visits}]" for n in best_path)

        final_text = (
            f"[MCTS Planner] Explored {graph_stats['total_nodes']} thoughts "
            f"across {graph_stats['max_depth'] + 1} depths "
            f"({graph_stats['merged_count']} merged, "
            f"avg score: {graph_stats['avg_score']:.2f}).\n"
            f"Best path: {path_desc}\n\n"
            f"{final_node.content}"
        )

        logger.info(
            "[MCTS] Selected path: %s (Q=%.2f, N=%d)",
            path_desc,
            final_node.q_value,
            final_node.visits,
        )

        # Cache the result
        self._cache_response(cache_key, final_text)

        return message.create_response(
            {
                "text": final_text,
                "domain": "planner",
                "thought_budget_status": budget.to_dict(),
                "best_path_utility": [n.metadata.get("utility_breakdown") for n in best_path],
                "total_nodes_explored": graph_stats["total_nodes"],
            }
        )

    # ── Cache helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key from prompt text."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def _cache_response(self, key: str, response: str) -> None:
        """Store a response in the LRU cache, evicting oldest if full."""
        import time

        self._response_cache[key] = (response, time.time())
        while len(self._response_cache) > self.MAX_CACHE_SIZE:
            self._response_cache.popitem(last=False)

    # ── Private helpers ───────────────────────────────────────────────────

    async def _generate_thought(
        self,
        query: str,
        branch_id: int,
        domain_hint: str = "general",
        budget: ThoughtBudget | None = None,
    ) -> tuple[str, int, float]:
        """Generate one diverse initial thought. Returns (content, tokens_used, latency_ms)."""
        import time

        start_t = time.time()
        tokens_used = 0

        prompt = f"Exploring Approach {branch_id} to solve: {query}"
        prompt_tokens = self._count_tokens(prompt)
        tokens_used += prompt_tokens
        if budget:
            budget.spend_branch(1)
            budget.spend_tokens(prompt_tokens)

        content = ""
        if self.llm:
            try:
                res = await self.llm.generate(prompt)
                content = str(res)
            except Exception as e:
                logger.warning("[GoT] Direct LLM thought generation failed: %s", e)

        if not content:
            topic = (
                f"domain.{domain_hint}.query"
                if domain_hint != "general"
                else "domain.general.query"
            )
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic=topic,
                payload={"text": prompt},
            )
            try:
                resp = await self.request(topic, req, timeout=90.0)
                content = str(resp.payload.get("text", ""))
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("[GoT] Branch %d generation timed out: %s", branch_id, e)
            except Exception:
                logger.exception("[GoT] Branch %d generation failed unexpectedly", branch_id)

        response_tokens = self._count_tokens(content)
        tokens_used += response_tokens
        if budget:
            budget.spend_tokens(response_tokens)

        latency_ms = (time.time() - start_t) * 1000.0
        return content, tokens_used, latency_ms

    async def _score_thought(
        self, graph: ThoughtGraph, node: ThoughtNode, budget: ThoughtBudget | None = None
    ) -> None:
        """Score a thought node by querying the evaluation domain or ExecutionNode/ToolRouter."""
        import time

        if node.metadata.get("is_observation"):
            node.score = 1.0  # Observations are always factual steps
            # Compute utility for observations too
            tokens_used = node.metadata.get("tokens_used", 0)
            latency_ms = node.metadata.get("latency_ms", 0.0)
            breakdown = self.utility_engine.calculate_utility(
                progress_score=node.score,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                risk_score=0.0,
            )
            node.metadata["utility_breakdown"] = breakdown.to_dict()
            node.metadata["utility_score"] = breakdown.utility
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
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("[GoT] Execution verification timed out: %s", e)
            except Exception:
                logger.exception("[GoT] Execution verification failed unexpectedly")

        # 1.5. Tool Call Interception
        tool_match = re.search(
            r"<tool_call\s+name=[\"'](.*?)[\"']>(.*?)</tool_call>",
            node.content,
            re.DOTALL | re.IGNORECASE,
        )
        if (
            tool_match
            and "execution_output" not in node.metadata
            and "execution_error" not in node.metadata
        ):
            tool_name = tool_match.group(1).strip()
            tool_args_str = tool_match.group(2).strip()
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="action.tool_call",
                payload={"tool_name": tool_name, "arguments": tool_args_str},
            )
            try:
                resp = await self.request("action.tool_call", req, timeout=90.0)
                if resp.type == MessageType.ERROR:
                    err = resp.payload.get("error", "Unknown tool error")
                    obs_content = f"Observation: Error: {err}"
                else:
                    obs_content = f"Observation:\n{resp.payload}"

                # Expand the tree deterministically with the observation
                graph.branch(node.id, obs_content, score=1.0, is_observation=True)
                node.score = 1.0  # Reward emitting a valid tool call
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("[GoT] Tool execution timed out: %s", e)
                obs_content = f"Observation: Execution timed out: {str(e)}"
                graph.branch(node.id, obs_content, score=0.1, is_observation=True)
                node.score = 0.5
            except Exception as e:
                logger.exception("[GoT] Tool execution failed unexpectedly")
                obs_content = f"Observation: Execution failed unexpectedly: {str(e)}"
                graph.branch(node.id, obs_content, score=0.1, is_observation=True)
                node.score = 0.5

        # 1.6. Skill Call Interception (SIL)
        skill_match = re.search(
            r"<skill_call\s+task=[\"'](.*?)[\"']>(.*?)</skill_call>",
            node.content,
            re.DOTALL | re.IGNORECASE,
        )
        if (
            skill_match
            and "execution_output" not in node.metadata
            and "execution_error" not in node.metadata
            and not tool_match
        ):
            task_desc = skill_match.group(1).strip()
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="action.sil_execute",
                payload={"task": task_desc},
            )
            try:
                resp = await self.request("action.sil_execute", req, timeout=120.0)
                if resp.type == MessageType.ERROR:
                    err = resp.payload.get("error", "Unknown SIL error")
                    obs_content = f"Observation: SIL Error: {err}"
                else:
                    status = resp.payload.get("status")
                    if status == "SUCCESS":
                        obs_content = f"Observation: SIL perfectly executed skill '{resp.payload.get('skill')}'"
                    else:
                        obs_content = f"Observation:\n{resp.payload}"

                graph.branch(node.id, obs_content, score=1.0, is_observation=True)
                node.score = 1.0  # Reward emitting a valid skill call
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("[GoT] SIL execution timed out: %s", e)
                obs_content = f"Observation: SIL execution timed out: {str(e)}"
                graph.branch(node.id, obs_content, score=0.1, is_observation=True)
                node.score = 0.5
            except Exception as e:
                logger.exception("[GoT] SIL execution failed unexpectedly")
                obs_content = f"Observation: SIL execution failed unexpectedly: {str(e)}"
                graph.branch(node.id, obs_content, score=0.1, is_observation=True)
                node.score = 0.5

        # If not evaluated by deterministic checkers, use Process Reward Model (PRM)
        if node.score == 0.0:
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="action.score_thought",
                payload={"content": node.content},
            )
            if budget:
                budget.spend_tokens(self._count_tokens(node.content))

            try:
                # PRM scoring is extremely fast (single forward pass)
                resp = await self.request("action.score_thought", req, timeout=5.0)
                score = resp.payload.get("score", 0.5)
                node.score = min(max(float(score), 0.0), 1.0)
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("[GoT] PRM evaluation timed out, defaulting to 0.5: %s", e)
                node.score = 0.5
            except Exception:
                logger.exception("[GoT] PRM evaluation failed unexpectedly, defaulting to 0.5")
                node.score = 0.5

        # Compute utility score
        tokens_used = node.metadata.get("tokens_used", 0)
        latency_ms = node.metadata.get("latency_ms", 0.0)
        risk_score = 1.0 if node.metadata.get("policy_blocked") else 0.0
        breakdown = self.utility_engine.calculate_utility(
            progress_score=node.score,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            risk_score=risk_score,
        )
        node.metadata["utility_breakdown"] = breakdown.to_dict()
        node.metadata["utility_score"] = breakdown.utility

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
        self,
        graph: ThoughtGraph,
        parent: ThoughtNode,
        query: str,
        refinement_id: int,
        domain_hint: str = "general",
        budget: ThoughtBudget | None = None,
    ) -> None:
        """Refine a thought by branching deeper with an adaptive context window."""
        import time

        start_t = time.time()
        tokens_used = 0

        trajectory = parent.trajectory_history + [parent.content]
        compressed_traj = [self._compress_text(txt) for txt in trajectory]

        MAX_CONTEXT_STEPS = 6
        if len(compressed_traj) > MAX_CONTEXT_STEPS:
            omitted_count = len(compressed_traj) - MAX_CONTEXT_STEPS
            head_step = f"Step 1: {compressed_traj[0]}"

            tail_steps = compressed_traj[-(MAX_CONTEXT_STEPS - 1) :]
            start_num = len(compressed_traj) - (MAX_CONTEXT_STEPS - 1) + 1

            tail_text = "\n\n".join(
                [f"Step {start_num + i}: {txt}" for i, txt in enumerate(tail_steps)]
            )

            history_text = (
                f"{head_step}\n\n"
                f"[... {omitted_count} intermediate logic steps dynamically omitted to preserve context bounds ...]\n\n"
                f"{tail_text}"
            )
        else:
            history_text = "\n\n".join(
                [f"Step {i + 1}: {txt}" for i, txt in enumerate(compressed_traj)]
            )

        prompt = (
            f"Solve the original query based on the following trajectory of thoughts and tool observations.\n"
            f"Original query: {query}\n"
            f"Trajectory:\n{history_text}\n\n"
            f'If you need to use a tool to continue reasoning, output exactly <tool_call name="tool_name">{{"arg":"val"}}</tool_call>.\n'
            f'If you want to delegate a complex sub-task to the Skill Intelligence Layer, output exactly <skill_call task="describe task">fallback args</skill_call>.\n'
            f"If the trajectory contains the final answer, provide a conclusive explanation without tool calls."
        )

        prompt_tokens = self._count_tokens(prompt)
        tokens_used += prompt_tokens
        if budget:
            budget.spend_branch(1)
            budget.spend_tokens(prompt_tokens)

        content = ""
        if self.llm:
            try:
                content = await self.llm.generate(prompt)
            except Exception as e:
                logger.warning("[GoT] Direct LLM refinement failed: %s", e)

        if not content:
            topic = (
                f"domain.{domain_hint}.query"
                if domain_hint != "general"
                else "domain.general.query"
            )
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic=topic,
                payload={"text": prompt},
            )
            try:
                resp = await self.request(topic, req, timeout=90.0)
                content = resp.payload.get("text", "")
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("[GoT] Refinement timed out: %s", e)
            except Exception:
                logger.exception("[GoT] Refinement failed unexpectedly")

        response_tokens = self._count_tokens(content)
        tokens_used += response_tokens
        if budget:
            budget.spend_tokens(response_tokens)

        latency_ms = (time.time() - start_t) * 1000.0

        if content:
            child = graph.branch(parent.id, content)
            child.metadata["tokens_used"] = tokens_used
            child.metadata["latency_ms"] = latency_ms

    async def _merge_thoughts(
        self,
        query: str,
        thought_a: str,
        thought_b: str,
        domain_hint: str = "general",
        budget: ThoughtBudget | None = None,
    ) -> tuple[str, int, float]:
        """Merge two complementary thoughts into a synthesis. Returns (content, tokens_used, latency_ms)."""
        import time

        start_t = time.time()
        tokens_used = 0

        prompt = (
            f"Synthesize these two approaches into a single, comprehensive solution:\n"
            f"Query: {query}\n"
            f"Approach A: {thought_a[:200]}\n"
            f"Approach B: {thought_b[:200]}\n"
            f"Provide a unified, improved answer."
        )

        prompt_tokens = self._count_tokens(prompt)
        tokens_used += prompt_tokens
        if budget:
            budget.spend_branch(1)
            budget.spend_tokens(prompt_tokens)

        content = ""
        if self.llm:
            try:
                res = await self.llm.generate(prompt)
                content = str(res)
            except Exception as e:
                logger.warning("[GoT] Direct LLM merge failed: %s", e)

        if not content:
            topic = (
                f"domain.{domain_hint}.query"
                if domain_hint != "general"
                else "domain.general.query"
            )
            req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic=topic,
                payload={"text": prompt},
            )
            try:
                resp = await self.request(topic, req, timeout=90.0)
                content = str(resp.payload.get("text", ""))
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("[GoT] Merge timed out: %s", e)
            except Exception:
                logger.exception("[GoT] Merge failed unexpectedly")

        response_tokens = self._count_tokens(content)
        tokens_used += response_tokens
        if budget:
            budget.spend_tokens(response_tokens)

        latency_ms = (time.time() - start_t) * 1000.0
        return content, tokens_used, latency_ms

    async def _contribute_to_workspace(self, message: Message) -> Message | None:
        """
        Participate in workspace deliberation by generating GoT thoughts.

        When a workspace.update arrives, runs a lightweight GoT process
        and posts the best-path result as a workspace.thought.

        Adaptive: skips expensive GoT for simple queries or fast-path flagged
        requests. Only runs full MCTS for complex reasoning tasks.
        """
        text = message.payload.get("text", "")
        if not text or len(text) < 10:
            return None

        # ── Adaptive execution: skip GoT for simple/fast-path queries ─────
        intent = message.payload.get("intent", "general_knowledge")
        is_fast_path = message.payload.get("is_fast_path", False)
        _skip_intents = {"general_knowledge", "smalltalk"}

        # Always skip for fast-path and simple intents (regardless of hardware)
        if is_fast_path or intent in _skip_intents:
            logger.debug(
                "[GoT] Skipping GoT for %s query (intent=%s, fast_path=%s)",
                "fast-path" if is_fast_path else "simple",
                intent,
                is_fast_path,
            )
            return None

        # Check cache first
        cache_key = self._cache_key(text)
        cached_content = None
        if cache_key in self._response_cache:
            import time

            cached, timestamp = self._response_cache[cache_key]
            if time.time() - timestamp <= self.cache_ttl_seconds:
                cached_content = cached
            else:
                self._response_cache.pop(cache_key, None)

        if cached_content is None:
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
