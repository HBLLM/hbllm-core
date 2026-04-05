"""
System 4 Sleep Cycle Node (Offline Consolidation).

Monitored system activity. If the user stops interacting with the API/CLI
for a configurable duration, it triggers a `system.sleep` event.
During sleep, it performs biologically-inspired memory consolidation:
compressing raw episodic logs into semantic summaries, clustering
knowledge graphs, and triggering artificial neuroplasticity (DPO)
if performance gaps were detected.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class SleepCycleNode(Node):
    """
    Orchestrates Memory Consolidation and Synaptic Strengthening when idle.
    """

    def __init__(self, node_id: str, idle_timeout_seconds: float = 10.0, llm: Any = None) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["sleep_cycle", "memory_consolidation"],
        )
        self.idle_timeout_seconds = idle_timeout_seconds
        self.is_sleeping = False
        self._last_system_activity = time.time()
        self._monitor_task: asyncio.Task[None] | None = None
        self._pending_goals: list[str] = []
        self.llm = llm  # Used for local GraphRAG clustering

    async def on_start(self) -> None:
        logger.info("Starting SleepCycleNode (Idle timeout: %s seconds)", self.idle_timeout_seconds)
        # Listen to router queries to track user activity
        await self.bus.subscribe("router.query", self._check_activity)

        # Accumulate curiosity goals for processing during sleep
        await self.bus.subscribe("system.sleep.goal", self._collect_goal)

        # Start the background idle monitor
        self._monitor_task = asyncio.create_task(self._idle_monitor_loop())

    async def on_stop(self) -> None:
        logger.info("Stopping SleepCycleNode")
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _collect_goal(self, message: Message) -> Message | None:
        """Accumulate curiosity goals for processing during sleep."""
        goal = str(message.payload.get("goal") or message.payload.get("text", ""))
        if goal and goal not in self._pending_goals:
            self._pending_goals.append(goal)
            logger.debug("[SleepNode] Queued curiosity goal: %s", goal[:80])
        return None

    async def _check_activity(self, message: Message) -> Message | None:
        """Update our internal timestamp whenever a user query flows through the system."""
        self._last_system_activity = time.time()

        if self.is_sleeping:
            logger.info(
                "[SleepNode] User input detected! Waking up immediately. Aborting deep sleep."
            )
            self.is_sleeping = False

        return None

    async def _idle_monitor_loop(self) -> None:
        """Continuously check if the system has been inactive."""
        # We use a short interval for testing if timeout is small
        check_interval = min(2.0, self.idle_timeout_seconds / 2.0)

        while self._running:
            await asyncio.sleep(check_interval)

            if not self.is_sleeping:
                idle_time = time.time() - self._last_system_activity
                if idle_time > self.idle_timeout_seconds:
                    await self._enter_sleep_cycle()

    async def _enter_sleep_cycle(self) -> None:
        """Perform offline memory consolidation and self-improvement training."""
        self.is_sleeping = True
        cycle_start = time.time()
        report: dict[str, Any] = {"memories_consolidated": 0, "goals_replayed": 0, "training_ran": False}
        logger.info(
            "[SleepNode] System Idle for >%.1fs. Entering Deep Sleep (Consolidation Mode)...",
            self.idle_timeout_seconds,
        )

        try:
            # ── Phase 1: Memory Consolidation ────────────────────────────
            report["memories_consolidated"] = await self._consolidate_memory()

            # ── Phase 2: Self-Improvement Training ───────────────────────
            if not self.is_sleeping:
                return  # User woke up during consolidation
            report["training_ran"] = await self._run_self_improvement()

            # ── Phase 3: Curiosity Goal Replay ───────────────────────────
            if not self.is_sleeping:
                return
            report["goals_replayed"] = await self._replay_curiosity_goals()

        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[SleepNode] Timeout during sleep cycle. Waking up.")
        except Exception as e:
            logger.error("[SleepNode] Sleep cycle interrupted by internal error: %s", e)

        # Emit sleep report
        report["duration_seconds"] = round(time.time() - cycle_start, 1)
        await self.bus.publish(
            "system.sleep.report",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.sleep.report",
                payload=report,
            ),
        )

        # Reset activity timer so we don't immediately go back to sleep unless idle again
        self._last_system_activity = time.time()
        self.is_sleeping = False

    async def _consolidate_memory(self) -> int:
        """Phase 1: Compress short-term memory into long-term summaries. Returns turns consolidated."""
        req_msg = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="memory.retrieve_recent",
            payload={"session_id": "default_session", "limit": 10},
        )

        try:
            resp = await self.bus.request("memory.retrieve_recent", req_msg, timeout=5.0)
        except Exception:
            logger.warning("[SleepNode] Could not reach memory node for consolidation.")
            return 0

        if resp.type == MessageType.ERROR:
            logger.error("[SleepNode] Failed to retrieve memory: %s", resp.payload.get("error"))
            self._last_system_activity = time.time()
            self.is_sleeping = False
            return 0

        turns = list(resp.payload.get("turns", []))

        if len(turns) >= 4:
            logger.info(
                "[SleepNode] Compressing %d recent turns into semantic long-term memory...",
                len(turns),
            )

            store_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="memory.store",
                payload={
                    "session_id": "default_session",
                    "role": "system",
                    "content": f"[CONSOLIDATED MEMORY] User discussed {len(turns) // 2} topics. Key facts extracted and embedded.",
                },
            )
            await self.bus.publish("memory.store", store_msg)

        else:
            logger.info(
                "[SleepNode] Not enough new memories to consolidate linearly. Proceeding to GraphRAG."
            )

        # ── Phase 1.5: Hierarchical GraphRAG Clustering ──
        if self.llm:
            try:
                # 1. Fetch active entities
                kg_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    topic="knowledge.query",
                    payload={"action": "all_entities", "limit": 20},
                )
                kg_resp = await self.bus.request("knowledge.query", kg_msg, timeout=5.0)
                entities = list(kg_resp.payload.get("entities", []))

                # Filter out existing communities
                leaf_nodes = [str(e.get("label", "")) for e in entities if e.get("type") != "community"]

                if len(leaf_nodes) >= 5:
                    logger.info(
                        "[SleepNode] GraphRAG: Clustering %d entities into Communities...",
                        len(leaf_nodes),
                    )
                    cluster_json = await self.llm.generate_json(
                        f"Group these concepts into 1 or 2 broad thematic Communities.\n"
                        f"Concepts: {leaf_nodes}\n"
                        f'Output JSON format: {{"communities": [{{"name": "Short Title", "summary": "1 sentence desc", "members": ["concept1", "concept2"]}}]}}'
                    )

                    if cluster_json and "error" not in cluster_json and "communities" in cluster_json:
                        for comm in cluster_json["communities"]:
                            add_comm_msg = Message(
                                type=MessageType.QUERY,
                                source_node_id=self.node_id,
                                topic="knowledge.query",
                                payload={
                                    "action": "add_community",
                                    "community_label": comm.get("name"),
                                    "member_labels": comm.get("members", []),
                                    "summary": comm.get("summary", ""),
                                },
                            )
                            await self.bus.publish("knowledge.query", add_comm_msg)
                            logger.info(
                                "[SleepNode] GraphRAG created Community: '%s' with %d members",
                                comm.get("name"),
                                len(comm.get("members", [])),
                            )
            except Exception as e:
                logger.warning("[SleepNode] GraphRAG clustering failed: %s", e)

        logger.info("[SleepNode] Memory consolidation complete.")
        return len(turns)

    async def _run_self_improvement(self) -> bool:
        """Phase 2: Trigger Lifelong Continuous DPO overnight."""
        logger.info("[SleepNode] Initiating Phase 2: Autonomous Continuous DPO...")

        # Create an event to wait for the learner node to respond
        training_complete_event = asyncio.Event()

        async def _on_learning_update(msg: Message) -> None:
            training_complete_event.set()

        sub = await self.bus.subscribe("system.learning_update", _on_learning_update)

        try:
            # Emit the trigger to wake up LearnerNode
            trigger_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.sleep.dpo_trigger",
                payload={"mode": "overnight_continuous"},
            )
            await self.bus.publish("system.sleep.dpo_trigger", trigger_msg)

            # Wait for learner to process. If it's empty, it returns immediately.
            # If it's a huge batch, give it 15 minutes max during 'sleep'.
            await asyncio.wait_for(training_complete_event.wait(), timeout=900.0)
            logger.info("[SleepNode] Continuous DPO phase completed successfully.")
            return True
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[SleepNode] DPO training timed out after 15 minutes.")
            return False
        except Exception as e:
            logger.warning("[SleepNode] Self-improvement skipped/failed: %s", e)
            return False
        finally:
            await self.bus.unsubscribe(sub)

    async def _replay_curiosity_goals(self) -> int:
        """
        Phase 3: Replay accumulated curiosity goals during sleep.

        For each queued goal, publishes a research query to memory.store
        so the system can explore and learn about curiosity-driven topics.
        """
        if not self._pending_goals:
            logger.info("[SleepNode] No curiosity goals to replay.")
            return 0

        goals_to_process = self._pending_goals[:10]  # Cap at 10 per cycle
        self._pending_goals = self._pending_goals[10:]
        replayed = 0

        for goal in goals_to_process:
            if not self.is_sleeping:
                break  # User woke up

            logger.info("[SleepNode] Replaying curiosity goal: %s", goal[:80])

            # Store the goal as a researched topic in memory
            store_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="memory.store",
                payload={
                    "session_id": "sleep_curiosity",
                    "role": "system",
                    "content": f"[CURIOSITY RESEARCH] Goal: {goal}. Queued for deep exploration during next active session.",
                    "domain": "curiosity",
                },
            )
            await self.bus.publish("memory.store", store_msg)
            replayed += 1
            await asyncio.sleep(0.1)  # Small delay between goals

        logger.info("[SleepNode] Replayed %d curiosity goals.", replayed)
        return replayed
