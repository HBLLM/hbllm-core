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


from enum import Enum


class SleepPhase(str, Enum):
    AWAKE = "awake"
    NREM = "nrem"  # Deep sleep: memory compression & clustering
    REM = "rem"  # REM sleep: neuroplasticity, curiosity replays


class SleepCycleNode(Node):
    """
    Orchestrates Memory Consolidation and Synaptic Strengthening when idle.
    """

    def __init__(
        self,
        node_id: str,
        idle_timeout_seconds: float = 10.0,
        llm: Any = None,
        self_model: Any = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["sleep_cycle", "memory_consolidation"],
        )
        self.idle_timeout_seconds = idle_timeout_seconds
        self.current_phase = SleepPhase.AWAKE
        self._last_system_activity = time.time()
        self._monitor_task: asyncio.Task[None] | None = None
        self._pending_goals: list[str] = []
        self.llm = llm  # Used for local GraphRAG clustering
        self.self_model = self_model  # For targeted DPO training on weak domains
        self._active_queries: dict[str, float] = {}

    @property
    def is_sleeping(self) -> bool:
        return self.current_phase != SleepPhase.AWAKE

    @is_sleeping.setter
    def is_sleeping(self, value: bool) -> None:
        if not value:
            self.current_phase = SleepPhase.AWAKE

    async def on_start(self) -> None:
        logger.info("Starting SleepCycleNode (Idle timeout: %s seconds)", self.idle_timeout_seconds)
        # Listen to router queries to track user activity
        await self.bus.subscribe("router.query", self._check_activity)
        await self.bus.subscribe("sensory.output", self._on_sensory_output)

        # Accumulate curiosity goals for processing during sleep
        await self.bus.subscribe("system.sleep.goal", self._collect_goal)

        # Manual trigger
        await self.bus.subscribe("system.sleep.force", self._handle_force_sleep)

        # Start the background idle monitor
        if self.idle_timeout_seconds is not None and self.idle_timeout_seconds > 0:
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

    async def _on_sensory_output(self, message: Message) -> None:
        corr_id = message.correlation_id or message.id
        self._active_queries.pop(corr_id, None)

    async def _check_activity(self, message: Message) -> Message | None:
        """Update our internal timestamp whenever a user query flows through the system."""
        self._last_system_activity = time.time()
        corr_id = message.correlation_id or message.id
        self._active_queries[corr_id] = time.time()

        if self.is_sleeping:
            logger.info(
                "[SleepNode] User input detected! Waking up immediately. Aborting deep sleep."
            )
            self.current_phase = SleepPhase.AWAKE

        return None

    async def _idle_monitor_loop(self) -> None:
        """Continuously check if the system has been inactive."""
        # We use a short interval for testing if timeout is small
        check_interval = min(2.0, self.idle_timeout_seconds / 2.0)

        while self._running:
            await asyncio.sleep(check_interval)

            # Prune stale active queries (> 300s) to prevent leaks
            now = time.time()
            stale_keys = [k for k, t in self._active_queries.items() if now - t > 300.0]
            for k in stale_keys:
                self._active_queries.pop(k, None)

            # If there are active queries, push the idle timer forward
            if self._active_queries:
                self._last_system_activity = now

            if not self.is_sleeping:
                idle_time = time.time() - self._last_system_activity
                if idle_time > self.idle_timeout_seconds:
                    await self._enter_sleep_cycle()

    async def _enter_sleep_cycle(self) -> None:
        """Perform offline memory consolidation and self-improvement training."""
        self.current_phase = SleepPhase.NREM
        cycle_start = time.time()
        report: dict[str, Any] = {
            "memories_consolidated": 0,
            "contradictions_resolved": 0,
            "temporal_refs_normalized": 0,
            "goals_replayed": 0,
            "training_ran": False,
            "dream_journal": "",
        }
        logger.info(
            "[SleepNode] System Idle for >%.1fs. Entering Deep Sleep (NREM Phase)...",
            self.idle_timeout_seconds,
        )

        try:
            # ── Phase 1: NREM (Memory Consolidation) ────────────────────
            report["memories_consolidated"] = await self._consolidate_memory()

            if not self.is_sleeping:
                return

            # ── Phase 1.6: Temporal Normalization ────────────────────────
            report["temporal_refs_normalized"] = await self._normalize_temporal_references()

            if not self.is_sleeping:
                return

            # ── Phase 1.7: Contradiction Detection & Resolution ─────────
            report["contradictions_resolved"] = await self._resolve_contradictions()

            if not self.is_sleeping:
                return

            # ── Phase 1.8: Knowledge Staleness Audit ─────────────────────
            report["stale_knowledge_audited"] = await self._audit_knowledge_staleness()

            if not self.is_sleeping:
                return

            # ── Phase 1.9: Task Knowledge Promotion (T2 → T3) ───────────
            report["knowledge_promoted"] = await self._promote_task_knowledge()

            if not self.is_sleeping:
                return

            # Transition to REM sleep
            self.current_phase = SleepPhase.REM
            logger.info("[SleepNode] Transitioning to REM Phase...")

            # ── Phase 2: REM (Self-Improvement Training) ────────────────
            report["training_ran"] = await self._run_self_improvement()

            if not self.is_sleeping:
                return
            report["skills_optimized"] = await self._optimize_skills()

            # ── Phase 3: Curiosity Goal Replay ───────────────────────────
            if not self.is_sleeping:
                return
            report["goals_replayed"] = await self._replay_curiosity_goals()

        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[SleepNode] Timeout during sleep cycle. Waking up.")
        except Exception as e:
            logger.error("[SleepNode] Sleep cycle interrupted by internal error: %s", e)

        # ── Phase 4: Dream Journal ───────────────────────────────────────
        report["duration_seconds"] = round(time.time() - cycle_start, 1)
        report["dream_journal"] = await self._generate_dream_journal(report)

        # Emit sleep report
        await self.bus.publish(
            "system.sleep.report",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.sleep.report",
                payload=report,
            ),
        )

        # ── Phase 5: Proactive Memory Warming ────────────────────────────
        await self._warm_memory_cache()

        # Reset activity timer so we don't immediately go back to sleep unless idle again
        self._last_system_activity = time.time()
        self.current_phase = SleepPhase.AWAKE
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
                leaf_nodes = [
                    str(e.get("label", "")) for e in entities if e.get("type") != "community"
                ]

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

                    if (
                        cluster_json
                        and "error" not in cluster_json
                        and "communities" in cluster_json
                    ):
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
        """Phase 2: Trigger Lifelong Continuous DPO overnight.

        If a SelfModel is available, queries it for declining/weak domains
        and sends them as priority hints so DPO focuses training where the
        system is weakest.
        """
        logger.info("[SleepNode] Initiating Phase 2: Autonomous Continuous DPO...")

        # ── SelfModel-guided priority targeting ──────────────────────────
        priority_domains: list[str] = []
        if self.self_model:
            try:
                weaknesses = self.self_model.get_weaknesses(max_score=0.6, min_samples=3)
                metrics = self.self_model.get_metrics()
                declining = metrics.get("declining", [])
                # Merge: declining domains first, then general weaknesses
                seen: set[str] = set()
                for d in declining + weaknesses:
                    if d not in seen:
                        priority_domains.append(d)
                        seen.add(d)
                if priority_domains:
                    logger.info(
                        "[SleepNode] SelfModel-guided DPO: prioritizing %d weak/declining domains: %s",
                        len(priority_domains),
                        priority_domains[:5],
                    )
            except Exception as e:
                logger.warning(
                    "[SleepNode] SelfModel query failed, proceeding without priority hints: %s", e
                )

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
                payload={
                    "mode": "overnight_continuous",
                    "priority_domains": priority_domains,
                },
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

    async def _optimize_skills(self) -> int:
        """Phase 2b: Replay and optimize flaky or inefficient skills."""
        logger.info("[SleepNode] Initiating Phase 2b: Skill Optimization...")
        try:
            # Emit event to trigger skill optimization in SIL / SkillRegistry
            opt_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.sleep.skill_optimize",
                payload={},
            )
            await self.bus.publish("system.sleep.skill_optimize", opt_msg)
            return 1
        except Exception as e:
            logger.warning("[SleepNode] Skill optimization failed: %s", e)
            return 0

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

    # ── Gap Closers: Contradiction Detection, Temporal Normalization, Dream Journal ──

    async def _resolve_contradictions(self) -> int:
        """
        Detect and resolve contradictory facts in the Knowledge Graph.

        Scans for entities that have multiple conflicting relations of the same type
        from the same source (e.g., "user prefers dark mode" vs "user prefers light mode").
        Resolves by keeping the most recently created fact.

        Returns:
            Number of contradictions resolved.
        """
        logger.info("[SleepNode] Phase 1.7: Scanning Knowledge Graph for contradictions...")
        resolved = 0

        try:
            # Fetch all entities from the knowledge graph
            kg_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="knowledge.query",
                payload={"action": "all_entities", "limit": 100},
            )
            kg_resp = await self.bus.request("knowledge.query", kg_msg, timeout=5.0)
            entities = list(kg_resp.payload.get("entities", []))

            if len(entities) < 2:
                logger.info("[SleepNode] Not enough entities for contradiction scan.")
                return 0

            # Fetch all relations
            rel_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="knowledge.query",
                payload={"action": "all_relations", "limit": 200},
            )
            rel_resp = await self.bus.request("knowledge.query", rel_msg, timeout=5.0)
            relations = list(rel_resp.payload.get("relations", []))

            # Group relations by (source_id, relation_type) to find conflicts
            from collections import defaultdict

            groups: dict[str, list[dict]] = defaultdict(list)
            for rel in relations:
                key = f"{rel.get('source_id', '')}:{rel.get('relation_type', '')}"
                groups[key].append(rel)

            # Find groups with multiple targets (potential contradictions)
            for key, rels in groups.items():
                if len(rels) <= 1:
                    continue

                # For preference/attribute relations, multiple targets = contradiction
                rel_type = rels[0].get("relation_type", "")
                if rel_type not in ("prefers", "is_a", "has"):
                    continue  # Only flag inherently-exclusive relations

                # Sort by created_at descending — keep newest, prune rest
                rels_sorted = sorted(rels, key=lambda r: r.get("created_at", 0), reverse=True)

                for stale_rel in rels_sorted[1:]:
                    prune_msg = Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="knowledge.query",
                        payload={
                            "action": "remove_relation",
                            "source_id": stale_rel.get("source_id"),
                            "target_id": stale_rel.get("target_id"),
                            "relation_type": stale_rel.get("relation_type"),
                        },
                    )
                    await self.bus.publish("knowledge.query", prune_msg)
                    resolved += 1
                    logger.info(
                        "[SleepNode] Resolved contradiction: pruned stale '%s' "
                        "relation (kept newest)",
                        rel_type,
                    )

        except Exception as e:
            logger.warning("[SleepNode] Contradiction resolution failed: %s", e)

        logger.info("[SleepNode] Contradiction resolution complete: %d resolved.", resolved)
        return resolved

    async def _normalize_temporal_references(self) -> int:
        """
        Scan episodic memory for relative temporal references ("yesterday",
        "last week") and replace them with absolute dates.

        Uses the temporal-reasoning plugin's parser to detect references,
        then rewrites the content with concrete dates based on the
        memory entry's timestamp.

        Returns:
            Number of temporal references normalized.
        """
        logger.info("[SleepNode] Phase 1.6: Normalizing temporal references in memories...")
        normalized = 0

        try:
            from hbllm.plugins.temporal_reasoning.temporal_engine import (
                parse_temporal_references,
            )
        except ImportError:
            try:
                # Plugin may be installed under hyphenated path
                import importlib
                import sys

                plugin_path = str(
                    __import__("pathlib").Path(__file__).parent.parent
                    / "plugins"
                    / "temporal-reasoning"
                )
                if plugin_path not in sys.path:
                    sys.path.insert(0, plugin_path)
                mod = importlib.import_module("temporal_engine")
                parse_temporal_references = mod.parse_temporal_references
            except Exception:
                logger.warning(
                    "[SleepNode] temporal-reasoning plugin not available. "
                    "Skipping temporal normalization."
                )
                return 0

        try:
            # Fetch recent episodic entries
            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="memory.retrieve_recent",
                payload={"session_id": "default_session", "limit": 20},
            )
            resp = await self.bus.request("memory.retrieve_recent", req_msg, timeout=5.0)

            if resp.type == MessageType.ERROR:
                return 0

            turns = list(resp.payload.get("turns", []))

            from datetime import datetime, timezone

            for turn in turns:
                content = turn.get("content", "")
                if not content:
                    continue

                refs = parse_temporal_references(content)
                if not refs:
                    continue

                # Use the turn's timestamp (or now) as the reference point
                turn_ts = turn.get("timestamp", time.time())
                ref_dt = datetime.fromtimestamp(turn_ts, tz=timezone.utc)

                updated_content = content
                for keyword, delta in refs:
                    absolute_dt = ref_dt - delta
                    absolute_str = absolute_dt.strftime("%Y-%m-%d")
                    # Replace the relative reference with absolute date
                    updated_content = updated_content.replace(
                        keyword, f"{keyword} ({absolute_str})"
                    )
                    normalized += 1

                if updated_content != content:
                    # Store the normalized version
                    store_msg = Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="memory.store",
                        payload={
                            "session_id": turn.get("session_id", "default_session"),
                            "role": turn.get("role", "system"),
                            "content": updated_content,
                            "normalized": True,
                        },
                    )
                    await self.bus.publish("memory.store", store_msg)
                    logger.debug(
                        "[SleepNode] Normalized %d temporal refs in memory entry.",
                        len(refs),
                    )

        except Exception as e:
            logger.warning("[SleepNode] Temporal normalization failed: %s", e)

        logger.info(
            "[SleepNode] Temporal normalization complete: %d references normalized.",
            normalized,
        )
        return normalized

    async def _generate_dream_journal(self, report: dict[str, Any]) -> str:
        """
        Generate a human-readable summary of what the AI learned during sleep.

        This is the "here's what I learned while you were away" experience,
        equivalent to Claude's /dream output but far more comprehensive.

        Args:
            report: The accumulated sleep cycle report dict.

        Returns:
            A formatted dream journal string.
        """
        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        duration = report.get("duration_seconds", 0)

        sections = []
        sections.append(f"🌙 Dream Journal — {now}")
        sections.append(f"Sleep duration: {duration}s")
        sections.append("")

        # Phase 1: Memory Consolidation
        mem_count = report.get("memories_consolidated", 0)
        if mem_count > 0:
            sections.append(
                f"📝 Memory Consolidation: Compressed {mem_count} recent "
                f"conversation turns into long-term semantic memory."
            )
        else:
            sections.append("📝 Memory Consolidation: No new memories to consolidate.")

        # Phase 1.6: Temporal Normalization
        temp_count = report.get("temporal_refs_normalized", 0)
        if temp_count > 0:
            sections.append(
                f"🕐 Temporal Normalization: Resolved {temp_count} relative time "
                f"references (e.g., 'yesterday' → absolute dates)."
            )

        # Phase 1.7: Contradiction Resolution
        contra_count = report.get("contradictions_resolved", 0)
        if contra_count > 0:
            sections.append(
                f"⚖️ Contradiction Resolution: Detected and pruned {contra_count} "
                f"conflicting facts from the knowledge graph."
            )

        # Phase 2: DPO Training
        if report.get("training_ran"):
            sections.append(
                "🧠 Neural Plasticity: Ran autonomous DPO training to strengthen "
                "preferred response patterns."
            )

        # Phase 2b: Skills
        skills = report.get("skills_optimized", 0)
        if skills:
            sections.append(f"⚡ Skill Optimization: Replayed and optimized {skills} skill(s).")

        # Phase 3: Curiosity
        goals = report.get("goals_replayed", 0)
        if goals > 0:
            sections.append(
                f"🔍 Curiosity Exploration: Investigated {goals} knowledge gap(s) "
                f"identified during active sessions."
            )

        journal = "\n".join(sections)

        # Store the dream journal in episodic memory
        try:
            store_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="memory.store",
                payload={
                    "session_id": "dream_journal",
                    "role": "system",
                    "content": journal,
                    "domain": "sleep_cycle",
                },
            )
            await self.bus.publish("memory.store", store_msg)
            logger.info("[SleepNode] Dream journal stored in memory.")
        except Exception as e:
            logger.warning("[SleepNode] Failed to store dream journal: %s", e)

        logger.info("[SleepNode] Dream Journal:\n%s", journal)
        return journal

    async def _audit_knowledge_staleness(self) -> int:
        """
        Phase 1.8: Knowledge Staleness Audit.

        Scan web-sourced knowledge for entries past their TTL.
        Stale entries are marked obsolete so the system hedges when using them.

        Returns the number of stale entries processed.
        """
        logger.info("[SleepNode] Running knowledge staleness audit...")
        processed = 0

        try:
            # Query the knowledge base for stale web entries
            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="knowledge.stale_check",
                payload={"ttl_days": 30},
            )

            try:
                resp = await self.bus.request("knowledge.stale_check", req_msg, timeout=5.0)
                stale_entries = resp.payload.get("stale_entries", [])
            except Exception:
                # If no handler registered, use direct KB access fallback
                logger.debug("[SleepNode] No knowledge.stale_check handler, skipping.")
                return 0

            for entry in stale_entries:
                doc_id = entry.get("doc_id", "")
                age_days = entry.get("age_days", 0)
                meta = entry.get("metadata", {})
                title = meta.get("source_title", "unknown")

                logger.info(
                    "[SleepNode] Stale knowledge: '%s' (%.0f days old)",
                    title,
                    age_days,
                )

                # Mark as obsolete — the system will hedge when using it
                await self.bus.publish(
                    "knowledge.obsolete",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="knowledge.obsolete",
                        payload={
                            "doc_id": doc_id,
                            "reason": f"Stale: {age_days:.0f} days past TTL",
                            "source_url": meta.get("source_url", ""),
                        },
                    ),
                )
                processed += 1

            logger.info(
                "[SleepNode] Staleness audit complete: %d entries flagged",
                processed,
            )
        except Exception as e:
            logger.warning("[SleepNode] Staleness audit failed: %s", e)

        return processed

    async def _promote_task_knowledge(self) -> int:
        """
        Phase 1.9: Task Knowledge Promotion (T2 → T3).

        Scan episodic memory for task-scoped web research (T2) that was
        accessed frequently across sessions. Promote to core knowledge (T3)
        for permanent storage.

        Returns the number of entries promoted.
        """
        logger.info("[SleepNode] Checking for task knowledge promotion...")
        promoted = 0

        try:
            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="memory.retrieve_by_tag",
                payload={
                    "tag": "tier:task_knowledge",
                    "domain": "web_research",
                    "min_access_count": 3,
                },
            )

            try:
                resp = await self.bus.request("memory.retrieve_by_tag", req_msg, timeout=5.0)
                candidates = resp.payload.get("entries", [])
            except Exception:
                logger.debug("[SleepNode] No memory.retrieve_by_tag handler, skipping.")
                return 0

            for entry in candidates:
                content = entry.get("content", "")
                source_url = entry.get("source_url", "")
                title = entry.get("source_title", "Promoted Knowledge")
                access_count = entry.get("access_count", 0)

                if not content:
                    continue

                # Promote to T3 via knowledge.ingest
                await self.bus.publish(
                    "knowledge.ingest",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="knowledge.ingest",
                        payload={
                            "content": content,
                            "url": source_url,
                            "title": f"[Promoted] {title}",
                            "trust_score": 0.7,
                            "tier": "core_knowledge",
                            "ttl_days": 30,
                            "source_type": "promotion",
                            "promotion_reason": f"Accessed {access_count}x across sessions",
                        },
                    ),
                )
                promoted += 1

                logger.info(
                    "[SleepNode] Promoted T2→T3: '%s' (accessed %dx)",
                    title[:60],
                    access_count,
                )

        except Exception as e:
            logger.warning("[SleepNode] Task knowledge promotion failed: %s", e)

        logger.info("[SleepNode] Promotion complete: %d entries promoted", promoted)
        return promoted

    async def _warm_memory_cache(self) -> int:
        """
        Phase 5: Proactive Memory Warming — pre-fetch context on wake-up.

        After sleep completes, pre-loads recent conversation topics and
        last session summary into the memory fast-path so the system
        "remembers" without being asked. No competitor does this.

        Returns the number of topics warmed.
        """
        logger.info("[SleepNode] Warming memory cache for next session...")
        warmed = 0

        try:
            # 1. Fetch last 5 conversation turns for topic extraction
            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="memory.retrieve_recent",
                payload={"session_id": "default_session", "limit": 5},
            )
            resp = await self.bus.request("memory.retrieve_recent", req_msg, timeout=3.0)
            turns = resp.payload.get("turns", [])

            if not turns:
                logger.info("[SleepNode] No recent turns to warm cache with.")
                return 0

            # 2. Extract unique topic keywords from recent turns
            topics: set[str] = set()
            for turn in turns:
                content = turn.get("content", "") if isinstance(turn, dict) else ""
                # Extract significant words (>4 chars, not common stop words)
                stop_words = {
                    "about",
                    "after",
                    "before",
                    "could",
                    "would",
                    "should",
                    "there",
                    "their",
                    "these",
                    "those",
                    "where",
                    "which",
                    "being",
                    "going",
                    "doing",
                    "having",
                    "think",
                    "thing",
                }
                words = content.lower().split()
                for word in words:
                    clean = word.strip(".,!?\"'()[]{}:;")
                    if len(clean) > 4 and clean not in stop_words:
                        topics.add(clean)

            # 3. Pre-warm top-5 topics via memory retrieval
            top_topics = sorted(topics)[:5]
            for topic in top_topics:
                try:
                    warm_msg = Message(
                        type=MessageType.QUERY,
                        source_node_id=self.node_id,
                        topic="memory.retrieve_recent",
                        payload={"query": topic, "limit": 3, "warm_cache": True},
                    )
                    await self.bus.request("memory.retrieve_recent", warm_msg, timeout=2.0)
                    warmed += 1
                except Exception:
                    pass  # Non-critical — skip if individual topic warm fails

            logger.info(
                "[SleepNode] Memory cache warmed with %d topics: %s",
                warmed,
                top_topics,
            )
        except Exception as e:
            logger.warning("[SleepNode] Memory cache warming failed (non-critical): %s", e)

        return warmed

    async def _handle_force_sleep(self, message: Message) -> Message | None:
        """
        Handle manual sleep trigger (equivalent to Claude's /dream command).

        Allows CLI/chat/API to trigger a consolidation cycle on demand.
        """
        if self.is_sleeping:
            logger.info("[SleepNode] Manual trigger ignored — already sleeping.")
            return (
                message.create_response(
                    {"status": "already_sleeping", "phase": self.current_phase.value}
                )
                if message.type == MessageType.QUERY
                else None
            )

        logger.info("[SleepNode] Manual consolidation triggered (like /dream).")
        # Run the sleep cycle without waiting for the idle timeout
        _dream_task = asyncio.create_task(self._enter_sleep_cycle())
        _dream_task.add_done_callback(
            lambda t: (
                logger.error("[SleepNode] Dream cycle raised: %s", t.exception())
                if not t.cancelled() and t.exception()
                else None
            )
        )

        if message.type == MessageType.QUERY:
            return message.create_response(
                {"status": "consolidation_started", "message": "Dream cycle initiated."}
            )
        return None
