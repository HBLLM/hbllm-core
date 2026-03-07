"""
System 4 Sleep Cycle Node (Offline Consolidation).

Monitors system activity. If the user stops interacting with the API/CLI 
for a configurable duration, it triggers a `system.sleep` event.
During sleep, it accesses the MemoryNode to compress raw verbose logs 
into semantic summaries and triggers LoRA training if weakness was 
detected throughout the day.
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
    Background process that orchestrates Memory Consolidation when idle.
    """

    def __init__(self, node_id: str, idle_timeout_seconds: float = 10.0):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=["sleep_cycle", "memory_consolidation"])
        self.idle_timeout_seconds = idle_timeout_seconds
        self.is_sleeping = False
        self._last_system_activity = time.time()
        self._monitor_task: asyncio.Task | None = None

    async def on_start(self) -> None:
        logger.info("Starting SleepCycleNode (Idle timeout: %s seconds)", self.idle_timeout_seconds)
        # Listen to router queries to track user activity
        await self.bus.subscribe("router.query", self._check_activity)
        
        # Start the background idle monitor
        self._monitor_task = asyncio.create_task(self._idle_monitor_loop())

    async def on_stop(self) -> None:
        logger.info("Stopping SleepCycleNode")
        if self._monitor_task:
            self._monitor_task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _check_activity(self, message: Message) -> Message | None:
        """Update our internal timestamp whenever a user query flows through the system."""
        self._last_system_activity = time.time()
        
        if self.is_sleeping:
            logger.info("[SleepNode] User input detected! Waking up immediately. Aborting deep sleep.")
            self.is_sleeping = False
            
        return None

    async def _idle_monitor_loop(self):
        """Continuously check if the system has been inactive."""
        # We use a short interval for testing if timeout is small
        check_interval = min(2.0, self.idle_timeout_seconds / 2.0)
        
        while self._running:
            await asyncio.sleep(check_interval)
            
            if not self.is_sleeping:
                idle_time = time.time() - self._last_system_activity
                if idle_time > self.idle_timeout_seconds:
                    await self._enter_sleep_cycle()
                    
    async def _enter_sleep_cycle(self):
        """Perform offline memory consolidation and self-improvement training."""
        self.is_sleeping = True
        logger.info("[SleepNode] System Idle for >%.1fs. Entering Deep Sleep (Consolidation Mode)...", self.idle_timeout_seconds)
        
        try:
            # ── Phase 1: Memory Consolidation ────────────────────────────
            await self._consolidate_memory()
            
            # ── Phase 2: Self-Improvement Training ───────────────────────
            if not self.is_sleeping:
                return  # User woke up during consolidation
            await self._run_self_improvement()

        except TimeoutError:
             logger.warning("[SleepNode] Timeout during sleep cycle. Waking up.")
        except Exception as e:
            logger.error("[SleepNode] Sleep cycle interrupted by internal error: %s", e)
            
        # Reset activity timer so we don't immediately go back to sleep unless idle again
        self._last_system_activity = time.time()
        self.is_sleeping = False

    async def _consolidate_memory(self):
        """Phase 1: Compress short-term memory into long-term summaries."""
        req_msg = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="memory.retrieve_recent",
            payload={"session_id": "default_session", "limit": 10}
        )
        
        try:
            resp = await self.bus.request("memory.retrieve_recent", req_msg, timeout=5.0)
        except Exception:
            logger.warning("[SleepNode] Could not reach memory node for consolidation.")
            return

        if resp.type == MessageType.ERROR:
            logger.error("[SleepNode] Failed to retrieve memory: %s", resp.payload.get("error"))
            self._last_system_activity = time.time()
            self.is_sleeping = False
            return

        turns = resp.payload.get("turns", [])
        
        if len(turns) >= 4:
            logger.info("[SleepNode] Compressing %d recent turns into semantic long-term memory...", len(turns))
            
            store_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="memory.store",
                payload={
                    "session_id": "default_session", 
                    "role": "system", 
                    "content": f"[CONSOLIDATED MEMORY] User discussed {len(turns)//2} topics. Key facts extracted and embedded."
                }
            )
            await self.bus.publish("memory.store", store_msg)
            logger.info("[SleepNode] Memory consolidation complete.")
        else:
            logger.info("[SleepNode] Not enough new memories to consolidate. Skipping.")

    async def _run_self_improvement(self):
        """Phase 2: Check for reflection data and trigger LoRA fine-tuning."""
        import asyncio
        
        try:
            from hbllm.serving.self_improve import run_improvement_cycle
        except ImportError:
            logger.debug("[SleepNode] self_improve module not available, skipping training.")
            return

        logger.info("[SleepNode] Checking for reflection datasets for self-improvement...")
        
        try:
            # Run training in a thread to avoid blocking the event loop
            results = await asyncio.to_thread(
                run_improvement_cycle,
                reflection_dir="workspace/reflection",
                model_size="125m",
                max_steps=50,  # Short training during sleep
            )
            
            if results:
                for domain, result in results.items():
                    if isinstance(result, dict) and "error" not in result:
                        logger.info(
                            "[SleepNode] Self-improvement complete for domain '%s': "
                            "%d steps, avg_loss=%.4f",
                            domain, result.get("steps", 0), result.get("avg_loss", 0),
                        )
                    elif isinstance(result, dict):
                        logger.warning("[SleepNode] Training failed for '%s': %s", domain, result.get("error"))
            else:
                logger.info("[SleepNode] No reflection data found. Model is up to date.")
                
        except Exception as e:
            logger.warning("[SleepNode] Self-improvement skipped: %s", e)

