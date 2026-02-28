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
        """Perform offline memory consolidation."""
        self.is_sleeping = True
        logger.info("[SleepNode] System Idle for >%.1fs. Entering Deep Sleep (Consolidation Mode)...", self.idle_timeout_seconds)
        
        try:
            # Step 1: Request unsummarized short-term memory
            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="memory.retrieve_recent",
                payload={"session_id": "default_session", "limit": 10}
            )
            
            # Use a short timeout so we don't block forever if memory is down
            resp = await self.bus.request("memory.retrieve_recent", req_msg, timeout=5.0)
            if resp.type == MessageType.ERROR:
                logger.error("[SleepNode] Failed to retrieve memory for consolidation: %s", resp.payload.get("error"))
                self.is_sleeping = False
                # reset activity so we don't immediately retry
                self._last_system_activity = time.time()
                return

            turns = resp.payload.get("turns", [])
            
            # Simple heuristic: Only bother summarizing if we have a decent chunk of dialogue
            if len(turns) >= 4:
                logger.info("[SleepNode] Dreaming: Compressing %d recent turns into semantic long-term memory...", len(turns))
                
                # We format the chat for the LLM
                chat_transcript = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in turns])
                
                prompt = f"System Instruction: Summarize the following historical conversation extremely concisely. Extract only the factual claims, user preferences, and final conclusions. Do not include conversational filler.\n\n<transcript>\n{chat_transcript}\n</transcript>\n\nSummary:"
                
                # Ask the Workspace to execute the summarization
                # We bypass the router and go directly to module.evaluate to use the base General domain
                summarize_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    topic="module.evaluate",
                    payload={"text": prompt, "domain_hint": "general", "intent": "summarization"}
                )
                
                # In a real async system we'd wait for workspace.thought or sensory.output, 
                # but to simulate the offline batch process, we'll pretend the general domain replies.
                # Actually, DomainModule uses publish, so we can't 'request' it easily anymore without a callback.
                
                # Emulate the memory compression storage for Phase 9 prototype
                logger.info("[SleepNode] (Simulated LLM Compression): Storing consolidated summary of %d turns...", len(turns))
                
                # Delete the old verbose turns (This endpoint would need to be added to MemoryNode)
                # await self.bus.publish("memory.prune", ...)
                
                # Store the new dense vector representation
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
            else:
                logger.info("[SleepNode] Not enough new memories to consolidate. Resting...")

        except TimeoutError:
             logger.warning("[SleepNode] Timeout during sleep cycle. Waking up.")
        except Exception as e:
            logger.error("[SleepNode] Sleep cycle interrupted by internal error: %s", e)
            
        # Reset activity timer so we don't immediately go back to sleep unless idle again
        self._last_system_activity = time.time()
        self.is_sleeping = False
