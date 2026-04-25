"""
Multi-Agent Swarm Plugin for HBLLM.

Manages sub-brains (lightweight pipelines) to execute decomposed tasks in parallel.
"""

import asyncio
import logging
import uuid
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger("hbllm_swarm")

__plugin__ = {
    "name": "hbllm_swarm",
    "version": "0.1.0",
    "description": "Multi-agent swarm manager for parallel sub-tasks.",
}


class SwarmManagerNode(HBLLMPlugin):
    """
    Subscribes to swarm actions or task decompositions and orchestrates
    parallel execution using lightweight mocked sub-brains.
    """

    def __init__(self, node_id: str = "swarm_manager") -> None:
        super().__init__(node_id=node_id, capabilities=["task_orchestration", "swarm"])
        self.active_swarms: dict[str, dict[str, Any]] = {}

    @subscribe("task_decompose")
    async def on_task_decompose(self, message: Message) -> None:
        """
        Takes a complex task, spawns parallel agents, and awaits completion.
        """
        sub_tasks = message.payload.get("sub_tasks", [])
        if not sub_tasks:
            logger.warning("[%s] No sub-tasks provided to swarm.", self.node_id)
            return

        swarm_id = str(uuid.uuid4())
        logger.info("[%s] Spawning %d agents for swarm %s", self.node_id, len(sub_tasks), swarm_id)

        # Mocking the spawning of separate CognitivePipelines by running parallel async tasks
        # In a real implementation, this would instantiate actual pipeline objects or process workers.
        results = await asyncio.gather(
            *(self._run_sub_agent(task) for task in sub_tasks), return_exceptions=True
        )

        aggregated_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                aggregated_results.append({"task": sub_tasks[i], "error": str(res)})
            else:
                aggregated_results.append(res)

        logger.info("[%s] Swarm %s completed.", self.node_id, swarm_id)

        # Publish the aggregated result back to the bus
        reply = message.create_response(
            payload={"swarm_id": swarm_id, "results": aggregated_results},
            msg_type=MessageType.TASK_AGGREGATE,
        )
        # Assuming we can publish back on a generic channel or the one requested
        if self.bus:
            await self.bus.publish("task_aggregate", reply)

    async def _run_sub_agent(self, task_def: dict[str, Any]) -> dict[str, Any]:
        """
        Simulates a lightweight sub-brain executing a task.
        """
        task_name = task_def.get("name", "unknown")
        # Simulate thinking delay
        await asyncio.sleep(1.0)
        return {
            "task": task_name,
            "status": "completed",
            "output": f"Successfully executed: {task_name}",
        }
