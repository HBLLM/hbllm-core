"""
Failure Analyzer Node — intercepts failures and synthesizes automated repairs.

Responsibilities:
1. Classify failure types (Timeout, Logic error, Tool failure, Null output, etc.)
2. Use an LLM to generate a repair strategy and modified skill steps.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.provider_adapter import ProviderLLM
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

class FailureAnalyzerNode(Node):
    """Diagnoses execution failures and proposes repaired action steps."""

    def __init__(self, node_id: str, llm: ProviderLLM | None = None) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["failure_analysis", "skill_repair", "diagnostic_reasoning"],
        )
        self.llm = llm

    async def on_start(self) -> None:
        logger.info("Starting FailureAnalyzerNode")
        await self.bus.subscribe("action.analyze_failure", self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping FailureAnalyzerNode")

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        # Expecting payload: { "skill_name": str, "steps": list[str], "execution_trace": list[dict], "error_message": str}
        payload = message.payload
        skill_name = payload.get("skill_name", "Unknown")
        steps = payload.get("steps", [])
        trace = payload.get("execution_trace", [])
        error_msg = payload.get("error_message", "Unknown error")

        logger.info("Analyzing failure for skill '%s': %s", skill_name, error_msg)

        # Classify Failure
        failure_type = self._classify_failure(error_msg)

        if not self.llm:
            return message.create_response({
                "failure_type": failure_type,
                "repaired": False,
                "reason": "No LLM available for repair",
                "new_steps": steps
            })

        # Ask LLM to generate a fixed step list
        prompt = (
            f"The skill '{skill_name}' failed to execute.\n"
            f"Failure Type: {failure_type}\n"
            f"Error Message: {error_msg}\n"
            f"Original Steps:\n{steps}\n"
            f"Execution Trace so far:\n{trace}\n"
            f"Please analyze the failure and provide an updated JSON list of steps that fixes this issue.\n"
            f"Respond ONLY with valid JSON formatting like `{{\"new_steps\": [\"step 1\", \"step 2\"]}}`."
        )

        try:
            repair_json = await self.llm.generate_json(prompt)
            new_steps = repair_json.get("new_steps", steps) if repair_json else steps

            repaired = new_steps != steps

            return message.create_response({
                "failure_type": failure_type,
                "repaired": repaired,
                "new_steps": new_steps,
                "reason": "Automated LLM repair strategy applied." if repaired else "Could not find a differing repair strategy."
            })
        except Exception as e:
            logger.error("Failed to generate repair strategy: %s", e)
            return message.create_error(f"Repair generation failed: {e}")

    def _classify_failure(self, error_message: str) -> str:
        error_lower = error_message.lower()
        if "timeout" in error_lower:
            return "Timeout"
        elif "syntax" in error_lower or "nameerror" in error_lower:
            return "Logic Error"
        elif "connection" in error_lower or "api" in error_lower or "http" in error_lower:
            return "Network Error"
        elif "missing" in error_lower or "not found" in error_lower:
            return "Missing Data / File"
        elif "tool" in error_lower or "invalid parameter" in error_lower:
            return "Tool Failure"
        else:
            return "General Execution Failure"
