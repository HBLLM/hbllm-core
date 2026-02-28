"""
System 2 Critic Node (Active Halting).

Monitors proposed thoughts on the Global Workspace Blackboard. 
Uses the base LLM to evaluate thoughts against the user's initial query,
detecting hallucination, flawed logic, or divergence. Emits a PASS/FAIL 
verdict to trigger backtracking if necessary.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class CriticNode(Node):
    """
    LLM-powered evaluator acting as a cognitive brake in the consensus loop.
    """

    def __init__(self, node_id: str, llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=["critic", "evaluation", "halting"])
        self.llm = llm  # LLMInterface instance
        
        # Cache the original queries so we can evaluate thoughts against them
        self._query_cache: dict[str, str] = {}

    async def on_start(self) -> None:
        logger.info("Starting CriticNode")
        await self.bus.subscribe("workspace.thought", self.evaluate_thought)
        await self.bus.subscribe("module.evaluate", self._cache_query)

    async def on_stop(self) -> None:
        logger.info("Stopping CriticNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _cache_query(self, message: Message) -> Message | None:
        """Cache original user queries so the Critic can compare thoughts against them."""
        if message.correlation_id:
            text = message.payload.get("text", "")
            if text:
                self._query_cache[message.correlation_id] = text
        return None

    async def evaluate_thought(self, message: Message) -> Message | None:
        """
        Uses the LLM to interrogate a thought proposal before the Workspace finalizes it.
        """
        proposal = message.payload
        thought_type = proposal.get("type", "intuition")
        
        # Don't critique meta-thoughts or our own critiques
        if thought_type in ("simulation_result", "critique", "symbolic_logic"):
            return None
        
        if not self.llm:
            return None

        content = str(proposal.get("content", ""))
        correlation_id = message.correlation_id or ""
        original_query = self._query_cache.get(correlation_id, "unknown query")
        
        # LLM Self-Reflection Evaluation
        evaluation = await self.llm.generate_json(
            f"You are a strict QA evaluator for an AI system. Given the user's original query "
            f"and a proposed response, determine if the response is:\n"
            f"(a) Relevant to the query\n"
            f"(b) Factually grounded (not hallucinated or fabricated)\n"
            f"(c) Not deflecting with clichés like 'As an AI...'\n"
            f"(d) Not attempting unsafe operations\n\n"
            f"Original Query: \"{original_query}\"\n"
            f"Proposed Response: \"{content[:500]}\"\n\n"
            f"Output JSON: {{\"verdict\": \"PASS\" or \"FAIL\", \"reason\": \"brief explanation\"}}"
        )
        
        status = evaluation.get("verdict", "PASS").upper()
        reason = evaluation.get("reason", "")
        
        # Normalize the verdict
        if status not in ("PASS", "FAIL"):
            status = "PASS"  # Default to pass if LLM output is ambiguous
        
        if status == "FAIL":
            logger.warning(
                "[CriticNode] LLM evaluation FAILED thought from %s. Reason: %s", 
                message.source_node_id, reason
            )
        
        # Emit the evaluation back to the Workspace
        critique_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.thought",
            payload={
                "type": "critique",
                "target_node": message.source_node_id,
                "status": status,
                "reason": reason,
                "confidence": 1.0,
                "original_content": proposal.get("content")
            },
            correlation_id=message.correlation_id
        )
        await self.bus.publish("workspace.thought", critique_msg)
        return None
