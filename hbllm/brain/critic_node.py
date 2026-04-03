"""
System 2 Critic Node (Active Halting).

Monitors proposed thoughts on the Global Workspace Blackboard.
Uses the base LLM to evaluate thoughts against the user's initial query,
detecting hallucination, flawed logic, or divergence. Emits a PASS/FAIL
verdict to trigger backtracking if necessary.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class CriticNode(Node):
    """
    LLM-powered evaluator acting as a cognitive brake in the consensus loop.
    """

    # Maximum number of queries to keep in the LRU cache
    MAX_CACHE_SIZE = 500

    def __init__(self, node_id: str, llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=["critic", "evaluation", "halting"])
        self.llm = llm  # LLMInterface instance

        # LRU cache for original queries (bounded to prevent memory leaks)
        self._query_cache: OrderedDict[str, str] = OrderedDict()

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
                # Evict oldest entries if cache exceeds max size
                while len(self._query_cache) > self.MAX_CACHE_SIZE:
                    self._query_cache.popitem(last=False)
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

        # LLM Self-Reflection Evaluation (fail-open: default to PASS on error)
        status = "PASS"
        reason = ""
        try:
            from hbllm.brain.constitutional_principles import (
                format_principles_for_prompt,
                get_principles,
            )

            principles = get_principles()
            principles_str = format_principles_for_prompt(principles)

            evaluation = await self.llm.generate_json(
                f"You are a strict Constitutional AI Evaluator. Given the user's original query "
                f"and a proposed response, determine if the response violates ANY of the following principles:\n\n"
                f"{principles_str}\n\n"
                f"Original Query: \"{original_query}\"\n"
                f"Proposed Response: \"{content[:1000]}\"\n\n"
                f"Output JSON:\n"
                f"{{\n"
                f"  \"violations\": [\"name_of_failed_principle_1\", ...],\n"
                f"  \"rationale\": \"brief explanation of why the violation occurred, or 'All clear' if none\"\n"
                f"}}"
            )

            violations = evaluation.get("violations", [])
            reason = evaluation.get("rationale", "")

            if violations and isinstance(violations, list) and len(violations) > 0:
                status = "FAIL"
                reason = f"Violated principles: {', '.join(str(v) for v in violations)}. Details: {reason}"
            else:
                status = "PASS"
                if not reason:
                    reason = "Passed all constitutional principles."

        except Exception as e:
            logger.warning("[CriticNode] LLM constitutional evaluation failed, defaulting to PASS: %s", e)
            status = "PASS"
            reason = "Critic evaluation skipped due to LLM error"

        if status == "FAIL":
            logger.warning(
                "[CriticNode] LLM evaluation FAILED thought from %s. Reason: %s",
                message.source_node_id, reason
            )

        # Note: Do NOT pop the cache here — multiple thoughts may share
        # the same correlation_id. LRU eviction handles cleanup.

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
