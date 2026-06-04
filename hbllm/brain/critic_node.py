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
from typing import TYPE_CHECKING

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.provider_adapter import ProviderLLM

logger = logging.getLogger(__name__)


class CriticNode(Node):
    """
    LLM-powered evaluator acting as a cognitive brake in the consensus loop.
    """

    # Maximum number of queries to keep in the LRU cache
    MAX_CACHE_SIZE = 500

    def __init__(self, node_id: str, llm: ProviderLLM | None = None) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["critic", "evaluation", "halting"],
        )
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
        confidence = 0.5  # Default when no evaluation runs

        # On slow CPU systems, use fast regex-based constitutional check instead of LLM evaluation
        from hbllm.brain.factory import _is_slow_cpu

        if _is_slow_cpu():
            logger.info("[CriticNode] Running lightweight rule-based constitutional check on CPU.")
            status, reason, confidence = self._fast_constitutional_check(content, original_query)

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
                    "confidence": confidence,
                    "original_content": proposal.get("content"),
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", critique_msg)
            return None

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
                f'Original Query: "{original_query}"\n'
                f'Proposed Response: "{content[:1000]}"\n\n'
                f"Output JSON:\n"
                f"{{\n"
                f'  "violations": ["name_of_failed_principle_1", ...],\n'
                f'  "rationale": "brief explanation of why the violation occurred, or \'All clear\' if none"\n'
                f"}}"
            )

            violations = evaluation.get("violations", [])
            reason = evaluation.get("rationale", "")

            if violations and isinstance(violations, list) and len(violations) > 0:
                status = "FAIL"
                # Confidence scales with violation count (more violations = higher certainty of failure)
                confidence = min(0.7 + len(violations) * 0.1, 1.0)
                reason = f"Violated principles: {', '.join(str(v) for v in violations)}. Details: {reason}"
            else:
                status = "PASS"
                # LLM explicitly found no violations — high confidence
                confidence = 0.95
                if not reason:
                    reason = "Passed all constitutional principles."

        except Exception as e:
            logger.warning(
                "[CriticNode] LLM constitutional evaluation failed, defaulting to FAIL (System Blocked): %s",
                e,
            )
            status = "FAIL"
            # Low confidence since we couldn't actually evaluate
            confidence = 0.4
            reason = f"Critic evaluation blocked due to internal system/LLM error: {e}"

        if status == "FAIL":
            logger.warning(
                "[CriticNode] LLM evaluation FAILED thought from %s. Reason: %s",
                message.source_node_id,
                reason,
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
                "confidence": confidence,
                "original_content": proposal.get("content"),
            },
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("workspace.thought", critique_msg)
        return None

    def _fast_constitutional_check(
        self, content: str, original_query: str
    ) -> tuple[str, str, float]:
        """
        Fast, zero-LLM constitutional check using regex patterns.

        Returns (status, reason, confidence) where status is 'PASS' or 'FAIL'.
        Covers the most critical safety violations without any model inference.
        """
        import re

        content_lower = content.lower().strip()

        # ── Harmlessness: detect clearly dangerous instruction patterns ──
        _harmful_patterns = [
            r"\b(?:how\s+to\s+(?:make|build|create)\s+(?:a\s+)?(?:bomb|explosive|weapon))",
            r"\b(?:synthesiz(?:e|ing)\s+(?:drugs?|meth|fentanyl|poison))",
            r"\b(?:hack(?:ing)?\s+(?:into|someone|password|account))",
            r"\b(?:(?:kill|murder|assassinate|harm)\s+(?:someone|a\s+person|yourself|people))",
            r"\b(?:credit\s*card\s*(?:number|info)|social\s*security\s*(?:number|ssn))\b",
            r"\b(?:child\s+(?:porn|abuse|exploitation))\b",
        ]
        for pattern in _harmful_patterns:
            if re.search(pattern, content_lower):
                return (
                    "FAIL",
                    f"Rule-based safety check: harmful content detected (pattern: {pattern[:40]}...)",
                    0.9,
                )

        # ── Honesty: detect fabricated authoritative claims ──
        _fabrication_patterns = [
            r"(?:according\s+to\s+my\s+(?:latest|real-?time)\s+(?:data|search|browse))",
            r"(?:i\s+(?:just\s+)?(?:searched|browsed|looked\s+up)\s+(?:the\s+)?(?:web|internet|google))",
        ]
        for pattern in _fabrication_patterns:
            if re.search(pattern, content_lower):
                return (
                    "FAIL",
                    "Rule-based honesty check: response claims live web access which is fabricated.",
                    0.8,
                )

        # ── Helpfulness: detect generic refusal/evasion responses ──
        _refusal_patterns = [
            r"(?:as\s+an?\s+ai\s+(?:language\s+)?model,?\s+i\s+(?:don'?t|cannot|can'?t))",
            r"(?:i(?:'m|\s+am)\s+(?:just\s+)?an?\s+ai\s+(?:and\s+)?(?:don'?t|cannot|can'?t))",
            r"(?:i\s+(?:don'?t|cannot|can'?t)\s+(?:help|assist|answer|provide)\s+(?:with\s+)?that)",
            r"(?:i'?m\s+not\s+(?:able|capable|programmed)\s+to\s+(?:help|assist|answer))",
        ]
        for pattern in _refusal_patterns:
            if re.search(pattern, content_lower):
                return (
                    "FAIL",
                    "Rule-based helpfulness check: response is a generic refusal/evasion that provides no value.",
                    0.8,
                )

        # ── Coherence: detect empty or degenerate responses ──
        if len(content_lower) < 5:
            return (
                "FAIL",
                "Rule-based coherence check: response is empty or near-empty.",
                0.85,
            )

        # Check if response is just the query echoed back
        if original_query and len(original_query) > 10:
            query_lower = original_query.lower().strip()
            if content_lower == query_lower or content_lower.startswith(query_lower):
                return (
                    "FAIL",
                    "Rule-based coherence check: response merely echoes the user query.",
                    0.8,
                )

        # All rule-based checks passed
        return (
            "PASS",
            "Passed rule-based constitutional check (lightweight CPU mode).",
            0.85,
        )

