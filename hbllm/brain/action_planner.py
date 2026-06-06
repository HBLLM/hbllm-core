"""
Action Planner — maps workspace consensus to structured ActionPlans.

This module is the single source of truth for deciding *what action* the
DecisionNode should take.  It replaces the ad-hoc ``if/elif`` chain that
previously lived inside ``DecisionNode._route_to_execution``.

The planner is intentionally **not** a bus Node — it is a pure function
layer that the DecisionNode calls synchronously.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hbllm.brain.action_schema import ActionPlan, ActionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vague-query detection: pure pronoun/referential check.
# We do NOT re-detect verbs like "search" / "google" here because the
# RouterNode has already resolved the intent to ``web_search`` upstream.
# We only ask: "is the query self-contained, or does it reference something
# from the conversation?"
# ---------------------------------------------------------------------------
_REFERENTIAL_PRONOUNS = re.compile(r"\b(?:it|this|that|them|those|these)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Code-block extraction (markdown fenced blocks)
# ---------------------------------------------------------------------------
_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)

# Minimum confidence to proceed — below this we ask the user to clarify.
MIN_CONFIDENCE = 0.3


class ActionPlanner:
    """
    Stateless planner that converts workspace consensus into an ``ActionPlan``.

    Usage::

        planner = ActionPlanner()
        plan = planner.plan(
            intent="web_search",
            thought_type="web_search",
            content="I will search it.",
            confidence=0.9,
            original_query={"text": "search it", "force_search": False, ...},
        )
    """

    # ── public API ─────────────────────────────────────────────────────────

    def plan(
        self,
        intent: str,
        thought_type: str,
        content: str,
        confidence: float,
        original_query: dict[str, Any],
    ) -> ActionPlan:
        """Return a structured ``ActionPlan`` for the given inputs."""

        # ── confidence gating ──────────────────────────────────────────────
        if confidence < MIN_CONFIDENCE:
            return ActionPlan(
                action_type=ActionType.CLARIFY,
                content=content,
                metadata={"reason": "confidence_below_threshold", "confidence": confidence},
            )

        # ── intent-based dispatch ──────────────────────────────────────────
        if intent == "speak" or original_query.get("force_audio", False):
            return self._plan_audio(content, original_query)

        if intent == "web_search" or original_query.get("force_search", False):
            return self._plan_web_search(content, original_query)

        if thought_type == "api_synthesis" or intent == "tool_synthesis":
            return self._plan_api_call(content, intent, original_query)

        if intent == "iot_command" or original_query.get("iot_topic"):
            return self._plan_iot_command(content, original_query)

        if intent == "mcp_tool" or original_query.get("mcp_tool_name"):
            return self._plan_mcp_tool(content, original_query)

        # ── content-based fallback (code detection) ────────────────────────
        if thought_type == "code_execution" or "```python" in content:
            return self._plan_code_execution(content, original_query)

        # ── default: text response ─────────────────────────────────────────
        return ActionPlan(
            action_type=ActionType.TEXT_RESPONSE,
            content=content,
            metadata={"thought_type": thought_type},
        )

    # ── private plan builders ──────────────────────────────────────────────

    def _plan_audio(self, content: str, original_query: dict[str, Any]) -> ActionPlan:
        return ActionPlan(
            action_type=ActionType.AUDIO_OUTPUT,
            content=content,
        )

    def _plan_web_search(self, content: str, original_query: dict[str, Any]) -> ActionPlan:
        query = original_query.get("text", content[:200])
        needs_resolution = self._is_vague_query(query)

        return ActionPlan(
            action_type=ActionType.WEB_SEARCH,
            content=query,
            metadata={
                "needs_context_resolution": needs_resolution,
                "max_results": 3,
            },
        )

    def _plan_code_execution(self, content: str, original_query: dict[str, Any]) -> ActionPlan:
        code = self._extract_code(content)
        return ActionPlan(
            action_type=ActionType.CODE_EXECUTION,
            content=code,
            metadata={"raw_content": content},
        )

    def _plan_api_call(
        self, content: str, intent: str, original_query: dict[str, Any]
    ) -> ActionPlan:
        return ActionPlan(
            action_type=ActionType.API_CALL,
            content=content,
            metadata={"intent": intent},
        )

    def _plan_iot_command(self, content: str, original_query: dict[str, Any]) -> ActionPlan:
        return ActionPlan(
            action_type=ActionType.IOT_COMMAND,
            content=content,
            metadata={
                "iot_topic": original_query.get("iot_topic", "hbllm/command"),
            },
        )

    def _plan_mcp_tool(self, content: str, original_query: dict[str, Any]) -> ActionPlan:
        return ActionPlan(
            action_type=ActionType.MCP_TOOL,
            content=content,
            metadata={
                "tool_name": original_query.get("mcp_tool_name", ""),
                "arguments": original_query.get("mcp_arguments", {}),
            },
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_vague_query(query: str) -> bool:
        """Detect whether a search query needs context resolution.

        A query is considered "vague" if it:
        - Is very short (< 4 words), OR
        - Contains referential pronouns (it, this, that, …) and is shorter
          than 8 words — meaning it likely refers to something in the
          conversation history rather than being self-contained.

        This intentionally does NOT check for verbs like "search" / "google"
        because the intent has already been resolved upstream by RouterNode.
        """
        words = query.split()
        word_count = len(words)

        if word_count < 4:
            return True

        if _REFERENTIAL_PRONOUNS.search(query) and word_count < 8:
            return True

        return False

    @staticmethod
    def _extract_code(content: str) -> str:
        """Extract Python code from markdown fenced code blocks."""
        match = _CODE_FENCE.search(content)
        if match:
            return match.group(1).strip()

        # Fallback: try splitting on the fence markers
        if "```python" in content:
            try:
                return content.split("```python")[1].split("```")[0].strip()
            except (IndexError, ValueError):
                pass

        # Last resort: return the raw content
        return content
