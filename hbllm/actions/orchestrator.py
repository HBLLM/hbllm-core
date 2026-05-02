"""
Core Multi-Agent Orchestrator — Planner → Executor → Reviewer pipeline.

Framework-agnostic orchestration. Platform-specific tool argument builders
should extend ``build_tool_args`` from ``hbllm.actions.agent_executor``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from hbllm.actions.agent_executor import (
    AgentResponse,
    AgentStep,
    ConfidenceScorer,
    build_tool_args,
)
from hbllm.actions.tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


_PLANNER_PROMPT_TEMPLATE = """You are a Task Planner. Break down the user's request into clear, numbered steps.
For each step, specify:
- What needs to be done
- Which tool to use (if any): {tool_names}
- Expected output

Format your plan as:
STEP 1: [description]
TOOL: [tool_name] or NONE
INPUT: [tool input if applicable]

STEP 2: ...

Be concise. Maximum 5 steps.
IMPORTANT: Only use the tools listed above. Do NOT reference tools that are not available."""

REVIEWER_PROMPT = """You are a Code Reviewer. Review the execution results and provide:
1. Whether the task was completed successfully
2. Any issues or improvements needed
3. A final summary for the user

Be concise and constructive."""


class MultiAgentOrchestrator:
    """Runs a Planner → Executor → Reviewer pipeline using a single LLM."""

    def __init__(self, llm: Any, tool_registry: ToolRegistry):
        self.llm = llm
        self.tools = tool_registry

    def _build_planner_prompt(self) -> str:
        """Build a planner prompt with the actual available tool names."""
        available = self.tools.list_tools(available_only=True)
        tool_names = ", ".join(t["name"] for t in available) or "NONE"
        return _PLANNER_PROMPT_TEMPLATE.format(tool_names=tool_names)

    async def execute(
        self,
        task: str,
        history: list[dict[str, str]] | None = None,
        kb_context: str = "",
    ) -> AgentResponse:
        start = time.monotonic()
        steps: list[AgentStep] = []
        step_num = 0

        # ── Phase 1: Planning ──
        step_num += 1
        planner_prompt = self._build_planner_prompt()
        plan_messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]
        if kb_context:
            plan_messages.insert(
                1, {"role": "system", "content": f"Context from knowledge base:\n{kb_context}"}
            )

        plan_text = await self._call_llm(plan_messages)
        steps.append(
            AgentStep(step_num=step_num, role="planner", action="think", content=plan_text)
        )

        # ── Phase 2: Execute each step ──
        parsed_steps = self._parse_plan(plan_text)
        execution_results: list[str] = []

        for ps in parsed_steps:
            step_num += 1
            desc = ps.get("description", "")
            tool_name = ps.get("tool", "").lower().strip()
            tool_input = ps.get("input", "").strip()

            if tool_name and tool_name != "none" and tool_input:
                # Pre-execution availability check — the tool could have gone
                # offline between planning and execution.
                availability = self.tools.get_availability(tool_name)
                if not availability.get("available", False) and availability.get(
                    "registered", False
                ):
                    tool_result = ToolResult(
                        tool=tool_name,
                        success=False,
                        output="",
                        error=f"Tool '{tool_name}' became unavailable during execution",
                    )
                else:
                    tool_result = await self.tools.invoke(
                        tool_name, **self._resolve_tool_args(tool_name, tool_input)
                    )
                steps.append(
                    AgentStep(
                        step_num=step_num,
                        role="executor",
                        action="tool_call",
                        content=f"Executing: {desc}",
                        tool_result=tool_result,
                    )
                )
                result_text = (
                    tool_result.output if tool_result.success else f"Error: {tool_result.error}"
                )
                execution_results.append(
                    f"Step {ps.get('num', '?')}: {desc}\nResult: {result_text}"
                )
            else:
                steps.append(
                    AgentStep(
                        step_num=step_num,
                        role="executor",
                        action="think",
                        content=f"Reasoning: {desc}",
                    )
                )
                execution_results.append(f"Step {ps.get('num', '?')}: {desc} (reasoning only)")

        # ── Phase 3: Review ──
        step_num += 1
        review_messages = [
            {"role": "system", "content": REVIEWER_PROMPT},
            {
                "role": "user",
                "content": f"Original task: {task}\n\nExecution results:\n"
                + "\n\n".join(execution_results),
            },
        ]
        review_text = await self._call_llm(review_messages)
        steps.append(
            AgentStep(step_num=step_num, role="reviewer", action="review", content=review_text)
        )

        # ── Final response ──
        step_num += 1
        final_messages = [
            {
                "role": "system",
                "content": "Synthesize the plan, execution results, and review into a clear final response for the user. Include relevant outputs.",
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nPlan:\n{plan_text}\n\nResults:\n"
                + "\n".join(execution_results)
                + f"\n\nReview:\n{review_text}",
            },
        ]
        final_text = await self._call_llm(final_messages)

        total_ms = (time.monotonic() - start) * 1000
        confidence = ConfidenceScorer.score(task, final_text, had_context=bool(kb_context))

        return AgentResponse(
            content=final_text,
            steps=steps,
            confidence=confidence["overall"],
            confidence_flags=confidence["flags"],
            multi_agent=True,
            total_duration_ms=total_ms,
        )

    def _resolve_tool_args(self, tool_name: str, tool_input: str) -> dict[str, str]:
        """Resolve tool arguments. Override in platform subclasses for extra tools."""
        return build_tool_args(tool_name, tool_input)

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM provider."""
        try:
            if hasattr(self.llm, "chat"):
                return await self.llm.chat(messages)
            elif hasattr(self.llm, "generate"):
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                return await self.llm.generate(prompt)
            else:
                return "[LLM not available]"
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"[LLM error: {e}]"

    @staticmethod
    def _parse_plan(plan_text: str) -> list[dict[str, str]]:
        """Parse a structured plan into steps."""
        steps: list[dict[str, str]] = []
        current: dict[str, str] = {}

        for line in plan_text.split("\n"):
            line = line.strip()
            step_match = re.match(r"STEP\s*(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
            tool_match = re.match(r"TOOL\s*:\s*(.+)", line, re.IGNORECASE)
            input_match = re.match(r"INPUT\s*:\s*(.+)", line, re.IGNORECASE)

            if step_match:
                if current:
                    steps.append(current)
                current = {"num": step_match.group(1), "description": step_match.group(2).strip()}
            elif tool_match and current:
                current["tool"] = tool_match.group(1).strip()
            elif input_match and current:
                current["input"] = input_match.group(1).strip()

        if current:
            steps.append(current)

        return steps[:5]  # Max 5 steps
