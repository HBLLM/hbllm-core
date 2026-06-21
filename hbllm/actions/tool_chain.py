"""
ReAct-style tool reasoning loop.

Implements the Observe → Think → Act → Observe cycle for multi-step
tool use with intermediate reasoning. Replaces the single-pass tool
invocation in AgentExecutor with a proper iterative loop.

Architecture:
    1. **Observe**: Collect context (memory, world state, tool results)
    2. **Think**: LLM reasons about what to do next
    3. **Act**: Execute a tool or deliver a final answer
    4. **Observe**: Incorporate tool results, loop back to Think

Supports:
    - Multi-step chaining (search → read → summarize → email)
    - Parallel tool execution where dependencies allow
    - Intermediate scratchpad for chain-of-thought
    - Budget limits (max iterations, max tokens, max wall-time)
    - Graceful degradation on tool failures
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.actions.tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ReActConfig:
    """Budget and behavior limits for the reasoning loop."""

    max_iterations: int = 8
    max_wall_time_seconds: float = 60.0
    max_parallel_tools: int = 3
    allow_parallel: bool = True
    include_scratchpad: bool = True
    verbose_logging: bool = False


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class Observation:
    """A single observation from a tool or the environment."""

    source: str  # tool name or "environment"
    content: str
    success: bool = True
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class ThoughtStep:
    """A single step in the reasoning chain."""

    step_num: int
    thought: str  # LLM's reasoning
    action: str | None = None  # tool name or "FINISH"
    action_input: str | None = None
    observation: Observation | None = None
    parallel_actions: list[dict[str, str]] | None = None
    duration_ms: float = 0.0


@dataclass
class ReActResult:
    """Complete result from a ReAct reasoning loop."""

    answer: str
    steps: list[ThoughtStep]
    total_iterations: int
    total_tool_calls: int
    total_duration_ms: float
    finished_reason: str  # "answer", "max_iterations", "timeout", "error"
    scratchpad: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "total_duration_ms": self.total_duration_ms,
            "finished_reason": self.finished_reason,
            "steps": [
                {
                    "step": s.step_num,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": {
                        "source": s.observation.source,
                        "content": s.observation.content[:500],
                        "success": s.observation.success,
                    }
                    if s.observation
                    else None,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
        }


# ── System Prompt ────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are a reasoning agent that solves tasks step-by-step using available tools.

For EVERY step, you MUST output in EXACTLY this format:

Thought: [your reasoning about what to do next]
Action: [tool_name OR "FINISH"]
Action Input: [input for the tool, OR your final answer if Action is FINISH]

Available tools:
{tool_descriptions}

Rules:
1. Always start with a Thought explaining your reasoning.
2. Use tools when you need external information or to perform actions.
3. After receiving an Observation (tool result), reason about it before acting again.
4. When you have enough information, use Action: FINISH with your complete answer.
5. If a tool fails, reason about alternatives — do not repeat the same failing call.
6. Be concise in thoughts. Be thorough in your final answer.
7. For parallel tools, output multiple Action/Action Input pairs prefixed with [PARALLEL].
"""


# ── Core Loop ────────────────────────────────────────────────────────────────


class ReActLoop:
    """
    ReAct-style iterative tool reasoning engine.

    Usage:
        loop = ReActLoop(llm=provider, tools=registry)
        result = await loop.run("What's the weather in Tokyo and the population?")
    """

    def __init__(
        self,
        llm: Any,
        tools: ToolRegistry,
        config: ReActConfig | None = None,
        rollback_registry: Any | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config or ReActConfig()
        self.rollback_registry = rollback_registry  # RollbackRegistry for undo

    async def run(
        self,
        task: str,
        context: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> ReActResult:
        """Execute the full ReAct loop for a given task."""
        start = time.monotonic()
        steps: list[ThoughtStep] = []
        scratchpad_lines: list[str] = []
        total_tool_calls = 0

        # Build tool descriptions
        tool_list = self.tools.list_tools(available_only=True)
        tool_desc = "\n".join(f"  - {t['name']}: {t['description']}" for t in tool_list)

        system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

        # Initial messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})

        if history:
            messages.extend(history[-6:])

        messages.append({"role": "user", "content": f"Task: {task}"})

        finished_reason = "max_iterations"
        answer = ""

        for iteration in range(1, self.config.max_iterations + 1):
            # Check timeout
            elapsed = time.monotonic() - start
            if elapsed > self.config.max_wall_time_seconds:
                finished_reason = "timeout"
                answer = self._extract_partial_answer(steps)
                break

            step_start = time.monotonic()

            # Get LLM reasoning
            llm_output = await self._call_llm(messages)

            # Parse the response
            thought, actions = self._parse_react_output(llm_output)

            step = ThoughtStep(
                step_num=iteration,
                thought=thought,
            )

            if self.config.verbose_logging:
                logger.info(
                    "ReAct step %d: thought=%s, actions=%s",
                    iteration,
                    thought[:80],
                    actions,
                )

            # Check for FINISH
            finish_action = next((a for a in actions if a["tool"].upper() == "FINISH"), None)
            if finish_action:
                answer = finish_action["input"]
                step.action = "FINISH"
                step.action_input = answer
                step.duration_ms = (time.monotonic() - step_start) * 1000
                steps.append(step)
                finished_reason = "answer"

                if self.config.include_scratchpad:
                    scratchpad_lines.append(
                        f"Step {iteration}: Thought: {thought}\n"
                        f"Action: FINISH\nAnswer: {answer[:200]}"
                    )
                break

            # Execute tool(s)
            tool_actions = [a for a in actions if a["tool"].upper() != "FINISH"]

            if not tool_actions:
                # LLM didn't output a valid action — nudge it
                messages.append({"role": "assistant", "content": llm_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Observation: You did not specify a valid Action. "
                            "Please use the format:\n"
                            "Thought: [reasoning]\n"
                            "Action: [tool_name or FINISH]\n"
                            "Action Input: [input]"
                        ),
                    }
                )
                step.duration_ms = (time.monotonic() - step_start) * 1000
                steps.append(step)
                continue

            # Parallel execution if multiple tools and allowed
            if len(tool_actions) > 1 and self.config.allow_parallel:
                observations = await self._execute_parallel(
                    tool_actions[: self.config.max_parallel_tools]
                )
                step.parallel_actions = tool_actions
                total_tool_calls += len(tool_actions)
            else:
                # Sequential — just the first action
                action = tool_actions[0]
                step.action = action["tool"]
                step.action_input = action["input"]
                obs = await self._execute_tool(action["tool"], action["input"])
                observations = [obs]
                total_tool_calls += 1

            # Build observation text
            obs_text_parts = []
            for obs in observations:
                status = "✓" if obs.success else "✗"
                obs_text_parts.append(f"[{status} {obs.source}]: {obs.content}")

            obs_text = "\n".join(obs_text_parts)
            step.observation = (
                observations[0]
                if len(observations) == 1
                else Observation(
                    source="parallel",
                    content=obs_text,
                    success=all(o.success for o in observations),
                )
            )

            step.duration_ms = (time.monotonic() - step_start) * 1000
            steps.append(step)

            # Scratchpad
            if self.config.include_scratchpad:
                action_desc = step.action or "parallel"
                scratchpad_lines.append(
                    f"Step {iteration}: Thought: {thought}\n"
                    f"Action: {action_desc}\n"
                    f"Observation: {obs_text[:300]}"
                )

            # Feed observation back to LLM
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {obs_text}"})

        # If we exhausted iterations without FINISH, extract best answer
        if not answer:
            answer = self._extract_partial_answer(steps)

        total_ms = (time.monotonic() - start) * 1000

        return ReActResult(
            answer=answer,
            steps=steps,
            total_iterations=len(steps),
            total_tool_calls=total_tool_calls,
            total_duration_ms=total_ms,
            finished_reason=finished_reason,
            scratchpad="\n---\n".join(scratchpad_lines),
        )

    # ── Tool Execution ───────────────────────────────────────────────────

    async def _execute_tool(self, tool_name: str, tool_input: str) -> Observation:
        """Execute a single tool and return an observation."""
        try:
            from hbllm.actions.agent_executor import build_tool_args

            args = build_tool_args(tool_name, tool_input)
            result: ToolResult = await self.tools.invoke(tool_name, **args)

            content = result.output if result.success else f"Error: {result.error}"
            return Observation(
                source=tool_name,
                content=content,
                success=result.success,
            )
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return Observation(
                source=tool_name,
                content=f"Tool execution failed: {e}",
                success=False,
            )

    async def _execute_parallel(self, actions: list[dict[str, str]]) -> list[Observation]:
        """Execute multiple tools concurrently."""
        tasks = [self._execute_tool(a["tool"], a["input"]) for a in actions]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # ── LLM Interface ────────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM and return the raw text response."""
        try:
            if hasattr(self.llm, "generate"):
                resp = await self.llm.generate(messages)
                # Handle both string and object responses
                if hasattr(resp, "content"):
                    return resp.content
                return str(resp)
            elif hasattr(self.llm, "chat"):
                return await self.llm.chat(messages)
            else:
                return "Thought: I don't have access to an LLM.\nAction: FINISH\nAction Input: Unable to process — no LLM configured."
        except Exception as e:
            logger.error("LLM call failed in ReAct loop: %s", e)
            return f"Thought: LLM call failed with error: {e}\nAction: FINISH\nAction Input: I encountered an error and cannot continue."

    # ── Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_react_output(text: str) -> tuple[str, list[dict[str, str]]]:
        """Parse Thought/Action/Action Input from LLM output.

        Returns:
            (thought, list of {tool, input} dicts)
        """
        import re

        thought = ""
        actions: list[dict[str, str]] = []

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract all Action/Action Input pairs
        action_pattern = re.compile(
            r"Action:\s*(.+?)\s*\n\s*Action Input:\s*(.+?)(?=\nThought:|\nAction:|\Z)",
            re.DOTALL,
        )

        for match in action_pattern.finditer(text):
            tool = match.group(1).strip()
            inp = match.group(2).strip()
            actions.append({"tool": tool, "input": inp})

        # Fallback: if no structured output, try to extract intent
        if not thought and not actions:
            thought = text.strip()
            # Check if the entire response looks like a final answer
            if not any(
                kw in text.lower() for kw in ["tool_call", "action:", "let me", "i need to"]
            ):
                actions.append({"tool": "FINISH", "input": text.strip()})

        return thought, actions

    @staticmethod
    def _extract_partial_answer(steps: list[ThoughtStep]) -> str:
        """Extract the best partial answer from completed steps."""
        # Use the last observation as context
        observations = [
            s.observation.content for s in steps if s.observation and s.observation.success
        ]

        if observations:
            return "Based on the information gathered:\n" + "\n".join(observations[-3:])

        # Fall back to last thought
        if steps:
            return steps[-1].thought

        return "Unable to complete the task within the allowed budget."
