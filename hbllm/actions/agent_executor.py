"""
Core Agent Executor — Framework-agnostic tool-augmented task execution.

Provides:
  - AgentExecutor: single-agent tool-augmented chat
  - AgentStep / AgentResponse: data structures for execution traces
  - CoreToolSet: registers framework-agnostic tools (python, shell, file, web)

Platform-specific tools (config, model management, scheduling) are NOT
registered here — they belong in the platform bridge layer.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.actions.builtin_tools import (
    SAFE_SHELL_COMMANDS,
    tool_file_read,
    tool_file_write,
    tool_python_exec,
    tool_shell_exec,
    tool_web_search,
)
from hbllm.actions.complexity import ComplexityDetector
from hbllm.actions.tool_registry import ToolRegistry, ToolResult
from hbllm.brain.confidence_estimator import ConfidenceEstimator

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class AgentStep:
    """A single step in an agent execution trace."""

    step_num: int
    role: str  # "planner", "executor", "reviewer", "user"
    action: str  # "think", "tool_call", "review", "respond"
    content: str
    tool_result: ToolResult | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_num": self.step_num,
            "role": self.role,
            "action": self.action,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_result:
            d["tool_result"] = {
                "tool": self.tool_result.tool,
                "success": self.tool_result.success,
                "output": self.tool_result.output,
                "error": self.tool_result.error,
                "duration_ms": self.tool_result.duration_ms,
            }
        return d


@dataclass
class AgentResponse:
    """Complete response from agent execution."""

    content: str
    steps: list[AgentStep]
    confidence: float = 0.0
    confidence_flags: list[str] = field(default_factory=list)
    multi_agent: bool = False
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "steps": [s.to_dict() for s in self.steps],
            "confidence": self.confidence,
            "confidence_flags": self.confidence_flags,
            "multi_agent": self.multi_agent,
            "total_duration_ms": self.total_duration_ms,
        }


# ── Confidence Wrapper ─────────────────────────────────────────────────────────


class ConfidenceScorer:
    """Wrapper around core ConfidenceEstimator for scoring responses."""

    _estimator = ConfidenceEstimator()

    @staticmethod
    def score(query: str, response: str, had_context: bool = False) -> dict[str, Any]:
        report = ConfidenceScorer._estimator.estimate(query=query, response=response)
        overall = report.overall
        if had_context:
            overall = min(1.0, overall + 0.1)
        return {
            "overall": round(overall, 2),
            "flags": report.flags,
        }


# ── Core Tool Registration ────────────────────────────────────────────────────


def register_core_tools(registry: ToolRegistry) -> None:
    """Register framework-agnostic tools into a ToolRegistry."""
    registry.register(
        "python_exec",
        "Execute Python code in a sandboxed subprocess with 5s timeout and 256MB memory limit",
        tool_python_exec,
        {"code": "Python code to execute"},
    )
    registry.register(
        "shell_exec",
        f"Execute safe shell commands. Allowed: {', '.join(sorted(SAFE_SHELL_COMMANDS))}",
        tool_shell_exec,
        {"command": "Shell command to execute"},
    )
    registry.register(
        "web_search",
        "Search the web using DuckDuckGo (no API key needed)",
        tool_web_search,
        {"query": "Search query"},
    )
    registry.register(
        "file_read",
        "Read a file from the local filesystem (max 500KB)",
        tool_file_read,
        {"path": "Absolute or home-relative file path"},
    )
    registry.register(
        "file_write",
        "Write content to a file (within home directory only)",
        tool_file_write,
        {"path": "File path", "content": "Content to write"},
    )


# ── Tool Argument Builder ─────────────────────────────────────────────────────


def build_tool_args(tool_name: str, tool_input: str) -> dict[str, str]:
    """Build keyword arguments for a tool call from raw input string."""
    if tool_name == "python_exec":
        return {"code": tool_input}
    elif tool_name == "shell_exec":
        return {"command": tool_input}
    elif tool_name == "web_search":
        return {"query": tool_input}
    elif tool_name == "file_read":
        return {"path": tool_input}
    elif tool_name == "file_write":
        if ":" in tool_input:
            path, content = tool_input.split(":", 1)
            return {"path": path.strip(), "content": content.strip()}
        return {"path": tool_input, "content": ""}
    elif tool_name == "kb_search":
        return {"query": tool_input}
    return {"input": tool_input}


# ── Agent Executor ─────────────────────────────────────────────────────────────


class AgentExecutor:
    """
    Core agent executor for tool-augmented chat.

    Handles single-agent tool calls by:
      1. Parsing TOOL_CALL/TOOL_INPUT pairs from LLM responses
      2. Executing tools and feeding results back
      3. Auto-detecting complexity for multi-agent pipeline

    To extend with platform-specific tools, subclass and override
    ``_register_platform_tools()``.
    """

    def __init__(self, llm: Any, kb: Any = None):
        self.llm = llm
        self.kb = kb
        self.tools = ToolRegistry()
        register_core_tools(self.tools)
        self._register_platform_tools()

    def _register_platform_tools(self) -> None:
        """Override in subclasses to register platform-specific tools."""
        pass

    def register_kb_search(self, search_fn: Any) -> None:
        """Register a knowledge base search function as a tool."""
        import asyncio

        async def _kb_search(query: str) -> ToolResult:
            try:
                results = await asyncio.to_thread(search_fn, query, 5)
                formatted = []
                for r in results:
                    formatted.append(
                        f"[{r.get('file_name', '?')}] (score: {r.get('score', 0):.2f})\n"
                        f"{r.get('content', '')[:300]}"
                    )
                output = "\n\n".join(formatted) if formatted else "No results found."
                return ToolResult(tool="kb_search", success=True, output=output)
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
                return ToolResult(tool="kb_search", success=False, output="", error=str(e))

        self.tools.register(
            "kb_search",
            "Search the knowledge base for relevant documents and code",
            _kb_search,
            {"query": "Search query"},
        )

    async def execute(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        agent_mode: bool = False,
        kb_context: str = "",
    ) -> AgentResponse:
        """
        Execute a user message.

        If agent_mode is True and the task is complex, delegates to the
        multi-agent orchestrator. For moderately complex tasks with tools,
        uses the ReAct iterative reasoning loop. Otherwise, single-pass.
        """
        start = time.monotonic()

        if agent_mode and ComplexityDetector.needs_multi_agent(message):
            from hbllm.actions.orchestrator import MultiAgentOrchestrator

            orchestrator = MultiAgentOrchestrator(self.llm, self.tools)
            return await orchestrator.execute(message, history, kb_context)

        # ── ReAct loop for tool-augmented reasoning ──
        if agent_mode:
            from hbllm.actions.tool_chain import ReActConfig, ReActLoop

            react = ReActLoop(
                llm=self.llm,
                tools=self.tools,
                config=ReActConfig(max_iterations=8, max_wall_time_seconds=60.0),
            )
            result = await react.run(
                task=message,
                context=kb_context,
                history=history,
            )

            # Convert ReAct steps to AgentSteps for API compatibility
            steps = []
            for rs in result.steps:
                tool_result = None
                if rs.observation and rs.action and rs.action != "FINISH":
                    tool_result = ToolResult(
                        tool=rs.action,
                        success=rs.observation.success,
                        output=rs.observation.content if rs.observation.success else "",
                        error=rs.observation.content if not rs.observation.success else "",
                        duration_ms=rs.duration_ms,
                    )
                steps.append(
                    AgentStep(
                        step_num=rs.step_num,
                        role="executor",
                        action="tool_call" if rs.action != "FINISH" else "respond",
                        content=rs.thought,
                        tool_result=tool_result,
                    )
                )

            total_ms = (time.monotonic() - start) * 1000
            confidence = ConfidenceScorer.score(
                message, result.answer, had_context=bool(kb_context)
            )

            return AgentResponse(
                content=result.answer,
                steps=steps,
                confidence=confidence["overall"],
                confidence_flags=confidence["flags"],
                multi_agent=False,
                total_duration_ms=total_ms,
            )

        # ── Simple single-pass (non-agent mode) ──
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        if kb_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant context from knowledge base:\n{kb_context}",
                }
            )
        if history:
            messages.extend(history[-10:])
        messages.append({"role": "user", "content": message})

        response_text = await self._call_llm(messages)
        total_ms = (time.monotonic() - start) * 1000
        confidence = ConfidenceScorer.score(message, response_text, had_context=bool(kb_context))

        return AgentResponse(
            content=response_text,
            steps=[],
            confidence=confidence["overall"],
            confidence_flags=confidence["flags"],
            multi_agent=False,
            total_duration_ms=total_ms,
        )

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        try:
            if hasattr(self.llm, "chat"):
                return await self.llm.chat(messages)
            elif hasattr(self.llm, "generate"):
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                return await self.llm.generate(prompt)
            else:
                return "[LLM not available — configure a provider]"
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("LLM call failed: %s", e)
            return f"[Error: {e}]"

    @staticmethod
    def _extract_tool_calls(text: str) -> list[dict[str, str]]:
        """Extract TOOL_CALL/TOOL_INPUT pairs from LLM response."""
        calls: list[dict[str, str]] = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match = re.match(r"TOOL_CALL:\s*(.+)", line, re.IGNORECASE)
            if match:
                tool = match.group(1).strip()
                inp = ""
                if i + 1 < len(lines):
                    inp_match = re.match(r"TOOL_INPUT:\s*(.+)", lines[i + 1].strip(), re.IGNORECASE)
                    if inp_match:
                        inp = inp_match.group(1).strip()
                        i += 1
                calls.append({"tool": tool, "input": inp})
            i += 1
        return calls
