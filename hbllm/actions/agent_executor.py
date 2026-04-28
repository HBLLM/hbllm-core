"""
Core Agent Executor — Framework-agnostic tool-augmented task execution.

Provides:
  - AgentExecutor: single-agent tool-augmented chat
  - AgentStep / AgentResponse: data structures for execution traces
  - CoreToolSet: registers framework-agnostic tools (python, shell, file, web)

Platform-specific tools (config, model management, scheduling) are NOT
registered here — they belong in the platform bridge (e.g. sentra.agent_executor).
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
            except Exception as e:
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
        multi-agent orchestrator. Otherwise, uses single-agent with tool support.
        """
        start = time.monotonic()

        if agent_mode and ComplexityDetector.needs_multi_agent(message):
            from hbllm.actions.orchestrator import MultiAgentOrchestrator

            orchestrator = MultiAgentOrchestrator(self.llm, self.tools)
            return await orchestrator.execute(message, history, kb_context)

        # Single agent with tool parsing
        steps: list[AgentStep] = []
        step_num = 0

        # Build messages for LLM
        tool_list = self.tools.list_tools()
        tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_list)

        system_prompt = (
            (
                "You are a helpful AI assistant with access to tools.\n\n"
                f"Available tools:\n{tool_desc}\n\n"
                "To use a tool, output EXACTLY:\n"
                "TOOL_CALL: tool_name\n"
                "TOOL_INPUT: input_value\n\n"
                "After receiving tool results, provide your final answer.\n"
                "Only use tools when needed — for simple questions, just answer directly."
            )
            if agent_mode
            else "You are a helpful AI assistant."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

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

        # Get LLM response
        response_text = await self._call_llm(messages)
        step_num += 1

        # Check for tool calls
        if agent_mode:
            tool_calls = self._extract_tool_calls(response_text)
            if tool_calls:
                for tc in tool_calls:
                    step_num += 1
                    result = await self.tools.invoke(
                        tc["tool"], **build_tool_args(tc["tool"], tc["input"])
                    )
                    steps.append(
                        AgentStep(
                            step_num=step_num,
                            role="executor",
                            action="tool_call",
                            content=f"Using {tc['tool']}",
                            tool_result=result,
                        )
                    )

                # Feed results back to LLM for final response
                tool_results_text = "\n".join(
                    f"Tool: {s.tool_result.tool}\n"
                    f"{'Output' if s.tool_result.success else 'Error'}: "
                    f"{s.tool_result.output if s.tool_result.success else s.tool_result.error}"
                    for s in steps
                    if s.tool_result
                )
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool results:\n{tool_results_text}\n\nProvide your final answer based on these results.",
                    }
                )
                response_text = await self._call_llm(messages)

        total_ms = (time.monotonic() - start) * 1000
        confidence = ConfidenceScorer.score(message, response_text, had_context=bool(kb_context))

        return AgentResponse(
            content=response_text,
            steps=steps,
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
        except Exception as e:
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
