"""
Context Window Manager — manages token budget for each request.

Ensures the combined context (system prompt + memory + RAG + conversation)
fits within the model's context window. Uses priority-based truncation:

  Priority 1 (highest): System prompt / identity
  Priority 2: Recent conversation history
  Priority 3: Retrieved memory context
  Priority 4: Curiosity goals / metadata
  Priority 5 (lowest): Older conversation history
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Rough token estimation: 1 token ≈ 4 characters (English text average)
_CHARS_PER_TOKEN = 4


@dataclass
class ContextBlock:
    """A block of content with priority for context window fitting."""
    content: str
    priority: int  # 1 = highest (keep first), 5 = lowest (truncate first)
    label: str = ""
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = estimate_tokens(self.content)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. Uses char-based heuristic."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


class ContextWindowManager:
    """
    Manages the token budget for model context windows.

    Fits multiple content blocks (system prompt, memory, conversation, etc.)
    into a fixed token budget using priority-based truncation.

    Usage:
        cwm = ContextWindowManager(max_tokens=2048)
        cwm.add("system", "You are helpful.", priority=1)
        cwm.add("memory", "User likes Python.", priority=3)
        cwm.add("conversation", long_history, priority=2)
        result = cwm.build()
        # result.text is the fitted context, result.used_tokens <= 2048
    """

    def __init__(self, max_tokens: int = 2048, reserve_for_output: int = 256):
        self.max_tokens = max_tokens
        self.reserve_for_output = reserve_for_output
        self._blocks: list[ContextBlock] = []

    @property
    def available_tokens(self) -> int:
        """Tokens available for context (minus output reserve)."""
        return self.max_tokens - self.reserve_for_output

    def add(
        self,
        label: str,
        content: str,
        priority: int = 3,
    ) -> None:
        """
        Add a content block to the context window.

        Args:
            label: Human-readable label (e.g., "system", "memory", "conversation")
            content: The text content
            priority: 1 (highest, kept first) to 5 (lowest, truncated first)
        """
        if not content or not content.strip():
            return

        block = ContextBlock(
            content=content.strip(),
            priority=priority,
            label=label,
        )
        self._blocks.append(block)

    def add_messages(
        self,
        messages: list[dict[str, str]],
        priority: int = 2,
        label: str = "conversation",
    ) -> None:
        """
        Add conversation messages (list of {role, content} dicts).
        Recent messages get higher priority.
        """
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # More recent messages = lower index = higher priority adjustment
            recency_bonus = max(0, len(messages) - i - 1)
            effective_priority = min(5, priority + (recency_bonus // 3))

            self.add(
                label=f"{label}:{role}:{i}",
                content=f"{role}: {content}",
                priority=effective_priority,
            )

    def build(self) -> ContextResult:
        """
        Build the final context string by fitting blocks within the token budget.

        Returns a ContextResult with the fitted text and usage stats.
        """
        budget = self.available_tokens

        # Sort by priority (highest first = lowest number)
        sorted_blocks = sorted(self._blocks, key=lambda b: b.priority)

        included: list[ContextBlock] = []
        truncated: list[str] = []
        used_tokens = 0

        for block in sorted_blocks:
            remaining = budget - used_tokens
            if remaining <= 0:
                truncated.append(block.label)
                continue

            if block.token_estimate <= remaining:
                # Block fits entirely
                included.append(block)
                used_tokens += block.token_estimate
            else:
                # Partial fit — truncate content to fit
                available_chars = remaining * _CHARS_PER_TOKEN
                truncated_content = block.content[:available_chars]

                # Try to truncate at a sentence boundary
                last_period = truncated_content.rfind(".")
                last_newline = truncated_content.rfind("\n")
                cut_point = max(last_period, last_newline)
                if cut_point > len(truncated_content) // 2:
                    truncated_content = truncated_content[:cut_point + 1]

                partial_block = ContextBlock(
                    content=truncated_content,
                    priority=block.priority,
                    label=f"{block.label} (truncated)",
                )
                included.append(partial_block)
                used_tokens += partial_block.token_estimate
                truncated.append(block.label)

        # Reconstruct in original order for natural flow
        # Group by priority for readability
        sections = []
        for block in included:
            sections.append(block.content)

        text = "\n\n".join(sections)

        return ContextResult(
            text=text,
            used_tokens=used_tokens,
            max_tokens=self.max_tokens,
            blocks_included=len(included),
            blocks_truncated=truncated,
            utilization=used_tokens / self.available_tokens if self.available_tokens > 0 else 0,
        )

    def clear(self) -> None:
        """Clear all blocks for reuse."""
        self._blocks.clear()

    def stats(self) -> dict[str, Any]:
        """Return current stats before building."""
        total_tokens = sum(b.token_estimate for b in self._blocks)
        return {
            "blocks": len(self._blocks),
            "total_tokens_needed": total_tokens,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
            "would_fit": total_tokens <= self.available_tokens,
        }


@dataclass
class ContextResult:
    """Result of context window fitting."""
    text: str
    used_tokens: int
    max_tokens: int
    blocks_included: int
    blocks_truncated: list[str]
    utilization: float  # 0.0 to 1.0

    @property
    def is_truncated(self) -> bool:
        return len(self.blocks_truncated) > 0

    def summary(self) -> str:
        """Human-readable summary."""
        trunc = f", truncated: {self.blocks_truncated}" if self.blocks_truncated else ""
        return (
            f"Context: {self.used_tokens}/{self.max_tokens} tokens "
            f"({self.utilization:.0%}), {self.blocks_included} blocks{trunc}"
        )
