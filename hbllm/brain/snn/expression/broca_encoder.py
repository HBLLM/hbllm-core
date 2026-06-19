"""
Broca's Encoder — ultra-minimal LLM interface.

The LLM is genuinely just Broca's area: it receives structured
content decisions from the SNN and produces grammatically correct
text.  No context, no reasoning, no query, no memory hints.

Prompt template (~80 tokens)::

    TYPE: explanation
    TONE: neutral
    SAY: [point 1, point 2, point 3]
    MAX: 50 tokens
    → produce one fluent sentence

This is an 87% token reduction from v1 deep prompts (~600 tokens)
and 73% reduction from v3 shallow prompts (~300 tokens).

Components:
    BrocaEncoder  — generates ~80-token prompts from ContentNodes
    BrocaPrompt   — the minimal prompt data structure

Usage::

    encoder = BrocaEncoder()
    prompt = encoder.encode(content_node)
    # prompt is ~80 tokens: "TYPE: assertion\\nTONE: emphatic\\nSAY: [...]\\n→ one fluent sentence"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.brain.snn.expression.content_planner import ContentNode

logger = logging.getLogger(__name__)


@dataclass
class BrocaPrompt:
    """Minimal prompt data structure for Broca's encoding.

    Attributes:
        prompt_text: The actual prompt string to send to the LLM.
        content_type: The content type being rendered.
        tone: The tone being requested.
        key_point_count: Number of key points included.
        estimated_tokens: Estimated token count of the prompt.
    """

    prompt_text: str = ""
    content_type: str = ""
    tone: str = ""
    key_point_count: int = 0
    estimated_tokens: int = 0


class BrocaEncoder:
    """Ultra-minimal LLM interface — text production only.

    Converts ``ContentNode`` objects into ~80-token prompts that
    tell the LLM exactly what to say.  The LLM's only job is
    grammar and fluency.

    The prompt format is deliberately terse:

    .. code-block:: text

        TYPE: explanation
        TONE: neutral
        SAY:
        - SNN architecture uses LIF neurons
        - STDP enables learning
        MAX: 50
        → one fluent sentence

    Args:
        max_tokens_per_node: Default token budget per node.
        sentence_instruction: The final instruction line.
    """

    def __init__(
        self,
        max_tokens_per_node: int = 60,
        sentence_instruction: str = "→ produce one fluent sentence",
    ) -> None:
        self._max_tokens = max_tokens_per_node
        self._instruction = sentence_instruction

    def encode(self, node: ContentNode) -> BrocaPrompt:
        """Generate a minimal prompt for one ContentNode.

        Args:
            node: The content decision to render.

        Returns:
            BrocaPrompt with the ultra-minimal prompt text.
        """
        lines: list[str] = []

        lines.append(f"TYPE: {node.content_type}")
        lines.append(f"TONE: {node.tone}")

        lines.append("SAY:")
        for point in node.key_points:
            lines.append(f"- {point}")

        if node.causal_basis:
            lines.append(f"BECAUSE: {node.causal_basis}")

        lines.append(f"MAX: {self._max_tokens}")
        lines.append(self._instruction)

        prompt_text = "\n".join(lines)

        return BrocaPrompt(
            prompt_text=prompt_text,
            content_type=node.content_type,
            tone=node.tone,
            key_point_count=len(node.key_points),
            estimated_tokens=max(1, len(prompt_text) // 4),
        )

    def encode_batch(self, nodes: list[ContentNode]) -> BrocaPrompt:
        """Generate a batch prompt for multiple ContentNodes.

        Produces a single prompt that asks for multiple sentences,
        one per content node.

        Args:
            nodes: Ordered list of ContentNodes.

        Returns:
            BrocaPrompt with a multi-sentence rendering prompt.
        """
        if not nodes:
            return BrocaPrompt()

        lines: list[str] = []
        lines.append("RENDER each item as one sentence:")
        lines.append("")

        for i, node in enumerate(nodes):
            lines.append(f"[{i + 1}] TYPE: {node.content_type} | TONE: {node.tone}")
            for point in node.key_points:
                lines.append(f"  - {point}")
            if node.causal_basis:
                lines.append(f"  BECAUSE: {node.causal_basis}")
            lines.append("")

        total_budget = self._max_tokens * len(nodes)
        lines.append(f"MAX: {total_budget} tokens total")
        lines.append("→ produce numbered sentences, one per item")

        prompt_text = "\n".join(lines)

        return BrocaPrompt(
            prompt_text=prompt_text,
            content_type="batch",
            tone="mixed",
            key_point_count=sum(len(n.key_points) for n in nodes),
            estimated_tokens=max(1, len(prompt_text) // 4),
        )

    def assemble(
        self,
        rendered_nodes: list[tuple[ContentNode, str]],
    ) -> str:
        """Assemble rendered nodes into final response text.

        Groups content by type and joins with appropriate paragraph
        breaks.

        Args:
            rendered_nodes: List of (ContentNode, rendered_text) tuples,
                in order.

        Returns:
            Assembled response text.
        """
        if not rendered_nodes:
            return ""

        parts: list[str] = []
        prev_type = ""

        for node, text in rendered_nodes:
            text = text.strip()
            if not text:
                continue

            # Add paragraph break on major type changes
            if prev_type and node.content_type != prev_type:
                if node.content_type in ("transition",) or prev_type in ("transition",):
                    pass  # Transitions flow naturally
                else:
                    parts.append("")  # paragraph break

            parts.append(text)
            prev_type = node.content_type

        return "\n".join(parts)

    @staticmethod
    def estimate_savings(
        broca_prompt: BrocaPrompt,
        shallow_tokens: int = 300,
        deep_tokens: int = 600,
    ) -> dict[str, float]:
        """Estimate token savings vs v3 shallow and v1 deep prompts.

        Returns:
            Dict with 'vs_shallow' and 'vs_deep' reduction percentages.
        """
        broca_tokens = broca_prompt.estimated_tokens
        return {
            "vs_shallow": 1.0 - (broca_tokens / max(1, shallow_tokens)),
            "vs_deep": 1.0 - (broca_tokens / max(1, deep_tokens)),
            "broca_tokens": broca_tokens,
        }
