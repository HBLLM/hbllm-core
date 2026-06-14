"""Dual LLM Router — routes generation to local or external model by task tier.

The brain uses two LLM backends:
  - **Local model**: Fast, low-latency, runs on-device. Used for content
    generation, shallow rendering, embeddings, and simple structured output.
  - **External model**: Powerful, high-quality. Used for complex reasoning,
    multi-step planning, code generation, and tool orchestration.

The router selects which backend to use based on:
  1. Explicit ``TaskTier`` hints from the caller
  2. The current ``CognitiveStateMachine`` state (``allow_heavy_llm``)
  3. Automatic complexity heuristics (prompt length, keyword detection)

Usage::

    from hbllm.brain.dual_llm_router import DualLLMRouter, TaskTier

    router = DualLLMRouter(
        local=ProviderLLM(local_provider),
        external=ProviderLLM(openai_provider),
    )

    # Auto-route based on task complexity
    text = await router.generate("Hello!", tier=TaskTier.AUTO)

    # Force local for fast chat
    text = await router.generate("Hi there", tier=TaskTier.LOCAL)

    # Force external for complex reasoning
    text = await router.generate(complex_prompt, tier=TaskTier.EXTERNAL)
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.brain.autonomy.state_machine import CognitiveStateMachine
    from hbllm.brain.provider_adapter import ProviderLLM

logger = logging.getLogger(__name__)


# ── Task Tier ────────────────────────────────────────────────────────────────


class TaskTier(StrEnum):
    """Which LLM tier to route a generation request to."""

    LOCAL = "local"  # Small on-device model
    EXTERNAL = "external"  # Large cloud/API model
    AUTO = "auto"  # Let the router decide


# ── Routing Decision ─────────────────────────────────────────────────────────


@dataclass
class RoutingDecision:
    """Record of why a particular tier was chosen."""

    tier: TaskTier
    reason: str
    prompt_length: int = 0
    complexity_score: float = 0.0
    state_allows_heavy: bool = True
    fallback_used: bool = False


# ── Complexity Heuristics ────────────────────────────────────────────────────

# Keywords that suggest complex reasoning requiring a powerful model
_COMPLEX_KEYWORDS = frozenset(
    {
        "analyze",
        "architect",
        "compare",
        "contrast",
        "debug",
        "decompose",
        "design",
        "evaluate",
        "explain why",
        "implement",
        "optimize",
        "prove",
        "reason",
        "refactor",
        "trade-off",
        "tradeoff",
    }
)

# Keywords that suggest simple tasks fine for a local model
_SIMPLE_KEYWORDS = frozenset(
    {
        "define",
        "greet",
        "hello",
        "help",
        "hi",
        "list",
        "name",
        "repeat",
        "say",
        "summarize",
        "thanks",
        "translate",
        "what is",
        "what's",
    }
)

# Patterns that suggest code generation (→ external)
_CODE_PATTERNS = re.compile(
    r"(write|create|build|implement|fix|debug)\s+(a\s+)?"
    r"(function|class|module|script|program|api|endpoint|test|code)",
    re.IGNORECASE,
)


def estimate_complexity(prompt: str) -> float:
    """Estimate task complexity from the prompt text (0.0 = trivial, 1.0 = hard).

    This is a fast heuristic — not a classifier. Used when ``TaskTier.AUTO``
    is requested and no cognitive state machine is available.

    Factors:
      - Prompt length (longer = more complex)
      - Presence of complexity keywords
      - Code generation patterns
      - Multi-step indicators (numbered lists, "then", "after")
    """
    score = 0.0
    prompt_lower = prompt.lower()

    # Length factor (0.0 – 0.3)
    word_count = len(prompt.split())
    if word_count > 200:
        score += 0.3
    elif word_count > 100:
        score += 0.2
    elif word_count > 50:
        score += 0.1

    # Complex keyword detection (0.0 – 0.3)
    complex_hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in prompt_lower)
    score += min(complex_hits * 0.1, 0.3)

    # Simple keyword suppression
    simple_hits = sum(1 for kw in _SIMPLE_KEYWORDS if kw in prompt_lower)
    score -= min(simple_hits * 0.1, 0.2)

    # Code generation (0.0 – 0.2)
    if _CODE_PATTERNS.search(prompt):
        score += 0.2

    # Multi-step indicators (0.0 – 0.2)
    step_indicators = len(re.findall(r"\b(then|after|next|step \d|finally)\b", prompt_lower))
    score += min(step_indicators * 0.05, 0.2)

    return max(0.0, min(1.0, score))


# ── Usage Stats ──────────────────────────────────────────────────────────────


@dataclass
class DualLLMStats:
    """Accumulated usage statistics for both tiers."""

    local_calls: int = 0
    external_calls: int = 0
    local_tokens: int = 0
    external_tokens: int = 0
    auto_routed_local: int = 0
    auto_routed_external: int = 0
    fallbacks: int = 0  # External → Local fallback count
    total_latency_local_ms: float = 0.0
    total_latency_external_ms: float = 0.0

    @property
    def total_calls(self) -> int:
        return self.local_calls + self.external_calls

    @property
    def local_ratio(self) -> float:
        """Fraction of calls handled by local model (0.0 – 1.0)."""
        if self.total_calls == 0:
            return 0.0
        return self.local_calls / self.total_calls

    @property
    def avg_latency_local_ms(self) -> float:
        if self.local_calls == 0:
            return 0.0
        return self.total_latency_local_ms / self.local_calls

    @property
    def avg_latency_external_ms(self) -> float:
        if self.external_calls == 0:
            return 0.0
        return self.total_latency_external_ms / self.external_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_calls": self.local_calls,
            "external_calls": self.external_calls,
            "local_tokens": self.local_tokens,
            "external_tokens": self.external_tokens,
            "auto_routed_local": self.auto_routed_local,
            "auto_routed_external": self.auto_routed_external,
            "fallbacks": self.fallbacks,
            "local_ratio": round(self.local_ratio, 3),
            "avg_latency_local_ms": round(self.avg_latency_local_ms, 1),
            "avg_latency_external_ms": round(self.avg_latency_external_ms, 1),
        }


# ── Dual LLM Router ─────────────────────────────────────────────────────────


class DualLLMRouter:
    """Routes LLM requests between a local and external model.

    Exposes the same ``generate`` / ``generate_json`` / ``generate_stream``
    interface as ``ProviderLLM``, making it a drop-in replacement.

    Routing logic:
      1. If ``tier=TaskTier.LOCAL`` → always use local
      2. If ``tier=TaskTier.EXTERNAL`` → use external (fallback to local if unavailable)
      3. If ``tier=TaskTier.AUTO``:
         a. Check CognitiveStateMachine.allow_heavy_llm
         b. Estimate prompt complexity
         c. Route to external if complexity > threshold, else local
    """

    # Complexity threshold — prompts scoring above this use external
    DEFAULT_COMPLEXITY_THRESHOLD = 0.4

    def __init__(
        self,
        local: ProviderLLM,
        external: ProviderLLM | None = None,
        state_machine: CognitiveStateMachine | None = None,
        complexity_threshold: float = DEFAULT_COMPLEXITY_THRESHOLD,
    ) -> None:
        self.local = local
        self.external = external
        self.state_machine = state_machine
        self.complexity_threshold = complexity_threshold
        self.stats = DualLLMStats()

        # Provider names for logging
        self._local_name = getattr(local, "provider", None)
        self._external_name = getattr(external, "provider", None) if external else None

        logger.info(
            "[DualLLMRouter] Initialized: local=%s, external=%s, threshold=%.2f",
            getattr(self._local_name, "name", "local") if self._local_name else "local",
            getattr(self._external_name, "name", "none") if self._external_name else "none",
            complexity_threshold,
        )

    # ── Routing Decision ──────────────────────────────────────────────

    def route(self, prompt: str, tier: TaskTier = TaskTier.AUTO) -> RoutingDecision:
        """Decide which tier to use for a given prompt.

        Args:
            prompt: The generation prompt.
            tier: Explicit tier hint (LOCAL, EXTERNAL, or AUTO).

        Returns:
            RoutingDecision with the chosen tier and reasoning.
        """
        prompt_length = len(prompt)

        # Explicit tier
        if tier == TaskTier.LOCAL:
            return RoutingDecision(
                tier=TaskTier.LOCAL,
                reason="explicit_local",
                prompt_length=prompt_length,
            )

        if tier == TaskTier.EXTERNAL:
            if self.external is None:
                return RoutingDecision(
                    tier=TaskTier.LOCAL,
                    reason="external_requested_but_unavailable",
                    prompt_length=prompt_length,
                    fallback_used=True,
                )
            return RoutingDecision(
                tier=TaskTier.EXTERNAL,
                reason="explicit_external",
                prompt_length=prompt_length,
            )

        # AUTO routing
        state_allows_heavy = True
        if self.state_machine is not None:
            state_allows_heavy = self.state_machine.current_profile.allow_heavy_llm

        # If no external model configured, always use local
        if self.external is None:
            return RoutingDecision(
                tier=TaskTier.LOCAL,
                reason="no_external_configured",
                prompt_length=prompt_length,
                state_allows_heavy=state_allows_heavy,
            )

        # If cognitive state forbids heavy LLM, use local
        if not state_allows_heavy:
            return RoutingDecision(
                tier=TaskTier.LOCAL,
                reason="state_forbids_heavy_llm",
                prompt_length=prompt_length,
                state_allows_heavy=False,
            )

        # Estimate complexity
        complexity = estimate_complexity(prompt)

        if complexity >= self.complexity_threshold:
            return RoutingDecision(
                tier=TaskTier.EXTERNAL,
                reason=f"auto_complexity_{complexity:.2f}",
                prompt_length=prompt_length,
                complexity_score=complexity,
                state_allows_heavy=state_allows_heavy,
            )
        else:
            return RoutingDecision(
                tier=TaskTier.LOCAL,
                reason=f"auto_simple_{complexity:.2f}",
                prompt_length=prompt_length,
                complexity_score=complexity,
                state_allows_heavy=state_allows_heavy,
            )

    def _get_llm(self, decision: RoutingDecision) -> ProviderLLM:
        """Get the ProviderLLM instance for a routing decision."""
        if decision.tier == TaskTier.EXTERNAL and self.external is not None:
            return self.external
        return self.local

    # ── Generate Interface ────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        tier: TaskTier = TaskTier.AUTO,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text, routing to the appropriate model.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tier: Which tier to use (AUTO, LOCAL, or EXTERNAL).
            system_prompt: Optional system prompt override.

        Returns:
            Generated text string.
        """
        decision = self.route(prompt, tier)
        llm = self._get_llm(decision)

        start = time.monotonic()
        try:
            result = await llm.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
        except Exception as e:
            # Fallback: if external fails, try local
            if decision.tier == TaskTier.EXTERNAL and self.external is not None:
                logger.warning("[DualLLMRouter] External failed (%s), falling back to local", e)
                self.stats.fallbacks += 1
                decision.fallback_used = True
                result = await self.local.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            else:
                raise

        latency_ms = (time.monotonic() - start) * 1000
        self._record_stats(decision, latency_ms)

        logger.debug(
            "[DualLLMRouter] %s → %s (%.1fms, reason=%s)",
            decision.tier.value,
            llm.provider.name if hasattr(llm, "provider") else "unknown",
            latency_ms,
            decision.reason,
        )

        return result

    async def generate_json(
        self,
        prompt: str,
        max_tokens: int = 64,
        tier: TaskTier = TaskTier.AUTO,
    ) -> dict[str, Any]:
        """Generate structured JSON, routing to the appropriate model.

        JSON extraction is generally fine on local models since the output
        is constrained and deterministic.
        """
        decision = self.route(prompt, tier)
        llm = self._get_llm(decision)

        start = time.monotonic()
        try:
            result = await llm.generate_json(prompt, max_tokens=max_tokens)
        except Exception as e:
            if decision.tier == TaskTier.EXTERNAL and self.external is not None:
                logger.warning(
                    "[DualLLMRouter] External JSON failed (%s), falling back to local", e
                )
                self.stats.fallbacks += 1
                result = await self.local.generate_json(prompt, max_tokens=max_tokens)
            else:
                raise

        latency_ms = (time.monotonic() - start) * 1000
        self._record_stats(decision, latency_ms)

        return result

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        tier: TaskTier = TaskTier.AUTO,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens, routing to the appropriate model."""
        decision = self.route(prompt, tier)
        llm = self._get_llm(decision)

        start = time.monotonic()
        try:
            async for token in llm.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            ):
                yield token
        except Exception as e:
            if decision.tier == TaskTier.EXTERNAL and self.external is not None:
                logger.warning(
                    "[DualLLMRouter] External stream failed (%s), falling back to local", e
                )
                self.stats.fallbacks += 1
                async for token in self.local.generate_stream(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                ):
                    yield token
            else:
                raise

        latency_ms = (time.monotonic() - start) * 1000
        self._record_stats(decision, latency_ms)

    # ── Stats ─────────────────────────────────────────────────────────

    def _record_stats(self, decision: RoutingDecision, latency_ms: float) -> None:
        """Record usage statistics for a generation call."""
        if decision.tier == TaskTier.LOCAL or decision.fallback_used:
            self.stats.local_calls += 1
            self.stats.total_latency_local_ms += latency_ms
        else:
            self.stats.external_calls += 1
            self.stats.total_latency_external_ms += latency_ms

        # Track auto-routing decisions
        if decision.reason.startswith("auto_"):
            if decision.tier == TaskTier.LOCAL or decision.fallback_used:
                self.stats.auto_routed_local += 1
            else:
                self.stats.auto_routed_external += 1

    @property
    def usage(self) -> dict[str, int]:
        """Compatibility: return combined usage like ProviderLLM.usage."""
        local_usage = self.local.usage if hasattr(self.local, "usage") else {}
        external_usage = (
            self.external.usage
            if self.external is not None and hasattr(self.external, "usage")
            else {}
        )
        return {
            "prompt_tokens": local_usage.get("prompt_tokens", 0)
            + external_usage.get("prompt_tokens", 0),
            "completion_tokens": local_usage.get("completion_tokens", 0)
            + external_usage.get("completion_tokens", 0),
            "total_tokens": local_usage.get("total_tokens", 0)
            + external_usage.get("total_tokens", 0),
            "call_count": local_usage.get("call_count", 0) + external_usage.get("call_count", 0),
        }

    def snapshot(self) -> dict[str, Any]:
        """Full routing statistics snapshot."""
        return {
            "stats": self.stats.to_dict(),
            "complexity_threshold": self.complexity_threshold,
            "local_provider": (
                self.local.provider.name if hasattr(self.local, "provider") else "unknown"
            ),
            "external_provider": (
                self.external.provider.name
                if self.external is not None and hasattr(self.external, "provider")
                else "none"
            ),
            "has_state_machine": self.state_machine is not None,
        }
