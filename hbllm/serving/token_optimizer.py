"""
Token Optimizer — reduces inference cost through context management.

Strategies:
1. Context summarization: compress long conversation histories
2. Prompt pruning: remove redundant or low-value tokens
3. Model routing: select cheapest model that can handle the task
4. Caching: reuse responses for repeated/similar queries
5. Budget enforcement: hard limits on token usage per request
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of token optimization."""
    original_tokens: int
    optimized_tokens: int
    savings_pct: float
    strategies_applied: list[str]
    recommended_model: str


class TokenOptimizer:
    """
    Reduces inference cost through intelligent token management.

    Cost reduction formula:
    cost = tokens × model_price_per_token

    Optimization levers:
    1. Reduce tokens (summarize, prune)
    2. Route to cheaper model (when possible)
    3. Cache responses (avoid redundant inference)
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        summary_ratio: float = 0.3,
        cache_size: int = 1000,
    ):
        self.max_context_tokens = max_context_tokens
        self.summary_ratio = summary_ratio
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size
        self._stats = {
            "total_optimized": 0,
            "tokens_saved": 0,
            "cache_hits": 0,
        }

        # Model cost tiers (relative cost per 1K tokens)
        self._model_costs = {
            "small": 0.01,    # e.g., 7B model
            "medium": 0.05,   # e.g., 30B model
            "large": 0.10,    # e.g., 70B model
            "specialist": 0.15,  # domain-specific fine-tuned
        }

    # ─── Full Optimization Pipeline ──────────────────────────────────

    def optimize(
        self,
        query: str,
        context: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> OptimizationResult:
        """
        Run the full optimization pipeline on a request.

        Args:
            query: User query
            context: Conversation history [{"role": "...", "content": "..."}]
            max_tokens: Optional token budget override
        """
        self._stats["total_optimized"] += 1
        budget = max_tokens or self.max_context_tokens
        strategies = []

        # Estimate original token count
        original_tokens = self._estimate_tokens(query)
        if context:
            original_tokens += sum(self._estimate_tokens(m.get("content", "")) for m in context)

        optimized_tokens = original_tokens

        # Strategy 1: Summarize long context
        if context and optimized_tokens > budget:
            context = self.summarize_context(context, budget)
            optimized_tokens = self._estimate_tokens(query) + sum(
                self._estimate_tokens(m.get("content", "")) for m in context
            )
            strategies.append("context_summarization")

        # Strategy 2: Prune query
        pruned_query = self.prune_prompt(query)
        if len(pruned_query) < len(query):
            savings = self._estimate_tokens(query) - self._estimate_tokens(pruned_query)
            optimized_tokens -= savings
            strategies.append("prompt_pruning")

        # Strategy 3: Select cheapest suitable model
        recommended = self.select_model(query, optimized_tokens)
        strategies.append(f"model_routing:{recommended}")

        savings_pct = (
            (original_tokens - optimized_tokens) / max(original_tokens, 1) * 100
        )
        self._stats["tokens_saved"] += max(0, original_tokens - optimized_tokens)

        return OptimizationResult(
            original_tokens=original_tokens,
            optimized_tokens=max(0, optimized_tokens),
            savings_pct=round(savings_pct, 1),
            strategies_applied=strategies,
            recommended_model=recommended,
        )

    # ─── Context Summarization ───────────────────────────────────────

    def summarize_context(
        self,
        messages: list[dict[str, str]],
        token_budget: int,
    ) -> list[dict[str, str]]:
        """
        Summarize conversation history to fit within token budget.

        Strategy:
        - Keep the most recent messages intact
        - Summarize older messages into a condensed form
        - Always keep system messages
        """
        if not messages:
            return []

        # Separate system messages (always keep)
        system = [m for m in messages if m.get("role") == "system"]
        conversation = [m for m in messages if m.get("role") != "system"]

        if not conversation:
            return system

        # Keep last N messages intact
        keep_recent = min(4, len(conversation))
        recent = conversation[-keep_recent:]
        older = conversation[:-keep_recent]

        if not older:
            return system + recent

        # Summarize older messages
        older_text = " ".join(m.get("content", "")[:200] for m in older)
        words = older_text.split()
        target_words = int(len(words) * self.summary_ratio)
        summary_text = " ".join(words[:target_words]) + "..."

        summary_msg = {
            "role": "system",
            "content": f"[Earlier conversation summary: {summary_text}]",
        }

        return system + [summary_msg] + recent

    # ─── Prompt Pruning ──────────────────────────────────────────────

    def prune_prompt(self, text: str) -> str:
        """Remove redundant tokens from a prompt."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove filler words that don't affect meaning
        fillers = [
            r'\bbasically\b', r'\bactually\b', r'\bjust\b',
            r'\breally\b', r'\bvery\b', r'\bquite\b',
            r'\bsimply\b', r'\bliterally\b',
        ]
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        # Clean up resulting double spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ─── Model Selection ─────────────────────────────────────────────

    def select_model(self, query: str, token_count: int = 0) -> str:
        """
        Select the cheapest model that can handle the query.

        Routing logic:
        - Simple queries (greetings, lookups) → small
        - Standard Q&A, summarization → medium
        - Complex reasoning, math, coding → large
        - Domain-specific (medical, legal) → specialist
        """
        query_lower = query.lower()

        # Simple greetings/short queries
        if len(query.split()) < 5 or any(
            w in query_lower for w in ["hello", "hi", "thanks", "bye", "ok"]
        ):
            return "small"

        # Complex reasoning indicators
        reasoning_signals = [
            "explain", "analyze", "compare", "prove", "derive",
            "calculate", "debug", "implement", "design", "architect",
        ]
        if any(s in query_lower for s in reasoning_signals):
            return "large"

        # Domain-specific
        domain_signals = [
            "medical", "legal", "financial", "clinical",
            "diagnosis", "contract", "compliance", "hipaa",
        ]
        if any(s in query_lower for s in domain_signals):
            return "specialist"

        # Default to medium
        return "medium"

    # ─── Response Caching ────────────────────────────────────────────

    def cache_check(self, query: str) -> str | None:
        """Check cache for a similar query response."""
        key = self._cache_key(query)
        if key in self._cache:
            self._stats["cache_hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def cache_store(self, query: str, response: str) -> None:
        """Store a response in cache."""
        key = self._cache_key(query)
        self._cache[key] = response
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _cache_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    # ─── Helpers ─────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)."""
        return max(1, len(text) // 4)

    def estimate_cost(self, model: str, tokens: int) -> float:
        """Estimate inference cost for a model and token count."""
        cost_per_1k = self._model_costs.get(model, 0.05)
        return round(cost_per_1k * tokens / 1000, 6)

    def stats(self) -> dict[str, Any]:
        return {
            "total_optimized": self._stats["total_optimized"],
            "tokens_saved": self._stats["tokens_saved"],
            "cache_hits": self._stats["cache_hits"],
            "cache_size": len(self._cache),
        }
