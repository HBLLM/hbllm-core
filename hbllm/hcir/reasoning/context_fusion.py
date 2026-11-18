"""
Context Fusion Engine — token-budgeted context assembly.

Merges all available context sources before LLM calls:
    1. Episodic memory (recent conversation + experiences)
    2. Semantic memory (relevant knowledge)
    3. World state (IoT sensors, weather, calendar, location)
    4. Emotional state (user's current mood, interaction style)
    5. Active goals and tasks
    6. Self-model (own capabilities and confidence)

Uses a priority-weighted budget allocator to fit the most relevant
context within the model's token limit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContextSlice:
    """A single slice of context from one source."""

    source: str  # e.g. "episodic_memory", "world_state"
    content: str
    priority: float = 0.5  # 0.0-1.0, higher = more important
    token_estimate: int = 0  # Estimated tokens (chars / 4 heuristic)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = max(1, len(self.content) // 4)


@dataclass
class FusedContext:
    """Complete fused context ready for LLM injection."""

    sections: list[ContextSlice]
    total_tokens: int
    budget_used_pct: float
    assembly_time_ms: float

    def to_system_prompt(self) -> str:
        """Format as a system prompt section."""
        if not self.sections:
            return ""

        parts = []
        for s in self.sections:
            header = _section_headers.get(s.source, s.source.replace("_", " ").title())
            parts.append(f"### {header}\n{s.content}")

        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "budget_used_pct": round(self.budget_used_pct, 1),
            "assembly_time_ms": round(self.assembly_time_ms, 1),
            "sections": [
                {"source": s.source, "priority": s.priority, "tokens": s.token_estimate}
                for s in self.sections
            ],
        }


# Human-readable section headers
_section_headers = {
    "episodic_memory": "Recent Context",
    "semantic_memory": "Relevant Knowledge",
    "world_state": "Current Environment",
    "emotion_state": "Interaction Context",
    "active_goals": "Active Goals & Tasks",
    "self_model": "System Capabilities",
    "calendar": "Schedule",
    "weather": "Weather",
    "location": "Location",
    "iot_devices": "Connected Devices",
    "user_model": "User Understanding",
    "active_project": "Active Project",
    "relationships": "Key People",
    "reality_graph": "World State",
}


class ContextFusionEngine:
    """
    Assembles and prioritizes context from multiple sources.

    Usage:
        engine = ContextFusionEngine(token_budget=4000)
        engine.register_source("world_state", world_state_provider)
        fused = await engine.fuse(query="What's the temperature?", tenant_id="user1")
    """

    def __init__(
        self,
        token_budget: int = 4000,
        priority_weights: dict[str, float] | None = None,
    ) -> None:
        self.token_budget = token_budget
        self._sources: dict[str, _ContextSource] = {}

        # Default priority weights (can be overridden)
        self._priority_weights = priority_weights or {
            "episodic_memory": 0.9,
            "user_model": 0.85,
            "active_project": 0.85,
            "semantic_memory": 0.8,
            "active_goals": 0.7,
            "world_state": 0.6,
            "reality_graph": 0.6,
            "relationships": 0.55,
            "emotion_state": 0.5,
            "self_model": 0.4,
            "calendar": 0.5,
            "iot_devices": 0.4,
        }

    def register_source(
        self,
        name: str,
        provider: Any,
        priority: float | None = None,
    ) -> None:
        """Register a context source.

        Provider must implement:
            async def get_context(query: str, tenant_id: str, budget: int) -> str
        Or be a callable: async (query, tenant_id, budget) -> str
        """
        if priority is None:
            priority = self._priority_weights.get(name, 0.5)

        self._sources[name] = _ContextSource(
            name=name,
            provider=provider,
            priority=priority,
        )

    async def fuse(
        self,
        query: str,
        tenant_id: str = "default",
        extra_context: dict[str, str] | None = None,
    ) -> FusedContext:
        """Fuse all registered context sources into a single context."""
        start = time.monotonic()
        slices: list[ContextSlice] = []

        # Collect from all sources (with per-source budget hints)
        import asyncio

        per_source_budget = max(200, self.token_budget // max(1, len(self._sources)))

        tasks = [
            self._fetch_source(source, query, tenant_id, per_source_budget)
            for source in self._sources.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, ContextSlice) and result.content.strip():
                slices.append(result)
            elif isinstance(result, Exception):
                logger.debug("Context source failed: %s", result)

        # Add any extra context passed directly
        if extra_context:
            for name, content in extra_context.items():
                if content.strip():
                    slices.append(
                        ContextSlice(
                            source=name,
                            content=content,
                            priority=self._priority_weights.get(name, 0.5),
                        )
                    )

        # Sort by priority (highest first)
        slices.sort(key=lambda s: s.priority, reverse=True)

        # Budget allocation — greedy by priority
        selected: list[ContextSlice] = []
        remaining_budget = self.token_budget

        for s in slices:
            if s.token_estimate <= remaining_budget:
                selected.append(s)
                remaining_budget -= s.token_estimate
            elif remaining_budget > 100:
                # Truncate to fit
                max_chars = remaining_budget * 4
                truncated = ContextSlice(
                    source=s.source,
                    content=s.content[:max_chars] + "…",
                    priority=s.priority,
                    token_estimate=remaining_budget,
                    metadata=s.metadata,
                )
                selected.append(truncated)
                remaining_budget = 0

        total_tokens = sum(s.token_estimate for s in selected)
        elapsed = (time.monotonic() - start) * 1000

        return FusedContext(
            sections=selected,
            total_tokens=total_tokens,
            budget_used_pct=(total_tokens / self.token_budget) * 100 if self.token_budget else 0,
            assembly_time_ms=elapsed,
        )

    async def _fetch_source(
        self,
        source: _ContextSource,
        query: str,
        tenant_id: str,
        budget: int,
    ) -> ContextSlice:
        """Fetch context from a single source."""
        try:
            provider = source.provider

            if hasattr(provider, "get_context"):
                content = await provider.get_context(query, tenant_id, budget)
            elif callable(provider):
                import inspect

                res = provider(query, tenant_id, budget)
                if inspect.isawaitable(res):
                    content = await res
                else:
                    content = res
            else:
                content = str(provider)

            return ContextSlice(
                source=source.name,
                content=str(content),
                priority=source.priority,
            )
        except Exception as e:
            logger.debug("Failed to fetch context from %s: %s", source.name, e)
            return ContextSlice(source=source.name, content="", priority=0)

    # ── Pre-built context providers ──────────────────────────────────────

    @staticmethod
    def memory_provider(memory_system: Any) -> Any:
        """Create a context provider from a MemorySystem."""

        async def _provider(query: str, tenant_id: str, budget: int) -> str:
            parts = []

            # Episodic memory (recent relevant episodes)
            if hasattr(memory_system, "episodic") and memory_system.episodic:
                try:
                    episodes = await memory_system.episodic.recall(
                        query=query, tenant_id=tenant_id, limit=5
                    )
                    if episodes:
                        formatted = "\n".join(
                            f"- {e.get('summary', e.get('content', ''))[:200]}"
                            for e in episodes[:5]
                        )
                        parts.append(f"**Recent experiences:**\n{formatted}")
                except Exception:
                    logger.debug("Episodic memory recall failed in context fusion", exc_info=True)

            # Semantic memory (relevant facts)
            if hasattr(memory_system, "semantic") and memory_system.semantic:
                try:
                    facts = await memory_system.semantic.search(
                        query=query, tenant_id=tenant_id, limit=5
                    )
                    if facts:
                        formatted = "\n".join(f"- {f.get('content', '')[:200]}" for f in facts[:5])
                        parts.append(f"**Known facts:**\n{formatted}")
                except Exception:
                    logger.debug("Semantic memory search failed in context fusion", exc_info=True)

            return "\n\n".join(parts) if parts else ""

        return _provider

    @staticmethod
    def world_state_provider(world_state: Any) -> Any:
        """Create a context provider from a WorldStateEngine."""

        async def _provider(query: str, _tenant_id: str, budget: int) -> str:
            if not hasattr(world_state, "_graph"):
                return ""

            parts = []
            # Get high-confidence entities
            entities = sorted(
                world_state._graph.values(),
                key=lambda e: e.confidence,
                reverse=True,
            )[:10]

            for entity in entities:
                if entity.confidence > 0.3:
                    props = ", ".join(f"{k}={v}" for k, v in list(entity.properties.items())[:5])
                    parts.append(
                        f"- {entity.entity_id}: {props} (confidence={entity.confidence:.0%})"
                    )

            return "\n".join(parts) if parts else ""

        return _provider

    @staticmethod
    def emotion_provider(emotion_engine: Any) -> Any:
        """Create a context provider from an EmotionEngine."""

        async def _provider(query: str, tenant_id: str, _budget: int) -> str:
            if not emotion_engine:
                return ""

            try:
                state = emotion_engine.get_state(tenant_id)
                if state:
                    mood = state.get("dominant_emotion", "neutral")
                    valence = state.get("valence", 0.0)
                    arousal = state.get("arousal", 0.5)
                    return f"User mood: {mood} (valence={valence:+.1f}, arousal={arousal:.1f})"
            except Exception:
                logger.debug("Emotion state retrieval failed in context fusion", exc_info=True)
            return ""

        return _provider

    @staticmethod
    def goals_provider(goal_manager: Any) -> Any:
        """Create a context provider from a GoalManager."""

        async def _provider(query: str, tenant_id: str, _budget: int) -> str:
            if not goal_manager:
                return ""

            try:
                goals = goal_manager.get_active_goals(tenant_id=tenant_id)
                if not goals:
                    return ""

                parts = []
                for g in goals[:5]:
                    status = g.get("status", "active")
                    parts.append(f"- [{status}] {g.get('description', '')[:100]}")

                return "**Active goals:**\n" + "\n".join(parts) if parts else ""
            except Exception:
                logger.debug("Goal retrieval failed in context fusion", exc_info=True)
            return ""

        return _provider

    @staticmethod
    def user_model_provider(user_model: Any) -> Any:
        """Create a context provider from a UserModelEngine."""

        async def _provider(query: str, tenant_id: str, budget: int) -> str:
            if not user_model:
                return ""
            try:
                return await user_model.get_context(query, tenant_id, budget)
            except Exception:
                return ""

        return _provider

    @staticmethod
    def project_provider(project_graph: Any) -> Any:
        """Create a context provider from a ProjectGraph."""

        async def _provider(query: str, tenant_id: str, budget: int) -> str:
            if not project_graph:
                return ""
            try:
                return await project_graph.get_context(query, tenant_id, budget)
            except Exception:
                return ""

        return _provider

    @staticmethod
    def relationship_provider(relationship_memory: Any) -> Any:
        """Create a context provider from a RelationshipMemory."""

        async def _provider(query: str, tenant_id: str, budget: int) -> str:
            if not relationship_memory:
                return ""
            try:
                return await relationship_memory.get_context(query, tenant_id, budget)
            except Exception:
                return ""

        return _provider

    @staticmethod
    def reality_graph_provider(reality_graph: Any) -> Any:
        """Create a context provider from a RealityGraph."""

        async def _provider(query: str, tenant_id: str, budget: int) -> str:
            if not reality_graph:
                return ""
            try:
                return await reality_graph.get_context(query, tenant_id, budget)
            except Exception:
                return ""

        return _provider


@dataclass
class _ContextSource:
    """Internal representation of a registered context source."""

    name: str
    provider: Any
    priority: float = 0.5
