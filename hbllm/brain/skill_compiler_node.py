"""
Skill Compiler Node — automatic skill extraction from repeated experience patterns.

Monitors experience and reflection events to detect recurring action patterns.
When a pattern appears frequently enough with consistent success, it extracts
a reusable skill and stores it in the SkillRegistry.

Pipeline:
  experience → detect repetition → extract pattern → build skill → store

This transforms HBLLM from a system that executes tasks to one that
*learns how to execute tasks faster* over time.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class ActionPattern:
    """A detected recurring action pattern."""

    pattern_hash: str
    actions: list[str]
    tools: list[str]
    category: str
    occurrences: int = 0
    successes: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    example_queries: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.occurrences, 1)


class SkillCompilerNode(Node):
    """
    Detects repeated action patterns and compiles them into reusable skills.

    Subscribes to:
        system.reflection — deep reflection events from ExperienceNode
        system.evaluation — evaluation reports from EvaluationNode
        system.experience — raw decision outputs

    Publishes:
        skill.extracted — when a new skill is compiled
        skill.updated — when an existing skill's metrics improve
    """

    def __init__(
        self,
        node_id: str,
        skill_registry: Any = None,
        min_occurrences: int = 3,
        min_success_rate: float = 0.7,
        pattern_window: int = 200,
        ngram_size: int = 3,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["skill_compilation", "pattern_detection"],
        )
        self.skill_registry = skill_registry
        self.min_occurrences = min_occurrences
        self.min_success_rate = min_success_rate
        self.pattern_window = pattern_window
        self.ngram_size = ngram_size

        # Pattern tracking
        self._patterns: dict[str, ActionPattern] = {}
        self._action_history: list[dict[str, Any]] = []
        self._compiled_hashes: set[str] = set()

        # Stats
        self._skills_compiled = 0
        self._patterns_detected = 0

    async def on_start(self) -> None:
        logger.info(
            "Starting SkillCompilerNode (min_occurrences=%d, min_success=%.0f%%)",
            self.min_occurrences,
            self.min_success_rate * 100,
        )
        await self.bus.subscribe("system.reflection", self._handle_reflection)
        await self.bus.subscribe("system.evaluation", self._handle_evaluation)
        await self.bus.subscribe("system.experience", self._handle_experience)
        await self.bus.subscribe("skill_compiler.query", self._handle_query)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping SkillCompilerNode — patterns=%d skills_compiled=%d",
            len(self._patterns),
            self._skills_compiled,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Event Handlers ───────────────────────────────────────────────

    async def _handle_experience(self, message: Message) -> None:
        """Record raw action from decision output."""
        payload = message.payload
        action = {
            "intent": payload.get("intent", "answer"),
            "thought_type": payload.get("thought_type", ""),
            "tools": payload.get("tools_used", []),
            "content_hash": hashlib.md5(payload.get("text", "")[:200].lower().encode()).hexdigest()[
                :8
            ],
            "query": payload.get("text", "")[:200],
            "timestamp": time.time(),
            "success": True,  # assumed until evaluation says otherwise
        }

        self._action_history.append(action)
        if len(self._action_history) > self.pattern_window:
            self._action_history = self._action_history[-self.pattern_window :]

        # Detect patterns in recent actions
        await self._detect_patterns()

    async def _handle_evaluation(self, message: Message) -> None:
        """Update action success based on evaluation results."""
        payload = message.payload
        overall = payload.get("overall_score", 0.5)

        # Mark recent matching actions as success/failure
        if self._action_history:
            latest = self._action_history[-1]
            latest["success"] = overall > 0.6

    async def _handle_reflection(self, message: Message) -> None:
        """Extract skill candidates from deep reflection events."""
        payload = message.payload
        rules = payload.get("rules", [])
        category = payload.get("category", "general")

        # Rules with high confidence are skill candidates
        for rule in rules:
            confidence = rule.get("confidence", 0)
            if confidence >= 0.6:
                condition = rule.get("condition", "")
                action_str = rule.get("action", "")

                pattern_hash = hashlib.md5(f"{condition}|{action_str}".encode()).hexdigest()[:12]

                if pattern_hash not in self._patterns:
                    self._patterns[pattern_hash] = ActionPattern(
                        pattern_hash=pattern_hash,
                        actions=[action_str],
                        tools=[],
                        category=category,
                    )
                    self._patterns_detected += 1

                pattern = self._patterns[pattern_hash]
                pattern.occurrences += 1
                pattern.last_seen = time.time()

                # Check if ready to compile
                await self._maybe_compile(pattern)

    async def _handle_query(self, message: Message) -> Message | None:
        """Return skill compiler stats."""
        return message.create_response(self.stats())

    # ── Pattern Detection ────────────────────────────────────────────

    async def _detect_patterns(self) -> None:
        """Detect recurring action sequences using n-gram analysis."""
        if len(self._action_history) < self.ngram_size:
            return

        # Build n-grams from recent action types
        recent = self._action_history[-50:]
        for i in range(len(recent) - self.ngram_size + 1):
            window = recent[i : i + self.ngram_size]

            # Create pattern signature from action intents and tool usage
            signature_parts = []
            all_tools: list[str] = []
            all_actions: list[str] = []
            is_success = True

            for action in window:
                intent = action.get("intent", "unknown")
                thought = action.get("thought_type", "unknown")
                sig = f"{intent}:{thought}"
                signature_parts.append(sig)
                all_actions.append(sig)

                tools = action.get("tools", [])
                if isinstance(tools, list):
                    for t in tools:
                        tool_name = t if isinstance(t, str) else str(t.get("name", ""))
                        if tool_name:
                            all_tools.append(tool_name)

                if not action.get("success", True):
                    is_success = False

            signature = "|".join(signature_parts)
            pattern_hash = hashlib.md5(signature.encode()).hexdigest()[:12]

            if pattern_hash not in self._patterns:
                # Determine category from most common intent
                intents = [a.get("intent", "general") for a in window]
                category = max(set(intents), key=intents.count)

                self._patterns[pattern_hash] = ActionPattern(
                    pattern_hash=pattern_hash,
                    actions=all_actions,
                    tools=list(set(all_tools)),
                    category=category,
                )
                self._patterns_detected += 1

            pattern = self._patterns[pattern_hash]
            pattern.occurrences += 1
            pattern.last_seen = time.time()
            if is_success:
                pattern.successes += 1

            # Store example query
            query = window[0].get("query", "")
            if query and len(pattern.example_queries) < 5:
                pattern.example_queries.append(query[:200])

            # Check if ready to compile
            await self._maybe_compile(pattern)

    # ── Skill Compilation ────────────────────────────────────────────

    async def _maybe_compile(self, pattern: ActionPattern) -> None:
        """Compile a pattern into a reusable skill if it meets thresholds."""
        if pattern.pattern_hash in self._compiled_hashes:
            return

        if pattern.occurrences < self.min_occurrences:
            return

        if pattern.success_rate < self.min_success_rate:
            return

        # Compile the skill
        skill = None
        if self.skill_registry:
            execution_trace = [{"action": a} for a in pattern.actions]
            skill = self.skill_registry.extract_and_store(
                task_description=f"Auto-compiled from {pattern.occurrences} occurrences: {' → '.join(pattern.actions[:5])}",
                execution_trace=execution_trace,
                tools_used=pattern.tools,
                success=True,
                category=pattern.category,
            )

        self._compiled_hashes.add(pattern.pattern_hash)
        self._skills_compiled += 1

        logger.info(
            "[SkillCompiler] Compiled skill from pattern: %s "
            "(occurrences=%d success_rate=%.0f%% tools=%s)",
            pattern.actions[:3],
            pattern.occurrences,
            pattern.success_rate * 100,
            pattern.tools[:3],
        )

        # Publish extraction event
        await self.publish(
            "skill.extracted",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="skill.extracted",
                payload={
                    "pattern_hash": pattern.pattern_hash,
                    "actions": pattern.actions,
                    "tools": pattern.tools,
                    "category": pattern.category,
                    "occurrences": pattern.occurrences,
                    "success_rate": round(pattern.success_rate, 3),
                    "skill_id": skill.skill_id if skill else None,
                },
            ),
        )

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "patterns_detected": self._patterns_detected,
            "skills_compiled": self._skills_compiled,
            "active_patterns": len(self._patterns),
            "compiled_patterns": len(self._compiled_hashes),
            "action_history_size": len(self._action_history),
            "top_patterns": [
                {
                    "hash": p.pattern_hash,
                    "actions": p.actions[:3],
                    "occurrences": p.occurrences,
                    "success_rate": round(p.success_rate, 3),
                }
                for p in sorted(
                    self._patterns.values(),
                    key=lambda x: x.occurrences,
                    reverse=True,
                )[:5]
            ],
        }
