"""
Rule Extractor Node — mines if→then rules from high-salience events.

Subscribes to system.salience for priority events, accumulates them,
and periodically extracts behavioral rules/patterns. Publishes
discovered rules to system.rules.new and stores them in memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Rule data structure ──────────────────────────────────────────────────────


@dataclass
class ExtractedRule:
    """An if→then rule mined from experience data."""

    rule_id: str
    condition: str
    action: str
    confidence: float
    source_events: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    occurrences: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "condition": self.condition,
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "occurrences": self.occurrences,
            "source_count": len(self.source_events),
        }


# ── Pattern templates for rule extraction ────────────────────────────────────

_RULE_PATTERNS: list[tuple[str, str, str]] = [
    # (regex_pattern, condition_group, action_group)
    (r"(?:when|whenever|if)\s+(.+?),?\s+(?:then|should|must|always)\s+(.+?)(?:\.|$)", "1", "2"),
    (
        r"(.+?)\s+(?:always|usually|typically)\s+(?:leads to|results in|causes)\s+(.+?)(?:\.|$)",
        "1",
        "2",
    ),
    (
        r"(?:user|they|he|she)\s+(?:prefers?|likes?|wants?)\s+(.+?)\s+(?:when|over|instead of)\s+(.+?)(?:\.|$)",
        "1",
        "2",
    ),
    (r"(?:every time|each time)\s+(.+?),?\s+(.+?)(?:\.|$)", "1", "2"),
    (r"(.+?)\s+(?:requires|needs|demands)\s+(.+?)(?:\.|$)", "1", "2"),
]


def _rule_id(condition: str, action: str) -> str:
    """Deterministic rule ID."""
    key = f"{condition.lower().strip()}|{action.lower().strip()}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def extract_rules_from_text(text: str) -> list[tuple[str, str, float]]:
    """
    Extract (condition, action, confidence) triples from text.

    Returns:
        List of (condition, action, confidence) tuples.
    """
    rules: list[tuple[str, str, float]] = []

    for pattern, _, _ in _RULE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            condition = match.group(1).strip()
            action = match.group(2).strip()

            # Basic quality filters
            if len(condition) < 5 or len(action) < 5:
                continue
            if len(condition) > 200 or len(action) > 200:
                continue

            # Confidence based on specificity
            confidence = 0.5
            if any(w in condition.lower() for w in ["user", "always", "every"]):
                confidence += 0.15
            if any(w in action.lower() for w in ["should", "must", "needs"]):
                confidence += 0.1

            rules.append((condition, action, min(confidence, 0.95)))

    return rules


# ── Rule Extractor Node ──────────────────────────────────────────────────────


class RuleExtractorNode(Node):
    """
    Mines if→then rules from accumulated high-salience events.

    Listens to system.salience for priority events, accumulates content,
    and periodically extracts rules. Discovered rules are published to
    system.rules.new and optionally stored in the knowledge graph.

    Supports:
    - Regex-based rule extraction from text
    - Rule deduplication and occurrence counting
    - Batch extraction with configurable interval
    - Publishing new rules to the bus
    """

    def __init__(
        self,
        node_id: str,
        extraction_interval: float = 30.0,
        min_events_for_extraction: int = 3,
        promotion_confidence: float = 0.85,
        promotion_occurrences: int = 5,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["rule_extraction", "pattern_mining"],
        )
        self.extraction_interval = extraction_interval
        self.min_events_for_extraction = min_events_for_extraction
        self.promotion_confidence = promotion_confidence
        self.promotion_occurrences = promotion_occurrences

        self._priority_buffer: list[dict[str, Any]] = []
        self._rules: dict[str, ExtractedRule] = {}
        self._promoted: set[str] = set()  # Rule IDs already promoted
        self._extraction_task: asyncio.Task[Any] | None = None

    @property
    def rules(self) -> list[ExtractedRule]:
        """All extracted rules, sorted by confidence."""
        return sorted(self._rules.values(), key=lambda r: r.confidence, reverse=True)

    async def on_start(self) -> None:
        logger.info("Starting RuleExtractorNode (interval=%.1fs)", self.extraction_interval)
        await self.bus.subscribe("system.salience", self._on_salience)
        self._extraction_task = asyncio.create_task(self._extraction_loop())

    async def on_stop(self) -> None:
        logger.info("Stopping RuleExtractorNode")
        if self._extraction_task:
            self._extraction_task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        """Handle direct rule extraction requests."""
        if message.type != MessageType.QUERY:
            return None

        action = message.payload.get("action", "list")
        if action == "list":
            return Message(
                type=MessageType.RESPONSE,
                source_node_id=self.node_id,
                topic="rules.response",
                payload={"rules": [r.to_dict() for r in self.rules]},
                correlation_id=message.id,
            )
        elif action == "extract_now":
            count = await self._run_extraction()
            return Message(
                type=MessageType.RESPONSE,
                source_node_id=self.node_id,
                topic="rules.response",
                payload={"extracted": count, "total_rules": len(self._rules)},
                correlation_id=message.id,
            )
        return None

    async def _on_salience(self, message: Message) -> Message | None:
        """Accumulate priority events for rule extraction."""
        payload = message.payload
        if not payload.get("is_priority", False):
            return None

        self._priority_buffer.append(
            {
                "content": payload.get("content", ""),
                "score": payload.get("score", 0),
                "message_id": payload.get("message_id", ""),
                "timestamp": time.time(),
            }
        )
        logger.debug(
            "[RuleExtractor] Buffered priority event (total=%d)",
            len(self._priority_buffer),
        )
        return None

    async def _extraction_loop(self) -> None:
        """Periodically extract rules from accumulated events."""
        while self._running:
            await asyncio.sleep(self.extraction_interval)
            if len(self._priority_buffer) >= self.min_events_for_extraction:
                await self._run_extraction()

    async def _run_extraction(self) -> int:
        """Extract rules from the buffer and publish new ones."""
        if not self._priority_buffer:
            return 0

        events = self._priority_buffer.copy()
        self._priority_buffer.clear()

        new_rules = 0

        for event in events:
            content = event.get("content", "")
            if not content:
                continue

            extracted = extract_rules_from_text(content)
            for condition, action, confidence in extracted:
                rid = _rule_id(condition, action)

                if rid in self._rules:
                    # Existing rule — boost confidence
                    existing = self._rules[rid]
                    existing.occurrences += 1
                    existing.confidence = min(existing.confidence + 0.05, 0.99)
                    existing.source_events.append(event.get("message_id", ""))

                    # Auto-promote if threshold reached
                    if (
                        existing.confidence >= self.promotion_confidence
                        and existing.occurrences >= self.promotion_occurrences
                        and rid not in self._promoted
                    ):
                        self._promoted.add(rid)
                        await self.bus.publish(
                            "owner.rules.suggest",
                            Message(
                                type=MessageType.EVENT,
                                source_node_id=self.node_id,
                                topic="owner.rules.suggest",
                                payload={
                                    "rule": existing.to_dict(),
                                    "suggested_text": f"When {existing.condition}, then {existing.action}",
                                    "source": "auto",
                                },
                            ),
                        )
                        logger.info(
                            "[RuleExtractor] Promoting rule %s (confidence=%.2f, occurrences=%d)",
                            rid,
                            existing.confidence,
                            existing.occurrences,
                        )
                else:
                    # New rule
                    rule = ExtractedRule(
                        rule_id=rid,
                        condition=condition,
                        action=action,
                        confidence=confidence,
                        source_events=[event.get("message_id", "")],
                    )
                    self._rules[rid] = rule
                    new_rules += 1

                    # Publish new rule discovery
                    await self.bus.publish(
                        "system.rules.new",
                        Message(
                            type=MessageType.EVENT,
                            source_node_id=self.node_id,
                            topic="system.rules.new",
                            payload=rule.to_dict(),
                        ),
                    )

        if new_rules:
            logger.info(
                "[RuleExtractor] Extracted %d new rules (total=%d)",
                new_rules,
                len(self._rules),
            )

        return new_rules
