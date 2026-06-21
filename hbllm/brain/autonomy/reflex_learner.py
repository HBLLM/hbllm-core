"""Reflex Learner — learns new reflexes from user corrections.

When a user says "Next time the temperature drops below 18°C, turn on
the heater automatically", the reflex learner:
    1. Extracts the trigger condition
    2. Extracts the action to take
    3. Creates a new reflex rule
    4. Stores it in the reflex store
    5. Registers it with AutonomyCore

Architecture:
    - LLM-powered extraction for condition → action parsing
    - Heuristic fallback for simple patterns
    - Confidence scoring (new reflexes start at low confidence)
    - User approval flow for learned reflexes
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LearnedReflex:
    """A reflex rule learned from user behavior or instruction."""

    reflex_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    tenant_id: str = "default"
    name: str = ""
    description: str = ""
    trigger_condition: str = ""  # Human-readable condition
    trigger_sensor: str = ""  # Sensor path (e.g., "iot.temperature")
    trigger_threshold: str = ""  # Comparison (e.g., "< 18")
    action: str = ""  # Action to take (e.g., "iot.heater.on")
    action_params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5  # Starts low, increases with positive feedback
    enabled: bool = True
    approved: bool = False  # Requires user approval before activation
    created_at: float = field(default_factory=time.time)
    last_triggered: float | None = None
    trigger_count: int = 0
    source: str = "user_instruction"  # "user_instruction", "behavior_pattern", "suggestion"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflex_id": self.reflex_id,
            "name": self.name,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "action": self.action,
            "confidence": round(self.confidence, 2),
            "enabled": self.enabled,
            "approved": self.approved,
            "trigger_count": self.trigger_count,
            "source": self.source,
        }


class ReflexStore:
    """Persistent storage for learned reflexes."""

    def __init__(self, db_path: str | Path = "data/learned_reflexes.db") -> None:
        self.db_path = Path(db_path)

    async def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_reflexes (
                    reflex_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    trigger_condition TEXT,
                    trigger_sensor TEXT,
                    trigger_threshold TEXT,
                    action TEXT NOT NULL,
                    action_params TEXT,
                    confidence REAL DEFAULT 0.5,
                    enabled INTEGER DEFAULT 1,
                    approved INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_triggered REAL,
                    trigger_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'user_instruction'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reflexes_tenant
                ON learned_reflexes(tenant_id, enabled)
            """)
            conn.commit()
        finally:
            conn.close()

    def save(self, reflex: LearnedReflex) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO learned_reflexes "
                "(reflex_id, tenant_id, name, description, trigger_condition, "
                "trigger_sensor, trigger_threshold, action, action_params, "
                "confidence, enabled, approved, created_at, last_triggered, "
                "trigger_count, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    reflex.reflex_id,
                    reflex.tenant_id,
                    reflex.name,
                    reflex.description,
                    reflex.trigger_condition,
                    reflex.trigger_sensor,
                    reflex.trigger_threshold,
                    reflex.action,
                    json.dumps(reflex.action_params),
                    reflex.confidence,
                    int(reflex.enabled),
                    int(reflex.approved),
                    reflex.created_at,
                    reflex.last_triggered,
                    reflex.trigger_count,
                    reflex.source,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_active(self, tenant_id: str) -> list[LearnedReflex]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT * FROM learned_reflexes "
                "WHERE tenant_id = ? AND enabled = 1 AND approved = 1 "
                "ORDER BY confidence DESC",
                (tenant_id,),
            )
            return [self._row_to_reflex(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_pending_approval(self, tenant_id: str) -> list[LearnedReflex]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT * FROM learned_reflexes "
                "WHERE tenant_id = ? AND approved = 0 "
                "ORDER BY created_at DESC",
                (tenant_id,),
            )
            return [self._row_to_reflex(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def approve(self, reflex_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE learned_reflexes SET approved = 1 WHERE reflex_id = ?",
                (reflex_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def record_trigger(self, reflex_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE learned_reflexes SET "
                "trigger_count = trigger_count + 1, "
                "last_triggered = ?, "
                "confidence = MIN(1.0, confidence + 0.05) "
                "WHERE reflex_id = ?",
                (time.time(), reflex_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _row_to_reflex(self, row: tuple[Any, ...]) -> LearnedReflex:
        return LearnedReflex(
            reflex_id=row[0],
            tenant_id=row[1],
            name=row[2],
            description=row[3] or "",
            trigger_condition=row[4] or "",
            trigger_sensor=row[5] or "",
            trigger_threshold=row[6] or "",
            action=row[7],
            action_params=json.loads(row[8]) if row[8] else {},
            confidence=row[9] or 0.5,
            enabled=bool(row[10]),
            approved=bool(row[11]),
            created_at=row[12],
            last_triggered=row[13],
            trigger_count=row[14] or 0,
            source=row[15] or "user_instruction",
        )


class ReflexLearner:
    """Learns new reflexes from user instructions and behavior.

    Usage::

        learner = ReflexLearner(store=reflex_store)

        # From explicit instruction
        reflex = await learner.learn_from_instruction(
            tenant_id="user1",
            instruction="When the humidity goes above 70%, turn on the dehumidifier",
        )

        # From observed behavior pattern
        reflex = learner.learn_from_pattern(
            tenant_id="user1",
            pattern={"trigger": "bedtime", "action": "lights.off"},
        )
    """

    EXTRACTION_PROMPT = """Extract a reflex rule from this user instruction.

Instruction: {instruction}

Respond in JSON:
{{
    "name": "Short descriptive name",
    "trigger_sensor": "sensor path (e.g., iot.humidity)",
    "trigger_threshold": "comparison (e.g., > 70)",
    "trigger_condition": "Human-readable condition",
    "action": "action path (e.g., iot.dehumidifier.on)",
    "action_params": {{}},
    "description": "What this reflex does"
}}"""

    def __init__(
        self,
        store: ReflexStore,
        provider: Any | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self._total_learned = 0

    async def learn_from_instruction(
        self,
        tenant_id: str,
        instruction: str,
    ) -> LearnedReflex:
        """Extract and create a reflex from a user instruction."""
        if self.provider:
            reflex = await self._extract_with_llm(tenant_id, instruction)
        else:
            reflex = self._extract_heuristic(tenant_id, instruction)

        reflex.source = "user_instruction"
        reflex.approved = False  # Requires approval

        self.store.save(reflex)
        self._total_learned += 1

        logger.info(
            "Learned reflex '%s': %s → %s (pending approval)",
            reflex.name,
            reflex.trigger_condition,
            reflex.action,
        )

        return reflex

    def learn_from_pattern(
        self,
        tenant_id: str,
        pattern: dict[str, Any],
    ) -> LearnedReflex:
        """Create a reflex from an observed behavior pattern."""
        reflex = LearnedReflex(
            tenant_id=tenant_id,
            name=f"Pattern: {pattern.get('trigger', 'unknown')}",
            trigger_condition=str(pattern.get("trigger", "")),
            trigger_sensor=pattern.get("sensor", ""),
            action=str(pattern.get("action", "")),
            action_params=pattern.get("params", {}),
            confidence=0.3,  # Low confidence for pattern-learned reflexes
            source="behavior_pattern",
            approved=False,
        )

        self.store.save(reflex)
        self._total_learned += 1
        return reflex

    async def _extract_with_llm(
        self,
        tenant_id: str,
        instruction: str,
    ) -> LearnedReflex:
        """Use LLM to extract reflex components from instruction."""
        prompt = self.EXTRACTION_PROMPT.format(instruction=instruction)

        try:
            response = await self.provider.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )

            content = response.get("content", "") if isinstance(response, dict) else str(response)

            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(content[json_start:json_end])
            else:
                return self._extract_heuristic(tenant_id, instruction)

            return LearnedReflex(
                tenant_id=tenant_id,
                name=data.get("name", instruction[:50]),
                description=data.get("description", ""),
                trigger_condition=data.get("trigger_condition", ""),
                trigger_sensor=data.get("trigger_sensor", ""),
                trigger_threshold=data.get("trigger_threshold", ""),
                action=data.get("action", ""),
                action_params=data.get("action_params", {}),
                confidence=0.6,
            )

        except Exception as e:
            logger.warning("LLM extraction failed: %s, using heuristic", e)
            return self._extract_heuristic(tenant_id, instruction)

    def _extract_heuristic(
        self,
        tenant_id: str,
        instruction: str,
    ) -> LearnedReflex:
        """Simple keyword-based extraction as LLM fallback."""
        lower = instruction.lower()

        # Try to find "when X, do Y" patterns
        trigger = instruction
        action = ""

        for separator in ["then ", "do ", "turn on ", "turn off ", "set ", "activate "]:
            if separator in lower:
                parts = lower.split(separator, 1)
                trigger = parts[0].strip()
                action = separator.strip() + " " + parts[1].strip()
                break

        return LearnedReflex(
            tenant_id=tenant_id,
            name=instruction[:50],
            description=instruction,
            trigger_condition=trigger,
            action=action or instruction,
            confidence=0.4,
        )

    def stats(self) -> dict[str, Any]:
        return {"total_learned": self._total_learned}
