"""
Owner Rules — natural language → structured policy conversion.

Allows robot/system owners to set long-term behavioral rules in plain
English. Rules are parsed into Policy objects with conditions, persisted
in SQLite, and enforced via the PolicyEngine.

Example rules:
  "Never open the door after 9pm for strangers"
  "Always speak softly when the baby is sleeping"
  "Don't discuss finances with guests"
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hbllm.brain.policy_engine import (
    Policy,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyType,
)

logger = logging.getLogger(__name__)


# ── NL Rule Parser ──────────────────────────────────────────────────────────

# Time patterns: "after 9pm", "before 6am", "between 10pm and 6am"
_TIME_RE = re.compile(
    r"(?:after|past)\s+(\d{1,2})\s*([ap]m)?",
    re.IGNORECASE,
)
_TIME_BEFORE_RE = re.compile(
    r"before\s+(\d{1,2})\s*([ap]m)?",
    re.IGNORECASE,
)

# Person patterns: "for strangers", "with guests", "to visitors"
_PERSON_RE = re.compile(
    r"(?:for|with|to)\s+(strangers?|guests?|visitors?|family|children|kids|unknown\s+(?:people|persons?))",
    re.IGNORECASE,
)

# Sensor/state patterns: "when the baby is sleeping", "when lights are off"
_STATE_RE = re.compile(
    r"when\s+(?:the\s+)?(\w+)\s+(?:is|are)\s+(sleeping|awake|off|on|open|closed|locked|unlocked|active|inactive|home|away)",
    re.IGNORECASE,
)

# Action keywords for deny
_DENY_KEYWORDS = {
    "never", "don't", "do not", "dont", "must not", "mustn't",
    "should not", "shouldn't", "cannot", "can't", "forbid",
    "prohibit", "block", "prevent", "disallow", "refuse",
}

# Action keywords for require/transform
_REQUIRE_KEYWORDS = {
    "always", "must", "should", "ensure", "make sure", "require",
}


@dataclass
class ParsedRule:
    """Result of parsing a natural language owner rule."""
    original_text: str
    policy_type: PolicyType
    action: PolicyAction
    description: str
    pattern: str = ""
    content: str = ""
    conditions: list[PolicyCondition] = field(default_factory=list)
    severity: str = "medium"
    confidence: float = 0.0  # How confident the parser is


def _to_24h(hour: int, ampm: str | None) -> int:
    """Convert 12h time to 24h."""
    if not ampm:
        return hour
    ampm = ampm.lower()
    if ampm == "pm" and hour != 12:
        return hour + 12
    if ampm == "am" and hour == 12:
        return 0
    return hour


def _extract_action_subject(text: str) -> str:
    """Extract the main action/subject from the rule for regex pattern generation."""
    # Remove time/person/state clauses to get the core action
    cleaned = text
    for pattern in [_TIME_RE, _TIME_BEFORE_RE, _PERSON_RE, _STATE_RE]:
        cleaned = pattern.sub("", cleaned)

    # Remove deny/require keywords
    for kw in sorted(_DENY_KEYWORDS | _REQUIRE_KEYWORDS, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(kw)}\b", "", cleaned, flags=re.IGNORECASE)

    # Clean up
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".,!;:")

    if not cleaned or len(cleaned) < 3:
        return ""

    # Build a regex: split into significant words and join with .*
    words = [w for w in cleaned.split() if len(w) > 2 and w.lower() not in {"the", "and", "for", "with"}]
    if not words:
        return ""

    return ".*".join(re.escape(w) for w in words[:4])


def parse_owner_rule(text: str) -> ParsedRule:
    """
    Parse a natural language owner rule into a structured ParsedRule.

    Handles patterns like:
      - "Never open the door after 9pm for strangers"
      - "Always speak softly when the baby is sleeping"
      - "Don't discuss finances with guests"

    Returns:
        ParsedRule with extracted type, conditions, pattern, and severity.
    """
    text_lower = text.lower().strip()
    conditions: list[PolicyCondition] = []
    confidence = 0.5

    # ── Determine policy type ──
    is_deny = any(kw in text_lower for kw in _DENY_KEYWORDS)
    is_require = any(kw in text_lower for kw in _REQUIRE_KEYWORDS)

    if is_deny:
        policy_type = PolicyType.DENY
        action = PolicyAction.BLOCK
        confidence += 0.2
    elif is_require:
        policy_type = PolicyType.TRANSFORM
        action = PolicyAction.PREPEND
        confidence += 0.15
    else:
        # Default: treat as a deny rule with warning
        policy_type = PolicyType.DENY
        action = PolicyAction.WARN
        confidence += 0.05

    # ── Extract time conditions ──
    time_match = _TIME_RE.search(text)
    if time_match:
        hour = _to_24h(int(time_match.group(1)), time_match.group(2))
        conditions.append(PolicyCondition(key="time_hour", operator="gte", value=hour))
        confidence += 0.1

    time_before_match = _TIME_BEFORE_RE.search(text)
    if time_before_match:
        hour = _to_24h(int(time_before_match.group(1)), time_before_match.group(2))
        conditions.append(PolicyCondition(key="time_hour", operator="lt", value=hour))
        confidence += 0.1

    # ── Extract person conditions ──
    person_match = _PERSON_RE.search(text)
    if person_match:
        person = person_match.group(1).lower().rstrip("s")  # Normalize plural
        if person in ("stranger", "guest", "visitor", "unknown person", "unknown people"):
            conditions.append(PolicyCondition(key="person_type", operator="neq", value="family"))
        elif person in ("family", "children", "kid"):
            conditions.append(PolicyCondition(key="person_type", operator="eq", value=person))
        confidence += 0.1

    # ── Extract state conditions ──
    state_match = _STATE_RE.search(text)
    if state_match:
        subject = state_match.group(1).lower()
        state = state_match.group(2).lower()
        state_key = f"{subject}_state"

        if state in ("sleeping", "on", "open", "locked", "active", "home"):
            conditions.append(PolicyCondition(key=state_key, operator="eq", value=state))
        elif state in ("awake", "off", "closed", "unlocked", "inactive", "away"):
            conditions.append(PolicyCondition(key=state_key, operator="eq", value=state))
        confidence += 0.1

    # ── Extract action pattern for regex matching ──
    pattern = _extract_action_subject(text)

    # ── Determine severity ──
    severity = "medium"
    if any(w in text_lower for w in ("never", "critical", "dangerous", "safety", "emergency")):
        severity = "critical"
    elif any(w in text_lower for w in ("important", "must", "always")):
        severity = "high"
    elif any(w in text_lower for w in ("prefer", "try to", "ideally")):
        severity = "low"

    # ── Build content for TRANSFORM rules ──
    content = ""
    if policy_type == PolicyType.TRANSFORM:
        # Extract what they want as the transformation
        content = f"[RULE: {text.strip()}]"

    return ParsedRule(
        original_text=text.strip(),
        policy_type=policy_type,
        action=action,
        description=text.strip(),
        pattern=pattern,
        content=content,
        conditions=conditions,
        severity=severity,
        confidence=min(confidence, 0.95),
    )


# ── Owner Rule Store ────────────────────────────────────────────────────────

class OwnerRuleStore:
    """
    SQLite-backed persistence for owner-defined rules.

    Each rule is stored with its parsed policy structure and can be
    loaded into a PolicyEngine at startup.
    """

    def __init__(self, db_path: str | Path = "owner_rules.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS owner_rules (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    policy_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    pattern TEXT DEFAULT '',
                    content TEXT DEFAULT '',
                    conditions_json TEXT DEFAULT '[]',
                    severity TEXT DEFAULT 'medium',
                    confidence REAL DEFAULT 0.0,
                    source TEXT DEFAULT 'owner',
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_owner_rules_tenant
                ON owner_rules(tenant_id)
            """)

    def add_rule(
        self,
        tenant_id: str,
        text: str,
        source: str = "owner",
    ) -> tuple[str, ParsedRule]:
        """
        Parse and store a natural language rule.

        Returns:
            (rule_id, parsed_rule) tuple.
        """
        parsed = parse_owner_rule(text)
        rule_id = uuid.uuid4().hex[:12]
        policy_name = f"owner_{rule_id}"
        now = datetime.now(timezone.utc).isoformat()

        conditions_json = json.dumps([c.to_dict() for c in parsed.conditions])

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO owner_rules
                    (id, tenant_id, original_text, policy_name, policy_type,
                     action, description, pattern, content, conditions_json,
                     severity, confidence, source, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                rule_id, tenant_id, parsed.original_text, policy_name,
                parsed.policy_type.value, parsed.action.value,
                parsed.description, parsed.pattern, parsed.content,
                conditions_json, parsed.severity, parsed.confidence,
                source, now, now,
            ))

        logger.info(
            "Added owner rule '%s' for tenant '%s' (type=%s, severity=%s, confidence=%.2f)",
            rule_id, tenant_id, parsed.policy_type.value, parsed.severity, parsed.confidence,
        )
        return rule_id, parsed

    def list_rules(self, tenant_id: str) -> list[dict[str, Any]]:
        """List all rules for a tenant."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM owner_rules WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "original_text": row["original_text"],
                "policy_type": row["policy_type"],
                "action": row["action"],
                "pattern": row["pattern"],
                "severity": row["severity"],
                "confidence": row["confidence"],
                "source": row["source"],
                "enabled": bool(row["enabled"]),
                "conditions": json.loads(row["conditions_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM owner_rules WHERE id = ?", (rule_id,),
            )
            return cursor.rowcount > 0

    def toggle_rule(self, rule_id: str, enabled: bool) -> bool:
        """Enable or disable a rule."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "UPDATE owner_rules SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), datetime.now(timezone.utc).isoformat(), rule_id),
            )
            return cursor.rowcount > 0

    def load_into_engine(self, tenant_id: str, engine: PolicyEngine) -> int:
        """
        Load all enabled rules for a tenant into a PolicyEngine.

        Returns the number of policies loaded.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM owner_rules WHERE tenant_id = ? AND enabled = 1",
                (tenant_id,),
            ).fetchall()

        loaded = 0
        for row in rows:
            conditions = [
                PolicyCondition.from_dict(c)
                for c in json.loads(row["conditions_json"])
            ]
            policy = Policy(
                name=row["policy_name"],
                type=PolicyType(row["policy_type"]),
                action=PolicyAction(row["action"]),
                description=row["description"],
                pattern=row["pattern"],
                content=row["content"],
                severity=row["severity"],
                conditions=conditions,
                source=row["source"],
                priority=100 if row["severity"] == "critical" else 50,
            )
            engine.add_policy(policy)
            loaded += 1

        logger.info("Loaded %d owner rules for tenant '%s'", loaded, tenant_id)
        return loaded

    def clear_tenant(self, tenant_id: str) -> int:
        """Remove all rules for a tenant."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM owner_rules WHERE tenant_id = ?", (tenant_id,),
            )
            return cursor.rowcount
