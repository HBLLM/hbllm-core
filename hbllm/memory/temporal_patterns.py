"""Temporal Pattern Detector — time-series analysis over episodic memory.

Detects recurring temporal patterns in user interactions:
    - Day-of-week patterns ("You ask about X on Monday mornings")
    - Time-of-day patterns ("You code between 10pm-2am")
    - Frequency patterns ("You check stocks every 30 minutes during market hours")
    - Periodicity via FFT analysis

Uses sliding windows and frequency domain analysis to find periodicity
without requiring ML models.

Architecture:
    1. Accumulates interaction timestamps per domain/topic
    2. Periodically runs pattern analysis (triggered by SleepNode)
    3. Stores detected patterns in SQLite
    4. Emits `memory.pattern.detected` events for autonomy

Usage::

    detector = TemporalPatternDetector(db_path="data/temporal_patterns.db")
    await detector.init_db()
    detector.record_interaction(tenant_id, domain="coding", timestamp=now)
    patterns = detector.detect_patterns(tenant_id)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TemporalPattern:
    """A detected temporal pattern in user behavior."""

    pattern_id: str = ""
    tenant_id: str = "default"
    domain: str = ""  # e.g., "coding", "cooking", "fitness"
    pattern_type: str = ""  # "day_of_week", "time_of_day", "frequency", "periodic"
    description: str = ""
    confidence: float = 0.0  # 0.0 - 1.0
    parameters: dict[str, Any] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "tenant_id": self.tenant_id,
            "domain": self.domain,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "parameters": self.parameters,
            "detected_at": self.detected_at,
            "sample_count": self.sample_count,
        }


class TemporalPatternDetector:
    """Detects temporal patterns from interaction history.

    Requires at least 10 data points per domain before attempting
    pattern detection to avoid false positives.
    """

    MIN_SAMPLES = 10  # Minimum interactions before detecting patterns
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to report a pattern

    def __init__(
        self,
        db_path: str | Path = "data/temporal_patterns.db",
    ) -> None:
        self.db_path = Path(db_path)
        self._pattern_cache: dict[str, list[TemporalPattern]] = {}

    async def init_db(self) -> None:
        """Create tables for interaction tracking and pattern storage."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    timestamp_unix REAL NOT NULL,
                    hour_of_day INTEGER NOT NULL,
                    day_of_week INTEGER NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_tenant_domain
                ON interactions(tenant_id, domain, timestamp_unix DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detected_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    description TEXT,
                    confidence REAL,
                    parameters TEXT,
                    detected_at REAL,
                    sample_count INTEGER
                )
            """)
            conn.commit()
        finally:
            conn.close()
        logger.debug("TemporalPatternDetector initialized at %s", self.db_path)

    def record_interaction(
        self,
        tenant_id: str,
        domain: str,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a user interaction with timestamp.

        Args:
            tenant_id: Tenant identifier.
            domain: Interaction category (e.g., "coding", "cooking").
            timestamp: Unix timestamp. Defaults to now.
            metadata: Optional additional data.
        """
        ts = timestamp or time.time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO interactions (tenant_id, domain, timestamp_unix, "
                "hour_of_day, day_of_week, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tenant_id,
                    domain,
                    ts,
                    dt.hour,
                    dt.weekday(),  # 0=Monday, 6=Sunday
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def detect_patterns(
        self,
        tenant_id: str,
        domain: str | None = None,
    ) -> list[TemporalPattern]:
        """Run pattern detection for a tenant.

        Args:
            tenant_id: Tenant identifier.
            domain: If set, only analyze this domain. Otherwise, all domains.

        Returns:
            List of detected patterns above confidence threshold.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Get all domains for tenant
            if domain:
                domains = [domain]
            else:
                cursor = conn.execute(
                    "SELECT DISTINCT domain FROM interactions WHERE tenant_id = ?",
                    (tenant_id,),
                )
                domains = [row[0] for row in cursor.fetchall()]

            all_patterns: list[TemporalPattern] = []

            for d in domains:
                # Fetch interactions
                cursor = conn.execute(
                    "SELECT timestamp_unix, hour_of_day, day_of_week "
                    "FROM interactions WHERE tenant_id = ? AND domain = ? "
                    "ORDER BY timestamp_unix",
                    (tenant_id, d),
                )
                rows = cursor.fetchall()

                if len(rows) < self.MIN_SAMPLES:
                    continue

                timestamps = [r[0] for r in rows]
                hours = [r[1] for r in rows]
                days = [r[2] for r in rows]

                # Detect various pattern types
                all_patterns.extend(
                    self._detect_day_of_week_patterns(tenant_id, d, days, len(rows))
                )
                all_patterns.extend(
                    self._detect_time_of_day_patterns(tenant_id, d, hours, len(rows))
                )
                all_patterns.extend(
                    self._detect_frequency_patterns(tenant_id, d, timestamps, len(rows))
                )

            # Filter by confidence
            patterns = [p for p in all_patterns if p.confidence >= self.CONFIDENCE_THRESHOLD]

            # Store detected patterns
            for p in patterns:
                self._store_pattern(conn, p)
            conn.commit()

            return patterns
        finally:
            conn.close()

    def get_stored_patterns(self, tenant_id: str) -> list[TemporalPattern]:
        """Retrieve previously detected patterns for a tenant."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT * FROM detected_patterns WHERE tenant_id = ? ORDER BY confidence DESC",
                (tenant_id,),
            )
            patterns = []
            for row in cursor.fetchall():
                patterns.append(
                    TemporalPattern(
                        pattern_id=row[0],
                        tenant_id=row[1],
                        domain=row[2],
                        pattern_type=row[3],
                        description=row[4] or "",
                        confidence=row[5] or 0.0,
                        parameters=json.loads(row[6]) if row[6] else {},
                        detected_at=row[7] or 0.0,
                        sample_count=row[8] or 0,
                    )
                )
            return patterns
        finally:
            conn.close()

    # ── Pattern Detectors ────────────────────────────────────────────────

    def _detect_day_of_week_patterns(
        self,
        tenant_id: str,
        domain: str,
        days: list[int],
        total_count: int,
    ) -> list[TemporalPattern]:
        """Detect day-of-week preferences."""
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        counter = Counter(days)
        total = len(days)

        patterns: list[TemporalPattern] = []

        for day_num, count in counter.most_common(3):
            # Expected uniform distribution: 1/7 ≈ 14.3%
            observed_pct = count / total
            expected_pct = 1.0 / 7
            ratio = observed_pct / expected_pct

            if ratio > 1.5:  # At least 50% more than expected
                confidence = min(1.0, (ratio - 1.0) / 2.0) * min(1.0, total / 30)
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    patterns.append(
                        TemporalPattern(
                            pattern_id=f"dow_{tenant_id}_{domain}_{day_num}",
                            tenant_id=tenant_id,
                            domain=domain,
                            pattern_type="day_of_week",
                            description=f"You tend to engage with {domain} on {day_names[day_num]}s "
                            f"({observed_pct:.0%} of interactions).",
                            confidence=confidence,
                            parameters={
                                "day_of_week": day_num,
                                "day_name": day_names[day_num],
                                "observed_pct": round(observed_pct, 3),
                                "ratio": round(ratio, 2),
                            },
                            sample_count=total_count,
                        )
                    )

        return patterns

    def _detect_time_of_day_patterns(
        self,
        tenant_id: str,
        domain: str,
        hours: list[int],
        total_count: int,
    ) -> list[TemporalPattern]:
        """Detect time-of-day preferences."""
        # Group into 4-hour blocks for more robust detection
        blocks = {
            "early_morning": (5, 8),
            "morning": (9, 12),
            "afternoon": (13, 16),
            "evening": (17, 20),
            "night": (21, 24),
            "late_night": (0, 4),
        }

        block_counts: dict[str, int] = defaultdict(int)
        for h in hours:
            for block_name, (start, end) in blocks.items():
                if start <= h < end or (block_name == "late_night" and h < end):
                    block_counts[block_name] += 1
                    break

        total = len(hours)
        patterns: list[TemporalPattern] = []

        for block_name, count in sorted(block_counts.items(), key=lambda x: -x[1]):
            observed_pct = count / total
            expected_pct = 4.0 / 24  # Each block covers ~4 hours

            ratio = observed_pct / expected_pct if expected_pct > 0 else 0

            if ratio > 1.8:  # 80% more than expected
                start_h, end_h = blocks[block_name]
                confidence = min(1.0, (ratio - 1.0) / 2.5) * min(1.0, total / 20)
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    patterns.append(
                        TemporalPattern(
                            pattern_id=f"tod_{tenant_id}_{domain}_{block_name}",
                            tenant_id=tenant_id,
                            domain=domain,
                            pattern_type="time_of_day",
                            description=f"You typically engage with {domain} during "
                            f"{block_name.replace('_', ' ')} ({start_h:02d}:00-{end_h:02d}:00).",
                            confidence=confidence,
                            parameters={
                                "block": block_name,
                                "start_hour": start_h,
                                "end_hour": end_h,
                                "observed_pct": round(observed_pct, 3),
                                "ratio": round(ratio, 2),
                            },
                            sample_count=total_count,
                        )
                    )

        return patterns

    def _detect_frequency_patterns(
        self,
        tenant_id: str,
        domain: str,
        timestamps: list[float],
        total_count: int,
    ) -> list[TemporalPattern]:
        """Detect interaction frequency patterns via inter-arrival time analysis."""
        if len(timestamps) < 5:
            return []

        # Compute inter-arrival times
        deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        # Filter out very small deltas (< 10s, likely duplicate events)
        deltas = [d for d in deltas if d > 10]
        if len(deltas) < 3:
            return []

        # Compute statistics
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
        std_delta = math.sqrt(variance) if variance > 0 else 0

        # Coefficient of variation (low = regular, high = irregular)
        cv = std_delta / mean_delta if mean_delta > 0 else float("inf")

        patterns: list[TemporalPattern] = []

        # Regular frequency pattern (CV < 0.5 means fairly regular)
        if cv < 0.5 and mean_delta < 86400:  # Less than 1 day between interactions
            # Determine human-readable interval
            if mean_delta < 3600:
                interval_str = f"every {mean_delta / 60:.0f} minutes"
            elif mean_delta < 86400:
                interval_str = f"every {mean_delta / 3600:.1f} hours"
            else:
                interval_str = f"every {mean_delta / 86400:.1f} days"

            confidence = max(0, 1.0 - cv) * min(1.0, len(deltas) / 20)

            if confidence >= self.CONFIDENCE_THRESHOLD:
                patterns.append(
                    TemporalPattern(
                        pattern_id=f"freq_{tenant_id}_{domain}",
                        tenant_id=tenant_id,
                        domain=domain,
                        pattern_type="frequency",
                        description=f"You interact with {domain} regularly — {interval_str}.",
                        confidence=confidence,
                        parameters={
                            "mean_interval_s": round(mean_delta, 1),
                            "std_interval_s": round(std_delta, 1),
                            "cv": round(cv, 3),
                            "interval_human": interval_str,
                        },
                        sample_count=total_count,
                    )
                )

        return patterns

    def _store_pattern(self, conn: sqlite3.Connection, pattern: TemporalPattern) -> None:
        """Upsert a detected pattern to the database."""
        conn.execute(
            "INSERT OR REPLACE INTO detected_patterns "
            "(pattern_id, tenant_id, domain, pattern_type, description, "
            "confidence, parameters, detected_at, sample_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pattern.pattern_id,
                pattern.tenant_id,
                pattern.domain,
                pattern.pattern_type,
                pattern.description,
                pattern.confidence,
                json.dumps(pattern.parameters),
                pattern.detected_at,
                pattern.sample_count,
            ),
        )

    def stats(self) -> dict[str, Any]:
        """Detector statistics."""
        conn = sqlite3.connect(self.db_path)
        try:
            interactions = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            patterns = conn.execute("SELECT COUNT(*) FROM detected_patterns").fetchone()[0]
            domains = conn.execute("SELECT COUNT(DISTINCT domain) FROM interactions").fetchone()[0]
            return {
                "total_interactions": interactions,
                "detected_patterns": patterns,
                "unique_domains": domains,
            }
        finally:
            conn.close()
