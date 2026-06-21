"""Spatial Memory — location-aware context storage.

Associates memories with physical locations, enabling:
    - Room-based context ("In the kitchen, you asked about recipes")
    - Device-location mapping ("Phone = mobile, Desktop = office")
    - Geofence events ("At the gym, you play workout playlists")

Location is inferred from:
    - IoT device room assignments (from MqttIoTNode)
    - Device ID → location mappings (from DeviceBridge)
    - WiFi SSID → location mappings (configurable)
    - Explicit user annotations

Uses SQLite for persistence with spatial indexing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LocationContext:
    """A physical location with associated metadata."""

    location_id: str = ""
    name: str = ""  # Human-readable: "Kitchen", "Office", "Gym"
    location_type: str = "room"  # "room", "building", "zone", "geofence"
    identifiers: dict[str, str] = field(default_factory=dict)  # WiFi SSID, device_id, room_name
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_id": self.location_id,
            "name": self.name,
            "location_type": self.location_type,
            "identifiers": self.identifiers,
            "metadata": self.metadata,
        }


@dataclass
class SpatialMemoryEntry:
    """A memory entry tagged with location."""

    id: str = ""
    tenant_id: str = "default"
    location_id: str = ""
    location_name: str = ""
    domain: str = ""
    content_summary: str = ""
    interaction_count: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SpatialMemory:
    """Location-aware memory system.

    Tracks what topics/domains are discussed in which locations,
    enabling contextual priming based on physical presence.

    Usage::

        spatial = SpatialMemory(db_path="data/spatial_memory.db")
        await spatial.init_db()

        # Register a location
        spatial.register_location("kitchen", identifiers={"room": "kitchen"})

        # Record an interaction
        spatial.record_interaction("user1", location_id="kitchen", domain="cooking")

        # Get context for a location
        context = spatial.get_location_context("user1", "kitchen")
    """

    def __init__(
        self,
        db_path: str | Path = "data/spatial_memory.db",
    ) -> None:
        self.db_path = Path(db_path)
        # WiFi SSID → location_id mapping
        self._wifi_map: dict[str, str] = {}
        # Device ID → location_id mapping
        self._device_map: dict[str, str] = {}

    async def init_db(self) -> None:
        """Create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    location_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_type TEXT NOT NULL DEFAULT 'room',
                    identifiers TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spatial_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    content_summary TEXT,
                    timestamp_unix REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (location_id) REFERENCES locations(location_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spatial_tenant_location
                ON spatial_interactions(tenant_id, location_id, timestamp_unix DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS location_domain_stats (
                    tenant_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    interaction_count INTEGER DEFAULT 0,
                    first_seen REAL,
                    last_seen REAL,
                    PRIMARY KEY (tenant_id, location_id, domain)
                )
            """)
            conn.commit()
        finally:
            conn.close()
        logger.debug("SpatialMemory initialized at %s", self.db_path)

    def register_location(
        self,
        location_id: str,
        name: str | None = None,
        location_type: str = "room",
        identifiers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register or update a physical location.

        Args:
            location_id: Unique identifier (e.g., "kitchen", "office_2nd_floor").
            name: Human-readable name. Defaults to location_id.
            location_type: "room", "building", "zone", "geofence".
            identifiers: Mapping keys to find this location (WiFi SSID, device_id, etc).
            metadata: Additional location data.
        """
        ids = identifiers or {}
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO locations "
                "(location_id, name, location_type, identifiers, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    location_id,
                    name or location_id,
                    location_type,
                    json.dumps(ids),
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Update lookup maps
        if "wifi_ssid" in ids:
            self._wifi_map[ids["wifi_ssid"]] = location_id
        if "device_id" in ids:
            self._device_map[ids["device_id"]] = location_id

    def resolve_location(
        self,
        wifi_ssid: str | None = None,
        device_id: str | None = None,
        room_name: str | None = None,
    ) -> str | None:
        """Resolve a physical location from available signals.

        Checks WiFi SSID, device ID, and room name in order of priority.
        """
        if wifi_ssid and wifi_ssid in self._wifi_map:
            return self._wifi_map[wifi_ssid]
        if device_id and device_id in self._device_map:
            return self._device_map[device_id]
        if room_name:
            # Check if room_name is a registered location_id
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT location_id FROM locations WHERE location_id = ? OR name = ?",
                    (room_name, room_name),
                ).fetchone()
                if row:
                    return row[0]
            finally:
                conn.close()
        return None

    def record_interaction(
        self,
        tenant_id: str,
        location_id: str,
        domain: str,
        content_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an interaction at a location."""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO spatial_interactions "
                "(tenant_id, location_id, domain, content_summary, timestamp_unix, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tenant_id,
                    location_id,
                    domain,
                    content_summary,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )

            # Update aggregate stats
            conn.execute(
                "INSERT INTO location_domain_stats "
                "(tenant_id, location_id, domain, interaction_count, first_seen, last_seen) "
                "VALUES (?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(tenant_id, location_id, domain) DO UPDATE SET "
                "interaction_count = interaction_count + 1, "
                "last_seen = ?",
                (tenant_id, location_id, domain, now, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_location_context(
        self,
        tenant_id: str,
        location_id: str,
        limit: int = 10,
    ) -> list[SpatialMemoryEntry]:
        """Get the top domains/topics for a given location.

        Returns what the user typically discusses/does at this location,
        sorted by interaction frequency.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT s.domain, s.interaction_count, s.first_seen, s.last_seen, "
                "l.name "
                "FROM location_domain_stats s "
                "JOIN locations l ON s.location_id = l.location_id "
                "WHERE s.tenant_id = ? AND s.location_id = ? "
                "ORDER BY s.interaction_count DESC LIMIT ?",
                (tenant_id, location_id, limit),
            )

            entries = []
            for row in cursor.fetchall():
                entries.append(
                    SpatialMemoryEntry(
                        tenant_id=tenant_id,
                        location_id=location_id,
                        location_name=row[4],
                        domain=row[0],
                        interaction_count=row[1],
                        first_seen=row[2],
                        last_seen=row[3],
                    )
                )
            return entries
        finally:
            conn.close()

    def get_domains_by_location(
        self,
        tenant_id: str,
    ) -> dict[str, list[str]]:
        """Get all location → domain mappings for a tenant.

        Returns:
            Dict mapping location_id to list of domains (sorted by frequency).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT location_id, domain FROM location_domain_stats "
                "WHERE tenant_id = ? ORDER BY interaction_count DESC",
                (tenant_id,),
            )
            result: dict[str, list[str]] = {}
            for row in cursor.fetchall():
                loc = row[0]
                if loc not in result:
                    result[loc] = []
                result[loc].append(row[1])
            return result
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        """Spatial memory statistics."""
        conn = sqlite3.connect(self.db_path)
        try:
            locations = conn.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
            interactions = conn.execute("SELECT COUNT(*) FROM spatial_interactions").fetchone()[0]
            return {
                "registered_locations": locations,
                "total_interactions": interactions,
                "wifi_mappings": len(self._wifi_map),
                "device_mappings": len(self._device_map),
            }
        finally:
            conn.close()
