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

Uses DatabasePool (aiosqlite) for persistence — consistent with all
other HBLLM memory backends.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.memory.interface import MemoryType
from hbllm.memory.pool import DatabasePool
from hbllm.memory.repository import MemoryRepository

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


class SpatialMemory(MemoryRepository):
    """Location-aware memory system.

    Tracks what topics/domains are discussed in which locations,
    enabling contextual priming based on physical presence.

    Usage::

        spatial = SpatialMemory(db_path="data/spatial_memory.db")
        await spatial.init_db()

        # Register a location
        await spatial.register_location("kitchen", identifiers={"room": "kitchen"})

        # Record an interaction
        await spatial.record_interaction("user1", location_id="kitchen", domain="cooking")

        # Get context for a location
        context = await spatial.get_location_context("user1", "kitchen")
    """

    def __init__(
        self,
        db_path: str | Path = "data/spatial_memory.db",
    ) -> None:
        self.db_path = Path(db_path)
        self.pool = DatabasePool(str(self.db_path))
        # WiFi SSID → location_id mapping
        self._wifi_map: dict[str, str] = {}
        # Device ID → location_id mapping
        self._device_map: dict[str, str] = {}

    async def init_db(self) -> None:
        """Create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    location_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_type TEXT NOT NULL DEFAULT 'room',
                    identifiers TEXT,
                    metadata TEXT
                )
            """)
            await conn.execute("""
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
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spatial_tenant_location
                ON spatial_interactions(tenant_id, location_id, timestamp_unix DESC)
            """)
            await conn.execute("""
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
            await conn.commit()
        logger.debug("SpatialMemory initialized at %s", self.db_path)

    async def close(self) -> None:
        """Close database connections."""
        await self.pool.close_all()

    async def register_location(
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
        async with self.pool.acquire() as conn:
            await conn.execute(
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
            await conn.commit()

        # Update lookup maps
        if "wifi_ssid" in ids:
            self._wifi_map[ids["wifi_ssid"]] = location_id
        if "device_id" in ids:
            self._device_map[ids["device_id"]] = location_id

    async def resolve_location(
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
            async with self.pool.acquire() as conn:
                async with conn.execute(
                    "SELECT location_id FROM locations WHERE location_id = ? OR name = ?",
                    (room_name, room_name),
                ) as cursor:
                    row = await cursor.fetchone()
                if row:
                    return row[0]
        return None

    async def record_interaction(
        self,
        tenant_id: str,
        location_id: str,
        domain: str,
        content_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an interaction at a location."""
        now = time.time()
        async with self.pool.acquire() as conn:
            await conn.execute(
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
            await conn.execute(
                "INSERT INTO location_domain_stats "
                "(tenant_id, location_id, domain, interaction_count, first_seen, last_seen) "
                "VALUES (?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(tenant_id, location_id, domain) DO UPDATE SET "
                "interaction_count = interaction_count + 1, "
                "last_seen = ?",
                (tenant_id, location_id, domain, now, now, now),
            )
            await conn.commit()

    async def get_location_context(
        self,
        tenant_id: str,
        location_id: str,
        limit: int = 10,
    ) -> list[SpatialMemoryEntry]:
        """Get the top domains/topics for a given location.

        Returns what the user typically discusses/does at this location,
        sorted by interaction frequency.
        """
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT s.domain, s.interaction_count, s.first_seen, s.last_seen, "
                "l.name "
                "FROM location_domain_stats s "
                "JOIN locations l ON s.location_id = l.location_id "
                "WHERE s.tenant_id = ? AND s.location_id = ? "
                "ORDER BY s.interaction_count DESC LIMIT ?",
                (tenant_id, location_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        entries = []
        for row in rows:
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

    async def get_domains_by_location(
        self,
        tenant_id: str,
    ) -> dict[str, list[str]]:
        """Get all location → domain mappings for a tenant.

        Returns:
            Dict mapping location_id to list of domains (sorted by frequency).
        """
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT location_id, domain FROM location_domain_stats "
                "WHERE tenant_id = ? ORDER BY interaction_count DESC",
                (tenant_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        result: dict[str, list[str]] = {}
        for row in rows:
            loc = row[0]
            if loc not in result:
                result[loc] = []
            result[loc].append(row[1])
        return result

    # ── MemoryRepository interface ───────────────────────────────────
    # Transitional adapters.

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.SPATIAL

    async def initialize(self) -> None:
        await self.init_db()

    async def shutdown(self) -> None:
        await self.close()

    async def store(self, content: str, tenant_id: str = "default", **kwargs: Any) -> str:
        """Store a spatial interaction.

        Keyword Args:
            location_id: Location identifier (required).
            domain: Domain/topic tag (default: "general").
        """
        location_id = kwargs.get("location_id", "unknown")
        domain = kwargs.get("domain", "general")
        await self.record_interaction(
            tenant_id=tenant_id,
            location_id=location_id,
            domain=domain,
            content_summary=content,
            metadata=kwargs.get("metadata"),
        )
        return f"{tenant_id}:{location_id}:{domain}"

    async def retrieve(
        self, memory_id: str, tenant_id: str = "default", **kwargs: Any
    ) -> dict[str, Any] | None:
        """Retrieve a location's registration info."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT * FROM locations WHERE location_id = ?",
                (memory_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "location_id": row[0],
            "name": row[1],
            "location_type": row[2],
            "identifiers": json.loads(row[3]) if row[3] else {},
            "metadata": json.loads(row[4]) if row[4] else {},
        }

    async def search(
        self, query: str, tenant_id: str = "default", **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search spatial interactions by location_id or domain."""
        raw_limit = kwargs.get("top_k")
        if raw_limit is None:
            raw_limit = kwargs.get("limit")

        limit = 10
        if raw_limit is not None:
            try:
                limit = int(raw_limit)
            except (ValueError, TypeError):
                pass

        entries = await self.get_location_context(tenant_id, query, limit=limit)
        return [
            {
                "location_id": e.location_id,
                "location_name": e.location_name,
                "domain": e.domain,
                "interaction_count": e.interaction_count,
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
                "score": float(e.interaction_count),
            }
            for e in entries
        ]

    async def stats(self, tenant_id: str = "default") -> dict[str, Any]:
        """Spatial memory statistics."""
        async with self.pool.acquire() as conn:
            async with conn.execute("SELECT COUNT(*) FROM locations") as cursor:
                loc_row = await cursor.fetchone()
            async with conn.execute("SELECT COUNT(*) FROM spatial_interactions") as cursor:
                int_row = await cursor.fetchone()
        return {
            "memory_type": self.memory_type.value,
            "registered_locations": loc_row[0] if loc_row else 0,
            "total_interactions": int_row[0] if int_row else 0,
            "wifi_mappings": len(self._wifi_map),
            "device_mappings": len(self._device_map),
        }
