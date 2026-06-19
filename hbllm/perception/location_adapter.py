"""
Location Adapter — Spatial awareness and geofence-based triggers.

Provides location tracking and geofence monitoring for location-aware
behaviors:
    "Remind me when I get home"
    "What's the weather here?"
    "Show me nearby restaurants"

Supports multiple location sources:
- GPS (via mobile client reporting)
- IP-based geolocation (fallback)
- Manual/static location setting

Bus Topics:
    perception.location.update   → Published on location change
    perception.location.geofence → Published when entering/leaving a geofence
    perception.location.set      → Subscribed (manual location updates)

Usage::

    adapter = LocationAdapter(node_id="location")
    await adapter.start(bus)

    # Register a geofence
    adapter.add_geofence(Geofence(
        id="home",
        name="Home",
        latitude=37.7749,
        longitude=-122.4194,
        radius_meters=100,
    ))

    # Update location (from mobile client)
    await bus.publish("perception.location.set", Message(
        payload={"latitude": 37.7749, "longitude": -122.4194}
    ))
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class GeoCoordinate:
    """A geographic coordinate."""

    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0  # meters above sea level
    accuracy: float = 0.0  # horizontal accuracy in meters
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"  # "gps", "ip", "manual", "network"

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "altitude": round(self.altitude, 1),
            "accuracy": round(self.accuracy, 1),
            "timestamp": self.timestamp,
            "source": self.source,
        }

    def is_valid(self) -> bool:
        """Check if coordinates are valid (not at null island)."""
        return not (self.latitude == 0.0 and self.longitude == 0.0)

    def distance_to(self, other: GeoCoordinate) -> float:
        """Calculate distance to another coordinate in meters (Haversine)."""
        return haversine_distance(
            self.latitude, self.longitude,
            other.latitude, other.longitude,
        )


@dataclass
class Geofence:
    """A circular geographic boundary for trigger events."""

    id: str
    name: str
    latitude: float
    longitude: float
    radius_meters: float = 100.0
    tenant_id: str = ""
    # Trigger configuration
    trigger_on_enter: bool = True
    trigger_on_exit: bool = True
    # Optional payload to include in trigger events
    metadata: dict[str, Any] = field(default_factory=dict)
    # Cooldown between triggers (avoid rapid toggling)
    cooldown_seconds: float = 60.0
    # State tracking
    _last_inside: bool | None = field(default=None, repr=False)
    _last_trigger_time: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "radius_meters": self.radius_meters,
            "tenant_id": self.tenant_id,
            "trigger_on_enter": self.trigger_on_enter,
            "trigger_on_exit": self.trigger_on_exit,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Geofence:
        return cls(
            id=data["id"],
            name=data["name"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            radius_meters=data.get("radius_meters", 100.0),
            tenant_id=data.get("tenant_id", ""),
            trigger_on_enter=data.get("trigger_on_enter", True),
            trigger_on_exit=data.get("trigger_on_exit", True),
            metadata=data.get("metadata", {}),
            cooldown_seconds=data.get("cooldown_seconds", 60.0),
        )

    def contains(self, coord: GeoCoordinate) -> bool:
        """Check if a coordinate is inside this geofence."""
        distance = haversine_distance(
            self.latitude, self.longitude,
            coord.latitude, coord.longitude,
        )
        return distance <= self.radius_meters


@dataclass
class GeofenceEvent:
    """A geofence trigger event."""

    geofence_id: str
    geofence_name: str
    event_type: str  # "enter" or "exit"
    coordinate: GeoCoordinate
    distance_meters: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "geofence_id": self.geofence_id,
            "geofence_name": self.geofence_name,
            "event_type": self.event_type,
            "coordinate": self.coordinate.to_dict(),
            "distance_meters": round(self.distance_meters, 1),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ── Haversine Distance ──────────────────────────────────────────────────────


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the distance between two points on Earth in meters.

    Uses the Haversine formula for great-circle distance.
    """
    R = 6_371_000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ── Location Adapter Node ───────────────────────────────────────────────────


class LocationAdapter(Node):
    """Spatial awareness and geofence monitoring.

    Tracks device location and fires events when the user enters or
    leaves configured geographic boundaries (geofences).

    Location updates can come from:
    1. Mobile client GPS reports (via ``perception.location.set``)
    2. IP-based geolocation (automatic fallback)
    3. Manual configuration
    """

    def __init__(
        self,
        node_id: str = "location_adapter",
        update_threshold_meters: float = 50.0,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=[
                "location_tracking",
                "geofencing",
                "spatial_awareness",
            ],
        )
        self._update_threshold = update_threshold_meters

        # Current location per tenant
        self._locations: dict[str, GeoCoordinate] = {}

        # Geofences: tenant_id → {fence_id: Geofence}
        self._geofences: dict[str, dict[str, Geofence]] = {}

        # Location history (limited)
        self._history: dict[str, list[GeoCoordinate]] = {}
        self._max_history = 100

        # Stats
        self._total_updates: int = 0
        self._total_geofence_events: int = 0

    async def on_start(self) -> None:
        """Subscribe to location update topics."""
        await self.bus.subscribe(
            "perception.location.set", self._on_location_set
        )
        await self.bus.subscribe(
            "perception.location.geofence.add", self._on_add_geofence
        )
        await self.bus.subscribe(
            "perception.location.geofence.remove", self._on_remove_geofence
        )
        logger.info("LocationAdapter started")

    async def on_stop(self) -> None:
        logger.info(
            "LocationAdapter stopped (updates=%d, geofence_events=%d)",
            self._total_updates,
            self._total_geofence_events,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Location Updates ─────────────────────────────────────────────────

    async def _on_location_set(self, message: Message) -> None:
        """Handle location update from a client device."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"

        coord = GeoCoordinate(
            latitude=float(payload.get("latitude", 0)),
            longitude=float(payload.get("longitude", 0)),
            altitude=float(payload.get("altitude", 0)),
            accuracy=float(payload.get("accuracy", 0)),
            source=payload.get("source", "client"),
        )

        if not coord.is_valid():
            logger.debug("Invalid location update (null island)")
            return

        # Check if significant movement
        prev = self._locations.get(tenant_id)
        if prev and prev.distance_to(coord) < self._update_threshold:
            return  # Not enough movement

        self._locations[tenant_id] = coord
        self._total_updates += 1

        # Record history
        if tenant_id not in self._history:
            self._history[tenant_id] = []
        self._history[tenant_id].append(coord)
        if len(self._history[tenant_id]) > self._max_history:
            self._history[tenant_id] = self._history[tenant_id][
                -self._max_history :
            ]

        logger.info(
            "📍 Location update [%s]: %.4f, %.4f (accuracy=%.0fm, source=%s)",
            tenant_id,
            coord.latitude,
            coord.longitude,
            coord.accuracy,
            coord.source,
        )

        # Publish location update
        await self.bus.publish(
            "perception.location.update",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=tenant_id,
                topic="perception.location.update",
                payload=coord.to_dict(),
            ),
        )

        # Check geofences
        await self._check_geofences(tenant_id, coord)

    # ── Geofence Management ──────────────────────────────────────────────

    def add_geofence(self, fence: Geofence) -> None:
        """Register a geofence for monitoring."""
        tenant_id = fence.tenant_id or "default"
        if tenant_id not in self._geofences:
            self._geofences[tenant_id] = {}
        self._geofences[tenant_id][fence.id] = fence
        logger.info(
            "Added geofence '%s' (%s) at %.4f, %.4f (r=%.0fm)",
            fence.id,
            fence.name,
            fence.latitude,
            fence.longitude,
            fence.radius_meters,
        )

    def remove_geofence(self, tenant_id: str, fence_id: str) -> bool:
        """Remove a geofence."""
        if tenant_id in self._geofences:
            removed = self._geofences[tenant_id].pop(fence_id, None)
            return removed is not None
        return False

    def list_geofences(self, tenant_id: str) -> list[dict[str, Any]]:
        """List all geofences for a tenant."""
        if tenant_id not in self._geofences:
            return []
        return [f.to_dict() for f in self._geofences[tenant_id].values()]

    async def _on_add_geofence(self, message: Message) -> None:
        """Handle geofence registration via bus."""
        payload = message.payload
        tenant_id = message.tenant_id or payload.get("tenant_id", "default")

        fence = Geofence(
            id=payload.get("id", f"fence_{time.time():.0f}"),
            name=payload.get("name", "Unnamed"),
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            radius_meters=float(payload.get("radius_meters", 100)),
            tenant_id=tenant_id,
            trigger_on_enter=payload.get("trigger_on_enter", True),
            trigger_on_exit=payload.get("trigger_on_exit", True),
            metadata=payload.get("metadata", {}),
        )
        self.add_geofence(fence)

    async def _on_remove_geofence(self, message: Message) -> None:
        """Handle geofence removal via bus."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        fence_id = payload.get("id", "")
        if fence_id:
            self.remove_geofence(tenant_id, fence_id)

    # ── Geofence Checking ────────────────────────────────────────────────

    async def _check_geofences(
        self, tenant_id: str, coord: GeoCoordinate
    ) -> None:
        """Check all geofences for this tenant against the new location."""
        fences = self._geofences.get(tenant_id, {})
        now = time.time()

        for fence in fences.values():
            is_inside = fence.contains(coord)
            was_inside = fence._last_inside

            # Skip if no state change
            if was_inside is not None and is_inside == was_inside:
                continue

            # Cooldown check
            if now - fence._last_trigger_time < fence.cooldown_seconds:
                continue

            # Determine event type
            if is_inside and (was_inside is None or not was_inside):
                event_type = "enter"
            elif not is_inside and was_inside:
                event_type = "exit"
            else:
                fence._last_inside = is_inside
                continue

            # Check if this trigger type is enabled
            if event_type == "enter" and not fence.trigger_on_enter:
                fence._last_inside = is_inside
                continue
            if event_type == "exit" and not fence.trigger_on_exit:
                fence._last_inside = is_inside
                continue

            fence._last_inside = is_inside
            fence._last_trigger_time = now
            self._total_geofence_events += 1

            distance = haversine_distance(
                fence.latitude, fence.longitude,
                coord.latitude, coord.longitude,
            )

            event = GeofenceEvent(
                geofence_id=fence.id,
                geofence_name=fence.name,
                event_type=event_type,
                coordinate=coord,
                distance_meters=distance,
                metadata=fence.metadata,
            )

            logger.info(
                "🏠 Geofence %s: '%s' (%s) — distance=%.0fm",
                event_type,
                fence.name,
                fence.id,
                distance,
            )

            await self.bus.publish(
                "perception.location.geofence",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    topic="perception.location.geofence",
                    payload=event.to_dict(),
                ),
            )

    # ── Public API ───────────────────────────────────────────────────────

    def get_location(self, tenant_id: str) -> GeoCoordinate | None:
        """Get the current location for a tenant."""
        return self._locations.get(tenant_id)

    def get_distance_to(
        self, tenant_id: str, latitude: float, longitude: float
    ) -> float | None:
        """Get distance from tenant's current location to a point."""
        loc = self._locations.get(tenant_id)
        if loc is None:
            return None
        return haversine_distance(
            loc.latitude, loc.longitude, latitude, longitude
        )

    def get_nearest_geofence(
        self, tenant_id: str
    ) -> tuple[Geofence, float] | None:
        """Get the nearest geofence and distance to it."""
        loc = self._locations.get(tenant_id)
        if loc is None:
            return None

        fences = self._geofences.get(tenant_id, {})
        if not fences:
            return None

        nearest = None
        nearest_dist = float("inf")

        for fence in fences.values():
            dist = haversine_distance(
                loc.latitude, loc.longitude,
                fence.latitude, fence.longitude,
            )
            if dist < nearest_dist:
                nearest = fence
                nearest_dist = dist

        if nearest is None:
            return None
        return nearest, nearest_dist

    def stats(self) -> dict[str, Any]:
        """Return location adapter statistics."""
        return {
            "tracked_tenants": len(self._locations),
            "total_updates": self._total_updates,
            "total_geofence_events": self._total_geofence_events,
            "active_geofences": sum(
                len(fences) for fences in self._geofences.values()
            ),
        }
