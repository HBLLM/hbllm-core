"""Tests for LocationAdapter — spatial awareness and geofencing."""



from hbllm.perception.location_adapter import (
    GeoCoordinate,
    Geofence,
    GeofenceEvent,
    LocationAdapter,
    haversine_distance,
)

# ── Known distances (approximate) ───────────────────────────────────────────
# San Francisco → Los Angeles ≈ 559 km
SF_LAT, SF_LON = 37.7749, -122.4194
LA_LAT, LA_LON = 34.0522, -118.2437

# Two points ~100m apart (for geofence testing)
HOME_LAT, HOME_LON = 37.7749, -122.4194
NEAR_HOME_LAT, NEAR_HOME_LON = 37.7750, -122.4193  # ~15m away
FAR_LAT, FAR_LON = 37.7800, -122.4194  # ~567m away


class TestHaversineDistance:
    def test_same_point(self):
        d = haversine_distance(SF_LAT, SF_LON, SF_LAT, SF_LON)
        assert d == 0.0

    def test_sf_to_la(self):
        d = haversine_distance(SF_LAT, SF_LON, LA_LAT, LA_LON)
        # Should be approximately 559 km
        assert 550_000 < d < 570_000

    def test_short_distance(self):
        d = haversine_distance(HOME_LAT, HOME_LON, NEAR_HOME_LAT, NEAR_HOME_LON)
        assert d < 50  # Should be < 50 meters

    def test_symmetry(self):
        d1 = haversine_distance(SF_LAT, SF_LON, LA_LAT, LA_LON)
        d2 = haversine_distance(LA_LAT, LA_LON, SF_LAT, SF_LON)
        assert abs(d1 - d2) < 0.01


class TestGeoCoordinate:
    def test_defaults(self):
        c = GeoCoordinate()
        assert c.latitude == 0.0
        assert c.source == "unknown"

    def test_is_valid(self):
        assert not GeoCoordinate().is_valid()
        assert GeoCoordinate(latitude=37.0, longitude=-122.0).is_valid()

    def test_distance_to(self):
        c1 = GeoCoordinate(latitude=SF_LAT, longitude=SF_LON)
        c2 = GeoCoordinate(latitude=LA_LAT, longitude=LA_LON)
        assert c1.distance_to(c2) > 500_000

    def test_to_dict(self):
        c = GeoCoordinate(latitude=37.7749, longitude=-122.4194, source="gps")
        d = c.to_dict()
        assert d["latitude"] == 37.7749
        assert d["source"] == "gps"


class TestGeofence:
    def test_contains_inside(self):
        fence = Geofence(
            id="home",
            name="Home",
            latitude=HOME_LAT,
            longitude=HOME_LON,
            radius_meters=100,
        )
        near = GeoCoordinate(latitude=NEAR_HOME_LAT, longitude=NEAR_HOME_LON)
        assert fence.contains(near)

    def test_contains_outside(self):
        fence = Geofence(
            id="home",
            name="Home",
            latitude=HOME_LAT,
            longitude=HOME_LON,
            radius_meters=100,
        )
        far = GeoCoordinate(latitude=FAR_LAT, longitude=FAR_LON)
        assert not fence.contains(far)

    def test_to_dict(self):
        fence = Geofence(
            id="office",
            name="Office",
            latitude=37.0,
            longitude=-122.0,
            radius_meters=200,
        )
        d = fence.to_dict()
        assert d["id"] == "office"
        assert d["radius_meters"] == 200

    def test_from_dict(self):
        data = {
            "id": "gym",
            "name": "Gym",
            "latitude": 37.0,
            "longitude": -122.0,
            "radius_meters": 50,
        }
        fence = Geofence.from_dict(data)
        assert fence.id == "gym"
        assert fence.radius_meters == 50

    def test_roundtrip(self):
        fence = Geofence(
            id="park",
            name="Park",
            latitude=37.5,
            longitude=-122.1,
            radius_meters=300,
            trigger_on_exit=False,
        )
        restored = Geofence.from_dict(fence.to_dict())
        assert restored.id == fence.id
        assert not restored.trigger_on_exit


class TestGeofenceEvent:
    def test_to_dict(self):
        ev = GeofenceEvent(
            geofence_id="home",
            geofence_name="Home",
            event_type="enter",
            coordinate=GeoCoordinate(latitude=37.0, longitude=-122.0),
            distance_meters=42.5,
        )
        d = ev.to_dict()
        assert d["event_type"] == "enter"
        assert d["distance_meters"] == 42.5


class TestLocationAdapter:
    def test_instantiation(self):
        adapter = LocationAdapter()
        assert "location_tracking" in adapter.capabilities
        assert "geofencing" in adapter.capabilities

    def test_add_geofence(self):
        adapter = LocationAdapter()
        fence = Geofence(
            id="home",
            name="Home",
            latitude=HOME_LAT,
            longitude=HOME_LON,
            radius_meters=100,
            tenant_id="t1",
        )
        adapter.add_geofence(fence)
        fences = adapter.list_geofences("t1")
        assert len(fences) == 1
        assert fences[0]["id"] == "home"

    def test_remove_geofence(self):
        adapter = LocationAdapter()
        fence = Geofence(
            id="home",
            name="Home",
            latitude=HOME_LAT,
            longitude=HOME_LON,
            tenant_id="t1",
        )
        adapter.add_geofence(fence)
        assert adapter.remove_geofence("t1", "home")
        assert adapter.list_geofences("t1") == []

    def test_remove_nonexistent(self):
        adapter = LocationAdapter()
        assert not adapter.remove_geofence("t1", "missing")

    def test_get_location_none(self):
        adapter = LocationAdapter()
        assert adapter.get_location("t1") is None

    def test_get_distance_to_none(self):
        adapter = LocationAdapter()
        assert adapter.get_distance_to("t1", 37.0, -122.0) is None

    def test_get_nearest_geofence_none(self):
        adapter = LocationAdapter()
        assert adapter.get_nearest_geofence("t1") is None

    def test_stats(self):
        adapter = LocationAdapter()
        stats = adapter.stats()
        assert stats["tracked_tenants"] == 0
        assert stats["total_updates"] == 0
        assert stats["active_geofences"] == 0

    def test_get_nearest_geofence(self):
        adapter = LocationAdapter()
        adapter._locations["t1"] = GeoCoordinate(latitude=HOME_LAT, longitude=HOME_LON)

        fence1 = Geofence(
            id="near",
            name="Near",
            latitude=NEAR_HOME_LAT,
            longitude=NEAR_HOME_LON,
            tenant_id="t1",
        )
        fence2 = Geofence(
            id="far",
            name="Far",
            latitude=FAR_LAT,
            longitude=FAR_LON,
            tenant_id="t1",
        )
        adapter.add_geofence(fence1)
        adapter.add_geofence(fence2)

        result = adapter.get_nearest_geofence("t1")
        assert result is not None
        nearest, dist = result
        assert nearest.id == "near"
        assert dist < 50
