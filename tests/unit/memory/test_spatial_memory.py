"""Tests for SpatialMemory — location-aware context storage."""

import pytest
import pytest_asyncio

from hbllm.memory.spatial_memory import SpatialMemory


@pytest_asyncio.fixture
async def spatial(tmp_path):
    s = SpatialMemory(db_path=tmp_path / "spatial.db")
    await s.init_db()
    return s


@pytest.mark.asyncio
async def test_register_location(spatial):
    """Locations can be registered and stored."""
    spatial.register_location("kitchen", name="Kitchen", identifiers={"wifi_ssid": "Home"})
    assert spatial.stats()["registered_locations"] == 1


@pytest.mark.asyncio
async def test_resolve_by_wifi(spatial):
    """Locations can be resolved from WiFi SSID."""
    spatial.register_location("office", identifiers={"wifi_ssid": "CorpWiFi"})
    assert spatial.resolve_location(wifi_ssid="CorpWiFi") == "office"
    assert spatial.resolve_location(wifi_ssid="Unknown") is None


@pytest.mark.asyncio
async def test_resolve_by_device_id(spatial):
    """Locations can be resolved from device ID."""
    spatial.register_location("desk", identifiers={"device_id": "macbook"})
    assert spatial.resolve_location(device_id="macbook") == "desk"


@pytest.mark.asyncio
async def test_resolve_by_room_name(spatial):
    """Locations can be resolved by room name."""
    spatial.register_location("bedroom", name="Bedroom")
    assert spatial.resolve_location(room_name="bedroom") == "bedroom"
    assert spatial.resolve_location(room_name="Bedroom") == "bedroom"


@pytest.mark.asyncio
async def test_record_interaction(spatial):
    """Interactions at locations are tracked."""
    spatial.register_location("kitchen")
    spatial.record_interaction("t1", "kitchen", "cooking")
    spatial.record_interaction("t1", "kitchen", "cooking")
    spatial.record_interaction("t1", "kitchen", "cleaning")
    assert spatial.stats()["total_interactions"] == 3


@pytest.mark.asyncio
async def test_get_location_context(spatial):
    """Location context returns domains sorted by frequency."""
    spatial.register_location("kitchen")
    spatial.record_interaction("t1", "kitchen", "cooking")
    spatial.record_interaction("t1", "kitchen", "cooking")
    spatial.record_interaction("t1", "kitchen", "cleaning")

    ctx = spatial.get_location_context("t1", "kitchen")
    assert len(ctx) == 2
    assert ctx[0].domain == "cooking"
    assert ctx[0].interaction_count == 2
    assert ctx[1].domain == "cleaning"


@pytest.mark.asyncio
async def test_get_domains_by_location(spatial):
    """Get all location→domain mappings for a tenant."""
    spatial.register_location("kitchen")
    spatial.register_location("office")
    spatial.record_interaction("t1", "kitchen", "cooking")
    spatial.record_interaction("t1", "office", "coding")

    mapping = spatial.get_domains_by_location("t1")
    assert "kitchen" in mapping
    assert "office" in mapping
    assert "cooking" in mapping["kitchen"]


@pytest.mark.asyncio
async def test_tenant_isolation(spatial):
    """Interactions are isolated per tenant."""
    spatial.register_location("kitchen")
    spatial.record_interaction("t1", "kitchen", "cooking")
    spatial.record_interaction("t2", "kitchen", "cleaning")

    t1_ctx = spatial.get_location_context("t1", "kitchen")
    t2_ctx = spatial.get_location_context("t2", "kitchen")
    assert len(t1_ctx) == 1
    assert t1_ctx[0].domain == "cooking"
    assert len(t2_ctx) == 1
    assert t2_ctx[0].domain == "cleaning"


@pytest.mark.asyncio
async def test_location_update(spatial):
    """Re-registering a location updates it."""
    spatial.register_location("kitchen", name="Kitchen v1")
    spatial.register_location("kitchen", name="Kitchen v2")
    assert spatial.stats()["registered_locations"] == 1
