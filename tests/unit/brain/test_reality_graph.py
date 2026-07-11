"""Unit tests for RealityGraph — unified world model facade."""

import time

import pytest

from hbllm.brain.reasoning.reality_graph import RealityEntity, RealityGraph


class TestRealityEntity:
    def test_defaults(self):
        entity = RealityEntity(entity_id="test_1", entity_type="concept", label="HBLLM")
        assert entity.confidence == 1.0
        assert entity.source == ""
        assert entity.ttl is None
        assert entity.is_expired is False

    def test_expiry_with_ttl(self):
        entity = RealityEntity(
            entity_id="test_1",
            entity_type="state",
            label="temp",
            ttl=60.0,
            last_updated=time.time() - 120,  # 2 minutes ago
        )
        assert entity.is_expired is True

    def test_not_expired_within_ttl(self):
        entity = RealityEntity(
            entity_id="test_1",
            entity_type="state",
            label="temp",
            ttl=600.0,
            last_updated=time.time(),
        )
        assert entity.is_expired is False

    def test_no_ttl_never_expires(self):
        entity = RealityEntity(
            entity_id="test_1",
            entity_type="concept",
            label="permanent",
            ttl=None,
            last_updated=time.time() - 999999,
        )
        assert entity.is_expired is False

    def test_merge_with(self):
        e1 = RealityEntity(
            entity_id="test_1",
            entity_type="device",
            label="MacBook",
            attributes={"battery": 85},
            confidence=0.7,
            source="kg",
        )
        e2 = RealityEntity(
            entity_id="test_1",
            entity_type="device",
            label="MacBook",
            attributes={"cpu_temp": 65},
            confidence=0.9,
            source="perception_ws",
        )
        merged = e1.merge_with(e2)
        assert merged.source == "merged"
        assert merged.confidence == 0.9  # Takes max
        assert "battery" in merged.attributes
        assert "cpu_temp" in merged.attributes

    def test_to_dict(self):
        entity = RealityEntity(
            entity_id="test_1",
            entity_type="concept",
            label="HBLLM",
            attributes={"type": "project"},
            confidence=0.95,
        )
        d = entity.to_dict()
        assert d["entity_id"] == "test_1"
        assert d["label"] == "HBLLM"
        assert d["confidence"] == 0.95
        assert d["expired"] is False


class TestRealityGraph:
    @pytest.fixture
    def graph(self):
        return RealityGraph()

    def test_no_backends(self, graph):
        entity = graph.query_entity("anything")
        assert entity is None

    def test_query_by_type_empty(self, graph):
        results = graph.query_by_type("device")
        assert results == []

    def test_stats_no_backends(self, graph):
        s = graph.stats()
        assert s["backends"] == []

    @pytest.mark.asyncio
    async def test_get_context_empty(self, graph):
        ctx = await graph.get_context("test", "default", 100)
        assert ctx == ""

    def test_get_user_context_empty(self, graph):
        ctx = graph.get_user_context("default")
        assert ctx == ""

    def test_tick(self, graph):
        expired = graph.tick()
        assert expired == 0

    # ── With Mock KG ─────────────────────────────────────────────────

    def test_query_entity_from_kg(self):
        class MockKG:
            def get_entity(self, label):
                if label == "python":
                    return {
                        "entity_type": "concept",
                        "attributes": {"category": "language"},
                        "confidence": 0.9,
                    }
                return None

        graph = RealityGraph(knowledge_graph=MockKG())
        entity = graph.query_entity("python")
        assert entity is not None
        assert entity.source == "kg"
        assert entity.label == "python"

    def test_query_entity_not_in_kg(self):
        class MockKG:
            def get_entity(self, label):
                return None

        graph = RealityGraph(knowledge_graph=MockKG())
        entity = graph.query_entity("nonexistent")
        assert entity is None

    def test_query_entity_kg_with_neighbors(self):
        class MockKG:
            def get_entity(self, label):
                return None

            def neighbors(self, label):
                if label == "hbllm":
                    return {"brain": 0.8, "cognition": 0.7}
                return {}

        graph = RealityGraph(knowledge_graph=MockKG())
        entity = graph.query_entity("hbllm")
        assert entity is not None
        assert entity.attributes.get("neighbors") == 2

    # ── With Mock Brain WS ───────────────────────────────────────────

    def test_query_entity_from_brain_ws(self):
        class MockEntity:
            properties = {"temperature": 22}
            confidence = 0.8
            last_updated = time.time()

        class MockBrainWS:
            _graph = {"room_temp": MockEntity()}

        graph = RealityGraph(brain_world_state=MockBrainWS())
        entity = graph.query_entity("room_temp")
        assert entity is not None
        assert entity.source == "brain_ws"
        assert entity.attributes.get("temperature") == 22

    def test_get_brain_ws_entities(self):
        class MockEntity:
            properties = {"value": 1}
            confidence = 0.9
            last_updated = time.time()

        class MockBrainWS:
            _graph = {"sensor_a": MockEntity(), "sensor_b": MockEntity()}

        graph = RealityGraph(brain_world_state=MockBrainWS())
        entities = graph._get_brain_ws_entities()
        assert len(entities) == 2

    # ── With Mock Perception WS ──────────────────────────────────────

    def test_query_entity_from_perception_ws(self):
        class MockPerceptionWS:
            _iot_devices = {"smart_lamp": {"state": "on", "brightness": 80}}

        graph = RealityGraph(perception_world_state=MockPerceptionWS())
        entity = graph.query_entity("smart_lamp")
        assert entity is not None
        assert entity.source == "perception_ws"
        assert entity.entity_type == "device"

    def test_get_perception_ws_entities(self):
        class MockPerceptionWS:
            _iot_devices = {"lamp": {"state": "on"}, "fan": {"state": "off"}}
            _hardware = {"battery": 85, "cpu": "M2"}

        graph = RealityGraph(perception_world_state=MockPerceptionWS())
        entities = graph._get_perception_ws_entities()
        assert len(entities) == 3  # 2 devices + 1 hardware state

    # ── Multi-Backend Merge ──────────────────────────────────────────

    def test_merged_entity(self):
        class MockKG:
            def get_entity(self, label):
                if label == "macbook":
                    return {
                        "entity_type": "device",
                        "attributes": {"owner": "dumith"},
                        "confidence": 0.8,
                    }
                return None

        class MockPerceptionWS:
            _iot_devices = {"macbook": {"battery": 85, "cpu_temp": 62}}

        graph = RealityGraph(
            knowledge_graph=MockKG(),
            perception_world_state=MockPerceptionWS(),
        )
        entity = graph.query_entity("macbook")
        assert entity is not None
        assert entity.source == "merged"
        assert "owner" in entity.attributes
        assert "battery" in entity.attributes

    # ── Context Generation ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_context_with_kg(self):
        class MockKG:
            def neighbors(self, label):
                if label == "hbllm":
                    return {"brain": 0.8, "architecture": 0.7}
                return {}

        graph = RealityGraph(knowledge_graph=MockKG())
        ctx = await graph.get_context("working on hbllm", "default", 500)
        assert isinstance(ctx, str)

    def test_get_user_context_with_devices(self):
        class MockPerceptionWS:
            _iot_devices = {"smart_lamp": {"state": "on"}}
            _hardware = {}

            def get_summary(self):
                return "Home environment: quiet"

        graph = RealityGraph(perception_world_state=MockPerceptionWS())
        ctx = graph.get_user_context("default")
        assert "Devices" in ctx or "Home" in ctx

    # ── Stats ────────────────────────────────────────────────────────

    def test_stats_with_backends(self):
        class MockKG:
            def stats(self):
                return {"entities": 100}

        class MockBrainWS:
            _graph = {"a": 1, "b": 2}

        graph = RealityGraph(
            knowledge_graph=MockKG(),
            brain_world_state=MockBrainWS(),
        )
        s = graph.stats()
        assert "knowledge_graph" in s["backends"]
        assert "brain_world_state" in s["backends"]
        assert s["brain_ws_entities"] == 2
