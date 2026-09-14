"""Integration and serving endpoint tests for Studio SNN and learning routes.

Decoupled from unit/memory and unit/snn to maintain strict layer separation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hbllm.network.metrics import MetricsCollector
from hbllm.serving.api import app


class TestPlasticityServingEndpoints:
    """Verifies FastAPI serving routes for self-learning in studio.py."""

    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "development")
        from hbllm.serving.state import _state

        monkeypatch.setitem(_state, "brain", None)
        monkeypatch.setitem(_state, "synapse_gateway", None)
        return TestClient(app)

    def test_studio_learning_weights(self, client):
        response = client.get("/studio/learning")
        assert response.status_code == 200
        data = response.json()

        assert "learner" in data
        assert "synaptic_weights" in data["learner"]
        # Default weight for coding-coding should be present and equal 1.0
        assert data["learner"]["synaptic_weights"]["coding"]["coding"] == pytest.approx(1.0)

    def test_studio_learning_reset_weights(self, client):
        response = client.post("/studio/learning/reset_weights")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestSNNServingEndpoints:
    """Verifies FastAPI serving routes for SNN studio."""

    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "development")
        from hbllm.serving.state import _state

        monkeypatch.setitem(_state, "brain", None)
        monkeypatch.setitem(_state, "synapse_gateway", None)
        return TestClient(app)

    def test_snn_status_endpoint(self, client):
        MetricsCollector.reset()
        collector = MetricsCollector.get_instance()
        collector.record_snn_potential("priming_coding", 0.65)
        collector.record_snn_potential("human_attention_fatigue", 0.35)

        response = client.get("/api/snn/status")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "priming_categories" in data
        assert "attention_fatigue" in data

        coding_data = data["priming_categories"]["coding"]
        assert coding_data["potential"] == pytest.approx(0.65)
        assert len(coding_data["history"]) > 0

        attn_data = data["attention_fatigue"]
        assert attn_data["potential"] == pytest.approx(0.35)

    def test_snn_stimulate_endpoint(self, client):
        response = client.post("/api/snn/stimulate", json={"category": "coding", "charge": 0.5})
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"

        # Verify the potential increased in metrics
        collector = MetricsCollector.get_instance()
        pot = collector._mem_gauges.get("snn_potential:priming_coding", 0.0)
        assert pot > 0.0

    def test_snn_replay_endpoint(self, client):
        # When brain is not loaded in TestClient, it runs the mock fallback comparison path
        payload = {
            "query": "quantum loop gravity coding",
            "priming_state": {"coding": 0.8, "physics": 0.4},
        }
        response = client.post("/api/snn/replay", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "unprimed" in data
        assert "primed" in data
        assert "differentials" in data
        assert len(data["differentials"]) > 0
