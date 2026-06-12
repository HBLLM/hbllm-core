"""
Unit and integration tests for SNN Telemetry Caching, Explainable Retrieval,
Ranking Differential Inspector, and SNN Studio serving routes.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hbllm.brain.snn import LIFConfig
from hbllm.memory.priming import WorkingMemoryPrimer
from hbllm.memory.semantic import SemanticMemory
from hbllm.network.metrics import MetricsCollector
from hbllm.serving.api import app


class TestSNNTelemetryCaching:
    """Verifies MetricsCollector potential history caching and deque limit truncation."""

    def test_potential_history_caching_and_truncation(self):
        # Reset to ensure fresh instance
        MetricsCollector.reset()
        collector = MetricsCollector.get_instance()

        # Cache rolling potentials for a test neuron
        neuron_id = "test_rolling_neuron"
        for val in range(120):
            collector.record_snn_potential(neuron_id, val * 0.01)

        history = collector.get_snn_history(neuron_id)
        # Deque capacity is capped at 100
        assert len(history) == 100

        # Assert correct order (oldest popped, newest kept)
        assert history[0]["potential"] == pytest.approx(0.20)
        assert history[-1]["potential"] == pytest.approx(1.19)


class TestExplainableRetrievalAndDifferentials:
    """Verifies SemanticMemory search score breakdowns and ranking differentials."""

    def test_search_score_breakdown(self):
        mem = SemanticMemory()

        doc_a = mem.store(
            "quantum computing theory gravity", metadata={"domain": "physics", "usefulness": 5}
        )
        mem.store("quantum calculus equations theorem", metadata={"domain": "math"})

        # Search with explain=True
        priming_boosts = {"physics": 0.8, "math": 0.0}
        env = mem.search(
            query="quantum",
            top_k=2,
            priming_boosts=priming_boosts,
            priming_boost_weight=0.15,
            explain=True,
        )

        assert isinstance(env, dict)
        assert "results" in env
        assert "explanations" in env
        assert "global_stats" in env

        results = env["results"]
        explanations = env["explanations"]

        assert len(results) == 2
        # Verify first document is physics
        assert results[0]["id"] == doc_a
        assert "score_breakdown" in results[0]

        breakdown = results[0]["score_breakdown"]
        assert breakdown["similarity"] > 0.0
        assert breakdown["usefulness_boost"] > 0.0
        assert breakdown["priming_boost"] == pytest.approx(0.12)  # 0.8 * 0.15

        # Verify explanations list structure
        assert explanations[0]["doc_id"] == doc_a
        assert explanations[0]["domain"] == "physics"

    def test_ranking_differential_inspector(self):
        mem = SemanticMemory()

        results = [
            {
                "id": "doc_winner",
                "content": "Winner content",
                "score": 0.85,
                "score_breakdown": {
                    "similarity": 0.70,
                    "usefulness_boost": 0.03,
                    "reward_boost": 0.0,
                    "priming_boost": 0.12,
                },
            },
            {
                "id": "doc_runnerup",
                "content": "Runner content",
                "score": 0.80,
                "score_breakdown": {
                    "similarity": 0.78,
                    "usefulness_boost": 0.02,
                    "reward_boost": 0.0,
                    "priming_boost": 0.0,
                },
            },
        ]

        differentials = mem.get_ranking_differential(results)
        assert len(differentials) == 1

        diff = differentials[0]
        assert diff["doc_a_id"] == "doc_winner"
        assert diff["doc_b_id"] == "doc_runnerup"
        assert diff["deltas"]["total"] == pytest.approx(0.05)
        # Winner has lower similarity (-0.08) but higher priming (+0.12)
        assert diff["deltas"]["similarity"] == pytest.approx(-0.08)
        assert diff["deltas"]["priming"] == pytest.approx(0.12)

        # Verify natural language explanation details
        assert "SNN priming bias advantage" in diff["explanation"]
        assert "lower base similarity" in diff["explanation"]


class TestSNNServingEndpoints:
    """Verifies FastAPI serving routes in studio.py."""

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
