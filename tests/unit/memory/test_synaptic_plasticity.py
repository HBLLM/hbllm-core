"""
Unit and integration tests for Hebbian Plasticity, LTP/LTD reinforcement,
homeostatic decay, split persistence, and self-learning serving routes in HBLLM.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hbllm.memory.semantic import SemanticMemory
from hbllm.serving.api import app


class TestSynapticPlasticityDynamics:
    """Verifies Hebbian weight reinforcement and homeostasis in SemanticMemory."""

    def test_default_synaptic_weights(self):
        mem = SemanticMemory()
        # Default weights: 1.0 for matching pairs, 0.0 for others
        assert mem.synaptic_weights["coding"]["coding"] == pytest.approx(1.0)
        assert mem.synaptic_weights["coding"]["physics"] == pytest.approx(0.0)
        assert mem.synaptic_weights["math"]["math"] == pytest.approx(1.0)

    def test_search_uses_dynamic_synaptic_weights(self):
        mem = SemanticMemory()
        doc_a = mem.store("quantum computing physics", metadata={"domain": "physics"})

        # Artificially alter connection weight between coding category and physics domain
        mem.synaptic_weights["coding"]["physics"] = 0.45

        # Search with coding category primed
        priming_boosts = {"coding": 0.8}
        env = mem.search(
            query="quantum",
            top_k=1,
            priming_boosts=priming_boosts,
            priming_boost_weight=0.15,
            explain=True,
        )

        results = env["results"]
        assert len(results) == 1
        assert results[0]["id"] == doc_a
        # Boost should be priming_boost_weight * weight * potential: 0.15 * 0.45 * 0.8 = 0.054
        assert results[0]["score_breakdown"]["priming_boost"] == pytest.approx(0.054)

    def test_ltp_reinforcement_and_homeostatic_decay(self):
        mem = SemanticMemory()
        doc_a = mem.store("writing rust programs compiler", metadata={"domain": "coding"})

        # Manually lower the matching weight to allow reinforcement headroom
        mem.synaptic_weights["coding"]["coding"] = 0.5

        # Prime coding heavily (0.8) and math slightly (0.4)
        priming_boosts = {"coding": 0.8, "math": 0.4}
        mem.search(query="rust compiler", top_k=1, priming_boosts=priming_boosts)

        # Before feedback, verify starting weights
        assert mem.synaptic_weights["coding"]["coding"] == pytest.approx(0.5)
        assert mem.synaptic_weights["coding"]["math"] == pytest.approx(0.0)

        # 1. Trigger positive feedback (useful=True) -> LTP
        mem.feedback(doc_a, useful=True)

        # Verification of LTP reinforcement for target domain (coding):
        # Homeostatic decay is applied first: 0.5 * (1 - 0.01) = 0.495
        # Then LTP is added: 0.495 + 0.05 * 0.8 * 1.0 = 0.495 + 0.04 = 0.535
        assert mem.synaptic_weights["coding"]["coding"] == pytest.approx(0.535)

        # Verification of Homeostatic Decay on non-reinforced domain connections:
        # Before: 1.0, Decay: 1.0 * (1 - 0.01) = 0.99
        # No LTP since postsynaptic was 'coding' (not 'math')
        assert mem.synaptic_weights["math"]["math"] == pytest.approx(0.99)

    def test_ltd_depression_reinforcement(self):
        mem = SemanticMemory()
        doc_a = mem.store("basic algebra geometry", metadata={"domain": "math"})

        # Prime math at 0.6
        mem.search(query="algebra", top_k=1, priming_boosts={"math": 0.6})

        # 2. Trigger negative feedback (useful=False) -> LTD
        mem.feedback(doc_a, useful=False)

        # Homeostatic decay: 1.0 * (1 - 0.01) = 0.99
        # LTD: 0.99 + 0.05 * 0.6 * (-0.6) = 0.99 - 0.018 = 0.972
        assert mem.synaptic_weights["math"]["math"] == pytest.approx(0.972)


class TestSplitPersistence:
    """Verifies that synaptic weights persist in synaptic_matrix.json separately from documents.json."""

    def test_split_persistence_save_and_load(self, tmp_path):
        mem = SemanticMemory()
        mem.store("test doc", metadata={"domain": "general"})

        # Adjust weights
        mem.synaptic_weights["general"]["coding"] = 0.72

        # Save to disk
        mem.save_to_disk(tmp_path)

        # Verify files on disk
        doc_file = tmp_path / "documents.json"
        syn_file = tmp_path / "synaptic_matrix.json"

        assert doc_file.exists()
        assert syn_file.exists()

        # Verify synaptic weights are not stored in documents.json
        with open(doc_file) as f:
            doc_data = json.load(f)
            assert "synaptic_weights" not in doc_data

        # Verify weights are stored in synaptic_matrix.json
        with open(syn_file) as f:
            syn_data = json.load(f)
            assert syn_data["general"]["coding"] == pytest.approx(0.72)

        # Load from disk
        loaded_mem = SemanticMemory.load_from_disk(tmp_path)
        assert loaded_mem.synaptic_weights["general"]["coding"] == pytest.approx(0.72)


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
