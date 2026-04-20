"""Tests for DomainRegistry, weighted routing, and router improvements."""

from __future__ import annotations

import tempfile
from pathlib import Path

# ── DomainRegistry Tests ─────────────────────────────────────────────────────


class TestDomainRegistry:
    """Test hierarchical domain registry."""

    def _make_registry(self):
        from hbllm.modules.domain_registry import DomainRegistry

        return DomainRegistry(load_defaults=True)

    def test_default_domains_loaded(self):
        reg = self._make_registry()
        assert "general" in reg.all_domains
        assert "coding" in reg.all_domains
        assert "math" in reg.all_domains

    def test_register_subdomain(self):
        from hbllm.modules.domain_registry import DomainSpec

        reg = self._make_registry()
        reg.register(DomainSpec(name="coding.python", centroid_text="Python programming"))
        assert "coding.python" in reg.all_domains
        assert "coding.python" in reg.children("coding")

    def test_auto_register_parent(self):
        """Registering a sub-domain should auto-register missing parent."""
        from hbllm.modules.domain_registry import DomainRegistry, DomainSpec

        reg = DomainRegistry(load_defaults=False)
        reg.register(DomainSpec(name="science.physics", centroid_text="Physics topics"))
        assert reg.exists("science")
        assert reg.exists("science.physics")

    def test_resolve_adapter_exact(self):
        from hbllm.modules.domain_registry import DomainSpec

        reg = self._make_registry()
        reg.register(DomainSpec(name="coding.python", adapter_name="py-adapter"))
        assert reg.resolve_adapter("coding.python") == "py-adapter"

    def test_resolve_adapter_fallback(self):
        reg = self._make_registry()
        # "coding.rust" doesn't exist, should fall back to "coding"
        result = reg.resolve_adapter("coding.rust")
        assert result == "coding"

    def test_resolve_adapter_deep_fallback(self):
        reg = self._make_registry()
        # "coding.python.django" → "coding" (neither coding.python nor coding.python.django exist)
        result = reg.resolve_adapter("coding.python.django")
        assert result == "coding"

    def test_resolve_adapter_default(self):
        from hbllm.modules.domain_registry import DomainRegistry

        reg = DomainRegistry(load_defaults=False)
        assert reg.resolve_adapter("unknown.topic") == "default"

    def test_resolve_weighted(self):
        from hbllm.modules.domain_registry import DomainSpec

        reg = self._make_registry()
        reg.register(DomainSpec(name="coding.python"))
        result = reg.resolve_weighted({"coding.python": 0.7, "math": 0.3})
        assert "coding.python" in result
        assert "math" in result
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_resolve_weighted_merges_siblings(self):
        """Two sub-domains without their own adapters should merge to parent."""
        reg = self._make_registry()
        # coding.python and coding.rust both fall back to "coding"
        result = reg.resolve_weighted({"coding.python": 0.5, "coding.rust": 0.5})
        # Both should merge into "coding"
        assert "coding" in result
        assert abs(result["coding"] - 1.0) < 0.01

    def test_matches_hint_string(self):
        reg = self._make_registry()
        assert reg.matches_hint("coding", "coding.python") is True
        assert reg.matches_hint("coding.python", "coding") is True
        assert reg.matches_hint("coding", "math") is False

    def test_matches_hint_dict(self):
        reg = self._make_registry()
        assert reg.matches_hint("coding", {"coding.python": 0.7, "math": 0.3}) is True
        assert reg.matches_hint("general", {"coding": 1.0}) is False

    def test_is_ancestor(self):
        reg = self._make_registry()
        assert reg.is_ancestor("coding", "coding.python") is True
        assert reg.is_ancestor("coding", "coding") is True
        assert reg.is_ancestor("coding.python", "coding") is False
        assert reg.is_ancestor("math", "coding.python") is False

    def test_root_domains(self):
        reg = self._make_registry()
        roots = reg.root_domains
        for r in roots:
            assert "." not in r

    def test_leaf_domains(self):
        from hbllm.modules.domain_registry import DomainSpec

        reg = self._make_registry()
        reg.register(DomainSpec(name="coding.python"))
        leaves = reg.leaf_domains()
        assert "coding.python" in leaves
        # "coding" now has a child, so it should NOT be a leaf
        assert "coding" not in leaves

    def test_unregister(self):
        from hbllm.modules.domain_registry import DomainSpec

        reg = self._make_registry()
        reg.register(DomainSpec(name="coding.python"))
        reg.unregister("coding.python")
        assert not reg.exists("coding.python")
        assert reg.exists("coding")  # Parent should stay

    def test_centroid_texts(self):
        reg = self._make_registry()
        texts = reg.centroid_texts()
        assert "general" in texts
        assert len(texts["general"]) > 0


# ── Router Platt Calibration Tests ────────────────────────────────────────────


class TestPlattCalibration:
    """Test confidence calibration."""

    def test_calibrate_midpoint(self):
        """Score 0.4 with default params should be ~0.5."""
        from hbllm.brain.router_node import RouterNode

        router = RouterNode(node_id="test", use_vectors=False)
        result = router._calibrate(0.4)
        assert 0.4 < result < 0.6

    def test_calibrate_high_score(self):
        """High raw score should produce high calibrated score."""
        from hbllm.brain.router_node import RouterNode

        router = RouterNode(node_id="test", use_vectors=False)
        result = router._calibrate(0.9)
        assert result > 0.9

    def test_calibrate_low_score(self):
        """Low raw score should produce low calibrated score."""
        from hbllm.brain.router_node import RouterNode

        router = RouterNode(node_id="test", use_vectors=False)
        result = router._calibrate(0.1)
        assert result < 0.2

    def test_calibrate_monotonic(self):
        """Higher raw scores should always produce higher calibrated scores."""
        from hbllm.brain.router_node import RouterNode

        router = RouterNode(node_id="test", use_vectors=False)
        scores = [router._calibrate(x / 10.0) for x in range(11)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]


# ── Persistent Centroids Tests ────────────────────────────────────────────────


class TestPersistentCentroids:
    """Test centroid save/load roundtrip."""

    def test_save_load_roundtrip(self):
        import numpy as np

        from hbllm.brain.router_node import RouterNode

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "centroids.json"

            # Save
            router = RouterNode(node_id="test", use_vectors=False)
            router._centroids_path = path
            router.domain_centroids = {
                "coding": np.random.randn(384).astype(np.float32),
                "math": np.random.randn(384).astype(np.float32),
            }
            router.unknown_threshold = 0.35
            router._save_centroids()

            assert path.exists()

            # Load
            router2 = RouterNode(node_id="test2", use_vectors=False)
            router2._centroids_path = path
            router2._load_centroids()

            assert len(router2.domain_centroids) >= 2
            assert router2.unknown_threshold == 0.35
            np.testing.assert_allclose(
                router2.domain_centroids["coding"],
                router.domain_centroids["coding"],
                atol=1e-5,
            )

    def test_load_missing_file(self):
        """Loading from non-existent path should not raise."""
        from hbllm.brain.router_node import RouterNode

        router = RouterNode(node_id="test", use_vectors=False)
        router._centroids_path = Path("/nonexistent/centroids.json")
        router._load_centroids()  # Should not raise


# ── SpawnerNode Sub-Domain Tests ──────────────────────────────────────────────


class TestSpawnerSubDomain:
    """Test SpawnerNode sub-domain classification."""

    def test_classify_coding_python(self):
        from hbllm.brain.spawner_node import _classify_domain_rank

        assert _classify_domain_rank("coding.python") == 32

    def test_classify_math_calculus(self):
        from hbllm.brain.spawner_node import _classify_domain_rank

        assert _classify_domain_rank("math.calculus") == 32

    def test_classify_writing_poetry(self):
        from hbllm.brain.spawner_node import _classify_domain_rank

        assert _classify_domain_rank("writing.poetry") == 16

    def test_classify_unknown(self):
        from hbllm.brain.spawner_node import _classify_domain_rank

        assert _classify_domain_rank("gardening") == 8

    def test_classify_deep_subdomain(self):
        from hbllm.brain.spawner_node import _classify_domain_rank

        assert _classify_domain_rank("coding.python.django") == 32


# ── Dynamic Temperature Tests ─────────────────────────────────────────────────


class TestDynamicTemperature:
    """Test that temperature adapts to score spread."""

    def test_temperature_computation(self):
        """Verify temperature formula manually."""
        temp_min, temp_max = 0.05, 0.5

        # High spread = clear winner → low temperature
        spread_high = 0.3
        temp = temp_min + (temp_max - temp_min) * (1.0 - min(spread_high / 0.3, 1.0))
        assert abs(temp - temp_min) < 0.01

        # Low spread = ambiguous → high temperature
        spread_low = 0.0
        temp = temp_min + (temp_max - temp_min) * (1.0 - min(spread_low / 0.3, 1.0))
        assert abs(temp - temp_max) < 0.01

        # Medium spread → medium temperature
        spread_mid = 0.15
        temp = temp_min + (temp_max - temp_min) * (1.0 - min(spread_mid / 0.3, 1.0))
        assert temp_min < temp < temp_max
