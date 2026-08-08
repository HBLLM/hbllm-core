"""Tests for DomainSpec and DomainRegistry purification.

These tests import only from hbllm.modules.domain_registry which has no
StrEnum dependency, so they work on Python 3.10+.
"""

from __future__ import annotations

import warnings

from hbllm.modules.domain_registry import DomainRegistry, DomainSpec


class TestDomainSpecPurified:
    def test_has_cognitive_fields(self) -> None:
        """DomainSpec should have cognitive fields."""
        spec = DomainSpec(name="medical")
        assert hasattr(spec, "ontology")
        assert hasattr(spec, "reasoning_rules")
        assert hasattr(spec, "centroid_text")
        assert hasattr(spec, "weight_multiplier")

    def test_ontology_and_reasoning(self) -> None:
        spec = DomainSpec(
            name="medical",
            ontology=["anatomy", "pharmacology", "diagnosis"],
            reasoning_rules=["require_evidence", "cite_sources"],
        )
        assert len(spec.ontology) == 3
        assert "cite_sources" in spec.reasoning_rules

    def test_adapter_name_still_works(self) -> None:
        """adapter_name is preserved for backward compat but deprecated."""
        spec = DomainSpec(name="coding")
        assert spec.adapter_name == "coding"  # Defaults to name

    def test_hierarchy_preserved(self) -> None:
        spec = DomainSpec(name="coding.python")
        assert spec.parent == "coding"
        assert spec.depth == 2


class TestDomainRegistryPurified:
    def test_resolve_adapter_deprecated(self) -> None:
        """resolve_adapter() should emit deprecation warning."""
        registry = DomainRegistry()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = registry.resolve_adapter("general")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ExecutionOrchestrator" in str(w[0].message)
            assert result == "general"

    def test_resolve_weighted_deprecated(self) -> None:
        """resolve_weighted() should emit deprecation warning."""
        registry = DomainRegistry()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.resolve_weighted({"general": 0.5, "coding": 0.5})
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_cognitive_methods_not_deprecated(self) -> None:
        """Cognitive methods should NOT emit deprecation warnings."""
        registry = DomainRegistry()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.get("general")
            registry.exists("general")
            _ = registry.all_domains
            _ = registry.root_domains
            _ = registry.centroid_texts()
            assert len(w) == 0
