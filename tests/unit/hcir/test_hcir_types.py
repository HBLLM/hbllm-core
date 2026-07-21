"""Unit tests for HCIR Cognitive Type System."""

import pytest
from pydantic import ValidationError

from hbllm.hcir.types import (
    Attention,
    DecayStrategy,
    Provenance,
    ReliabilitySource,
    Scope,
    SecurityLevel,
    UncertaintyVector,
)


class TestUncertaintyVector:
    """Test constrained type validation on UncertaintyVector."""

    def test_defaults(self):
        uv = UncertaintyVector()
        assert uv.confidence == 0.5
        assert uv.freshness_ms == 0
        assert uv.reliability == ReliabilitySource.INFERRED
        assert uv.volatility == 0.0

    def test_valid_boundary_values(self):
        uv = UncertaintyVector(confidence=0.0, volatility=1.0)
        assert uv.confidence == 0.0
        assert uv.volatility == 1.0

    def test_invalid_confidence_above(self):
        with pytest.raises(ValidationError):
            UncertaintyVector(confidence=1.5)

    def test_invalid_confidence_below(self):
        with pytest.raises(ValidationError):
            UncertaintyVector(confidence=-0.1)

    def test_invalid_freshness_negative(self):
        with pytest.raises(ValidationError):
            UncertaintyVector(freshness_ms=-1)

    def test_invalid_volatility_above(self):
        with pytest.raises(ValidationError):
            UncertaintyVector(volatility=2.0)


class TestAttention:
    def test_defaults(self):
        a = Attention()
        assert a.salience == 0.5
        assert a.decay_strategy == DecayStrategy.EXPONENTIAL
        assert a.focus_area == ""

    def test_custom_values(self):
        a = Attention(salience=0.9, focus_area="planning")
        assert a.salience == 0.9
        assert a.focus_area == "planning"

    def test_invalid_salience(self):
        with pytest.raises(ValidationError):
            Attention(salience=1.5)


class TestProvenance:
    def test_defaults(self):
        p = Provenance()
        assert p.created_by == ""
        assert p.simulation_branch == "main"
        assert p.timestamp > 0

    def test_custom(self):
        p = Provenance(created_by="planner_01", engine="gemini-3.5")
        assert p.created_by == "planner_01"


class TestScope:
    def test_defaults(self):
        s = Scope()
        assert s.tenant_id == "default"
        assert s.security_level == SecurityLevel.TENANT

    def test_system_level(self):
        s = Scope(security_level=SecurityLevel.SYSTEM)
        assert s.security_level == SecurityLevel.SYSTEM
