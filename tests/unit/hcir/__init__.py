"""Unit tests for HCIR Cognitive Type System."""

import pytest
from pydantic import ValidationError

from hbllm.hcir.types import (
    Attention,
    Confidence,
    DecayStrategy,
    Priority,
    Provenance,
    ReliabilitySource,
    Scope,
    SecurityLevel,
    UncertaintyVector,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constrained Scalar Types
# ═══════════════════════════════════════════════════════════════════════════


class TestConstrainedTypes:
    """Test that constrained types reject out-of-range values."""

    def test_uncertainty_vector_defaults(self):
        uv = UncertaintyVector()
        assert uv.confidence == 0.5
        assert uv.freshness_ms == 0
        assert uv.reliability == ReliabilitySource.INFERRED
        assert uv.volatility == 0.0

    def test_uncertainty_vector_confidence_range(self):
        uv = UncertaintyVector(confidence=0.0)
        assert uv.confidence == 0.0
        uv = UncertaintyVector(confidence=1.0)
        assert uv.confidence == 1.0

    def test_uncertainty_vector_invalid_confidence(self):
        with pytest.raises(ValidationError):
            UncertaintyVector(confidence=1.5)
        with pytest.raises(ValidationError):
            UncertaintyVector(confidence=-0.1)

    def test_uncertainty_vector_invalid_freshness(self):
        with pytest.raises(ValidationError):
            UncertaintyVector(freshness_ms=-1)

    def test_uncertainty_vector_invalid_volatility(self):
        with pytest.raises(ValidationError):
            UncertaintyVector(volatility=2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Attention
# ═══════════════════════════════════════════════════════════════════════════


class TestAttention:
    def test_defaults(self):
        a = Attention()
        assert a.salience == 0.5
        assert a.activation == 0.5
        assert a.decay_rate == 0.05
        assert a.decay_strategy == DecayStrategy.EXPONENTIAL
        assert a.focus_area == ""

    def test_custom_values(self):
        a = Attention(
            salience=0.9,
            activation=0.8,
            decay_rate=0.1,
            decay_strategy=DecayStrategy.LINEAR,
            focus_area="planning",
        )
        assert a.salience == 0.9
        assert a.focus_area == "planning"

    def test_invalid_salience(self):
        with pytest.raises(ValidationError):
            Attention(salience=1.5)


# ═══════════════════════════════════════════════════════════════════════════
# Provenance
# ═══════════════════════════════════════════════════════════════════════════


class TestProvenance:
    def test_defaults(self):
        p = Provenance()
        assert p.created_by == ""
        assert p.engine == ""
        assert p.simulation_branch == "main"
        assert p.reasoning_step == 0
        assert p.timestamp > 0

    def test_custom(self):
        p = Provenance(
            created_by="planner_node",
            engine="gemini-3.5-flash",
            prompt_hash="abc123",
        )
        assert p.created_by == "planner_node"
        assert p.engine == "gemini-3.5-flash"


# ═══════════════════════════════════════════════════════════════════════════
# Scope
# ═══════════════════════════════════════════════════════════════════════════


class TestScope:
    def test_defaults(self):
        s = Scope()
        assert s.tenant_id == "default"
        assert s.security_level == SecurityLevel.TENANT

    def test_system_scope(self):
        s = Scope(security_level=SecurityLevel.SYSTEM)
        assert s.security_level == SecurityLevel.SYSTEM

    def test_simulation_scope(self):
        s = Scope(simulation_id="sim_branch_42")
        assert s.simulation_id == "sim_branch_42"
