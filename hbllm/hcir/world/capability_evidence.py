"""Domain-General Capability Evidence Model & Verification Ledger.

Implements structured 3-dimensional capability classification:
1. Implementation Status: Code existence, edge cases, typed interfaces.
2. Runtime Integration Status: Invoked directly by active cognitive loop.
3. Empirical Generalization Status: Held-out validation, novel compositions, controlled ablations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ImplementationStatus(StrEnum):
    """Maturity of underlying algorithm and code artifacts."""

    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    GAP = "GAP"


class IntegrationStatus(StrEnum):
    """Runtime wiring into active cognitive loop / decision flow."""

    ACTIVE_RUNTIME = "ACTIVE_RUNTIME"
    OFFLINE_STANDALONE = "OFFLINE_STANDALONE"
    UNWIRED = "UNWIRED"


class UnitTestStatus(StrEnum):
    """Unit test suite status."""

    PASSING = "PASSING"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


class BenchmarkStatus(StrEnum):
    """Status on empirical benchmarks (ARC-AGI-1/2 or ARC-AGI-3)."""

    BENCHMARKED = "BENCHMARKED"
    UNTESTED = "UNTESTED"


class GeneralizationStatus(StrEnum):
    """Generalization to unseen, held-out problem instances."""

    HELD_OUT_EVIDENCED = "HELD_OUT_EVIDENCED"
    IN_SAMPLE_ONLY = "IN_SAMPLE_ONLY"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class CapabilityEvidence:
    """Structured three-tier verification evidence for cognitive world-model capabilities."""

    capability_id: str
    name: str
    domain: str
    implementation_status: str  # IMPLEMENTED | PARTIAL | GAP
    integration_status: str  # ACTIVE_RUNTIME | OFFLINE_STANDALONE | UNWIRED
    unit_test_status: str  # PASSING | PARTIAL | NONE
    benchmark_status: str  # BENCHMARKED | UNTESTED
    generalization_status: str  # HELD_OUT_EVIDENCED | IN_SAMPLE_ONLY | UNVERIFIED
    evidence_refs: list[str] = field(default_factory=list)
    notes: str = ""

    def is_fully_verified(self) -> bool:
        """A capability receives verified status ONLY when implementation, integration, and generalization hold."""
        return (
            self.implementation_status == ImplementationStatus.IMPLEMENTED
            and self.integration_status == IntegrationStatus.ACTIVE_RUNTIME
            and self.unit_test_status == UnitTestStatus.PASSING
            and self.generalization_status == GeneralizationStatus.HELD_OUT_EVIDENCED
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence record to dictionary."""
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "domain": self.domain,
            "implementation_status": self.implementation_status,
            "integration_status": self.integration_status,
            "unit_test_status": self.unit_test_status,
            "benchmark_status": self.benchmark_status,
            "generalization_status": self.generalization_status,
            "fully_verified": self.is_fully_verified(),
            "evidence_refs": list(self.evidence_refs),
            "notes": self.notes,
        }


class CapabilityLedger:
    """Registry and evidence tracker across cognitive capabilities."""

    def __init__(self) -> None:
        self._entries: dict[str, CapabilityEvidence] = {}

    def register(self, record: CapabilityEvidence) -> None:
        """Register or update capability evidence."""
        self._entries[record.capability_id] = record

    def get(self, capability_id: str) -> CapabilityEvidence | None:
        """Retrieve capability evidence record."""
        return self._entries.get(capability_id)

    def summary(self) -> dict[str, Any]:
        """Aggregate summary counts across independent status dimensions."""
        total = len(self._entries)
        impl_counts = {s.value: 0 for s in ImplementationStatus}
        integ_counts = {s.value: 0 for s in IntegrationStatus}
        gen_counts = {s.value: 0 for s in GeneralizationStatus}
        fully_verified = 0

        for entry in self._entries.values():
            impl_counts[entry.implementation_status] = (
                impl_counts.get(entry.implementation_status, 0) + 1
            )
            integ_counts[entry.integration_status] = (
                integ_counts.get(entry.integration_status, 0) + 1
            )
            gen_counts[entry.generalization_status] = (
                gen_counts.get(entry.generalization_status, 0) + 1
            )
            if entry.is_fully_verified():
                fully_verified += 1

        return {
            "total_capabilities": total,
            "implementation": impl_counts,
            "integration": integ_counts,
            "generalization": gen_counts,
            "fully_verified": fully_verified,
        }
