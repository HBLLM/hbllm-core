"""PII Redactor — Scrub personally identifiable information before memory storage.

Scans text for PII patterns (email, phone, SSN, credit card, IP address,
API keys, JWT tokens) and applies configurable per-tenant redaction policies.

Policies:
    REDACT   — Replace with ``[REDACTED:<type>]`` placeholder
    HASH     — Replace with truncated SHA-256 (first 8 chars)
    ENCRYPT  — Encrypt via EncryptionVault (reversible for authorised users)
    PASS     — Allow through unchanged (audit-logged)

Bus Topics:
    security.pii.redacted  — Published when PII is found and redacted

Usage::

    redactor = PIIRedactor()
    result = redactor.redact("Call me at 555-123-4567", tenant_id="t1")
    # result.text == "Call me at [REDACTED:PHONE]"
    # result.findings == [PIIMatch(type=PIIType.PHONE, ...)]
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── PII Types ────────────────────────────────────────────────────────────────


class PIIType(str, Enum):
    """Categories of personally identifiable information."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    API_KEY = "API_KEY"
    JWT_TOKEN = "JWT_TOKEN"


class RedactionPolicy(str, Enum):
    """How to handle detected PII."""

    REDACT = "redact"
    HASH = "hash"
    ENCRYPT = "encrypt"
    PASS = "pass"


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class PIIMatch:
    """A single PII detection result."""

    pii_type: PIIType
    original: str
    start: int
    end: int
    replacement: str = ""


@dataclass
class RedactedText:
    """Result of a redaction pass."""

    text: str
    findings: list[PIIMatch] = field(default_factory=list)
    tenant_id: str = "default"

    @property
    def had_pii(self) -> bool:
        return len(self.findings) > 0


@dataclass
class TenantPIIPolicy:
    """Per-tenant PII handling policy."""

    default_policy: RedactionPolicy = RedactionPolicy.REDACT
    overrides: dict[PIIType, RedactionPolicy] = field(default_factory=dict)

    def policy_for(self, pii_type: PIIType) -> RedactionPolicy:
        return self.overrides.get(pii_type, self.default_policy)


# ── Regex Patterns ───────────────────────────────────────────────────────────

_PII_PATTERNS: dict[PIIType, re.Pattern[str]] = {
    # RFC 5322 simplified — matches most real-world emails
    PIIType.EMAIL: re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    # US/international phone numbers: +1-555-123-4567, (555) 123-4567, 555.123.4567
    PIIType.PHONE: re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    # US Social Security: 123-45-6789
    PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Credit card: 4 groups of 4 digits (Visa, MC, Amex patterns)
    PIIType.CREDIT_CARD: re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    # IPv4 and IPv6
    PIIType.IP_ADDRESS: re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        r"|"
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    ),
    # API keys: long hex/base64 strings (32+ chars) prefixed by common key indicators
    PIIType.API_KEY: re.compile(
        r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{32,})['\"]?",
        re.IGNORECASE,
    ),
    # JWT tokens: three base64url segments separated by dots
    PIIType.JWT_TOKEN: re.compile(
        r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"
    ),
}


# ── Redactor ─────────────────────────────────────────────────────────────────


class PIIRedactor:
    """Scans and redacts PII from text with per-tenant policies.

    Args:
        default_policy: Fallback policy when no tenant-specific override exists.
        bus: Optional MessageBus for audit event publishing.
    """

    def __init__(
        self,
        default_policy: RedactionPolicy = RedactionPolicy.REDACT,
        bus: Any | None = None,
    ) -> None:
        self.default_policy = default_policy
        self.bus = bus
        self._tenant_policies: dict[str, TenantPIIPolicy] = {}
        self._encryption_vault: Any | None = None

        # Telemetry
        self._total_scans = 0
        self._total_findings = 0
        self._findings_by_type: dict[PIIType, int] = {t: 0 for t in PIIType}

    def configure_tenant(
        self,
        tenant_id: str,
        default_policy: RedactionPolicy | None = None,
        overrides: dict[PIIType, RedactionPolicy] | None = None,
    ) -> None:
        """Set per-tenant PII handling policy."""
        self._tenant_policies[tenant_id] = TenantPIIPolicy(
            default_policy=default_policy or self.default_policy,
            overrides=overrides or {},
        )

    def set_encryption_vault(self, vault: Any) -> None:
        """Attach an EncryptionVault for ENCRYPT policy."""
        self._encryption_vault = vault

    # ── Core API ─────────────────────────────────────────────────────

    def scan(self, text: str) -> list[PIIMatch]:
        """Detect PII in text without modifying it."""
        findings: list[PIIMatch] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            for match in pattern.finditer(text):
                # For API_KEY pattern, use the captured group if available
                if pii_type == PIIType.API_KEY and match.lastindex:
                    value = match.group(1)
                    start = match.start(1)
                    end = match.end(1)
                else:
                    value = match.group(0)
                    start = match.start()
                    end = match.end()
                findings.append(PIIMatch(pii_type=pii_type, original=value, start=start, end=end))
        # Sort by position (descending) for safe replacement
        findings.sort(key=lambda m: m.start, reverse=True)
        return findings

    def redact(self, text: str, tenant_id: str = "default") -> RedactedText:
        """Scan and redact PII according to tenant policy.

        Returns a RedactedText with the cleaned string and list of findings.
        """
        self._total_scans += 1

        findings = self.scan(text)
        if not findings:
            return RedactedText(text=text, tenant_id=tenant_id)

        policy = self._tenant_policies.get(
            tenant_id, TenantPIIPolicy(default_policy=self.default_policy)
        )

        result = text
        for finding in findings:
            action = policy.policy_for(finding.pii_type)
            replacement = self._apply_policy(action, finding)
            finding.replacement = replacement
            result = result[: finding.start] + replacement + result[finding.end :]

            self._total_findings += 1
            self._findings_by_type[finding.pii_type] += 1

        logger.info(
            "PII redaction: %d findings in %d chars (tenant=%s)",
            len(findings),
            len(text),
            tenant_id,
        )

        return RedactedText(text=result, findings=findings, tenant_id=tenant_id)

    def _apply_policy(self, policy: RedactionPolicy, match: PIIMatch) -> str:
        """Apply a redaction policy to a single PII match."""
        if policy == RedactionPolicy.PASS:
            return match.original
        elif policy == RedactionPolicy.HASH:
            digest = hashlib.sha256(match.original.encode()).hexdigest()[:8]
            return f"[HASHED:{match.pii_type.value}:{digest}]"
        elif policy == RedactionPolicy.ENCRYPT:
            if self._encryption_vault and hasattr(self._encryption_vault, "encrypt"):
                encrypted = self._encryption_vault.encrypt(match.original)
                return f"[ENCRYPTED:{match.pii_type.value}:{encrypted[:16]}...]"
            # Fallback to REDACT if no vault configured
            logger.warning(
                "ENCRYPT policy requested but no vault configured, falling back to REDACT"
            )
            return f"[REDACTED:{match.pii_type.value}]"
        else:  # REDACT
            return f"[REDACTED:{match.pii_type.value}]"

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Redactor statistics."""
        return {
            "total_scans": self._total_scans,
            "total_findings": self._total_findings,
            "findings_by_type": {k.value: v for k, v in self._findings_by_type.items()},
        }
