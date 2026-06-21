"""PII Redactor — automatic personally identifiable information protection.

Scans text before memory storage to detect and redact sensitive information:
    - Email addresses
    - Phone numbers (international formats)
    - Social Security Numbers (US format)
    - Credit card numbers (Luhn-validated)
    - IP addresses (v4 and v6)
    - Named entities via regex heuristics (names following salutation patterns)

Configurable per-tenant redaction policies:
    - REDACT:  replace with [REDACTED_TYPE]
    - HASH:    replace with deterministic hash (allows dedup)
    - ENCRYPT: replace with reversible Fernet token (owner can decrypt)
    - PASS:    no redaction (trusted internal data)

All redaction events are logged for auditability.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── PII Types ────────────────────────────────────────────────────────────


class PIIType(str, Enum):
    """Supported PII categories."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PERSON_NAME = "person_name"


# ── Redaction Policies ───────────────────────────────────────────────────


class RedactionAction(str, Enum):
    """What to do when PII is detected."""

    REDACT = "redact"  # Replace with [REDACTED_TYPE]
    HASH = "hash"  # Replace with deterministic hash
    ENCRYPT = "encrypt"  # Replace with Fernet-encrypted token
    PASS = "pass"  # No redaction


@dataclass
class RedactionPolicy:
    """Per-tenant redaction configuration.

    Default: REDACT everything except PASS for person_name (too many false positives).
    """

    email: RedactionAction = RedactionAction.REDACT
    phone: RedactionAction = RedactionAction.REDACT
    ssn: RedactionAction = RedactionAction.REDACT
    credit_card: RedactionAction = RedactionAction.REDACT
    ip_address: RedactionAction = RedactionAction.HASH
    person_name: RedactionAction = RedactionAction.PASS

    def get_action(self, pii_type: PIIType) -> RedactionAction:
        """Get the redaction action for a PII type."""
        return getattr(self, pii_type.value, RedactionAction.REDACT)


@dataclass
class PIIMatch:
    """A detected PII occurrence in text."""

    pii_type: PIIType
    original: str
    start: int
    end: int
    replacement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.pii_type.value,
            "start": self.start,
            "end": self.end,
            "replacement": self.replacement,
            # Never log the original PII value
        }


# ── Regex Patterns ───────────────────────────────────────────────────────

# Email: standard RFC-like pattern
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Phone: international formats with optional country code
_PHONE_RE = re.compile(
    r"(?<!\d)"  # not preceded by digit
    r"(?:\+?\d{1,3}[-.\s]?)?"  # optional country code
    r"(?:\(?\d{2,4}\)?[-.\s]?)"  # area code
    r"(?:\d{3,4}[-.\s]?)"  # first group
    r"\d{3,4}"  # last group
    r"(?!\d)"  # not followed by digit
)

# SSN: XXX-XX-XXXX format
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit card: 13-19 digits, optionally space/dash separated
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# IPv4
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 (simplified — common formats)
_IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b")

# Person name heuristic: salutation + capitalized words
_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)


# ── Luhn Validation ──────────────────────────────────────────────────────


def _luhn_check(number: str) -> bool:
    """Validate a number string using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False

    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ── PIIRedactor ──────────────────────────────────────────────────────────


class PIIRedactor:
    """Scans and redacts PII from text before memory storage.

    Usage::

        redactor = PIIRedactor()

        # Redact with default policy
        clean_text, matches = redactor.redact(
            "Call me at john@example.com or 555-123-4567"
        )
        # clean_text = "Call me at [REDACTED_EMAIL] or [REDACTED_PHONE]"

        # Custom per-tenant policy
        policy = RedactionPolicy(email=RedactionAction.HASH)
        clean_text, matches = redactor.redact(text, policy=policy)
    """

    def __init__(
        self,
        default_policy: RedactionPolicy | None = None,
        hash_salt: str = "hbllm_pii_salt",
        encryption_key: bytes | None = None,
    ) -> None:
        self.default_policy = default_policy or RedactionPolicy()
        self._hash_salt = hash_salt
        self._encryption_key = encryption_key
        self._fernet = None

        # Initialize Fernet if encryption key is provided
        if encryption_key:
            try:
                from cryptography.fernet import Fernet  # type: ignore[import-not-found]

                self._fernet = Fernet(encryption_key)
            except ImportError:
                logger.warning(
                    "cryptography package not installed — ENCRYPT policy "
                    "will fall back to HASH. Install: pip install cryptography"
                )

        # Telemetry
        self._total_scanned = 0
        self._total_redacted = 0
        self._by_type: dict[str, int] = {}

    def redact(
        self,
        text: str,
        policy: RedactionPolicy | None = None,
        tenant_id: str = "default",
    ) -> tuple[str, list[PIIMatch]]:
        """Scan text for PII and apply redaction policy.

        Args:
            text: The input text to scan.
            policy: Redaction policy (uses default if None).
            tenant_id: For audit logging.

        Returns:
            Tuple of (redacted_text, list of PIIMatch detections).
        """
        if not text or not text.strip():
            return text, []

        self._total_scanned += 1
        pol = policy or self.default_policy

        # Detect all PII
        matches = self._detect_all(text)

        if not matches:
            return text, []

        # Apply redaction (process from end to preserve indices)
        matches.sort(key=lambda m: m.start, reverse=True)
        redacted = text

        for match in matches:
            action = pol.get_action(match.pii_type)
            if action == RedactionAction.PASS:
                continue

            replacement = self._apply_action(match.original, match.pii_type, action)
            match.replacement = replacement
            redacted = redacted[: match.start] + replacement + redacted[match.end :]

            self._total_redacted += 1
            self._by_type[match.pii_type.value] = self._by_type.get(match.pii_type.value, 0) + 1

        logger.debug(
            "PII scan: tenant=%s, found=%d, redacted=%d",
            tenant_id,
            len(matches),
            sum(1 for m in matches if m.replacement),
        )

        return redacted, matches

    def _detect_all(self, text: str) -> list[PIIMatch]:
        """Detect all PII occurrences in text."""
        matches: list[PIIMatch] = []
        seen_ranges: set[tuple[int, int]] = set()

        def _add_matches(regex: re.Pattern[str], pii_type: PIIType) -> None:
            for m in regex.finditer(text):
                span = (m.start(), m.end())
                # Skip overlapping matches
                if any(s <= span[0] < e or s < span[1] <= e for s, e in seen_ranges):
                    continue
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        original=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )
                seen_ranges.add(span)

        # Order matters — more specific patterns first
        _add_matches(_SSN_RE, PIIType.SSN)
        _add_matches(_EMAIL_RE, PIIType.EMAIL)

        # Credit cards: validate with Luhn to reduce false positives
        for m in _CC_RE.finditer(text):
            digits_only = re.sub(r"\D", "", m.group())
            if _luhn_check(digits_only) and len(digits_only) >= 13:
                span = (m.start(), m.end())
                if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_ranges):
                    matches.append(
                        PIIMatch(
                            pii_type=PIIType.CREDIT_CARD,
                            original=m.group(),
                            start=m.start(),
                            end=m.end(),
                        )
                    )
                    seen_ranges.add(span)

        _add_matches(_PHONE_RE, PIIType.PHONE)
        _add_matches(_IPV4_RE, PIIType.IP_ADDRESS)
        _add_matches(_IPV6_RE, PIIType.IP_ADDRESS)
        _add_matches(_NAME_RE, PIIType.PERSON_NAME)

        return matches

    def _apply_action(
        self,
        original: str,
        pii_type: PIIType,
        action: RedactionAction,
    ) -> str:
        """Apply a redaction action to a PII value."""
        if action == RedactionAction.REDACT:
            return f"[REDACTED_{pii_type.value.upper()}]"

        elif action == RedactionAction.HASH:
            salted = f"{self._hash_salt}:{original}"
            h = hashlib.sha256(salted.encode()).hexdigest()[:16]
            return f"[HASH_{pii_type.value.upper()}:{h}]"

        elif action == RedactionAction.ENCRYPT:
            if self._fernet:
                encrypted = self._fernet.encrypt(original.encode()).decode()
                return f"[ENC_{pii_type.value.upper()}:{encrypted}]"
            else:
                # Fall back to hash if Fernet not available
                return self._apply_action(original, pii_type, RedactionAction.HASH)

        return original  # PASS

    def decrypt(self, token: str) -> str | None:
        """Decrypt an encrypted PII value (owner use only).

        Args:
            token: The [ENC_TYPE:...] token to decrypt.

        Returns:
            The original PII value, or None if decryption fails.
        """
        if not self._fernet:
            return None

        # Extract the encrypted payload
        match = re.match(r"\[ENC_[A-Z_]+:(.+)\]", token)
        if not match:
            return None

        try:
            return self._fernet.decrypt(match.group(1).encode()).decode()
        except Exception:
            return None

    def scan_only(self, text: str) -> list[PIIMatch]:
        """Detect PII without redacting (for reporting/preview)."""
        return self._detect_all(text)

    def stats(self) -> dict[str, Any]:
        """Redactor statistics."""
        return {
            "total_scanned": self._total_scanned,
            "total_redacted": self._total_redacted,
            "by_type": dict(self._by_type),
            "default_policy": {
                pii.value: self.default_policy.get_action(pii).value for pii in PIIType
            },
        }
