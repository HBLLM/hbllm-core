"""Tests for PIIRedactor — PII detection and redaction."""

import pytest

from hbllm.security.pii_redactor import (
    PIIRedactor,
    PIIType,
    RedactionAction,
    RedactionPolicy,
)


class TestPIIDetection:
    """Tests for PII pattern detection."""

    @pytest.fixture
    def redactor(self):
        return PIIRedactor()

    def test_detect_email(self, redactor):
        text, matches = redactor.redact("Email me at john@example.com please")
        assert len(matches) >= 1
        email_matches = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 1
        assert "[REDACTED_EMAIL]" in text

    def test_detect_phone(self, redactor):
        text, matches = redactor.redact("Call me at 555-123-4567")
        phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) >= 1
        assert "[REDACTED_PHONE]" in text

    def test_detect_ssn(self, redactor):
        text, matches = redactor.redact("SSN: 123-45-6789")
        ssn_matches = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) == 1
        assert "[REDACTED_SSN]" in text

    def test_detect_ip_address(self, redactor):
        policy = RedactionPolicy(ip_address=RedactionAction.REDACT)
        text, matches = redactor.redact("Server at 192.168.1.100", policy=policy)
        ip_matches = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ip_matches) == 1
        assert "[REDACTED_IP_ADDRESS]" in text

    def test_detect_multiple_pii(self, redactor):
        text, matches = redactor.redact("john@example.com, SSN 123-45-6789, call 555-987-6543")
        assert len(matches) >= 3

    def test_no_pii(self, redactor):
        text, matches = redactor.redact("Hello, how are you today?")
        assert len(matches) == 0
        assert text == "Hello, how are you today?"

    def test_empty_text(self, redactor):
        text, matches = redactor.redact("")
        assert text == ""
        assert matches == []

    def test_person_name_default_pass(self, redactor):
        """Person names default to PASS policy (too many false positives)."""
        text, matches = redactor.redact("Contact Dr. Smith today")
        name_matches = [m for m in matches if m.pii_type == PIIType.PERSON_NAME]
        # Names are detected but not redacted by default
        if name_matches:
            assert "Dr. Smith" in text  # Not redacted


class TestRedactionPolicies:
    """Tests for different redaction modes."""

    def test_hash_mode(self):
        policy = RedactionPolicy(email=RedactionAction.HASH)
        redactor = PIIRedactor(default_policy=policy)
        text, matches = redactor.redact("john@example.com")
        assert "[HASH_EMAIL:" in text

    def test_hash_is_deterministic(self):
        policy = RedactionPolicy(email=RedactionAction.HASH)
        redactor = PIIRedactor(default_policy=policy)
        text1, _ = redactor.redact("john@example.com")
        text2, _ = redactor.redact("john@example.com")
        assert text1 == text2

    def test_pass_mode_no_redaction(self):
        policy = RedactionPolicy(email=RedactionAction.PASS)
        redactor = PIIRedactor(default_policy=policy)
        text, matches = redactor.redact("john@example.com")
        assert "john@example.com" in text

    def test_per_tenant_policy(self):
        strict = RedactionPolicy(email=RedactionAction.REDACT)
        relaxed = RedactionPolicy(email=RedactionAction.PASS)
        redactor = PIIRedactor()

        text1, _ = redactor.redact("john@example.com", policy=strict)
        assert "[REDACTED_EMAIL]" in text1

        text2, _ = redactor.redact("john@example.com", policy=relaxed)
        assert "john@example.com" in text2


class TestScanOnly:
    """Tests for scan-only mode (no modification)."""

    def test_scan_only_returns_matches(self):
        redactor = PIIRedactor()
        matches = redactor.scan_only("john@example.com and 555-123-4567")
        assert len(matches) >= 2

    def test_scan_only_no_modification(self):
        redactor = PIIRedactor()
        original = "john@example.com"
        matches = redactor.scan_only(original)
        assert matches[0].original == "john@example.com"


class TestPIIRedactorStats:
    """Tests for telemetry."""

    def test_stats_tracking(self):
        redactor = PIIRedactor()
        redactor.redact("john@example.com")
        redactor.redact("No PII here")
        s = redactor.stats()
        assert s["total_scanned"] == 2
        assert s["total_redacted"] >= 1
