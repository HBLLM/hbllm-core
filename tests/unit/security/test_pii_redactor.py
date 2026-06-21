"""Tests for PIIRedactor — PII detection and redaction."""

import pytest

from hbllm.security.pii_redactor import (
    PIIRedactor,
    PIIType,
    RedactionPolicy,
)


@pytest.fixture
def redactor():
    return PIIRedactor()


# ── Detection Tests ──────────────────────────────────────────────────────────


class TestPIIDetection:
    def test_detect_email(self, redactor):
        findings = redactor.scan("Contact john.doe@example.com for info")
        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.EMAIL
        assert findings[0].original == "john.doe@example.com"

    def test_detect_phone_us(self, redactor):
        findings = redactor.scan("Call 555-123-4567 today")
        assert any(f.pii_type == PIIType.PHONE for f in findings)

    def test_detect_phone_international(self, redactor):
        findings = redactor.scan("Reach me at +1-555-123-4567")
        assert any(f.pii_type == PIIType.PHONE for f in findings)

    def test_detect_ssn(self, redactor):
        findings = redactor.scan("SSN: 123-45-6789")
        assert any(f.pii_type == PIIType.SSN for f in findings)

    def test_detect_credit_card(self, redactor):
        findings = redactor.scan("Card: 4111 1111 1111 1111")
        assert any(f.pii_type == PIIType.CREDIT_CARD for f in findings)

    def test_detect_ipv4(self, redactor):
        findings = redactor.scan("Server at 192.168.1.100")
        assert any(f.pii_type == PIIType.IP_ADDRESS for f in findings)

    def test_detect_api_key(self, redactor):
        findings = redactor.scan("api_key=sk_test_FAKE00000000000000000000000000")
        assert any(f.pii_type == PIIType.API_KEY for f in findings)

    def test_detect_jwt(self, redactor):
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        findings = redactor.scan(f"Token: {token}")
        assert any(f.pii_type == PIIType.JWT_TOKEN for f in findings)

    def test_no_pii(self, redactor):
        findings = redactor.scan("The quick brown fox jumps over the lazy dog")
        assert len(findings) == 0

    def test_multiple_pii_types(self, redactor):
        text = "Email john@test.com, call 555-123-4567, SSN 123-45-6789"
        findings = redactor.scan(text)
        types_found = {f.pii_type for f in findings}
        assert PIIType.EMAIL in types_found
        assert PIIType.PHONE in types_found
        assert PIIType.SSN in types_found


# ── Redaction Tests ──────────────────────────────────────────────────────────


class TestRedaction:
    def test_redact_email(self, redactor):
        result = redactor.redact("Contact john@test.com")
        assert "[REDACTED:EMAIL]" in result.text
        assert "john@test.com" not in result.text
        assert result.had_pii

    def test_no_pii_passthrough(self, redactor):
        original = "Hello world, no PII here"
        result = redactor.redact(original)
        assert result.text == original
        assert not result.had_pii

    def test_hash_policy(self):
        redactor = PIIRedactor(default_policy=RedactionPolicy.HASH)
        result = redactor.redact("Email: test@example.com")
        assert "[HASHED:EMAIL:" in result.text
        assert "test@example.com" not in result.text

    def test_pass_policy(self):
        redactor = PIIRedactor(default_policy=RedactionPolicy.PASS)
        result = redactor.redact("Email: test@example.com")
        assert "test@example.com" in result.text
        assert result.had_pii  # Still detected, just not removed

    def test_encrypt_fallback_to_redact(self):
        """ENCRYPT without vault falls back to REDACT."""
        redactor = PIIRedactor(default_policy=RedactionPolicy.ENCRYPT)
        result = redactor.redact("Email: test@example.com")
        assert "[REDACTED:EMAIL]" in result.text

    def test_multiple_redactions(self, redactor):
        text = "Email john@test.com and jane@test.com"
        result = redactor.redact(text)
        assert result.text.count("[REDACTED:EMAIL]") == 2
        assert len(result.findings) == 2


# ── Tenant Policy Tests ──────────────────────────────────────────────────────


class TestTenantPolicy:
    def test_per_tenant_override(self):
        redactor = PIIRedactor()
        redactor.configure_tenant(
            "tenant_a",
            default_policy=RedactionPolicy.REDACT,
            overrides={PIIType.EMAIL: RedactionPolicy.HASH},
        )
        result = redactor.redact("Email: test@example.com", tenant_id="tenant_a")
        assert "[HASHED:EMAIL:" in result.text

    def test_default_tenant_uses_global_policy(self):
        redactor = PIIRedactor(default_policy=RedactionPolicy.HASH)
        result = redactor.redact("Email: test@example.com", tenant_id="unknown_tenant")
        assert "[HASHED:EMAIL:" in result.text

    def test_tenant_policy_isolation(self):
        redactor = PIIRedactor()
        redactor.configure_tenant("t1", default_policy=RedactionPolicy.HASH)
        redactor.configure_tenant("t2", default_policy=RedactionPolicy.REDACT)

        r1 = redactor.redact("Email: a@b.com", tenant_id="t1")
        r2 = redactor.redact("Email: a@b.com", tenant_id="t2")

        assert "[HASHED:" in r1.text
        assert "[REDACTED:" in r2.text


# ── Stats Tests ──────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_tracking(self, redactor):
        redactor.redact("Email: a@b.com and 555-123-4567")
        stats = redactor.stats()
        assert stats["total_scans"] == 1
        assert stats["total_findings"] >= 2
