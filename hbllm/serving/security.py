"""
Security Hardening — Input validation, body limits, prompt injection detection.

Provides:
  - sanitize_input(): strip control characters, truncate to limit
  - detect_injection(): regex-based prompt injection detection
  - BodySizeLimitMiddleware: per-endpoint body size enforcement
  - AuthRateLimiter: brute-force protection for auth endpoints
  - validate_cors_config(): block wildcard CORS in production
  - hash_password() / verify_password(): PBKDF2-SHA256
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


# ─── Input Validation ────────────────────────────────────────────────────────

MAX_CHAT_BODY = 1 * 1024 * 1024
MAX_UPLOAD_BODY = 50 * 1024 * 1024
MAX_DEFAULT_BODY = 5 * 1024 * 1024

INJECTION_PATTERNS = [
    r"ignore\s+(?:\w+\s+)*(?:instructions|prompts)",
    r"you\s+are\s+now\s+(?:a|an|in)\s+\w+\s+mode",
    r"disregard\s+(?:all|any|your)\s+(?:previous|prior)",
    r"system\s*:\s*you\s+are",
    r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>",
]
_injection_re = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_control_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_input(text: str, max_length: int = 50000) -> str:
    """Sanitize user input: strip control chars, truncate."""
    text = _control_re.sub("", text)
    return text[:max_length]


def detect_injection(text: str) -> dict[str, Any]:
    """Detect potential prompt injection attempts."""
    matches = _injection_re.findall(text)
    return {
        "detected": len(matches) > 0,
        "patterns_matched": len(matches),
        "risk_level": "high" if len(matches) >= 2 else "low" if matches else "none",
    }


# ─── Password Hashing ───────────────────────────────────────────────────────


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password with PBKDF2-SHA256 (100k iterations)."""
    if salt is None:
        salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"{salt.hex()}:{key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored PBKDF2 hash."""
    try:
        salt_hex, key_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(key_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, AttributeError):
        return False


# ─── CSRF Protection ────────────────────────────────────────────────────────

_csrf_secret = os.environ.get("HBLLM_CSRF_SECRET", secrets.token_hex(32))


def generate_csrf_token(session_id: str) -> str:
    """Generate a CSRF token tied to a session."""
    payload = f"{session_id}:{time.time():.3f}"
    sig = hmac.new(_csrf_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}:{sig}"


def validate_csrf_token(token: str, session_id: str, max_age: int = 3600) -> bool:
    """Validate a CSRF token."""
    try:
        parts = token.rsplit(":", 2)
        if len(parts) != 3:
            return False
        stored_session, ts_str, sig = parts
        if stored_session != session_id:
            return False
        ts = float(ts_str)
        if time.time() - ts > max_age:
            return False
        expected_payload = f"{session_id}:{ts_str}"
        expected_sig = hmac.new(
            _csrf_secret.encode(), expected_payload.encode(), hashlib.sha256
        ).hexdigest()[:32]
        return hmac.compare_digest(sig, expected_sig)
    except (ValueError, AttributeError):
        return False


# ─── Body Size Limit Middleware ──────────────────────────────────────────────


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding size limits."""

    def __init__(
        self,
        app: Any,
        chat_limit: int = MAX_CHAT_BODY,
        upload_limit: int = MAX_UPLOAD_BODY,
        default_limit: int = MAX_DEFAULT_BODY,
    ):
        super().__init__(app)
        self.chat_limit = chat_limit
        self.upload_limit = upload_limit
        self.default_limit = default_limit

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        content_length = request.headers.get("content-length")
        path = request.url.path

        if "/upload" in path or "/knowledge" in path:
            limit = self.upload_limit
        elif "/chat" in path:
            limit = self.chat_limit
        else:
            limit = self.default_limit

        if content_length:
            try:
                size = int(content_length)
            except (ValueError, TypeError):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid Content-Length header"},
                )
            if size > limit:
                logger.warning("Body size %d exceeds limit %d for %s", size, limit, path)
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request body too large",
                        "max_bytes": limit,
                        "received_bytes": size,
                    },
                )
        elif request.method in ("POST", "PUT", "PATCH"):
            # No Content-Length header — read actual body to enforce limits
            body = await request.body()
            if len(body) > limit:
                logger.warning("Body size %d exceeds limit %d for %s (no CL header)", len(body), limit, path)
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request body too large",
                        "max_bytes": limit,
                        "received_bytes": len(body),
                    },
                )

        return await call_next(request)


# ─── Auth Rate Limiting ─────────────────────────────────────────────────────


class AuthRateLimiter:
    """Rate limiter for auth endpoints. Prevents brute-force attacks.

    Uses an OrderedDict with bounded size to prevent memory exhaustion
    from attackers sending requests with millions of unique identifiers.
    """

    _MAX_TRACKED_IDENTIFIERS = 50_000

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._attempts: OrderedDict[str, list[float]] = OrderedDict()

    def check(self, identifier: str) -> tuple[bool, int]:
        """Check if an auth attempt is allowed. Returns (allowed, remaining)."""
        now = time.time()
        cutoff = now - self.window

        if identifier not in self._attempts:
            self._attempts[identifier] = []

        self._attempts[identifier] = [t for t in self._attempts[identifier] if t > cutoff]
        attempts = self._attempts[identifier]

        # Move to end (most recently accessed) for LRU ordering
        self._attempts.move_to_end(identifier)

        # Evict oldest entries if we exceed the max tracked identifiers
        while len(self._attempts) > self._MAX_TRACKED_IDENTIFIERS:
            self._attempts.popitem(last=False)

        # Evict empty entries to prevent unbounded dict growth
        if not attempts:
            self._attempts.pop(identifier, None)
            return True, self.max_attempts

        remaining = max(0, self.max_attempts - len(attempts))

        if len(attempts) >= self.max_attempts:
            logger.warning(
                "Auth rate limit exceeded for %s (%d attempts in %ds)",
                identifier,
                len(attempts),
                self.window,
            )
            return False, 0
        return True, remaining

    def record_attempt(self, identifier: str) -> None:
        """Record an auth attempt."""
        if identifier not in self._attempts:
            self._attempts[identifier] = []
        self._attempts[identifier].append(time.time())

    def reset(self, identifier: str) -> None:
        """Reset attempts for an identifier."""
        self._attempts.pop(identifier, None)


# ─── CORS Validator ──────────────────────────────────────────────────────────


def validate_cors_config(origins: list[str]) -> list[str]:
    """Validate CORS origins. Wildcard * is forbidden."""
    if "*" in origins:
        logger.error(
            "CORS wildcard '*' is not allowed. Set HBLLM_CORS_ORIGINS to specific domains."
        )
        filtered = [o for o in origins if o != "*"]
        return filtered if filtered else ["http://localhost", "https://localhost"]
    return origins


# ─── Legacy Implementations for Tests ───────────────────────────────────────


@dataclass
class ApiKey:
    """An API key with associated permissions."""

    key_hash: str
    tenant_id: str
    name: str = ""
    scopes: list[str] = field(default_factory=lambda: ["chat", "memory", "health"])
    active: bool = True
    created_at: float = field(default_factory=time.time)


class ApiKeyManager:
    """
    Manages API key validation and tenant resolution.
    Keys are stored as SHA-256 hashes.
    """

    def __init__(self) -> None:
        self._keys: dict[str, ApiKey] = {}  # hash -> ApiKey
        self._enabled = True

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Hash a raw API key for secure storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def add_key(
        self,
        raw_key: str,
        tenant_id: str,
        name: str = "",
        scopes: list[str] | None = None,
    ) -> ApiKey:
        """Register a new API key."""
        key = ApiKey(
            key_hash=self.hash_key(raw_key),
            tenant_id=tenant_id,
            name=name,
            scopes=scopes or ["chat", "memory", "health"],
        )
        self._keys[key.key_hash] = key
        return key

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> dict[str, ApiKey]:
        return self._keys

    def validate(self, raw_key: str) -> ApiKey | None:
        if not self._enabled:
            # Bypass mode: grant basic access only — never admin
            return ApiKey(
                key_hash="bypass",
                tenant_id="dev",
                name="disabled_mode",
                scopes=["chat", "memory", "health"],
            )

        key_hash = self.hash_key(raw_key)
        key = self._keys.get(key_hash)
        if key and key.active:
            return key
        return None

    def has_scope(self, key: ApiKey, scope: str) -> bool:
        return scope in key.scopes

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def wait_time(self) -> float:
        """Seconds until the requested number of tokens becomes available."""
        if self.tokens >= 1:
            return 0.0
        deficit = 1 - self.tokens
        return deficit / self.refill_rate if self.refill_rate > 0 else float("inf")


class RateLimiter:
    """
    Per-tenant rate limiting using token bucket algorithm.
    """

    def __init__(
        self,
        requests_per_minute: float = 60.0,
        burst_size: float = 10.0,
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._enabled = True

    async def check(self, tenant_id: str) -> tuple[bool, float]:
        """
        Check if request is allowed for a tenant.
        Returns (allowed, retry_after_seconds)
        """
        if not self._enabled:
            return True, 0.0

        async with self._lock:
            if tenant_id not in self._buckets:
                self._buckets[tenant_id] = TokenBucket(
                    capacity=self.burst_size,
                    refill_rate=self.requests_per_minute / 60.0,
                    tokens=self.burst_size,
                )

            bucket = self._buckets[tenant_id]
            allowed = bucket.consume()
            return allowed, bucket.wait_time if not allowed else 0.0

    async def reset(self, tenant_id: str) -> None:
        """Reset rate limit for a tenant."""
        async with self._lock:
            self._buckets.pop(tenant_id, None)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


class InputSanitizer:
    """
    Sanitizes user input to prevent injection and abuse.
    """

    _INJECTION_PATTERNS = [
        re.compile(r"<script\b", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"\{\{.*\}\}"),
        re.compile(r"\$\{.*\}"),
    ]

    def __init__(self, max_length: int = 10000, strip_html: bool = True):
        self.max_length = max_length
        self.strip_html = strip_html

    def sanitize(self, text: str) -> tuple[str, list[str]]:
        warnings: list[str] = []

        if not text:
            return "", []

        try:
            text.encode("utf-8").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            warnings.append("invalid_encoding")
            text = text.encode("utf-8", errors="replace").decode("utf-8")

        if len(text) > self.max_length:
            warnings.append(f"truncated_from_{len(text)}_to_{self.max_length}")
            text = text[: self.max_length]

        cleaned = []
        for ch in text:
            if ch in ("\n", "\t", "\r") or (ord(ch) >= 32):
                cleaned.append(ch)
            else:
                warnings.append("stripped_control_char")
        text = "".join(cleaned)

        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(text):
                warnings.append(f"injection_pattern_detected: {pattern.pattern}")

        if self.strip_html:
            stripped = re.sub(r"<[^>]+>", "", text)
            if stripped != text:
                warnings.append("html_stripped")
                text = stripped

        # Strip markdown injection patterns: ![](url), [text](javascript:...)
        md_image = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        if md_image != text:
            warnings.append("markdown_image_stripped")
            text = md_image
        md_link = re.sub(r"\[[^\]]*\]\(javascript:[^)]*\)", "", text, flags=re.IGNORECASE)
        if md_link != text:
            warnings.append("markdown_js_link_stripped")
            text = md_link

        return text.strip(), warnings
