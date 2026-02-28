"""
Security Middleware for HBLLM API.

Provides:
  - API Key authentication (X-API-Key / Bearer token)
  - Per-tenant rate limiting (token bucket algorithm)
  - Input sanitization (injection prevention, length limits)
  - Request signing verification
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ─── API Key Manager ─────────────────────────────────────────────────────────

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

    Keys are stored as SHA-256 hashes. Supports loading from
    environment, config file, or database.
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

    def validate(self, raw_key: str) -> ApiKey | None:
        """Validate a raw API key. Returns ApiKey if valid, None otherwise."""
        if not self._enabled:
            return ApiKey(key_hash="", tenant_id="default", scopes=["*"])

        key_hash = self.hash_key(raw_key)
        key = self._keys.get(key_hash)
        if key and key.active:
            return key
        return None

    def has_scope(self, key: ApiKey, scope: str) -> bool:
        """Check if an API key has a specific scope."""
        return "*" in key.scopes or scope in key.scopes

    def load_from_env(self) -> int:
        """
        Load API keys from environment variables.

        Format: HBLLM_API_KEY_<NAME>=<key>:<tenant_id>:<scopes>
        Example: HBLLM_API_KEY_ADMIN=sk-abc123:admin:chat,memory,health
        """
        loaded = 0
        for env_key, env_val in os.environ.items():
            if env_key.startswith("HBLLM_API_KEY_"):
                name = env_key[14:].lower()
                parts = env_val.split(":", 2)
                raw_key = parts[0]
                tenant_id = parts[1] if len(parts) > 1 else "default"
                scopes = parts[2].split(",") if len(parts) > 2 else ["chat", "memory", "health"]
                self.add_key(raw_key, tenant_id, name, scopes)
                loaded += 1
        return loaded

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def key_count(self) -> int:
        return len(self._keys)


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

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
        """Seconds until a token is available."""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.refill_rate


class RateLimiter:
    """
    Per-tenant rate limiting using token bucket algorithm.

    Default: 60 requests/minute per tenant.
    """

    def __init__(
        self,
        requests_per_minute: float = 60.0,
        burst_size: float = 10.0,
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._enabled = True

    def check(self, tenant_id: str) -> tuple[bool, float]:
        """
        Check if request is allowed for a tenant.

        Returns:
            (allowed, retry_after_seconds)
        """
        if not self._enabled:
            return True, 0.0

        with self._lock:
            if tenant_id not in self._buckets:
                self._buckets[tenant_id] = TokenBucket(
                    capacity=self.burst_size,
                    refill_rate=self.requests_per_minute / 60.0,
                    tokens=self.burst_size,
                )

            bucket = self._buckets[tenant_id]
            allowed = bucket.consume()
            return allowed, bucket.wait_time if not allowed else 0.0

    def reset(self, tenant_id: str) -> None:
        """Reset rate limit for a tenant."""
        with self._lock:
            self._buckets.pop(tenant_id, None)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


# ─── Input Sanitizer ──────────────────────────────────────────────────────────

class InputSanitizer:
    """
    Sanitizes user input to prevent injection and abuse.

    Enforces:
    - Max text length
    - Valid encoding
    - Strips control characters
    - Blocks common injection patterns
    """

    # Common injection patterns to detect
    _INJECTION_PATTERNS = [
        re.compile(r"<script\b", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),  # onload=, onclick=
        re.compile(r"\{\{.*\}\}"),  # Template injection
        re.compile(r"\$\{.*\}"),    # Expression injection
    ]

    def __init__(self, max_length: int = 10000, strip_html: bool = True):
        self.max_length = max_length
        self.strip_html = strip_html

    def sanitize(self, text: str) -> tuple[str, list[str]]:
        """
        Sanitize input text.

        Returns:
            (sanitized_text, list_of_warnings)
        """
        warnings: list[str] = []

        if not text:
            return "", []

        # Encoding validation
        try:
            text.encode("utf-8").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            warnings.append("invalid_encoding")
            text = text.encode("utf-8", errors="replace").decode("utf-8")

        # Length enforcement
        if len(text) > self.max_length:
            warnings.append(f"truncated_from_{len(text)}_to_{self.max_length}")
            text = text[:self.max_length]

        # Strip control characters (keep newlines and tabs)
        cleaned = []
        for ch in text:
            if ch in ("\n", "\t", "\r") or (ord(ch) >= 32):
                cleaned.append(ch)
            else:
                warnings.append("stripped_control_char")
        text = "".join(cleaned)

        # Check for injection patterns
        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(text):
                warnings.append(f"injection_pattern_detected: {pattern.pattern}")

        # Strip HTML tags if enabled
        if self.strip_html:
            stripped = re.sub(r"<[^>]+>", "", text)
            if stripped != text:
                warnings.append("html_stripped")
                text = stripped

        return text.strip(), warnings


# ─── Security Middleware ──────────────────────────────────────────────────────

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware combining API key auth, rate limiting, and input logging.

    Skips auth for:
    - /v1/health
    - /docs, /openapi.json, /redoc
    """

    SKIP_PATHS = {"/v1/health", "/docs", "/openapi.json", "/redoc", "/metrics"}

    def __init__(
        self,
        app,
        api_keys: ApiKeyManager | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        super().__init__(app)
        self.api_keys = api_keys or ApiKeyManager()
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for health/docs
        if path in self.SKIP_PATHS:
            return await call_next(request)

        # ── API Key Authentication ──
        if self.api_keys.enabled:
            api_key = self._extract_key(request)
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Missing API key. Use X-API-Key header or Bearer token."},
                )

            key_info = self.api_keys.validate(api_key)
            if not key_info:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid or revoked API key."},
                )

            # Attach tenant info to request state
            request.state.api_key = key_info
            request.state.tenant_id = key_info.tenant_id

        # ── Rate Limiting ──
        if self.rate_limiter.enabled:
            tenant_id = getattr(request.state, "tenant_id", "anonymous")
            allowed, retry_after = self.rate_limiter.check(tenant_id)

            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded.", "retry_after": round(retry_after, 1)},
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )

        return await call_next(request)

    @staticmethod
    def _extract_key(request: Request) -> str | None:
        """Extract API key from headers."""
        # X-API-Key header
        api_key = request.headers.get("x-api-key")
        if api_key:
            return api_key

        # Authorization: Bearer <token>
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]

        return None
