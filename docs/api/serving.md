---
title: "API Reference — REST & MCP Server"
description: "API documentation for the HBLLM FastAPI serving layer, MCP Server, and chat completions."
---

# REST API & MCP Server

HBLLM exposes a **FastAPI-based REST API** with OpenAI-compatible chat completions, plus a **Model Context Protocol (MCP) Server** for tool integration.

## Starting the Server

```bash
# Default: http://localhost:8000
hbllm serve --port 8000 --host 0.0.0.0

# Or directly:
python -m hbllm.serving.api
```

## Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HBLLM_API_KEY" \
  -d '{
    "model": "hbllm-500m",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is quantum computing?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### `POST /v1/brain/process`

Full cognitive pipeline endpoint (activates all brain nodes):

```bash
curl -X POST http://localhost:8000/v1/brain/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Design a microservices architecture for our e-commerce platform",
    "tenant_id": "tenant-001"
  }'
```

### `GET /v1/brain/stats`

Live cognitive subsystem metrics:

```bash
curl http://localhost:8000/v1/brain/stats
```

### `GET /health`

Health check endpoint for load balancers.

---

## MCP Server

The MCP Server (`hbllm.serving.mcp_server`) exposes HBLLM as a tool provider for AI agents:

```bash
# Start MCP server (stdio mode for IDE integration)
python -m hbllm.serving.mcp_server
```

---

## Security & Middleware Stack

The serving layer includes a comprehensive, layered security middleware stack.

### Middleware Order (outermost → innermost)

```
Request → BodySizeLimitMiddleware → JWTAuthMiddleware → [Cloud: ApiSecurityMiddleware] → Route Handler
```

### JWT Authentication (`serving/auth.py`)

`JWTAuthMiddleware` validates Bearer tokens and injects the identity triplet into `request.state`:

```python
request.state.tenant_id  # Organization
request.state.user_id    # User within tenant
request.state.device_id  # Device (edge/IoT)
```

- **Production mode**: Requires `HBLLM_JWT_SECRET` env var — refuses to start without it.
- **Development mode**: Auto-generates an ephemeral secret per boot (with warning).
- **Skip paths**: `/health`, `/metrics`, `/docs`, `/openapi.json`, admin/studio routes.

### Input Sanitization (`serving/security.py`)

| Function | Protection |
|----------|-----------|
| `sanitize_input(text, max_length)` | Strip control chars (keeps `\n`, `\t`), truncate |
| `detect_injection(text)` | Regex-based prompt injection detection with risk levels |

### Body Size Limits

`BodySizeLimitMiddleware` prevents memory exhaustion from oversized payloads:

| Route | Max Size |
|-------|----------|
| `/v1/chat` | 1 MB |
| `/v1/upload`, `/v1/knowledge` | 50 MB |
| Default | 5 MB |

### API Key Manager

`ApiKeyManager` provides SHA-256 hashed key validation with per-key scoping:

```python
from hbllm.serving.security import ApiKeyManager

akm = ApiKeyManager()
key = akm.add_key("sk-prod-key", tenant_id="acme", scopes=["chat", "knowledge"])
validated = akm.validate("sk-prod-key")  # Returns ApiKey or None
```

### Rate Limiting

Two rate limiters are available:

| Module | Use Case |
|--------|----------|
| `serving/security.py` → `AuthRateLimiter` | Brute-force protection for auth endpoints (per-IP) |
| `serving/rate_limiter.py` → `RateLimiter` | Per-tenant/user request throttling (token bucket) |

### CORS Hardening

`validate_cors_config()` blocks wildcard CORS (`*`) in production, preventing cross-origin credential theft.

### Password Security

`hash_password()` / `verify_password()` — PBKDF2-SHA256, 100k iterations, 16-byte random salts.

### CSRF Protection

`generate_csrf_token()` / `validate_csrf_token()` — HMAC-based form protection for admin endpoints.

> 📖 **[Full Security Architecture →](../security.md)** — Identity triplet, tenant guard, audit log, encryption at rest

---

## Serving Architecture

| Module | File | Purpose |
|---|---|---|
| REST API | `serving/api.py` | FastAPI app with OpenAI-compatible endpoints |
| Chat | `serving/chat.py` | Chat completion logic and streaming |
| MCP Server | `serving/mcp_server.py` | MCP tool provider |
| Pipeline | `serving/pipeline.py` | Cognitive pipeline orchestration |
| Providers | `serving/provider.py` | LLM provider abstraction (OpenAI, Anthropic, Local, Ollama) |
| Auth | `serving/auth.py` | JWT authentication middleware |
| Security | `serving/security.py` | Input sanitization, body limits, CORS, API keys |
| Rate Limiter | `serving/rate_limiter.py` | Per-tenant/user token bucket rate limiting |
| KV Cache | `serving/kv_cache.py` | Efficient key-value cache for inference |
| Token Optimizer | `serving/token_optimizer.py` | Token budget management |
| Self-Improve | `serving/self_improve.py` | Background self-improvement loop |
| Launcher | `serving/launcher.py` | Server startup and configuration |

---

## Security Package (`hbllm.security`)

Core security modules that are platform-independent (no serving dependency):

| Module | File | Purpose |
|---|---|---|
| Tenant Guard | `security/tenant_guard.py` | `contextvars`-based identity propagation, `@require_tenant` decorator |
| Audit Log | `security/audit_log.py` | Append-only SQLite audit trail with identity context |
| Encryption | `security/encryption.py` | Field-level symmetric encryption (XOR keystream + HMAC-SHA256) |
