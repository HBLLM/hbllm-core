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
Request → APIVersionMiddleware → PrometheusMiddleware → HTTPRateLimitMiddleware
       → BodySizeLimitMiddleware → JWTAuthMiddleware → [Cloud: ApiSecurityMiddleware]
       → Route Handler
```

| Layer | Module | Purpose |
|-------|--------|---------|
| API Versioning | `middleware/api_version.py` | Validates `Accept-Version` header, injects `X-API-Version` |
| Prometheus | `middleware/prometheus.py` | Request count, latency histogram, in-flight gauge |
| Rate Limiting | `middleware/rate_limit.py` | Per-tenant token bucket (configurable RPM) |
| Body Size | `serving/security.py` | Rejects oversized payloads (1–50 MB) |
| JWT Auth | `serving/auth.py` | Token validation, identity injection |

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

Three rate limiters are available:

| Module | Use Case |
|--------|----------|
| `middleware/rate_limit.py` → `HTTPRateLimitMiddleware` | **Primary** — Per-tenant HTTP rate limiting (token bucket, `HBLLM_RATE_LIMIT_RPM`) |
| `serving/security.py` → `AuthRateLimiter` | Brute-force protection for auth endpoints (per-IP) |
| `serving/rate_limiter.py` → `RateLimiter` | Per-tenant/user request throttling (application-level) |

When the rate limit is exceeded, the middleware returns:

```json
{"detail": "Rate limit exceeded. Try again in 42s"}
```

With `429 Too Many Requests` status and `Retry-After` header.

### CORS Hardening

`validate_cors_config()` blocks wildcard CORS (`*`) in production, preventing cross-origin credential theft.

### Password Security

`hash_password()` / `verify_password()` — PBKDF2-SHA256, 100k iterations, 16-byte random salts.

### CSRF Protection

`generate_csrf_token()` / `validate_csrf_token()` — HMAC-based form protection for admin endpoints.

> 📖 **[Full Security Architecture →](../security.md)** — Identity triplet, tenant guard, audit log, encryption at rest

## KV Cache & Persistence

**Module:** `hbllm.serving.kv_cache.KVCache`

Manages pre-allocated key-value tensors for autoregressive decoding, featuring Sliding Window Attention (SWA), Attention Sinks, and persistent serialization.

### Persistent Serialization

The KV Cache can be serialized to disk to preserve active cognitive contexts between boots:
- **`save_cache(file_path, model_config, tokenizer)`** — Saves active KV tensor histories to a `.kvc` file. Generates a unique SHA-256 integrity signature from the active architecture configuration (`num_layers`, `hidden_size`, `num_kv_heads`, `head_dim`, `vocab_size`, and SWA options).
- **`load_cache(file_path, model_config, tokenizer)`** — Reloads KV history. Performs a strict integrity validation signature check. If there is any discrepancy (e.g. mismatched model dimensions or vocabularies), it raises a descriptive `ValueError` to prevent silent state corruption.

---

## Serving Architecture

### Route Modules

Endpoints are organized into modular routers:

| Router | Module | Endpoints |
|--------|--------|-----------|
| `health_router` | `routes/health.py` | `/health`, `/health/live`, `/health/ready`, `/routing/stats` |
| `memory_router` | `routes/memory.py` | `/v1/memory/*`, `/v1/sync/*`, `/v1/feedback/*`, `/v1/knowledge/*`, `/v1/rules` |
| `studio_router` | `studio.py` | `/api/*`, `/studio/*` (Studio UI endpoints) |
| Core (api.py) | `api.py` | `/v1/chat`, `/v1/audio/*`, `/v1/benchmarks/*`, `/v1/cognitive/*`, `/studio/voice/*` |

### Dependency Injection

`serving/deps.py` provides `Depends()`-based state injection:

```python
from hbllm.serving.deps import get_brain, get_bus

@router.get("/my-endpoint")
async def my_endpoint(brain = Depends(get_brain)):
    return brain.stats()
```

### Module Map

| Module | File | Purpose |
|---|---|---|
| REST API | `serving/api.py` | FastAPI app, core endpoints, middleware registration |
| Route: Health | `serving/routes/health.py` | Health probes, routing stats |
| Route: Memory | `serving/routes/memory.py` | Memory, sync, feedback, knowledge |
| Dependencies | `serving/deps.py` | `Depends()` injection for routes |
| Chat | `serving/chat.py` | Chat completion logic and streaming |
| MCP Server | `serving/mcp_server.py` | MCP tool provider |
| Pipeline | `serving/pipeline.py` | Cognitive pipeline orchestration |
| Providers | `serving/provider.py` | LLM provider abstraction (OpenAI, Anthropic, Local, Ollama) |
| Auth | `serving/auth.py` | JWT authentication middleware |
| Security | `serving/security.py` | Input sanitization, body limits, CORS, API keys |
| Middleware: Rate Limit | `serving/middleware/rate_limit.py` | Per-tenant HTTP rate limiting |
| Middleware: Prometheus | `serving/middleware/prometheus.py` | Prometheus metrics collection |
| Middleware: Versioning | `serving/middleware/api_version.py` | API version headers |
| KV Cache | `serving/kv_cache.py` | Efficient key-value cache for inference |
| Token Optimizer | `serving/token_optimizer.py` | Token budget management |
| Self-Improve | `serving/self_improve.py` | Background self-improvement loop |
| Launcher | `serving/launcher.py` | Server startup and configuration |
| **Cognitive Daemon** | `serving/daemon.py` | Always-on cognitive process with autonomy heartbeat |
| **Proactive Processor** | `serving/proactive.py` | Routes autonomy actions to user-facing output via SSE |
| **Device Bridge** | `serving/device_bridge.py` | Cross-device session continuity and presence tracking |
| **Notification Routes** | `serving/routes/notifications.py` | REST + SSE notification API |

---

## Cognitive Daemon

**Module:** `hbllm.serving.daemon.CognitiveDaemon`

Long-running daemon process that boots the full Brain with AutonomyCore running
and keeps it alive as a background service. Manages the complete lifecycle:

1. Boot Brain via BrainFactory
2. Start AutonomyCore (cognitive heartbeat)
3. Start ProactiveProcessor (output bridge)
4. Optionally start HTTP server (FastAPI)
5. Run until shutdown signal (SIGTERM/SIGINT)
6. Graceful drain and persist state

### CLI Usage

```bash
# Foreground (development)
python -m hbllm.serving.daemon

# With specific provider
python -m hbllm.serving.daemon --provider openai/gpt-4o-mini

# Local model
python -m hbllm.serving.daemon --local --model-size 1.5b

# As a systemd service (journal-compatible logging)
python -m hbllm.serving.daemon --mode systemd
```

### Programmatic Usage

```python
from hbllm.serving.daemon import CognitiveDaemon

daemon = CognitiveDaemon(
    provider="openai/gpt-4o-mini",
    host="0.0.0.0",
    port=8000,
    data_dir="data",
)
await daemon.start()

# Get telemetry
snap = daemon.snapshot()
# {"uptime_s": 3600, "brain_nodes": 28, "autonomy": {...}, "proactive": {...}}

await daemon.stop()
```

---

## Proactive Processor & SSE Channel

**Module:** `hbllm.serving.proactive.ProactiveProcessor`

Routes AutonomyCore cognitive actions to user-facing output. When the AutonomyCore
generates a cognitive action (proactive reminder, anomaly detection, background
insight), this processor enriches it via the CognitivePipeline and delivers the
result through multiple channels.

### Delivery Channels

| Channel | Class | Description |
|---------|-------|-------------|
| NotificationGateway | Persistent | Pollable notification store |
| SSE Channel | Real-time | Per-tenant Server-Sent Events |
| Bus broadcast | Internal | `proactive.output` topic |

### SSEChannel

```python
from hbllm.serving.proactive import SSEChannel, ProactiveEvent

channel = SSEChannel(max_queue_size=100)

# Push an event
await channel.push(ProactiveEvent(
    tenant_id="user1",
    title="Goal Complete",
    body="Your research task finished",
))

# Stream events (used by API endpoint)
async for event in channel.stream("user1"):
    yield f"data: {event.to_dict()}\n\n"
```

---

## Device Bridge

**Module:** `hbllm.serving.device_bridge.DeviceBridge`

Enables seamless conversation handoff between devices — start a conversation on
your phone, continue on your laptop. Tracks device presence via heartbeat and
routes notifications to the best available device.

### Usage

```python
from hbllm.serving.device_bridge import DeviceBridge, DeviceInfo

bridge = DeviceBridge(bus=message_bus)
await bridge.start()

# Register a device
bridge.register_device(DeviceInfo(
    device_id="iphone-1",
    tenant_id="user1",
    device_type="mobile",
    capabilities=["audio", "display"],
    push_token="fcm:abc123",
))

# Find best device for notification
best = bridge.get_best_device("user1", required_capabilities=["display"])

# Handoff session between devices
await bridge.handoff_session("session-1", from_device="iphone-1", to_device="macbook-1")

# Stats
bridge.stats()
# {"total_devices": 3, "active_devices": 2, "tenants": 1, "handoffs": 1}
```

### Bus Events

| Topic | When |
|-------|------|
| `device.register` | New device connects |
| `device.heartbeat` | Device sends presence ping |
| `device.handoff` | Session transfer requested |
| `device.inactive` | Device went stale (5-minute timeout) |

## Speaker Identification API

HBLLM supports per-tenant voice identification — enrolling speakers, identifying them in real-time, and retroactively matching unknowns.

### `POST /studio/voice/speakers/enroll`

Enroll a speaker's voice for identification:

```bash
curl -X POST http://localhost:8000/studio/voice/speakers/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_id": "dumith",
    "speaker_name": "Dumith",
    "audio_hex": "<hex-encoded 16-bit PCM, 3+ seconds>",
    "sample_rate": 16000,
    "tenant_id": "default"
  }'
```

### `GET /studio/voice/speakers`

List all enrolled speakers for a tenant:

```bash
curl http://localhost:8000/studio/voice/speakers?tenant_id=default
```

### `DELETE /studio/voice/speakers/{speaker_id}`

Delete an enrolled speaker profile:

```bash
curl -X DELETE http://localhost:8000/studio/voice/speakers/dumith?tenant_id=default
```

---

## Security Package (`hbllm.security`)

Core security modules that are platform-independent (no serving dependency):

| Module | File | Purpose |
|---|---|---|
| Tenant Guard | `security/tenant_guard.py` | `contextvars`-based identity propagation, `@require_tenant` decorator |
| Trust Interceptor | `security/trust.py` | MessageBus interceptor for signature verification, replay protection, and internal node trust |
| Audit Log | `security/audit_log.py` | Append-only SQLite audit trail with identity context |
| Encryption | `security/encryption.py` | Field-level symmetric encryption (XOR keystream + HMAC-SHA256) |
