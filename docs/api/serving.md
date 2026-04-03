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

## Security

The serving layer (`hbllm.serving.security`) provides:

- **API Key authentication** via Bearer tokens
- **Per-tenant rate limiting**
- **Request validation** and input sanitization
- **CORS configuration** for web frontends

---

## Serving Architecture

| Module | File | Purpose |
|---|---|---|
| REST API | `serving/api.py` | FastAPI app with OpenAI-compatible endpoints |
| Chat | `serving/chat.py` | Chat completion logic and streaming |
| MCP Server | `serving/mcp_server.py` | MCP tool provider |
| Pipeline | `serving/pipeline.py` | Cognitive pipeline orchestration |
| Providers | `serving/provider.py` | LLM provider abstraction (OpenAI, Anthropic, Local, Ollama) |
| Security | `serving/security.py` | Auth, rate limiting, CORS |
| KV Cache | `serving/kv_cache.py` | Efficient key-value cache for inference |
| Token Optimizer | `serving/token_optimizer.py` | Token budget management |
| Self-Improve | `serving/self_improve.py` | Background self-improvement loop |
| Launcher | `serving/launcher.py` | Server startup and configuration |
