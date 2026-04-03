---
title: "Configuration — Environment & Settings"
description: "How to configure HBLLM Core using environment variables, .env files, and pyproject.toml settings."
---

# Configuration

HBLLM can be configured via environment variables, `.env` files, or programmatically.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HBLLM_PROVIDER` | `local` | LLM backend: `local`, `openai`, `anthropic`, `ollama` |
| `HBLLM_MODEL` | `hbllm/base-500m` | Model identifier |
| `HBLLM_PORT` | `8000` | API server port |
| `HBLLM_LOG_LEVEL` | `INFO` | Logging verbosity |
| `HBLLM_MEMORY_DIR` | `~/.hbllm/memory` | Memory database location |
| `HBLLM_ADAPTER_DIR` | `~/.hbllm/adapters` | LoRA adapter cache |
| `HBLLM_BUS_TYPE` | `inprocess` | Message bus: `inprocess` or `redis` |
| `HBLLM_REDIS_URL` | — | Redis connection URL for distributed bus |
| `HBLLM_REDIS_HMAC_KEY` | — | HMAC signing key for Redis messages |
| `HBLLM_ROS2_ENABLED` | `0` | Enable ROS2 integration |
| `HBLLM_QUANTIZE` | — | Quantization mode: `int4`, `int8`, or empty |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |

## .env File

Create a `.env` file in the project root:

```env
HBLLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
HBLLM_LOG_LEVEL=DEBUG
HBLLM_BUS_TYPE=redis
HBLLM_REDIS_URL=redis://localhost:6379
```

## Multi-Tenant Configuration

Each tenant is isolated by default. Tenant-specific settings are managed via the API:

```python
# Tenant memory is automatically isolated
brain = await BrainFactory.create(
    "openai/gpt-4o",
    tenant_id="tenant-001",
    memory_dir="/data/tenants/001"
)
```
