---
title: "Configuration — Environment & Settings"
description: "How to configure HBLLM Core using environment variables, .env files, and pyproject.toml settings."
---

# Configuration

HBLLM can be configured via environment variables, `.env` files, or programmatically.

## Environment Variables

### Core

| Variable | Default | Description |
|---|---|---|
| `HBLLM_PROVIDER` | `local` | LLM backend: `local`, `openai`, `anthropic`, `ollama` |
| `HBLLM_MODEL` | `hbllm/base-500m` | Model identifier |
| `HBLLM_PORT` | `8000` | API server port |
| `HBLLM_ENV` | `development` | Environment mode: `development` or `production` |
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

### Security & Production

| Variable | Default | Description |
|---|---|---|
| `HBLLM_JWT_SECRET` | *(auto-gen in dev)* | JWT signing secret. **Required** in production mode |
| `HBLLM_TENANT_GUARD_MODE` | `WARN` | Tenant isolation: `WARN` or `STRICT` (blocks cross-tenant) |
| `HBLLM_CORS_ORIGINS` | `*` | Allowed CORS origins. Wildcard blocked in production |

### Rate Limiting & Quotas

| Variable | Default | Description |
|---|---|---|
| `HBLLM_RATE_LIMIT_RPM` | `60` | Max requests per minute per tenant |
| `HBLLM_DB_MAX_PER_TENANT` | `50000` | Max episodic memory turns per tenant |

### Infrastructure

| Variable | Default | Description |
|---|---|---|
| `HBLLM_SHUTDOWN_DRAIN_SEC` | `15` | Seconds to drain in-flight requests on shutdown |
| `HBLLM_OTEL_ENDPOINT` | — | OpenTelemetry collector endpoint for distributed tracing |
| `HBLLM_CONFIG_PATH` | `hbllm.yaml` | Path to YAML configuration file |

## .env File

Create a `.env` file in the project root:

```env
HBLLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
HBLLM_LOG_LEVEL=DEBUG
HBLLM_BUS_TYPE=redis
HBLLM_REDIS_URL=redis://localhost:6379

# Production settings
HBLLM_ENV=production
HBLLM_JWT_SECRET=your-secret-here
HBLLM_RATE_LIMIT_RPM=120
HBLLM_DB_MAX_PER_TENANT=100000
```

## BrainConfig (Pydantic)

The `BrainConfig` class uses Pydantic `BaseModel` with validators for type-safe configuration:

```python
from hbllm.brain.config import BrainConfig

config = BrainConfig(
    provider="openai",
    model_name="gpt-4o",
    env="production",
    jwt_secret="my-production-secret",
    rate_limit_rpm=120,
    db_max_per_tenant=100_000,
    shutdown_drain_sec=30,
)
```

### Validators

| Field | Validator | Behavior |
|-------|-----------|----------|
| `env` | — | `development` or `production` |
| `jwt_secret` | `model_validator` | **Must be set** when `env=production` |
| `rate_limit_rpm` | `field_validator` | Must be `> 0` |
| `db_max_per_tenant` | `field_validator` | Must be `> 0` |
| `shutdown_drain_sec` | `field_validator` | Must be `>= 0` |

## YAML Configuration

HBLLM loads its system configuration from `hbllm.yaml` (or the path in `HBLLM_CONFIG_PATH`):

```yaml
env: production

cluster:
  node_id: "brain-01"
  
adapters:
  enabled: true
  cache_dir: "./checkpoints/adapters"
  auto_download: true
  require_sha256: true
  sources:
    - domain: coding
      repo_id: hbllm/coding-lora-v2
      revision: v2.1.0

checkpoints_dir: ./checkpoints
data_dir: ./data
```

```python
from hbllm.config import HBLLMCoreConfig

# Load from hbllm.yaml (auto-discovered) or explicit path
config = HBLLMCoreConfig.load("hbllm.yaml")
```

## Multi-Tenant Configuration

Each tenant is isolated at the `brain.process()` level. Memory is automatically partitioned per tenant:

```python
brain = await BrainFactory.create("openai/gpt-4o")

# Tenant isolation happens at query time
result = await brain.process(
    "Summarize our Q3 earnings",
    tenant_id="tenant-001",
    session_id="session-abc",
)
```

