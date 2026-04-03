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
