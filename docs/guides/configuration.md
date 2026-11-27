---
title: "Configuration — Runtime, Environment & Settings"
description: "How to configure HBLLM Core via CognitiveSystemConfig, environment variables, .env files, and YAML settings."
---

# Configuration

HBLLM Core can be configured programmatically through `CognitiveSystemConfig` (recommended), via environment variables / `.env` files, or via YAML (`hbllm.yaml.example`).

---

## 1. `CognitiveSystemConfig` (Pure-HCIR Composition Root)

**Module:** `hbllm.runtime.bootstrap.CognitiveSystemConfig`

```python
from pathlib import Path
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

config = CognitiveSystemConfig(
    world_id="prod_world",
    seed=42,
    exploration_budget=150,
    enable_logging=True,
    memory_dir=Path("./data/worlds/prod_world"),
    enable_grounding=True,
    enable_fuzzy_grounding=False,  # Set True for F1 multi-concept & adaptive relation SGE
    enable_fuzzy_reasoning=False,  # Set True for F2/F3 soft VM unification & composition policy
    perception_mode="handcrafted",  # "handcrafted" | "snn" | "learned"
    enable_brain_llm=False,  # Keep False for Pure-HCIR
    enable_autonomous_curiosity=False,
    max_events_per_store=2000,
)
system = build_cognitive_system(config)
```

| Field | Default | Description |
|---|---|---|
| `world_id` | `"default"` | Context/world partition identifier |
| `seed` | `None` | Deterministic random seed |
| `exploration_budget` | `150` | Epistemic probing step budget |
| `memory_dir` | `~/.hbllm/worlds/<world_id>` | Persistent `SQLiteEventStore` directory (`None` for in-memory) |
| `enable_grounding` | `True` | Attach `CorticalGroundingBridge` (SGE) |
| `enable_fuzzy_grounding` | `False` | Enable ADR 006 Phase `F1` graded concepts and adaptive relations |
| `enable_fuzzy_reasoning` | `False` | Enable ADR 006 Phase `F2`/`F3` soft VM unification, `ALPHA_CUT`, and `RelationCompositionPolicy` |
| `perception_mode` | `"handcrafted"` | `FeatureExtractor` backend (`handcrafted`, `snn`, `learned`) |
| `enable_brain_llm` | `False` | Allow optional external LLM scaffold outside `hcir/` |
| `enable_autonomous_curiosity` | `False` | Enable `EpistemicCuriosityController` web fallback on unknown queries |
| `max_events_per_store` | `2000` | Bounded compaction ceiling for `SQLiteEventStore` |

---

## 2. Environment Variables

### Runtime & Storage

| Variable | Default | Description |
|---|---|---|
| `HBLLM_ENV` | `development` | `development` or `production` |
| `HBLLM_LOG_LEVEL` | `INFO` | Logging verbosity |
| `HBLLM_DATA_DIR` | `./data` | Data directory for persistent stores and audit trails |
| `HBLLM_BUS_BACKEND` | `inprocess` | Message bus: `inprocess` or `redis` |
| `HBLLM_REDIS_URL` | — | Redis connection URL for distributed deployments |
| `HBLLM_REDIS_HMAC_KEY` | — | HMAC signing key for Redis messages |
| `HBLLM_DATABASE_URL` | — | PostgreSQL URL when running multi-replica server storage |
| `HBLLM_CONFIG_PATH` | `hbllm.yaml.example` | Path to YAML configuration file |

### Security & Serving (`hbllm.serving`)

| Variable | Default | Description |
|---|---|---|
| `HBLLM_JWT_SECRET` | *(ephemeral in dev)* | **Required** in `HBLLM_ENV=production`; must be shared across replicas |
| `HBLLM_UNSAFE_DEV_AUTH` | `0` | Explicit opt-in (`1`) for unauthenticated loopback dev mode (`JWTAuthMiddleware` fails closed by default) |
| `HBLLM_TENANT_GUARD_MODE` | `WARN` | Tenant isolation: `OFF`, `WARN`, or `STRICT` |
| `HBLLM_CORS_ORIGINS` | `*` | Allowed CORS origins (wildcard blocked in production) |
| `HBLLM_RATE_LIMIT_RPM` | `60` | Max requests per minute per tenant |
| `HBLLM_SHUTDOWN_DRAIN_SEC` | `15` | Graceful drain timeout on shutdown |
| `HBLLM_OTEL_ENDPOINT` | — | OpenTelemetry collector endpoint |

---

## 3. `.env` Example (Production)

```env
HBLLM_ENV=production
HBLLM_JWT_SECRET=replace-with-strong-32byte-secret
HBLLM_TENANT_GUARD_MODE=STRICT
HBLLM_CORS_ORIGINS=https://app.example.com
HBLLM_RATE_LIMIT_RPM=120
HBLLM_BUS_BACKEND=redis
HBLLM_REDIS_URL=redis://redis:6379/0
HBLLM_LOG_LEVEL=INFO
```

---

## 4. YAML & `HBLLMCoreConfig` (`hbllm.config`)

Copy `hbllm.yaml.example` as a starting point and load via `hbllm.config`:

```python
from hbllm.config import HBLLMCoreConfig

cfg = HBLLMCoreConfig.load("hbllm.yaml.example")
```
