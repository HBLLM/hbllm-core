---
title: "API Reference — Brain Factory"
description: "API documentation for BrainFactory and Brain, the primary entry points for creating and using HBLLM cognitive instances."
---

# Brain Factory API

The `BrainFactory` is the primary entry point for creating and managing HBLLM brain instances.

## `BrainFactory.create()`

```python
@staticmethod
async def create(
    provider: str | LLMProvider = "openai/gpt-4o-mini",
    config: BrainConfig | None = None,
    bus: MessageBus | None = None,
    **provider_kwargs,
) -> Brain
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str \| LLMProvider` | `"openai/gpt-4o-mini"` | Provider string (e.g., `"openai/gpt-4o"`, `"anthropic"`) or an `LLMProvider` instance |
| `config` | `BrainConfig \| None` | `None` | Brain configuration. Defaults to `BrainConfig()` |
| `bus` | `MessageBus \| None` | `None` | Custom message bus. Defaults to `InProcessBus` |
| `**provider_kwargs` | `Any` | — | Extra args passed to `get_provider()` |

**Returns:** `Brain` — A fully initialized brain instance with all nodes started.

## `BrainFactory.create_local()`

For running entirely on local hardware without API keys:

```python
@staticmethod
async def create_local(
    checkpoint_path: str | Path | None = None,
    model_size: str = "125m",
    config: BrainConfig | None = None,
    bus: MessageBus | None = None,
    device: str = "auto",
    lora_adapter_path: str | Path | None = None,
) -> Brain
```

## `BrainConfig`

Controls which cognitive subsystems are injected:

```python
from hbllm.brain.factory import BrainConfig

config = BrainConfig(
    inject_memory=True,          # Memory systems
    inject_identity=True,        # Ethics/personality
    inject_curiosity=True,       # Exploratory goals
    inject_perception=False,     # Audio/Vision (requires ML models)
    inject_revision=True,        # Self-critique loop
    inject_goals=True,           # Autonomous goal system
    inject_self_model=True,      # Capability tracking
    inject_metrics=True,         # Live cognitive metrics
    inject_cost_optimizer=True,  # Token optimization
    inject_policy_engine=True,   # Governance enforcement
    inject_owner_rules=True,     # Owner behavioral rules
    inject_sentinel=True,        # Proactive monitoring
    inject_fuzzy_logic=False,    # Fuzzy reasoning
    inject_symbolic_logic=False, # Z3 theorem prover
    total_timeout=60.0,
    system_prompt="You are a helpful AI assistant.",
)
```

## `Brain.process()`

```python
async def process(
    self,
    text: str,
    tenant_id: str = "default",
    session_id: str = "default",
) -> PipelineResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | The input query or goal |
| `tenant_id` | `str` | `"default"` | Tenant ID for memory isolation |
| `session_id` | `str` | `"default"` | Session correlation ID |

**Returns:** `PipelineResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The generated response |
| `correlation_id` | `str` | Unique request ID |
| `source_node` | `str` | Node that produced the final output |
| `confidence` | `float` | Confidence score (0.0–1.0) |
| `latency_ms` | `float` | End-to-end latency |
| `stages_completed` | `list[str]` | Pipeline stages that ran |
| `metadata` | `dict` | Additional context |
| `error` | `bool` | Whether an error occurred |

## `Brain.shutdown()`

```python
async def shutdown(self) -> None
```

Gracefully stops all nodes, flushes memory, and closes the message bus.

## `Brain.cognitive_stats()`

```python
def cognitive_stats(self) -> dict
```

Returns stats from all cognitive subsystems: metrics, self-model, skills, goals, tool memory, token optimizer, and rewards.

## Example

```python
import asyncio
from hbllm.brain.factory import BrainFactory, BrainConfig

async def main():
    config = BrainConfig(
        inject_perception=False,  # Skip heavy ML models
        inject_metrics=True,
    )
    
    brain = await BrainFactory.create(
        provider="openai/gpt-4o",
        config=config,
    )
    
    result = await brain.process(
        text="Analyze our server logs and design a firewall rule.",
        tenant_id="tenant-001",
    )
    
    print(f"Decision: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Stages: {result.stages_completed}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    
    # Get cognitive subsystem stats
    stats = brain.cognitive_stats()
    
    await brain.shutdown()

asyncio.run(main())
```
