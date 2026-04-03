---
title: "API Reference — Brain Factory"
description: "API documentation for BrainFactory, the primary entry point for creating HBLLM cognitive instances."
---

# Brain Factory API

The `BrainFactory` is the primary entry point for creating and managing HBLLM brain instances.

## `BrainFactory.create()`

```python
@classmethod
async def create(
    cls,
    model: str,
    tenant_id: str = "default",
    memory_dir: str | None = None,
    bus_type: str = "inprocess",
    redis_url: str | None = None,
) -> Brain
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — | Model identifier (e.g., `"openai/gpt-4o"`, `"local/hbllm-500m"`) |
| `tenant_id` | `str` | `"default"` | Tenant identifier for memory isolation |
| `memory_dir` | `str \| None` | `None` | Custom memory database directory |
| `bus_type` | `str` | `"inprocess"` | Message bus type: `"inprocess"` or `"redis"` |
| `redis_url` | `str \| None` | `None` | Redis URL for distributed bus |

**Returns:** `Brain` — A fully initialized brain instance with all nodes started.

**Example:**

```python
import asyncio
from hbllm.brain.factory import BrainFactory

async def main():
    brain = await BrainFactory.create(
        model="openai/gpt-4o",
        tenant_id="user-001",
    )
    
    result = await brain.process("What is quantum computing?")
    print(result.text)
    
    await brain.shutdown()

asyncio.run(main())
```

## `Brain.process()`

```python
async def process(
    self,
    text: str,
    images: list[str] | None = None,
    stream: bool = False,
) -> BrainResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | The input query or goal |
| `images` | `list[str] \| None` | `None` | Optional image paths for multimodal input |
| `stream` | `bool` | `False` | Enable streaming response |

**Returns:** `BrainResult` with `.text`, `.path`, `.latency_ms`, `.confidence`.

## `Brain.shutdown()`

```python
async def shutdown(self) -> None
```

Gracefully stops all nodes, flushes memory, and closes the message bus.
