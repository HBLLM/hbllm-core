---
title: "Plugin SDK — Mastering Extensions"
description: "How to safely extend HBLLM with the high-level Plugin SDK, featuring auto-discovery and declarative event binding."
---

# The HBLLM Plugin SDK

HBLLM is designed to be lean at its core. While you can write [Custom Nodes](custom-nodes.md) manually, the **Plugin SDK** provides a high-level, declarative way to build third-party tools, integrations, and UI adapters without touching the core codebase.

## Why Use the Plugin SDK?

- **Isolation**: Keep your custom logic separate from the HBLLM core.
- **Scaffolding**: Single-command setup for new projects.
- **Auto-Discovery**: HBLLM automatically finds and loads your plugins from the `plugins/` directory.
- **Declarative**: Use decorators to bind logic to MessageBus events effortlessly.

---

## 🛠️ Quick Start: "Hello World" Plugin

The easiest way to start is with the CLI:

```bash
hbllm plugin new my-logger
```

This creates a new package at `plugins/my_logger/`. Let's look at a simple implementation:

```python
from hbllm.network.messages import Message
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

__plugin__ = {
    "name": "my_logger",
    "version": "0.1.0",
    "description": "Logs every query seen on the bus."
}

class MyLoggerPlugin(HBLLMPlugin):
    """A simple plugin that prints bus traffic."""

    @subscribe("router.query")
    async def on_query(self, message: Message) -> None:
        text = message.payload.get("text", "")
        print(f"🔍 [MyLogger] User asked: {text}")

    @subscribe("sensory.output")
    async def on_output(self, message: Message) -> None:
        print(f"🧠 [MyLogger] Brain replied: {message.payload.get('text')}")
```

---

## 🧬 Core Concepts

### 1. `HBLLMPlugin` Base Class
Every plugin must inherit from `hbllm.plugin.sdk.HBLLMPlugin`. This base class handles registration with the [Service Registry](../architecture/network-resilience.md) and provides a pre-configured `self.bus` reference.

### 2. The `@subscribe` Decorator
Instead of manually calling `self.bus.subscribe()` in `on_start()`, use the `@subscribe` decorator. The SDK handles the heavy lifting of attaching your handlers once the node joins the network.

### 3. Plugin Metadata
The `__plugin__` dictionary is used for discovery and versioning. This information appears when you run `hbllm plugin list`.

---

## 🔧 CLI Commands

HBLLM provides built-in tools to manage your extension library.

### List Plugins
See all currently installed and discovered plugins:
```bash
hbllm plugin list
```

### Scaffold New Plugin
Create a boilerplate folder structure:
```bash
hbllm plugin new <name>
```

---

## 💡 Best Practices

!!! tip "Use Scoped Node IDs"
    When inheriting from `HBLLMPlugin`, the SDK automatically generates a unique `node_id` based on your plugin name. Avoid hardcoding IDs if you plan to run multiple instances of a plugin.

!!! warning "Avoid Blocking Code"
    HBLLM is fully asynchronous. Never use blocking `time.sleep()` or synchronous requests. Use `asyncio.sleep()` and `httpx` for external API calls.

!!! info "Lifecycle Hooks"
    You can still override `on_start()` and `on_stop()` for manual resource management (like closing database connections or cleaning up temporary files). Always call `await super().on_start()` if you maintain the override.

---

## 📦 Plugin Bundle Format

Plugins are distributed as directory bundles with a `plugin.json` manifest:

```
plugins/my-plugin/
├── plugin.json        # Manifest v2 (required)
├── my_engine.py       # Entry point (HBLLMPlugin subclass)
├── knowledge/         # Auto-ingested knowledge files
├── skills/            # Skill YAML definitions
├── policies/          # Governance policy files
├── prompts/           # Prompt templates
└── tests/
    └── test_my_plugin.py
```

### Manifest (`plugin.json`)

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "manifest_version": 2,
  "description": "What this plugin does",
  "author": "your-name",
  "entry_point": "my_engine.py",
  "dependencies": [],
  "tags": ["category"],
  "capabilities": ["what_it_provides"],
  "knowledge_dir": "knowledge",
  "skills_file": "skills/skills.yaml",
  "policies_file": "policies/policies.yaml"
}
```

---

## 🧠 Built-in Cognitive Plugins

HBLLM ships with three cognitive plugins that extend the brain's reasoning capabilities:

### Emotion Modeling (`emotion-modeling`)

Tracks emotional valence across conversations using a VAD (Valence-Arousal-Dominance) model with lexicon-based text analysis and exponential decay.

```python
from hbllm.plugin.sdk import HBLLMPlugin, subscribe
from hbllm.network.messages import Message

class EmotionEngine(HBLLMPlugin):
    @subscribe("system.experience")
    async def on_experience(self, message: Message) -> None:
        # Analyze text for emotional content using VAD lexicon
        text = message.payload.get("text", "")
        self._update_from_text(text)
        await self._publish_state()  # → "emotion.state"
```

**Topics**: `system.experience`, `system.evaluation` → `emotion.state`
**Capabilities**: Tone adaptation hints (empathetic, enthusiastic, focused, clarifying)

### Temporal Reasoning (`temporal-reasoning`)

Adds time-aware context to conversations. Recognizes temporal references ("yesterday", "last week") and provides deadline tracking.

**Topics**: `system.experience` → `temporal.context`
**Capabilities**: Event history, temporal reference parsing, deadline tracking

### Swarm Orchestrator (`swarm-orchestrator`)

Decomposes complex tasks into parallelizable subtasks, dispatches them to independent workers, and aggregates results.

```python
engine = SwarmEngine(max_workers=4, task_timeout=60.0)
result = await engine.execute("1. Research topic\n2. Write draft\n3. Review")
# → Runs steps 1-3 with dependency resolution and parallel execution
```

**Topics**: `swarm.request` → `swarm.complete`
**Capabilities**: Heuristic task decomposition, dependency resolution, configurable parallelism

---

## 🧪 Testing Plugins

Each plugin should include localized tests in a `tests/` directory:

```bash
# Run a single plugin's tests
python -m pytest hbllm/plugins/emotion-modeling/tests/ -v --timeout=30

# Run all plugin tests
python -m pytest hbllm/plugins/*/tests/ -v --timeout=30
```

Tests import the engine directly via `sys.path` manipulation (since plugin directories use hyphens):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_engine import EmotionEngine
```

