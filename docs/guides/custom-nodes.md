---
title: "Custom Nodes — Extending the Brain"
description: "Step-by-step guide to creating custom cognitive nodes for HBLLM's async message bus architecture."
---

# Writing Custom Nodes

HBLLM's architecture is designed for extensibility. Any new capability — sensors, APIs, reasoning engines — is added as a **Node**.

## Node Anatomy

Every node inherits from `hbllm.network.node.Node` and communicates via the message bus:

```python
from hbllm.network.node import Node, NodeType
from hbllm.network.messages import Message, MessageType

class WeatherNode(Node):
    """Fetches weather data and publishes to the cognitive stream."""

    def __init__(self, api_key: str):
        super().__init__(
            node_id="weather-sensor",
            node_type=NodeType.DETECTOR,
            capabilities=["weather", "temperature", "forecast"]
        )
        self.api_key = api_key

    async def on_start(self, bus):
        """Called when the node joins the bus."""
        await bus.subscribe("query.weather", self.handle_weather_query)

    async def handle_weather_query(self, topic: str, message: Message):
        """Respond to weather queries."""
        city = message.payload.get("city", "London")
        
        # Fetch weather data (your logic here)
        weather = await self._fetch_weather(city)
        
        # Publish response back to the bus
        await self.publish("response.weather", Message(
            type=MessageType.RESPONSE,
            source_node_id=self.node_id,
            payload=weather,
            correlation_id=message.correlation_id,
        ))
```

## Node Types

| Constant | Value | Examples |
|---|---|---|
| `NodeType.ROUTER` | `router` | RouterNode |
| `NodeType.CORE` | `core` | Critic, Decision, Workspace |
| `NodeType.DOMAIN_MODULE` | `domain_module` | Domain LoRA modules |
| `NodeType.MEMORY` | `memory` | MemoryNode |
| `NodeType.PLANNER` | `planner` | PlannerNode |
| `NodeType.LEARNER` | `learner` | LearnerNode |
| `NodeType.DETECTOR` | `detector` | CuriosityNode, Sensors |
| `NodeType.SPAWNER` | `spawner` | SpawnerNode |
| `NodeType.META` | `meta` | MetaReasoningNode, ExperienceNode |
| `NodeType.PERCEPTION` | `perception` | VisionNode, AudioInputNode |

## Registering Your Node

Add your node to the brain via `BrainFactory`:

```python
from hbllm.brain.factory import BrainFactory

brain = await BrainFactory.create("openai/gpt-4o")

# Register and start your custom node
weather = WeatherNode(api_key="your-key")
await weather.start(brain.bus)
```

## Best Practices

!!! tip "Keep Nodes Focused"
    Each node should do **one thing well**. If your node is growing complex, split it into multiple nodes that communicate via the bus.

!!! warning "Never Call Nodes Directly"
    Nodes must communicate **only** through published messages. Direct function calls break the architecture's decoupling guarantees.

!!! info "Use Correlation IDs"
    When responding to a query, always set `correlation_id` to link the response back to the original request.
