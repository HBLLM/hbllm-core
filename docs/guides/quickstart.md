---
title: "Quick Start Guide — HBLLM Core"
description: "Get HBLLM Core running in under 5 minutes on any hardware — no GPU required. Installation, CLI usage, API server, and Python SDK."
---

# Quick Start

Get HBLLM Core running in under 5 minutes.

## Prerequisites

- Python 3.10+
- pip or uv package manager
- **No GPU required** — HBLLM runs on CPU-only machines (laptops, Raspberry Pi 5, servers)
- (Optional) CUDA-capable GPU for faster local model inference
- (Optional) Rust toolchain for building SIMD compute kernels

!!! tip "Minimal Hardware"
    The 125M model runs in under 500MB RAM on any modern CPU. You can start building cognitive agents today on any laptop — no cloud GPU rentals needed.

## Installation

```bash
# Clone the repository
git clone https://github.com/hbllm/hbllm-core.git
cd HBLLM/core

# Install in development mode
pip install -e ".[dev]"

# Optional integrations
pip install paho-mqtt         # IoT / MQTT
export HBLLM_ROS2_ENABLED=1  # ROS2 Robotics (requires rclpy)
```

## Verify Installation

```bash
# Run the test suite
python -m pytest tests/ -v

# Check brain architecture
hbllm info
```

Expected output:
```
🧠 HBLLM Core v0.1.0
Nodes:   28 specialized brain nodes
Memory:  6 systems (Episodic, Semantic, Procedural, Value, Working, KG)
Model:   Zoning — shared base + LoRA domain adapters
Sizes:   125M / 500M / 1.5B parameters
```

## Start the API Server

=== "Local Model"

    ```bash
    # Requires downloaded safetensors
    python -m hbllm.serving.api
    ```

=== "OpenAI Backend"

    ```bash
    HBLLM_PROVIDER=openai \
    OPENAI_API_KEY=sk-... \
    python -m hbllm.serving.api
    ```

=== "Anthropic Backend"

    ```bash
    HBLLM_PROVIDER=anthropic \
    ANTHROPIC_API_KEY=sk-ant-... \
    python -m hbllm.serving.api
    ```

The server starts on `http://localhost:8000` with auto-generated OpenAPI docs at `/docs`.

## Python SDK

```python
import asyncio
from hbllm.brain.factory import BrainFactory

async def main():
    # Create a brain with OpenAI backend
    brain = await BrainFactory.create("openai/gpt-4o")
    
    # Process a complex query
    result = await brain.process(
        "Analyze our server logs and design a firewall rule.",
        tenant_id="tenant-001",
    )
    
    print(f"Decision: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    
    await brain.shutdown()

asyncio.run(main())
```

## CLI Commands

```bash
hbllm info                # View active brain architecture
hbllm nodes               # List all loaded cognitive nodes
hbllm serve --port 8000   # Start FastAPI + MCP Server
hbllm train --model-size 125m  # Start local pre-training loop
hbllm data --dataset fineweb   # Run data preparation pipeline
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) — Understand the system design.
- [Configuration Guide](configuration.md) — Customize HBLLM for your environment.
- [Custom Nodes](custom-nodes.md) — Extend the brain with your own nodes.
- [Deployment Guide](deployment.md) — Deploy to production.
