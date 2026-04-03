---
title: "LoRA Routing — Dynamic Domain Expert Selection"
description: "How HBLLM's Adapter Registry resolves, downloads, and hot-swaps LoRA adapters with SHA-256 integrity verification."
---

# LoRA Routing & Adapter Registry

## Adapter Lifecycle

```mermaid
sequenceDiagram
    participant Q as Query
    participant R as Router
    participant AR as AdapterRegistry
    participant HF as HuggingFace Hub
    participant M as Base Model

    Q->>R: "Explain quantum entanglement"
    R->>AR: resolve("physics")
    AR->>AR: Check local cache
    alt Cache hit
        AR-->>R: Return cached adapter
    else Cache miss
        AR->>HF: Download adapter (pinned revision)
        HF-->>AR: Adapter weights
        AR->>AR: Verify SHA-256 checksum
        AR->>AR: Convert PEFT → HBLLM state_dict
        AR-->>R: Return validated adapter
    end
    R->>M: Hot-swap LoRA weights
    M-->>Q: Domain-specialized response
```

## AdapterRegistry Configuration

```python
from hbllm.modules.adapter_registry import (
    AdapterRegistry,
    AdapterRegistryConfig,
    AdapterSource,
)

config = AdapterRegistryConfig(
    enabled=True,
    cache_dir="./checkpoints/adapters",
    auto_download=True,
    require_sha256=True,
    max_adapter_size_mb=100,
    sources=[
        AdapterSource(
            domain="coding",
            repo_id="hbllm/coding-lora-v2",
            revision="v2.1.0",  # Pinned Git tag
            sha256="abc123...",
            rank=8,
            peft_format=False,
        ),
        AdapterSource(
            domain="math",
            repo_id="hbllm/math-lora-v1",
            revision="main",
        ),
    ],
)

registry = AdapterRegistry(config)
```

## Security

- All downloaded adapters are verified against their SHA-256 checksum
- `weights_only=True` enforced on all `torch.load()` calls
- PEFT format conversion is sandboxed
- Revisions can be pinned to specific Git tags, branches, or commit SHAs
