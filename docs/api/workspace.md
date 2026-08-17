---
title: "API Reference — Workspace Management"
description: "API reference for WorkspaceManager, session directory structures, and cognitive scratchpad isolation."
---

# Workspace Management API

The **Workspace Management Subsystem** provides filesystem scratchpad isolation, temporary code artifact execution areas, and session workspace lifecycles.

**Package:** `hbllm.workspace`

---

## Subsystem Index

| Class | Module | Role |
|---|---|---|
| `WorkspaceManager` | `hbllm.workspace.workspace_manager` | Allocates and isolates tenant and session directories |
| `WorkspaceSession` | `hbllm.workspace.workspace_manager` | Context manager handling cleanup and artifact preservation |

---

## `WorkspaceManager`

```python
from pathlib import Path
from hbllm.workspace.workspace_manager import WorkspaceManager

manager = WorkspaceManager(base_dir=Path("./workspace"))

# Acquire an isolated workspace for a session
with manager.session_workspace(
    tenant_id="tenant-alpha", session_id="session-42"
) as ws:
    scratch_file = ws.path / "temp_script.py"
    scratch_file.write_text("print('Hello from sandboxed workspace')")
    print(f"Executing in isolated workspace: {ws.path}")
```
