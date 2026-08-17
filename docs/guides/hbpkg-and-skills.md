---
title: "Packages (.hbpkg) & Skills SDK — Distribution & Induction"
description: "How to package, sign, distribute, and install modular HBLLM capabilities using the .hbpkg archive format and declarative Skill specifications."
---

# Packages (`.hbpkg`) & Skills SDK

HBLLM provides a standardized archive format (`.hbpkg`) and a declarative Skill definition standard for packaging, sharing, and hot-loading cognitive modules, domain LoRAs, and procedural skills.

---

## The `.hbpkg` Package Standard

An `.hbpkg` file is an authenticated, compressed container containing cognitive code, declarative metadata, and optional model adapter weights.

### Package Structure

```
my-package.hbpkg (compressed archive)
├── manifest.json         # Package metadata, permissions, entry points
├── skills/               # Declarative SKILL.md definition files
│   └── weather_check.md
├── adapters/             # Domain LoRA safetensors (optional)
│   └── adapter_model.safetensors
└── nodes/                # Custom Python or native cognitive nodes
    └── custom_node.py
```

### `manifest.json` Specification

```json
{
  "schema_version": "1.0.0",
  "name": "devops-incident-responder",
  "version": "1.2.0",
  "author": {
    "name": "HBLLM Core Team",
    "email": "dev@hbllm.org"
  },
  "description": "Autonomous Kubernetes and cloud infrastructure incident response package",
  "capabilities_required": [
    "network:egress",
    "process:sandbox",
    "memory:procedural"
  ],
  "entry_points": {
    "node": "nodes.incident_node:IncidentNode",
    "skills": ["skills/triage.md", "skills/rollback.md"]
  },
  "min_core_version": "0.1.0"
}
```

---

## Packaging an `.hbpkg`

Use the Python SDK `hbllm.hbpkg.packager` to bundle and validate packages:

```python
from pathlib import Path
from hbllm.hbpkg.manifest import PackageManifest
from hbllm.hbpkg.packager import PackageBuilder

# 1. Define manifest
manifest = PackageManifest(
    name="devops-incident-responder",
    version="1.2.0",
    description="Autonomous incident response package",
    capabilities_required=["network:egress"],
)

# 2. Build the .hbpkg archive
builder = PackageBuilder(source_dir=Path("./my_package_source"))
package_path = builder.build(
    output_path=Path("./dist/devops-incident-responder.hbpkg"),
    manifest=manifest,
)
print(f"Built package at: {package_path}")
```

---

## Declarative Skill Definitions (`SKILL.md`)

Skills represent procedural knowledge that can be parsed, registered into `ProceduralMemory`, and invoked during planning.

### Anatomy of a `SKILL.md`

```markdown
---
name: "kubernetes-pod-restart"
description: "Gracefully restart unhealthy Kubernetes pods and verify restoration."
triggers:
  - "restart pod"
  - "fix crashing deployment"
required_tools:
  - "shell_exec"
  - "k8s_api"
parameters:
  namespace:
    type: "string"
    default: "default"
  pod_name:
    type: "string"
    required: true
---

# Execution Steps

1. Check pod status and recent termination reason via `kubectl describe pod {{pod_name}} -n {{namespace}}`.
2. Save crash logs to diagnostic archive.
3. Delete the pod gracefully with `--grace-period=30` to trigger deployment recreation.
4. Poll pod status until condition `Ready == True` with a 60s timeout.
```

---

## Installing & Loading Skills

### Programmatic Installation

```python
import asyncio
from hbllm.skills.installer import SkillInstaller
from hbllm.skills.parser import SkillParser


async def main():
    # 1. Parse markdown skill
    parser = SkillParser()
    skill = parser.parse_file("skills/kubernetes-pod-restart.md")
    print(f"Parsed skill: {skill.name} with triggers: {skill.triggers}")

    # 2. Install package into local runtime
    installer = SkillInstaller(target_dir="~/.hbllm/packages")
    await installer.install_package("./dist/devops-incident-responder.hbpkg")
    print("Package installed successfully!")


asyncio.run(main())
```

---

## Autonomous Skill Induction

HBLLM does not only execute pre-written skills; the **SkillCompilerNode** (`hbllm.brain.skills.skill_compiler_node`) and **SkillInductionNode** (`hbllm.brain.skills.skill_induction_node`) automatically synthesize new procedural skills when successful multi-step tool executions recur in `EpisodicMemory`.
