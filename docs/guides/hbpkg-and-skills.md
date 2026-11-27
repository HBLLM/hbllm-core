---
title: "Packages (.hbpkg) & Skills SDK — Distribution & Induction"
description: "How to package, sign, distribute, and install modular HBLLM capabilities using the .hbpkg archive format and declarative Skill specifications."
---

# Packages (`.hbpkg`) & Skills SDK

HBLLM provides a standardized archive format (`.hbpkg`), declarative neuro-symbolic skills (`hbllm.hcir.skills.declarative`), and `SKILL.md` definitions (`hbllm.skills`) for packaging, sharing, and hot-loading cognitive modules and procedural skills.

---

## The `.hbpkg` Package Standard

An `.hbpkg` file is an authenticated, compressed container containing cognitive code, declarative metadata, and optional skill/capability declarations.

### Package Structure

```
my-package.hbpkg (compressed archive)
├── manifest.json         # Package metadata, permissions, entry points
├── skills/               # Declarative SKILL.md definition files
│   └── weather_check.md
└── nodes/                # Custom Python or declarative skill modules
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
from hbllm.hbpkg.manifest import HBPkgManifest
from hbllm.hbpkg.packager import Packager

manifest = HBPkgManifest(
    name="devops-incident-responder",
    version="1.2.0",
    description="Autonomous incident response package",
)
packager = Packager()
print(manifest.name, manifest.version)
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

Beyond pre-written `SKILL.md` bundles, HBLLM automatically synthesizes declarative skills inside the Pure-HCIR core:
- **`DeclarativeNeuroSymbolicSkill`** (`hbllm.hcir.skills.declarative`): Expresses skills as composable `SymbolicPredicate` signatures (`evaluate_degree()`) paired with `SubgoalSequence` instructions compiled into `InstructionStream`s.
- **`SkillInductionEngine`** (`hbllm.hcir.skills.induction`), **`LibraryLearner`** (`hbllm.hcir.skills.library_learner`), & **`SkillMacroChunker`** (`hbllm.hcir.reasoning.operators.skill_chunker`): Induce and anti-unify reusable parameterized `SkillNode` bytecode programs from recurring trajectories and store them in `HCIRSkillStore`, `PrefrontalWorkingMemory.procedural_library`, and `TieredWorkspace`.
