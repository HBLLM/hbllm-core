---
title: "API Reference — Packages (.hbpkg) & Skills System"
description: "API reference for PackageBuilder, PackageManifest, SkillInstaller, SkillParser, and SkillRegistry."
---

# Packages & Skills API

The **Packages and Skills Subsystem** defines APIs for declarative neuro-symbolic skills (`hbllm.hcir.skills`) alongside `.hbpkg` package compilation, signing, and installation (`hbllm.hbpkg`, `hbllm.skills`).

**Packages:** `hbllm.hcir.skills`, `hbllm.hbpkg`, `hbllm.skills`, `hbllm.brain.skills`

---

## Subsystem Index

| Class | Module | Purpose |
|---|---|---|
| `DeclarativeNeuroSymbolicSkill` | `hbllm.hcir.skills.declarative` | Composable `SymbolicPredicate` + `SubgoalSequence` skill DSL |
| `SkillInductionEngine` | `hbllm.hcir.skills.induction` | Autonomous skill synthesis from recurring trajectories |
| `HBPkgManifest` | `hbllm.hbpkg.manifest` | Dataclass representing `manifest.json` metadata, permissions & lifecycle hooks |
| `Packager` | `hbllm.hbpkg.packager` | Compiles, verifies, and installs `.hbpkg` archives |
| `SkillParser` | `hbllm.skills.parser` | Parses YAML frontmatter and markdown instructions from `SKILL.md` |
| `SkillInstaller` | `hbllm.skills.installer` | Manages local package installation and version upgrades |
| `SkillRegistry` | `hbllm.brain.skills.skill_registry` | Runtime procedural skill lookup and invocation index |

---

## `DeclarativeNeuroSymbolicSkill` (`hbllm.hcir.skills.declarative`)

```python
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    CutSetPredicate,
    DeclarativeNeuroSymbolicSkill,
    SpatialActionIntent,
    SubgoalSequence,
    SymbolicSubgoal,
)


class KeyDoorSkill(DeclarativeNeuroSymbolicSkill):
    name = "key_door_skill"
    signature = ActionAffordancePredicate(required={1, 2, 3, 4}) & CutSetPredicate(min_partitions=2)
    program = SubgoalSequence(
        SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE_TO, target_query={"role": "cargo"}),
        SymbolicSubgoal(intent=SpatialActionIntent.INTERACT, target_query={"role": "barrier"}),
    )
```

---

## `Packager` & `HBPkgManifest`

```python
from pathlib import Path
from hbllm.hbpkg.manifest import HBPkgManifest
from hbllm.hbpkg.packager import Packager

manifest = HBPkgManifest(
    name="system-auditor",
    version="1.0.0",
    description="Security auditing skill package",
)

packager = Packager()
print(manifest.name, manifest.version)
```

---

## `SkillParser` & `SkillInstaller`

```python
import asyncio
from hbllm.skills.installer import SkillInstaller
from hbllm.skills.parser import SkillParser

# Parse a markdown skill
parser = SkillParser()
skill = parser.parse_file("skills/diagnose_network.md")
print(f"Skill name: {skill.name}, Triggers: {skill.triggers}")

# Install package
installer = SkillInstaller(target_dir="~/.hbllm/packages")
await installer.install_package("./dist/system-auditor.hbpkg")
```

---

## `SkillRegistry`

```python
from hbllm.brain.skills.skill_registry import SkillRegistry

registry = SkillRegistry()

# Register a skill
registry.register_skill(skill)

# Find skill by prompt similarity
matched_skill = registry.find_skill("Diagnose slow network latency")
if matched_skill:
    print(f"Found matching skill: {matched_skill.name}")
```
