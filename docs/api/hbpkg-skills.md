---
title: "API Reference — Packages (.hbpkg) & Skills System"
description: "API reference for PackageBuilder, PackageManifest, SkillInstaller, SkillParser, and SkillRegistry."
---

# Packages & Skills API

The **Packages and Skills Subsystem** defines APIs for declarative neuro-symbolic skills (`hbllm.hcir.skills`) alongside `.hbpkg` package compilation, signing, and installation (`hbllm.hbpkg`, `hbllm.skills`).

**Packages:** `hbllm.hcir.skills`, `hbllm.hbpkg`, `hbllm.skills`

---

## Subsystem Index

| Class | Module | Purpose |
|---|---|---|
| `DeclarativeNeuroSymbolicSkill` | `hbllm.hcir.skills.declarative` | Composable `SymbolicPredicate` + `SubgoalSequence` skill DSL (`evaluate_degree()`) |
| `SkillInductionEngine` | `hbllm.hcir.skills.induction` | Autonomous skill synthesis from recurring trajectories |
| `LibraryLearner` | `hbllm.hcir.skills.library_learner` | Wake–Sleep MDL anti-unification into parameterized `SkillNode` programs |
| `HCIRSkillStore` | `hbllm.hcir.skills.skill_store` | Persistent lookup and invocation index for HCIR skill bytecode streams |
| `HBPkgManifest` | `hbllm.hbpkg.manifest` | Dataclass representing `manifest.json` metadata, permissions & lifecycle hooks |
| `Packager` | `hbllm.hbpkg.packager` | Compiles, verifies, and installs `.hbpkg` archives |
| `SkillParser` | `hbllm.skills.parser` | Parses YAML frontmatter and markdown instructions from `SKILL.md` |
| `SkillInstaller` | `hbllm.skills.installer` | Manages local package installation and version upgrades |

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

## `HCIRSkillStore`

```python
from hbllm.hcir.skills.skill_store import HCIRSkillStore
from hbllm.hcir.workspace_tiers import TieredWorkspace

tiers = TieredWorkspace()
store = HCIRSkillStore(workspace=tiers.working_state, tiered_workspace=tiers)
records = store.list_skills()
```
