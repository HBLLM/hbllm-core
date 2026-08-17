---
title: "API Reference — Packages (.hbpkg) & Skills System"
description: "API reference for PackageBuilder, PackageManifest, SkillInstaller, SkillParser, and SkillRegistry."
---

# Packages & Skills API

The **Packages and Skills Subsystem** defines the APIs for compiling, packaging, signing, and installing modular extensions into the HBLLM runtime.

**Packages:** `hbllm.hbpkg`, `hbllm.skills`, `hbllm.brain.skills`

---

## Subsystem Index

| Class | Module | Purpose |
|---|---|---|
| `PackageManifest` | `hbllm.hbpkg.manifest` | Dataclass representing `manifest.json` metadata & dependencies |
| `PackageBuilder` | `hbllm.hbpkg.packager` | Compiles source directories into signed `.hbpkg` archives |
| `PackageExtractor`| `hbllm.hbpkg.packager` | Extracts and verifies `.hbpkg` archives into runtime directories |
| `SkillParser` | `hbllm.skills.parser` | Parses YAML frontmatter and markdown instructions from `SKILL.md` |
| `SkillInstaller` | `hbllm.skills.installer` | Manages local package installation and version upgrades |
| `SkillRegistry` | `hbllm.brain.skills.skill_registry` | Runtime procedural skill lookup and invocation index |

---

## `PackageBuilder` & `PackageManifest`

```python
from pathlib import Path
from hbllm.hbpkg.manifest import PackageManifest
from hbllm.hbpkg.packager import PackageBuilder

manifest = PackageManifest(
    name="system-auditor",
    version="1.0.0",
    description="Security auditing skill package",
    capabilities_required=["filesystem:read", "process:sandbox"],
)

builder = PackageBuilder(source_dir=Path("./src/auditor"))
archive_path = builder.build(
    output_path=Path("./dist/system-auditor.hbpkg"), manifest=manifest
)
print(f"Archive built at: {archive_path}")
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
