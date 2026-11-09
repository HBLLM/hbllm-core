# System Entry Points & Bootstrapping Architecture

> **Document Status:** Authoritative Architectural Specification  
> **Subsystem:** `hbllm.runtime.bootstrap`, `hbllm.cli`, `hbllm.__main__`  
> **Invariants:** 1 (State Ownership), 4 (Constitutional Governance), 7 (Domain-Agnostic Core)

---

## 1. Overview

HBLLM enforces a **single unified composition root** across all runtime contexts. Whether invoked via the Typer CLI, background service workers, evaluation harnesses, or integration tests, the cognitive system is initialized exclusively through `build_cognitive_system()`.

Duplicate kernel initializations, unverified `TransactionManager` instances, and ad-hoc wiring scripts are strictly prohibited.

---

## 2. The Single Composition Root (`hbllm.runtime.bootstrap`)

The canonical entry point for initializing the entire cognitive operating system:

```python
from hbllm.runtime.bootstrap import (
    CognitiveSystem,
    CognitiveSystemConfig,
    build_cognitive_system,
)

config = CognitiveSystemConfig(
    tenant_id="primary",
    enable_grounding=True,
    enable_governance=True,
    perception_mode="handcrafted",
)

system: CognitiveSystem = build_cognitive_system(config)
```

### 2.1 Initialization Lifecycle

1. **Workspace & Substrate Initialization:**
   - Instantiates `TieredWorkspace` with Working, Brain, Persistent, and Meta tiers.
   - Attaches `CognitiveGraph` (with native Rust PyO3 acceleration when compiled, falling back to pure Python).
2. **Constitutional Governance:**
   - Instantiates `ConstitutionalVerifier` with security, safety, and policy verification stages.
   - Instantiates `TransactionManager` bound to the verifier.
3. **Kernel Services:**
   - Assembles `KernelServices` containing `TransactionManager`, `EventBus`, `CapabilityResolver`, and `TieredWorkspace`.
   - Strictly asserts `not tm.unverified` upon completion.
4. **Autonomous Epistemic Engine:**
   - Instantiates `AutonomousEpistemicEngine` with explicit `KernelServices`.
5. **Cortical Grounding Bridge:**
   - Binds the Symbol Grounding Engine (`CorticalGroundingBridge`) sharing the identical `KernelServices`.

---

## 3. Unified Typer CLI (`hbllm.cli`)

The CLI is structured into clean cognitive commands replacing legacy transformer-pretraining entry points.

Executable via:
```bash
python -m hbllm [COMMAND] [ARGS]...
# or
hbllm [COMMAND] [ARGS]...
```

### 3.1 Command Hierarchy

| Command Group | Command | Description |
| :--- | :--- | :--- |
| **`kernel`** | `hbllm kernel boot` | Bootstraps the HCIR kernel and reports verified subsystem readiness. |
| | `hbllm kernel status` | Queries live graph statistics, dirty sets, substrate acceleration, and active tenants. |
| | `hbllm kernel replay <tx_log>` | Replays transactional event streams to verify deterministic state reconstruction. |
| **`mind`** | `hbllm mind run --driver <name>` | Runs continuous active inference loop driven by environment driver. |
| | `hbllm mind step --obs <file.json>` | Executes a deterministic single-cycle cognitive decision step. |
| **`bench`** | `hbllm bench dev` | Executes local development evaluation benchmark suite. |
| | `hbllm bench heldout` | Executes anti-tamper held-out generalization benchmark (gated by Git SHA). |
| | `hbllm bench local-suite` | Executes standard local game evaluation suite. |
| | `hbllm bench ablation` | Measures faculty-level ablation impact on task success. |
| | `hbllm bench sge` | Benchmarks symbol grounding precision, latency, and memory bounds. |
| **`diagnostics`**| `hbllm diagnostics` | Full system integrity check: verifier health, native substrate, memory discipline. |

---

## 4. Architectural Layering Guarantees

All entry points adhere to strict AST-level dependency boundaries validated in `tests/unit/architecture/test_layering.py`:
- `hbllm.hcir` never imports from `hbllm.brain`.
- Core OS never imports from `plugins`.
- Direct `sqlite3.connect` and `aiosqlite.connect` calls are confined strictly to infrastructure persistence modules (`hbllm.persistence`, `hbllm.hcir.stores`, `hbllm.security`).
