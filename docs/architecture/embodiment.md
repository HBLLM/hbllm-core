# Embodiment & Real-World Actuation

HBLLM Core interacts with the physical or operating system environment safely through the **Execution Reality Layer**. This system ensures that when the agent takes an action, it is not merely relying on the theoretical success of a tool, but actually verifying that the physical or digital state was altered as expected.

## Core Components

### 1. Device Reality Integration (`hbllm/brain/embodiment/os_adapter.py`)
The `os_adapter` acts as a secure boundary between the cognitive loop and the host OS. 
- It maps semantic intentions (e.g., "turn off the wifi") into safe actuator/sensor interfaces.
- Provides isolated contexts for the agent to probe the OS state without risking unconstrained shell execution.

### 2. Execution Verification Engine (`hbllm/brain/embodiment/verifier.py`)
Large Language Models are prone to hallucinating the success of their actions based on API responses. 
- The `ExecutionVerifier` class implements **asynchronous non-blocking polling**. 
- After an action is executed, the verifier probes the actual physical or digital state (e.g., checking if the light is actually off, or if the file actually exists) to confirm the state change.
- Re-triggers the agent if reality diverges from the expected simulation.

### 3. Idempotency Tracking (`hbllm/brain/embodiment/idempotency.py`)
During complex, multi-step execution, retries or crashes are inevitable. 
- To prevent duplicate mutating actions, the system generates deterministic action hashes based on the parameters and context.
- Maintains a lock-table of executed actions. 
- If a crash occurs and the system reboots, the agent will not repeat a destructive or state-mutating action it had successfully executed immediately prior to the crash.

### 4. Platform-Specific Adapters

The OS Adapter delegates to platform-specific backends for deep OS integration:

| Platform | Module | Capabilities |
|----------|--------|-------------|
| **macOS** | `platform_mac.py` | AppleScript automation, Spotlight search, system preferences, Notification Center, Finder integration |
| **Linux** | `platform_linux.py` | D-Bus integration, systemd service control, NetworkManager, udev device events, X11/Wayland window management |

Each adapter implements a common `PlatformAdapter` interface so the cognitive pipeline remains platform-agnostic while leveraging native capabilities.

### 5. Confirmation Gate (`hbllm/actions/confirmation.py`)

Human-in-the-loop approval for high-risk actions before execution:

- **Risk classification** — Actions are classified into trust tiers (SAFE, MODERATE, SENSITIVE, CRITICAL).
- **Escalation** — SENSITIVE and CRITICAL actions require explicit user confirmation.
- **Timeout** — Pending confirmations expire after a configurable timeout (default: 5 minutes).
- **Audit integration** — All confirmation decisions are logged to the audit trail.

### 6. Rollback Engine (`hbllm/actions/rollback.py`)

Undo support for reversible actions with snapshot-based state recovery:

- **Pre-action snapshots** — Captures relevant state before executing mutating actions.
- **Rollback execution** — Applies inverse operations to restore previous state.
- **Cascade rollback** — Multi-step action chains can be rolled back as a unit.
- **Retention** — Snapshots are retained for a configurable period (default: 24 hours).

## Safety Architecture

```
User Request → Risk Classifier → Confirmation Gate (if needed) → Idempotency Check
    → Pre-Action Snapshot → OS Adapter (Platform-Specific) → Execution
    → Verification Polling → Success/Failure → Rollback (if failed)
```

All embodiment actions follow this pipeline to ensure safety, reversibility, and verification at every step.
