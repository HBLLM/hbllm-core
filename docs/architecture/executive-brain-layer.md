# Executive Brain Layer Architecture

> **Module**: `core/hbllm/brain/autonomy/`  
> **Status**: Phase 1 & 2 complete  
> **Branch**: `feature/executive-brain-layer-phase1`

## Overview

The Executive Brain Layer transforms HBLLM from a reactive request→response system
into a **continuously operating cognitive organism**. It provides:

- **Continuous cognition** via a hybrid event + tick loop
- **Attentional control** via multi-factor event scoring
- **Adaptive resource management** via hierarchical state machine
- **Persistent goal tracking** via DAG-backed task graphs

## Architecture Diagram

```
                    ┌──────────────────────┐
                    │  External Events     │
                    │  (user input, sensor │
                    │   anomaly, device)   │
                    └──────────┬───────────┘
                               │
                    ╔══════════▼═══════════╗
                    ║   FAST PATH          ║
                    ║   (Event-Driven)     ║
                    ║   MessageBus →       ║
                    ║   asyncio.Event wake ║
                    ╚══════════╤═══════════╝
                               │
              ┌────────────────▼────────────────┐
              │         AutonomyCore            │
              │    (Cognitive Heartbeat)         │
              │                                 │
              │  ┌───────────┐ ┌──────────────┐ │
              │  │ Tier 1    │ │ Tier 2/3     │ │
              │  │ Reflexes  │ │ LLM Router   │ │
              │  │ (0 cost)  │ │ (escalation) │ │
              │  └───────────┘ └──────────────┘ │
              │                                 │
              │  ┌────────────────────────────┐ │
              │  │ Internal Thought Queue     │ │
              │  │ (deferred goals, reminders)│ │
              │  └────────────────────────────┘ │
              └──┬──────────────┬───────────────┘
                 │              │
    ┌────────────▼──┐  ┌───────▼──────────┐
    │  Attention    │  │ Cognitive State  │
    │  System       │  │ Machine          │
    │               │  │                  │
    │ • Scoring     │  │ • State hierarchy│
    │ • Decay       │  │ • Tick profiles  │
    │ • Budgets     │  │ • Guards/hooks   │
    │ • Context     │  │ • Interruption   │
    └───────────────┘  └─────────────────┘
                 │
    ┌────────────▼──────────────┐
    │   TaskGraphRuntime        │
    │   (Persistent Goals)      │
    │                           │
    │ • DAG execution           │
    │ • SQLite persistence      │
    │ • Retry / failure cascade │
    │ • Boot recovery           │
    └───────────────────────────┘
```

## Component Reference

### 1. CognitiveStateMachine (`state_machine.py`)

Controls the system's operating mode with a hierarchical state model.

#### State Hierarchy

| Category       | States                                           | Tick Rate   |
|----------------|--------------------------------------------------|-------------|
| **ACTIVE**     | `OBSERVING`, `FOCUSED`, `PLANNING`, `EXECUTING`  | 0.5–5s      |
| **PASSIVE**    | `IDLE`, `REFLECTING`, `LOW_POWER`                | 10–30s      |
| **TRANSITIONAL** | `INTERRUPTED`, `RECOVERING`, `SLEEPING`        | 1–60s       |

#### Adaptive Tick Profiles

Each state has a `TickProfile` that controls:
- `tick_interval_s` — how often the slow path runs
- `allow_heavy_llm` — whether Tier 3 reasoning is permitted
- `allow_fast_router` — whether Tier 2 routing is permitted
- `max_concurrent_thoughts` — parallel processing cap
- `interruption_threshold` — how hard it is to interrupt (0.0 = easy, 1.0 = impossible)

#### Interruption Flow

```
User Input → AttentionSystem.score_event() → priority_score
  → CognitiveStateMachine.should_allow_interruption(priority_score)
    → if True: save current state, transition to INTERRUPTED
    → on resolution: resume_from_interruption()
```

---

### 2. AttentionSystem (`attention.py`)

Multi-factor event prioritization and cognitive resource control.

#### Scoring Model

```
priority_score =
    0.30 × urgency
  + 0.20 × user_focus_weight
  + 0.10 × emotional_weight
  + 0.15 × temporal_relevance
  + 0.15 × goal_alignment
  - 0.05 × interruption_cost
  - 0.05 × cognitive_load_penalty
  - event_decay
```

#### Safety Mechanisms

| Mechanism              | Purpose                               | Default        |
|------------------------|---------------------------------------|----------------|
| **Event Decay**        | Debounce repeated similar events      | 30s window     |
| **Thought Budget**     | Cap thoughts per minute               | 30/min         |
| **Cooldown**           | Pause after budget burst              | 5s             |

#### Incremental Context Window

Maintains a rolling, salience-weighted view of active entities (people, topics,
devices) without expensive full-context rebuilds. Entities decay over time and
are pruned when salience drops below threshold.

---

### 3. AutonomyCore (`loop.py`)

The cognitive heartbeat — a hybrid event + tick daemon.

#### Dual-Path Architecture

| Path       | Trigger             | Latency  | Use Case                      |
|------------|---------------------|----------|-------------------------------|
| **Fast**   | MessageBus event    | Instant  | User input, sensor anomaly    |
| **Slow**   | Periodic tick       | Adaptive | Reflection, planning, pruning |

#### Tiered LLM Invocation

| Tier | Name            | Cost     | When Used                          |
|------|-----------------|----------|------------------------------------|
| 1    | Reflex          | Zero     | Deterministic rules, heuristics    |
| 2    | Fast Router     | Low      | Intent classification, urgency     |
| 3    | Heavy Reasoning | High     | Complex planning, synthesis        |

---

### 4. TaskGraphRuntime (`task_graph.py`)

Persistent, resumable goal execution engine.

#### Goal Lifecycle

```
PENDING → ACTIVE → COMPLETED
                 → FAILED
         → PAUSED → ACTIVE (resume)
         → CANCELLED
```

#### Task DAG Execution

- Root tasks (no dependencies) auto-promote to `READY`
- Completing a task cascades readiness to dependent tasks
- Failed tasks (after retries exhausted) `BLOCK` all dependents
- Goal auto-completes when all tasks reach terminal state

#### Boot Recovery

Tasks left in `RUNNING` state (from a crash/reboot) are automatically
reset to `READY` on startup via `recover_on_boot()`.

## Test Coverage

| Module                     | Tests | Status |
|----------------------------|-------|--------|
| CognitiveStateMachine      | 20    | ✅     |
| IncrementalContextWindow   | 8     | ✅     |
| AttentionSystem            | 8     | ✅     |
| AutonomyCore               | 11    | ✅     |
| TaskGraphRuntime           | 25    | ✅     |
| **Total**                  | **72**| ✅     |

## Cross-References

- [Adaptive Network Architecture](./adaptive-network.md) — the transport layer below this
- `hbllm/brain/scheduler_node.py` — the legacy scheduler to be migrated
- `hbllm/brain/attention_manager.py` — memory-focused attention (complementary)
