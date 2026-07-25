# HCIR Cognitive OS Kernel & Executive Runtime

> **Human-Cognitive Intermediate Representation (HCIR) & Kernel Architecture**
> 
> The HCIR Kernel is the central operating system kernel for HBLLM Core. It provides structured execution, dependency injection via `KernelServices`, priority thread execution, managed persistence, and sliding-window performance telemetry.

---

## Architecture

The Cognitive OS Kernel coordinates system execution through a decoupled services container:

```
  ┌───────────────────────────────────────────────────────────┐
  │                    ExecutiveRuntime                       │
  │     (Lifecycle Owner, Cycle Scheduler & Execution Mode)   │
  └─────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                     KernelServices                        │
  │    (Dependency Container for Cognitive OS ABI Execution)  │
  ├──────────────┬──────────────┬──────────────┬──────────────┤
  │ Workspace    │ TxManager    │ Resolver     │ Scheduler    │
  │ State        │ (Transactions│ (Capability  │ (Instruction │
  │ (HCIR Graph) │ & Rollbacks) │ Sandboxing)  │ Scheduler)   │
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ Executor     │ Persistence  │ Telemetry    │ EventBus     │
  │ (CPU / I/O   │ (Managed DB  │ (Sliding     │ (Kernel Pub/ │
  │ Pools)       │ Pools)       │ Latencies)   │ Sub Trie)    │
  └──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## Key Subsystems

### 1. `KernelServices` Dependency Container
Every cognitive node receives `KernelServices` via the ABI `execute()` contract:
- `workspace`: Active `HCIRWorkspaceState` (graph nodes, edges, goals).
- `transaction_manager`: Manages state modifications with atomic rollbacks.
- `capability_resolver`: Enforces permissions (filesystem, network, subprocess, db).
- `scheduler`: Attention-driven instruction dispatcher (`CognitiveScheduler`).
- `executor`: Priority thread pools for CPU- and I/O-bound tasks (`KernelExecutor`).
- `persistence`: Managed database connection pools (`PersistenceService`).
- `telemetry`: Micro-second latency and throughput metrics (`KernelTelemetry`).

### 2. `KernelExecutor` Thread Pools
Offloads heavy synchronous operations off the main `asyncio` event loop:
- **`run_cpu_bound()`**: Embedding calculations, matrix operations, tokenization.
- **`run_io_bound()`**: File I/O, SQLite operations, disk reads.

### 3. `PersistenceService` & Workload-Specific SQLite Profiles
Managed persistence layer providing workload-tuned SQLite connection PRAGMAs:
- `semantic_memory`: 512MB mmap, 64MB cache, WAL mode.
- `knowledge_graph`: 1GB mmap, 64MB cache, WAL mode.
- `event_log`: 32MB mmap, 4MB cache, WAL mode.
- `scheduler`: 2MB cache, NORMAL synchronous mode.

### 4. `Tenant-Aware EmbeddingCache`
Isolated embedding cache keyed composite tuple `(tenant_id, model_name, adapter_id, text_hash)`:
- Guarantees strict multi-tenant data isolation.
- Prevents stale cache hits when fine-tuned LoRA adapters or underlying embedding models change.

### 5. `KernelEventBus` Prefix Indexing & Batch Dispatch
Pub/sub kernel event bus featuring:
- Pre-cached wildcard and prefix matching (`transaction.*`).
- High-throughput batch dispatch (`publish_many()`).

### 6. Message Object Pooling (`MessagePool`)
Recycles `Message` instances on high-frequency channels to eliminate garbage collection overhead.

### 7. Dynamic Adaptive Throttling
`CognitiveStateMachine.tick_interval` dynamically scales tick intervals (1.5×–2.5×) during high system CPU load (>60% / >80%), preventing event loop starvation.

---

## Diagnostics CLI

Inspect system runtime health, event loop engine (`uvloop`), SQLite version, and native Rust acceleration crates:

```bash
python -m hbllm.cli.diagnostics
```
