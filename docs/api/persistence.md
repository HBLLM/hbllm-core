---
title: "API Reference — Persistence & Disaster Recovery"
description: "API reference for PersistenceService, DBPool, SQLiteProfile, and BackupManager."
---

# Persistence & Disaster Recovery API

The **Persistence Subsystem** provides managed async connection pools, workload-tuned SQLite PRAGMAs, and disaster recovery snapshot capabilities.

**Packages:** `hbllm.persistence`, `hbllm.backup`

---

## Subsystem Index

| Class | Module | Purpose |
|---|---|---|
| `PersistenceService` | `hbllm.persistence.service` | Central database service coordinating connection pools & profiles |
| `DBPool` | `hbllm.persistence.db_pool` | Async SQLite connection pool with acquisition context managers |
| `SQLiteProfile` | `hbllm.persistence.sqlite_profiles` | Tuned PRAGMA sets (WAL, mmap_size, cache_size, synchronous) |
| `ProfileManager` | `hbllm.persistence.sqlite_profiles` | Workload-to-profile resolver |
| `BackupManager` | `hbllm.backup` | Point-in-time database snapshotting, compression, and verification |
| `BackupConfig` | `hbllm.backup` | Retention count, compression level, and target directory settings |

---

## `PersistenceService` & `DBPool`

```python
from hbllm.persistence.service import PersistenceService

service = PersistenceService(data_dir="./data")
await service.initialize()

# Get connection pool for a specific database
pool = service.get_pool("working_memory.db", profile="semantic_memory")

async with pool.acquire() as conn:
    await conn.execute("CREATE TABLE IF NOT EXISTS sample (id INT, val TEXT)")
    await conn.commit()

await service.shutdown()
```

---

## `BackupManager`

```python
import asyncio
from hbllm.backup import BackupConfig, BackupManager

manager = BackupManager(
    config=BackupConfig(
        data_dir="./data",
        backup_dir="./backups",
        retention_count=10,
        compress=True,
    )
)

# Create an atomic snapshot
metadata = await manager.create_backup(label="pre-migration")
print(f"Created backup {metadata.backup_id} (size: {metadata.size_bytes} bytes)")

# Verify checksum
is_valid = await manager.verify_backup(metadata.backup_id)
print(f"Checksum valid: {is_valid}")
```
