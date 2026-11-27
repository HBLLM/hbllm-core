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
| `BackupManifest` | `hbllm.backup` | Manifest metadata tracking backup ID, files, size, and SHA-256 checksum |

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
from hbllm.backup import BackupManifest, BackupManager

manager = BackupManager(data_dir="./data", backup_dir="./backups")

# Create an atomic compressed archive
archive_path = manager.create_backup(label="pre-migration")
print(f"Created backup archive: {archive_path}")

# Verify archive checksum & manifest integrity
verification = manager.verify_backup(archive_path)
print(f"Checksum valid: {verification}")
```
