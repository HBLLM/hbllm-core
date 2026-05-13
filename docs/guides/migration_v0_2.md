# Migrating to HBLLM Core v0.2

Version 0.2 introduces several significant architectural improvements, especially around security, memory unified interface, and execution sandboxing. This guide details the breaking changes and how to adapt your plugins or customizations.

## 1. Security Hardening & Encryption (Breaking Change)

The custom XOR-based keystream encryption has been fully replaced with industry-standard `Fernet` (AES128 CBC + SHA256 HMAC) from the `cryptography` package.

**What you need to do:**
If you have an existing `working_memory.db` or other artifacts that contain encrypted values (like API keys) encrypted with v0.1:
1. Load your old key into a legacy decryption script (you can copy the `EncryptionVault` code from v0.1).
2. Decrypt the values.
3. Generate a new v0.2 key (which is a url-safe base64 encoded 32-byte string) or let the system auto-generate one.
4. Encrypt your values using the new v0.2 `EncryptionVault`.

*Note:* `EncryptionVault` now supports `.from_env()` to securely load keys without placing them on disk.

## 2. Unified Memory Interface

The memory subsystem has been refactored behind a strict `UnifiedMemoryInterface`. `MemoryNode` now implements this interface directly.

**What you need to do:**
- Stop calling `MemoryNode.db.store_turn()` directly in your custom nodes.
- Instead, use the unified async methods:
  - `await memory_node.store(MemoryType.EPISODIC, data, session_id="...")`
  - `await memory_node.search(query, memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC])`

## 3. Database Connection Pooling

The SQLite memory backends now use a connection pool built on `aiosqlite`.
- Concurrent reads and writes are now fully non-blocking.
- The `DatabasePool` automatically manages the `PRAGMA` optimizations (`journal_mode=WAL`).

## 4. Execution Sandbox Enhancements

The `ExecutionNode` for running Python code now has stricter OS-level and AST-level sandboxing.
- You can now specify `allowed_modules` in the config to create strict whitelists for code generation targets.
- By default, network access is stripped out if running on Linux (`unshare -n`).
- Hardware quotas (`RLIMIT_AS`, `RLIMIT_CPU`) are strictly enforced via the new `max_memory_mb` parameter.

## 5. Resilience: Circuit Breakers

The `CircuitBreaker` implementation now uses Exponential Backoff with Jitter. If a remote node goes down, the recovery timeouts will progressively lengthen, reducing load on flapping services.
