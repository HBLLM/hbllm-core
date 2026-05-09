# HBLLM Security & Governance

HBLLM is designed for enterprise environments where data isolation, secure code execution, and model integrity are mission-critical. This document outlines the security architecture implemented in the `core` package.

## 1. Code Sandboxing (`ExecutionNode`)

Allowing an LLM to generate and execute code autonomously is a massive security risk. HBLLM mitigates this via a rigorous, multi-layered sandboxing pipeline found in `core/hbllm/actions/execution_node.py`:

- **Static AST Validation**: Before any Python code is executed, it is parsed by an internal Abstract Syntax Tree (AST) walker. This static analysis strictly rejects dangerous imports (e.g., `os`, `sys`, `subprocess`) and built-in functions (e.g., `exec()`, `eval()`, `open()`).
- **Restricted Subprocess Execution**: Validated code is never run in the main process. It is executed in an isolated, restricted subprocess.
- **Strict Timeouts**: Every execution is bound by a strict timeout (default: 5.0 seconds). If the code enters an infinite loop or attempts to halt the system, the subprocess is ruthlessly killed.

## 2. Multi-Tenant Identity & Data Isolation

HBLLM enforces strict multi-tenant data isolation using an **identity triplet** — every operation is scoped to a `(tenant_id, user_id, device_id)` context.

### Identity Triplet

All security-sensitive operations require a fully resolved identity:

| Field | Purpose | Source |
|-------|---------|--------|
| `tenant_id` | Organization-level isolation | JWT payload or API key |
| `user_id` | User-level isolation within a tenant | JWT payload |
| `device_id` | Device-level isolation (edge/IoT) | JWT payload or device registration |

### Tenant Guard (`hbllm.security.tenant_guard`)

The `TenantGuard` module uses Python `contextvars` to propagate identity through async call chains. It supports three configurable enforcement modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `OFF` | No enforcement — all access allowed | Local development, single-user |
| `WARN` | Log violations but allow access | Staging, migration monitoring |
| `STRICT` | Raise `TenantIsolationError` on violations | Production, compliance |

Configure via `hbllm.yaml`:

```yaml
security:
  tenant_guard_mode: WARN  # OFF | WARN | STRICT
  isolation_level: TENANT  # TENANT | USER | DEVICE
```

Or via environment variable:

```bash
export HBLLM_TENANT_GUARD_MODE=STRICT
```

### `@require_tenant` Decorator

All data-access methods are protected with the `@require_tenant` decorator, which validates the identity context before any read or write:

```python
from hbllm.security.tenant_guard import require_tenant

@require_tenant
async def store(self, content: str, metadata: dict, *, tenant_id: str, **kwargs):
    # tenant_id is validated before this code runs
    ...
```

**Protected layers:**

- `MemoryNode` — `handle_store()`, `handle_search()`
- `SemanticMemory` — `store()`, `astore()`, `search()`
- `BrainState` / `AsyncBrainState` — all KV store, message history, tool log, and checkpoint methods

### Logical Data Isolation

- Every message on the `CognitiveEventBus` carries a mandatory `tenant_id` and `session_id`.
- The `MemoryNode` (Episodic, Semantic, Procedural) partitions all storage and vector databases by `tenant_id`. An agent reasoning on behalf of Tenant A cannot access or "remember" data from Tenant B.
- The `WorkspaceNode` blackboard tracks thought hierarchies individually per session.

*(Verified via automated tests: `test_integration.py::TestEndToEndPipeline::test_multi_tenant_isolation`, `test_e2e_tenant.py`, `test_identity_isolation.py`)*

## 3. LoRA Adapter Integrity

The HBLLM architecture allows for "In-Flight LoRA Hot-Swapping"—dynamically downloading and loading fine-tuned domain adapters from the Sentra Marketplace at runtime. 

To prevent supply-chain attacks or malicious weight injection:
- **SHA-256 Validation**: The `AdapterRegistry` enforces strict cryptographic checksums. Before a new `.safetensors` or `.bin` adapter is loaded into the base LLM, its SHA-256 hash is verified against a known, trusted ledger. 
- **Read-Only Model Memory**: Adapter weights are loaded into read-only memory buffers to prevent runtime manipulation.

## 4. Input Sanitization & Serving Security

The serving layer (`hbllm.serving.security`) provides defense-in-depth for all HTTP and WebSocket endpoints:

### Input Sanitization

| Function | Protection |
|----------|-----------|
| `sanitize_input()` | Strips control characters, truncates to configurable max length |
| `detect_injection()` | Regex-based prompt injection detection with risk scoring (`none`, `low`, `high`) |

Injection patterns detected include:
- `ignore previous instructions`
- `you are now in X mode`
- `disregard all prior` 
- Raw model tokens (`[INST]`, `<|im_start|>`)

### Request Body Limits

`BodySizeLimitMiddleware` enforces per-endpoint body size limits:

| Endpoint Pattern | Max Size |
|-----------------|----------|
| `/v1/chat` | 1 MB |
| `/v1/upload`, `/v1/knowledge` | 50 MB |
| All other endpoints | 5 MB |

### Authentication

- **JWT Authentication** (`hbllm.serving.auth`): `JWTAuthMiddleware` validates Bearer tokens and injects the identity triplet into `request.state`.
- **API Key Manager** (`hbllm.serving.security`): SHA-256 hashed API keys with per-key scoping and tenant resolution.

### CORS Hardening

`validate_cors_config()` blocks wildcard CORS (`*`) in production environments, preventing cross-origin credential theft.

### Auth Rate Limiting

`AuthRateLimiter` uses per-IP sliding windows to prevent brute-force attacks on authentication endpoints (default: 5 attempts per 15 minutes per IP).

### Password Security

PBKDF2-SHA256 with 100,000 iterations and 16-byte random salts for admin password storage.

## 5. Audit Logging (`hbllm.security.audit_log`)

An append-only, SOC2/GDPR-compliant audit trail captures every security-sensitive operation with the full identity triplet:

```python
from hbllm.security.audit_log import AuditLog, AuditAction, AuditSeverity

audit = AuditLog(db_path="data/audit.db")

audit.log(
    action=AuditAction.AUTH_FAILED,
    tenant_id="acme",
    user_id="user-123",
    device_id="edge-01",
    severity=AuditSeverity.WARNING,
    ip_address="192.168.1.100",
    details={"reason": "invalid_key"},
)
```

### Tracked Events

| Category | Events |
|----------|--------|
| **Auth** | Login, failed login, logout, token refresh, key generated/revoked |
| **WebSocket** | Connect, disconnect, capability registration |
| **Tenant** | Created, updated, deactivated, data purged |
| **Data** | Accessed, created, updated, deleted, exported |
| **Policy** | Created, updated, deleted |
| **Chat** | Message, conversation created/deleted |
| **Tools** | Executed, failed |
| **Admin** | Config changed, admin actions |
| **Webhooks** | Registered, delivered, failed |

### Compliance Features

- **Query by tenant/user/action/severity/time range** — indexed for fast lookups
- **Export JSON** — full audit trail for a tenant (GDPR data portability)
- **Purge old entries** — configurable retention period (default: 365 days)
- **Failed login tracking** — `failed_logins(hours=24)` for security alerting

## 6. Encryption at Rest (`hbllm.security.encryption`)

Field-level encryption for sensitive tenant data using symmetric encryption with HMAC-SHA256 authentication:

```python
from hbllm.security.encryption import EncryptionVault

vault = EncryptionVault.from_key_file("data/encryption.key")
encrypted = vault.encrypt("sensitive-api-key")
original = vault.decrypt(encrypted)
```

- **Key derivation**: PBKDF2-SHA256 (100k iterations) from password, or random 32-byte key
- **Key rotation**: `vault.rotate_key()` creates a new vault with a fresh key
- **Key fingerprinting**: `vault.key_fingerprint` for safe logging (first 12 chars of SHA-256)
- **Dict encryption**: `vault.encrypt_dict()` / `vault.decrypt_dict()` for structured data

## 7. Policy Engine

- **Constitutional AI layer** intercepting all outputs. Responses are evaluated against safety policies (`DENY`, `TRANSFORM`, `SCOPE`) before delivery.
- Per-tenant policy scoping with configurable `BLOCK`, `WARN`, `APPEND`, `PREPEND`, `REPLACE`, `RESTRICT` actions.

> 📖 **[Full Policy Engine Reference →](docs/api/governance.md)**

## 8. Configuration Reference

All security settings are in `hbllm.yaml` under the `security:` key:

```yaml
security:
  tenant_guard_mode: WARN     # OFF | WARN | STRICT
  isolation_level: TENANT     # TENANT | USER | DEVICE
  audit_enabled: true
  audit_db_path: data/audit.db
  rate_limiting_enabled: true
  rate_limit_rpm: 60
  encryption_enabled: false
  encryption_key_file: data/encryption.key
```

---
*For reporting vulnerabilities, please do not open a public issue. Email our security team directly.*
