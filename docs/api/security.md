---
title: "Security API — HBLLM Trust, Tenancy, and Audit"
description: "API reference for security modules: identity, RBAC, tenant isolation, encryption, audit logging, trust chains, and secrets management."
---

# Security API

HBLLM's security layer provides defense-in-depth for multi-tenant
cognitive systems: identity management, role-based access control,
tenant isolation, encryption at rest, and immutable audit trails.

## Module Overview

| Module | Class | Purpose |
|--------|-------|---------|
| `identity.py` | `NodeIdentity` | Ed25519 cryptographic identity per node |
| `identity_resolver.py` | `resolve_sovereign_identity` | Resolve node identity and provenance tuples |
| `rbac.py` | `RBACGuard` | Role-based access control (`Role`, `Permission`) |
| `tenant_guard.py` | `TenantContext`, `require_tenant` | Per-tenant context isolation and enforcement |
| `tenant_interceptor.py` | `TenantInterceptor` | Bus-level tenant context injection |
| `tenant_registry.py` | `TenantRegistry` | Tenant lifecycle management |
| `audit_trail.py` | `AuditTrail` | Cryptographically hash-chained immutable audit ledger |
| `encryption.py` | `EncryptionVault` | Fernet / PBKDF2 encryption at rest |
| `secrets.py` | `SecretProvider`, `get_secret_provider` | Pluggable secret resolution (env, Vault, AWS Secrets Manager) |
| `trust.py` | `TrustInterceptor` | Bus trust validation middleware |
| `trust_chain.py` | `TrustChain` | Hierarchical trust delegation |

## NodeIdentity

Every HBLLM node has an Ed25519 keypair for signing messages and
authenticating with peers.

```python
from hbllm.security.identity import NodeIdentity

identity = NodeIdentity.generate(node_id="homeserver")

# Sign data
signature = identity.sign(b"message payload")

# Verify a peer's signature
is_valid = identity.verify(peer_public_key, b"message payload", signature)

# Export public key for sharing
pub_key_hex = identity.public_key_hex
```

## `RBACGuard`

Role-based access control with hierarchical roles (`Role`) and permissions (`Permission`):

```python
from hbllm.security.rbac import Permission, RBACGuard, Role

rbac = RBACGuard(db_path="data/rbac.db")

# Assign roles
rbac.assign_role(tenant_id="tenant_1", user_id="alice", role=Role.ADMIN)

# Check permissions
allowed = rbac.check(
    tenant_id="tenant_1",
    user_id="alice",
    permission=Permission.MEMORY_WRITE,
)  # True
```

## `TenantContext` & `require_tenant`

Enforces strict data isolation between tenants at every layer:

```python
from hbllm.security.tenant_guard import TenantContext, require_tenant


@require_tenant
def read_tenant_record(tenant_id: str) -> str:
    return tenant_id


with TenantContext(tenant_id="tenant_1", user_id="alice"):
    assert read_tenant_record(tenant_id="tenant_1") == "tenant_1"
```

### Isolation Guarantees

| Layer | Mechanism |
|-------|-----------|
| **Memory** | Tenant-scoped SQLite databases and `Scope(tenant_id=...)` workspace partitions |
| **Bus** | `TenantInterceptor` tags all messages with tenant context |
| **API** | JWT claims carry tenant_id, validated on every request |
| **Encryption** | Vault encryption via `EncryptionVault` |
| **Quotas** | Per-tenant resource limits (DB rows, rate limits) |

## AuditTrail

Immutable, cryptographically hash-chained append-only audit ledger for compliance, forensics, and safety governance:

```python
from hbllm.security.audit_trail import AuditAction, AuditTrail

audit = AuditTrail(db_path="data/audit_trail.db")

# Log an action
audit.log(
    tenant_id="tenant_1",
    actor="alice",
    action="tool.execute",
    resource="shell_node",
    details={"command": "ls -la"},
    outcome="success",
)

# Query audit trail
entries = audit.query(
    tenant_id="tenant_1",
    action="tool.execute",
    since=time.time() - 86400,
    limit=100,
)
```

## `EncryptionVault`

Authenticated encryption for sensitive payloads at rest:

```python
from hbllm.security.encryption import EncryptionVault

vault = EncryptionVault.from_env()

ciphertext = vault.encrypt("sensitive memory content")
plaintext = vault.decrypt(ciphertext)
```

## `SecretProvider`

Secure credential lookup across environment variables, HashiCorp Vault, and AWS Secrets Manager:

```python
from hbllm.security.secrets import get_secret_provider

secrets = get_secret_provider()
api_key = secrets.get("OPENAI_API_KEY")
```

## `TrustInterceptor` & `TrustChain`

Validates peer message authenticity and delegates hierarchical trust on the message bus (`hbllm.security.trust`, `hbllm.security.trust_chain`):

```python
from hbllm.security.trust_chain import TrustChain

chain = TrustChain()
```

## Bus Topics

| Topic | Publisher | Description |
|-------|----------|-------------|
| `security.audit` | `AuditTrail` | New audit log entry |
| `security.alert` | `TenantContext` | Tenant isolation violation attempt |
| `security.trust.change` | `TrustInterceptor` | Peer trust level changed |
| `security.key.rotate` | `EncryptionVault` | Encryption key rotation event |
