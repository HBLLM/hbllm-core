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
| `identity_resolver.py` | `IdentityResolver` | Resolve node IDs to public keys |
| `rbac.py` | `RBACEngine` | Role-based access control with permissions |
| `tenant_guard.py` | `TenantGuard` | Per-tenant data isolation enforcer |
| `tenant_interceptor.py` | `TenantInterceptor` | Bus-level tenant context injection |
| `tenant_registry.py` | `TenantRegistry` | Tenant lifecycle management |
| `audit_log.py` | `AuditLog` | Immutable append-only audit trail |
| `encryption.py` | `EncryptionManager` | AES-256-GCM encryption at rest |
| `secrets.py` | `SecretStore` | Secure credential storage |
| `trust.py` | `TrustStore` | Peer trust level management |
| `trust_chain.py` | `TrustChain` | Hierarchical trust delegation |

## NodeIdentity

Every HBLLM node has an Ed25519 keypair for signing messages and
verifying peers in distributed swarms.

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

## RBACEngine

Role-based access control with hierarchical permissions:

```python
from hbllm.security.rbac import RBACEngine

rbac = RBACEngine()

# Define roles
rbac.create_role("admin", permissions=["*"])
rbac.create_role("user", permissions=["read", "write", "query"])
rbac.create_role("viewer", permissions=["read", "query"])

# Assign roles
rbac.assign_role(tenant_id="tenant_1", user_id="alice", role="admin")

# Check permissions
allowed = rbac.check_permission(
    tenant_id="tenant_1",
    user_id="alice",
    permission="write",
)  # True
```

## TenantGuard

Enforces strict data isolation between tenants at every layer:

```python
from hbllm.security.tenant_guard import TenantGuard

guard = TenantGuard()

# Validate tenant context
guard.enforce(tenant_id="tenant_1", resource_tenant="tenant_1")  # OK
guard.enforce(tenant_id="tenant_1", resource_tenant="tenant_2")  # Raises

# Scope a database query
query = guard.scope_query(base_query, tenant_id="tenant_1")
```

### Isolation Guarantees

| Layer | Mechanism |
|-------|-----------|
| **Memory** | Tenant-scoped SQLite databases |
| **Bus** | `TenantInterceptor` tags all messages with tenant context |
| **API** | JWT claims carry tenant_id, validated on every request |
| **Encryption** | Per-tenant encryption keys via `EncryptionManager` |
| **Quotas** | Per-tenant resource limits (DB rows, rate limits) |

## AuditLog

Immutable, append-only audit trail for compliance and forensics:

```python
from hbllm.security.audit_log import AuditLog

audit = AuditLog(db_path="data/audit.db")

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

## EncryptionManager

AES-256-GCM encryption for data at rest:

```python
from hbllm.security.encryption import EncryptionManager

enc = EncryptionManager(master_key="...")

# Encrypt sensitive data
ciphertext = enc.encrypt(b"sensitive memory content", tenant_id="tenant_1")

# Decrypt
plaintext = enc.decrypt(ciphertext, tenant_id="tenant_1")
```

## SecretStore

Secure credential storage with environment variable fallback:

```python
from hbllm.security.secrets import SecretStore

store = SecretStore()

# Store a secret
store.set("OPENAI_API_KEY", "sk-...", tenant_id="tenant_1")

# Retrieve (checks store → env vars → raises)
key = store.get("OPENAI_API_KEY", tenant_id="tenant_1")
```

## TrustStore & TrustChain

Manage peer trust levels in distributed swarms:

```python
from hbllm.security.trust import TrustStore

trust = TrustStore()

# Set trust level for a peer (0-100)
trust.set_trust("peer_node_1", level=80)

# Check if peer is trusted above threshold
is_trusted = trust.is_trusted("peer_node_1", min_level=50)  # True

# Revoke trust
trust.revoke("peer_node_1")
```

## Bus Topics

| Topic | Publisher | Description |
|-------|----------|-------------|
| `security.audit` | AuditLog | New audit log entry |
| `security.alert` | TenantGuard | Tenant isolation violation attempt |
| `security.trust.change` | TrustStore | Peer trust level changed |
| `security.key.rotate` | EncryptionManager | Encryption key rotation event |
