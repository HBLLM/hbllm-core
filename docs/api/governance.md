# Governance & Security

HBLLM enforces constitutional governance directly inside the **Pure-HCIR Kernel Governance** layer (`hbllm.hcir.kernel.governance`): `ConstitutionalVerifier`, `CommitAuthorityRegistry`, `EpistemicSafetyGate`, `GovernanceEngine`, `OwnerRuleStore`, and `PolicyEngine` govern every `TransactionManager` commit and capability invocation.

---

## Architecture Overview

```mermaid
graph TB
    subgraph "HCIR Kernel Governance (hbllm.hcir.kernel.governance)"
        TX["TransactionOperation / CapabilityCall"] --> CV["ConstitutionalVerifier + Verification Stages"]
        CV --> GE["GovernanceEngine + EpistemicSafetyGate"]
        GE -->|"Approved"| TM["TransactionManager / CognitiveKernel"]
        GE -->|"Denied"| ROLL["Compensating Rollback + Audit"]
    end

    subgraph "Policy &amp; Rules"
        OR["OwnerRuleStore"]
        PE["PolicyEngine"]
        OR --> PE --> CV &amp; GE
    end
```

---

## `ConstitutionalVerifier` & `EpistemicSafetyGate`

**Modules:** `hbllm.hcir.kernel.governance.constitutional_verifier`, `hbllm.hcir.kernel.governance.epistemic_gate`, `hbllm.hcir.kernel.governance.governance_engine`

```python
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.workspace import HCIRWorkspaceState

registry = CommitAuthorityRegistry()
verifier = ConstitutionalVerifier(authority_registry=registry)
tx_mgr = TransactionManager(
    workspace=HCIRWorkspaceState(),
    verification_stages=create_default_pipeline(constitutional_verifier=verifier),
)
```

---

## `PolicyEngine`

**Module:** `hbllm.hcir.kernel.governance.policies.policy_engine.PolicyEngine`

Evaluates declarative safety, scope, transform, and rate policies across cognitive outputs and capabilities.

### Policy Types

| Type | Purpose | Example |
|---|---|---|
| `DENY` | Block responses matching a pattern | Block harmful instructions |
| `REQUIRE` | Ensure responses include required elements | Medical responses must include disclaimers |
| `TRANSFORM` | Modify responses | Append safety warnings |
| `SCOPE` | Restrict domain access per tenant | Tenant X can only access "coding" domain |
| `RATE` | Per-tenant capability limits | Max 10 code executions per hour |

### Policy Actions

| Action | Behavior |
|---|---|
| `BLOCK` | Block the response entirely |
| `WARN` | Allow but log a warning |
| `APPEND` | Append text to the response |
| `PREPEND` | Prepend text to the response |
| `REPLACE` | Replace matched content |
| `RESTRICT` | Restrict access to domains/tools |

### Usage

```python
from hbllm.hcir.kernel.governance.policies.policy_engine import (
    Policy,
    PolicyAction,
    PolicyEngine,
    PolicyType,
)

engine = PolicyEngine()

# Add a deny policy
engine.add_policy(
    Policy(
        name="no-harmful-instructions",
        type=PolicyType.DENY,
        action=PolicyAction.BLOCK,
        pattern=r"how to (hack|exploit|attack)",
        description="Block requests for harmful activities",
        severity="critical",
    )
)

# Add a transform policy (auto-append disclaimer)
engine.add_policy(
    Policy(
        name="medical-disclaimer",
        type=PolicyType.TRANSFORM,
        action=PolicyAction.APPEND,
        content="This is not medical advice. Consult a healthcare professional.",
        domains=["health"],
    )
)

# Evaluate a response
result = engine.evaluate(
    text="Here is how to hack a server...",
    tenant_id="user-123",
    domain="security",
)

if not result.passed:
    print(result.violations)
else:
    print(result.modified_text)
```

### Loading from YAML

```yaml
# config/policies.yaml
policies:
  - name: no-harmful-instructions
    type: deny
    action: block
    pattern: "how to (hack|exploit|attack)"
    description: Block harmful activity guidance
    severity: critical
    tenant_ids: ["*"]

  - name: medical-disclaimer
    type: transform
    action: append
    content: "This is not medical advice."
    domains: ["health"]
```

```python
loaded = engine.load_from_yaml("config/policies.yaml")
print(f"Loaded {loaded} policies")
```

### Context-Aware Conditions

Policies can have runtime conditions that check against sensor/state context:

```python
from hbllm.hcir.kernel.governance.policies.policy_engine import (
    Policy,
    PolicyAction,
    PolicyCondition,
    PolicyType,
)

# Only activate after 9 PM
policy = Policy(
    name="quiet-hours",
    type=PolicyType.DENY,
    action=PolicyAction.BLOCK,
    pattern=r"play (music|video|audio)",
    conditions=[
        PolicyCondition(key="time_hour", operator="gte", value=21),
        PolicyCondition(key="baby_sleeping", operator="eq", value=True),
    ],
)
```

Supported operators: `eq`, `neq`, `gt`, `lt`, `gte`, `lte`, `in`, `not_in`.

### PolicyResult

```python
@dataclass
class PolicyResult:
    passed: bool  # Did all policies pass?
    original_text: str  # Unmodified response
    modified_text: str  # After transformations
    violations: list[str]  # Blocking violations
    warnings: list[str]  # Non-blocking warnings
    applied_policies: list[str]  # Names of policies that fired
```

---

## Owner Rules (`OwnerRuleStore`)

**Module:** `hbllm.hcir.kernel.governance.policies.owner_rules`

User-defined rules parsed into structured `ParsedRule` / `Policy` objects stored per `tenant_id` in SQLite (`add_rule()`, `list_rules()`, `toggle_rule()`, `load_into_engine()`):

```python
from hbllm.hcir.kernel.governance.policies.owner_rules import OwnerRuleStore
from hbllm.hcir.kernel.governance.policies.policy_engine import PolicyEngine

store = OwnerRuleStore(db_path="data/owner_rules.db")
engine = PolicyEngine()
count = store.load_into_engine(tenant_id="tenant_1", engine=engine)
```

---

## Constitutional Principles (`ConstitutionalVerifier`)

Core invariants enforced on every `TransactionManager` commit regardless of tenant or configuration:

- **Truthfulness** — never commit unverified or hallucinated epistemic beliefs
- **Governance Boundaries** — enforce `CommitAuthorityRegistry` and `SyscallAllowlist(strict=True)`
- **Privacy & Multi-Tenancy** — enforce `Scope(tenant_id=...)` isolation across all `TieredWorkspace` tiers
- **System Prior Protection** — preserve `SecurityLevel.SYSTEM` / `"core_prior"` seed nodes against eviction
- **Epistemic Safety** — fail-closed gating via `EpistemicSafetyGate` when hazard/risk thresholds are exceeded
