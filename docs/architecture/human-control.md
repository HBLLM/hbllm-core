# Human Control Layer

While autonomous systems benefit from proactive execution, they require robust safety boundaries and human-in-the-loop oversight. The Human Control Layer is designed to keep human oversight at the center of the cognitive loop without overwhelming the user.

## Trust Boundaries & Scoped Tokens (`permissions.py`)

Every tool and action in the system is tagged with an **ActionClass**:
- `SAFE`: Non-mutating or easily reversible actions (e.g., reading a configuration).
- `USER_AWARE`: Actions the user should know about, but don't strictly require pre-approval.
- `SENSITIVE`: Actions requiring explicit pre-approval (e.g., sending an email, changing a password).
- `CRITICAL`: High-risk actions that require multi-factor or elevated trust verification.

The system issues contextual **TrustGrants**, which are expiring and revocable tokens that permit the agent to carry out specific classes of action for a defined period.

## Explanation-First Mode & Intent Integrity (`guard.py`)

For `SENSITIVE` and `CRITICAL` tasks, the agent enters an **Explanation-First Mode**.
- The `SecurityGuard` intercepts the action before execution and generates an `IntentEnvelope` summarizing what it plans to do and why.
- The user is prompted to approve the intent.
- The `IntentIntegrityEngine` mathematically hashes the approval. This prevents the planner graph from mutating its intent or taking a different action under the guise of an approved request.

## Intervention & Reversibility Model (`intervention.py`)

Humans must be able to gracefully stop or reverse autonomous behavior.
- Provides semantic **pause** and **stop** APIs that halt cognitive execution cleanly rather than terminating the process aggressively.
- The **ReversibilityPolicy** enforces undo semantics and compensation steps for actions that have already been executed but were subsequently intervened upon.

## Human Cognitive Load Model (`attention.py`)

Constant interruptions lead to alert fatigue, causing users to blindly approve actions.
- Tracks a `user_attention_budget` and `approval_fatigue` metric.
- The system will batch `SENSITIVE` requests or defer non-urgent tasks to prevent overwhelming the user with permission prompts.
