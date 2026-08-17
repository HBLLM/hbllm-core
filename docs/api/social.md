---
title: "API Reference — Social Cognition & Persona Subsystem"
description: "API reference for UserModelEngine, PersonaEngine, RelationshipMemory, SocialTiming, and ActivityDigest modules."
---

# Social Cognition & Persona API

The **Social Cognition Subsystem** provides long-term human operator modeling, relationship graph persistence, dynamic persona modulation, and interruption timing control.

**Package:** `hbllm.brain.social`

---

## Subsystem Index

| Class | Module | Purpose |
|---|---|---|
| `UserModelEngine` | `user_model.py` | Predictive human modeling (expertise, fatigue, trust, preferences) |
| `UserModelNode` | `user_model_node.py` | MessageBus-connected node for real-time user state tracking |
| `PersonaEngine` | `persona_engine.py` | Adaptive tone, formality, and behavioral persona modulation |
| `RelationshipMemory` | `relationship_memory.py` | Persistent multi-user social graph and interaction milestones |
| `RelationshipNode` | `relationship_node.py` | Bus-connected social interaction and bond recorder |
| `SocialTiming` | `social_timing.py` | Interruption risk scoring and non-intrusive notification pacing |
| `ActivityDigest` | `activity_digest.py` | Periodic cognitive summarization of background events |
| `IdentityNode` | `identity_node.py` | Core constitutional identity and ethical invariant enforcement |

---

## `UserModelEngine`

**Module:** `hbllm.brain.social.user_model.UserModelEngine`

```python
from hbllm.brain.social.user_model import UserModelEngine

engine = UserModelEngine(tenant_id="tenant-alpha")

# Record an interaction turn
await engine.record_turn(
    user_id="user-01",
    query="Show me the Rust SIMD micro-benchmark results",
    latency_seconds=1.5,
)

# Retrieve current profile
profile = await engine.get_profile("user-01")
print(f"Domain Expertise (Systems): {profile.expertise.get('systems', 0.5):.2f}")
print(f"Cognitive Fatigue: {profile.fatigue_level:.2f}")
print(f"Trust Level: {profile.trust_score:.2f}")
```

---

## `PersonaEngine`

**Module:** `hbllm.brain.social.persona_engine.PersonaEngine`

```python
from hbllm.brain.social.persona_engine import PersonaEngine

persona = PersonaEngine()

# Get adapted tone style based on user profile and incident urgency
style = persona.resolve_style(
    user_id="user-01",
    urgency=0.9,  # High urgency
    formality_override=None,
)

print(f"Formality: {style.formality}")
print(f"Brevity Factor: {style.brevity_factor}")
print(f"Instruction Prompt Prefix: {style.system_prompt_addon}")
```

---

## `SocialTiming`

**Module:** `hbllm.brain.social.social_timing.SocialTiming`

```python
from hbllm.brain.social.social_timing import SocialTiming

timing = SocialTiming()

# Evaluate whether to interrupt user right now
decision = timing.evaluate_interruption(
    user_id="user-01",
    notification_urgency=0.3,
    user_current_activity="active_coding",
)

if decision.should_interrupt:
    print("Sending immediate notification")
else:
    print(f"Deferring to digest. Reason: {decision.suppression_reason}")
```
