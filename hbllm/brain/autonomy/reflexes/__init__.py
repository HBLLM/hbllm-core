"""Reflex Library — deterministic, zero-LLM response rules.

Provides a library of pre-built reflex rules for the AutonomyCore.
Reflexes are Tier 1 responses: instant, deterministic, no LLM cost.

Categories:
    environment — smart home/IoT awareness
    system      — host device health monitoring
    routine     — schedule and habit awareness
    security    — threat detection and access control

Usage::

    from hbllm.brain.autonomy.reflexes import get_all_reflexes

    for name, rule in get_all_reflexes().items():
        autonomy_core.add_reflex(name, rule)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hbllm.brain.autonomy.attention import AttentionEvent
from hbllm.network.messages import Message

# Type alias matching AutonomyCore's ReflexRule
ReflexRule = Callable[[AttentionEvent], Message | None]


def get_all_reflexes() -> dict[str, ReflexRule]:
    """Load all built-in reflex rules from all categories.

    Returns:
        Dict mapping reflex name to rule function.
    """
    from hbllm.brain.autonomy.reflexes.environment import get_environment_reflexes
    from hbllm.brain.autonomy.reflexes.routine import get_routine_reflexes
    from hbllm.brain.autonomy.reflexes.security import get_security_reflexes
    from hbllm.brain.autonomy.reflexes.system import get_system_reflexes

    all_rules: dict[str, ReflexRule] = {}
    all_rules.update(get_system_reflexes())
    all_rules.update(get_environment_reflexes())
    all_rules.update(get_routine_reflexes())
    all_rules.update(get_security_reflexes())
    return all_rules
