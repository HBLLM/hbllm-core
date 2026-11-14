"""Protocols for HCIR Kernel and Executive Runtime Subsystems.

Provides formal typing.Protocol definitions for offline consolidation,
replay, pruning, and maintenance, eliminating ad-hoc hasattr checks.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Replayable(Protocol):
    """Protocol for memory systems supporting forward and reverse trajectory replay."""

    def replay_forward_and_reverse(self, trajectory: Any, terminal_reward: float = 1.0) -> Any: ...


@runtime_checkable
class ConceptMaintainable(Protocol):
    """Protocol for concept learners supporting periodic offline maintenance."""

    def maintain(self) -> Any: ...


@runtime_checkable
class Prunable(Protocol):
    """Protocol for causal theories supporting pruning of stale or falsified hypotheses."""

    def prune_stale_hypotheses(self) -> Any: ...


@runtime_checkable
class SleepConsolidatable(Protocol):
    """Protocol for library learners supporting sleep consolidation into tiered workspace."""

    def sleep_consolidation(self, tiered_workspace: Any, tenant_id: str = "default") -> Any: ...


@runtime_checkable
class Consolidatable(Protocol):
    """Protocol for cognitive engines exporting transactional operations for persistence."""

    def to_transaction_operations(self) -> list[Any]: ...
