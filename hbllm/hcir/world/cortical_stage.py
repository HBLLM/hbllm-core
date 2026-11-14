from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hbllm.hcir.transactions import TransactionOperation

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext


@dataclass(frozen=True)
class WorkingMemoryDelta:
    """Explicit declarative delta operation for working memory mutations (Invariant 2 & 8, WP4)."""

    field: str
    value: Any
    op: str = "set"


@dataclass(frozen=True)
class StageResult:
    """Pure result emitted by a CorticalStage during active inference.

    Contains proposed transactional mutations to HCIR state, transient signals
    passed forward to downstream stages, and any proposed motor actuation.
    """

    author: str
    operations: list[TransactionOperation] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    proposed_action: Any = None
    annotations: dict[str, Any] = field(default_factory=dict)
    # WP1: the stage's journaled faculty writes (committed by the pipeline)
    journal: Any = None
    # WP4: structured WorkingMemoryDelta operations
    wm_deltas: list[WorkingMemoryDelta] = field(default_factory=list)


@runtime_checkable
class CorticalStage(Protocol):
    """Protocol for pure cortical processing stages in the 5-stage active inference pipeline.

    Invariants:
    1. A CorticalStage MUST NOT mutate the engine directly (`engine.X = ...`).
    2. A CorticalStage receives an immutable `CognitiveContext` snapshot.
    3. State mutations are proposed via `StageResult.operations` and committed by HCIR.
    """

    name: str

    def observe(self, ctx: CognitiveContext) -> StageResult:
        """Observe immutable context and emit state mutation intent & signals."""
        ...

    def bind_engine(self, engine: Any) -> None:
        """Bind underlying engine execution context."""
        return None
