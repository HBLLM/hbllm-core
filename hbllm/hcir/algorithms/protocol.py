from __future__ import annotations

"""GraphView Protocol for Pure Classical Graph Algorithms (WP5)."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EdgeView:
    """Read-only view of a graph edge."""

    id: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    weight: float = 1.0


@runtime_checkable
class GraphView(Protocol):
    """Abstract graph protocol for decoupled graph traversal and topology algorithms."""

    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    def has_node(self, node_id: str) -> bool: ...

    def neighbors(self, node_id: str) -> Iterable[str]: ...

    def edges_from(self, node_id: str) -> Iterable[Any]: ...

    def weight(self, u: str, v: str) -> float: ...


def native_of(graph: Any) -> Any | None:
    """Return the native accelerator attached to a graph object, if any.

    Algorithms dispatch to this adapter first when present; views and
    plain Python graphs simply do not define ``_native_adapter``.
    """
    return getattr(graph, "_native_adapter", None)
