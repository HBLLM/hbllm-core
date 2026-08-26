"""Native HCIR Graph Adapter.

Provides a unified interface between Python HCIR and the native persistent Rust GraphState substrate.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import HCIREdgeType, HCIRNodeType

logger = logging.getLogger(__name__)

# Attempt to load native PyO3 module
try:
    from hbllm_hcir_graph import (
        NativeGraph as _PyNativeGraph,  # type: ignore[import-not-found,no-redef]
    )
except ImportError:
    _PyNativeGraph = None  # type: ignore[assignment]


class NativeGraphAdapter:
    """Three-layer adapter wrapping native Rust GraphState.

    Preserves 100% of the high-level HCIR graph contract while delegating
    storage, traversal, chunked snapshotting, and canonical BLAKE3 hashing
    to the native Rust substrate when available.
    """

    def __init__(self, native_instance: Any = None) -> None:
        if native_instance is not None:
            self._native = native_instance
        elif _PyNativeGraph is not None:
            self._native = _PyNativeGraph()
        else:
            self._native = None

    @property
    def is_native(self) -> bool:
        """Whether this adapter is backed by the compiled Rust substrate."""
        return self._native is not None

    def add_node(
        self,
        node_id: str,
        node_type: str | HCIRNodeType,
        lifecycle: str = "ACTIVE",
        properties: dict[str, str] | None = None,
        created_at: float = 0.0,
    ) -> None:
        type_str = node_type.value if hasattr(node_type, "value") else str(node_type)
        if self._native is not None:
            props = {k: str(v) for k, v in (properties or {}).items()}
            self._native.add_node(node_id, type_str, lifecycle, props, created_at)

    def add_edge(
        self,
        edge_id: str,
        edge_type: str | HCIREdgeType,
        sources: list[str],
        targets: list[str],
        weight: float = 1.0,
        properties: dict[str, str] | None = None,
        created_at: float = 0.0,
    ) -> None:
        type_str = edge_type.value if hasattr(edge_type, "value") else str(edge_type)
        if self._native is not None:
            props = {k: str(v) for k, v in (properties or {}).items()}
            self._native.add_edge(edge_id, type_str, sources, targets, weight, props, created_at)

    def has_node(self, node_id: str) -> bool:
        if self._native is not None:
            return bool(self._native.has_node(node_id))
        return False

    def has_edge(self, edge_id: str) -> bool:
        if self._native is not None:
            return bool(self._native.has_edge(edge_id))
        return False

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        if self._native is not None:
            return self._native.get_node(node_id)
        return None

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        if self._native is not None:
            return self._native.get_edge(edge_id)
        return None

    def edges_from(self, node_id: str) -> list[dict[str, Any]]:
        if self._native is not None:
            return list(self._native.edges_from(node_id))
        return []

    def edges_to(self, node_id: str) -> list[dict[str, Any]]:
        if self._native is not None:
            return list(self._native.edges_to(node_id))
        return []

    def nodes_of_type(self, node_type: str | HCIRNodeType) -> list[dict[str, Any]]:
        type_str = node_type.value if hasattr(node_type, "value") else str(node_type)
        if self._native is not None:
            return list(self._native.nodes_of_type(type_str))
        return []

    def bfs_path(self, start: str, end: str, max_depth: int = 10) -> list[str]:
        if self._native is not None:
            return list(self._native.bfs_path(start, end, max_depth))
        return []

    def subgraph_bfs(
        self, center_id: str, max_depth: int = 3
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._native is not None:
            nodes, edges = self._native.subgraph_bfs(center_id, max_depth)
            return list(nodes), list(edges)
        return [], []

    def snapshot(self) -> NativeGraphAdapter:
        """Create an O(1) root snapshot sharing chunked persistent storage."""
        if self._native is not None:
            return NativeGraphAdapter(native_instance=self._native.snapshot())
        return NativeGraphAdapter()

    def canonical_hash(self) -> str:
        """Compute deterministic BLAKE3 versioned canonical state hash."""
        if self._native is not None:
            return str(self._native.canonical_hash())
        return ""
