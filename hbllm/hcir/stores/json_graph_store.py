"""JSON File-Backed Graph Store — HCIR Persistent Knowledge Storage.

Implements IGraphStore using atomic JSON file serialization.
Allows the CognitiveGraph (SkillNode, EpisodeNode, LearnedRuleNode, ProcedureNode)
to persist across process restarts, multi-stage runs, and distributed environments.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIRNode
from hbllm.hcir.stores import IGraphStore

logger = logging.getLogger("json_graph_store")


class JsonFileGraphStore(IGraphStore):
    """File-backed JSON store for CognitiveGraph with atomic write semantics."""

    def __init__(self, file_path: str | Path, auto_save: bool = True) -> None:
        self.file_path = Path(file_path).resolve()
        self.auto_save = auto_save
        self._cached_graph: CognitiveGraph | None = None
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_graph(self, graph: CognitiveGraph) -> None:
        """Persist the full graph state to JSON file atomically."""
        self._cached_graph = graph
        data = graph.to_dict()

        import uuid

        temp_path = self.file_path.with_name(
            f"{self.file_path.stem}_{os.getpid()}_{uuid.uuid4().hex[:6]}.tmp"
        )
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
            os.replace(temp_path, self.file_path)
            logger.debug(f"Saved {len(list(graph.all_nodes()))} nodes to {self.file_path}")
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            logger.error(f"Failed to save CognitiveGraph to {self.file_path}: {e}")
            raise

    def load_graph(self) -> CognitiveGraph:
        """Load CognitiveGraph from JSON file. Returns empty graph if file doesn't exist."""
        if not self.file_path.exists():
            logger.debug(f"Graph file {self.file_path} not found. Initializing empty graph.")
            graph = CognitiveGraph()
            self._cached_graph = graph
            return graph

        try:
            with open(self.file_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            graph = CognitiveGraph.from_dict(data)
            self._cached_graph = graph
            logger.debug(f"Loaded {len(list(graph.all_nodes()))} nodes from {self.file_path}")
            return graph
        except Exception as e:
            logger.warning(
                f"Error loading CognitiveGraph from {self.file_path}: {e}. Returning empty graph."
            )
            graph = CognitiveGraph()
            self._cached_graph = graph
            return graph

    def save_node(self, node: HCIRNode) -> None:
        """Upsert a single node in memory and optionally persist graph."""
        if self._cached_graph is None:
            self._cached_graph = self.load_graph()
        self._cached_graph.add_node(node)
        if self.auto_save:
            self.save_graph(self._cached_graph)

    def save_edge(self, edge: HCIREdge) -> None:
        """Upsert a single edge in memory and optionally persist graph."""
        if self._cached_graph is None:
            self._cached_graph = self.load_graph()
        self._cached_graph.add_edge(edge)
        if self.auto_save:
            self.save_graph(self._cached_graph)

    def delete_node(self, node_id: str) -> None:
        """Delete a node from graph and optionally persist."""
        if self._cached_graph is None:
            self._cached_graph = self.load_graph()
        self._cached_graph.remove_node(node_id)
        if self.auto_save:
            self.save_graph(self._cached_graph)

    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge from graph and optionally persist."""
        if self._cached_graph is None:
            self._cached_graph = self.load_graph()
        self._cached_graph.remove_edge(edge_id)
        if self.auto_save:
            self.save_graph(self._cached_graph)
