"""
Knowledge Graph — entity-relation graph layer for HBLLM memory.

Stores entities and labeled relationships between them, providing
neighbor lookups, shortest-path queries, and subgraph extraction.
Used by the experience pipeline to build structured knowledge from
high-salience events.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional Rust Acceleration ───────────────────────────────────────────────

try:
    from hbllm_knowledge_graph import (  # type: ignore[import-not-found]
        disambiguate_entities as _rust_disambiguate,
    )

    _HAS_RUST_KG = True
    logger.debug("Using Rust-accelerated knowledge graph")
except ImportError:
    _HAS_RUST_KG = False


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    label: str
    entity_type: str = "concept"
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class Relation:
    """A directed edge in the knowledge graph."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    valid_from: float | None = None
    valid_until: float | None = None

    @property
    def key(self) -> str:
        return f"{self.source_id}--{self.relation_type}-->{self.target_id}"


# ── Simple NLP entity extraction ─────────────────────────────────────────────

# Common relation patterns (regex-based, no heavy NLP dependency)
_RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"(\w[\w\s]{1,30}?)\s+(?:is a|is an|is the)\s+(\w[\w\s]{1,30})", "is_a"),
    (r"(\w[\w\s]{1,30}?)\s+(?:has|contains|includes)\s+(\w[\w\s]{1,30})", "has"),
    (r"(\w[\w\s]{1,30}?)\s+(?:causes|triggers|leads to)\s+(\w[\w\s]{1,30})", "causes"),
    (r"(\w[\w\s]{1,30}?)\s+(?:uses|requires|needs)\s+(\w[\w\s]{1,30})", "uses"),
    (r"(\w[\w\s]{1,30}?)\s+(?:depends on|relies on)\s+(\w[\w\s]{1,30})", "depends_on"),
    (r"(\w[\w\s]{1,30}?)\s+(?:produces|generates|creates)\s+(\w[\w\s]{1,30})", "produces"),
    (r"(\w[\w\s]{1,30}?)\s+(?:connects to|links to|relates to)\s+(\w[\w\s]{1,30})", "relates_to"),
    (r"(\w[\w\s]{1,30}?)\s+(?:prefers|likes|wants)\s+(\w[\w\s]{1,30})", "prefers"),
]

# Stopwords to filter out from entity extraction
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "not",
        "no",
        "so",
        "very",
        "just",
        "also",
        "too",
    }
)


def _entity_id(label: str) -> str:
    """Deterministic ID from normalized label."""
    normalized = label.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def extract_entities_from_text(text: str) -> list[tuple[str, str, str]]:
    """
    Extract (source, relation, target) triples from text using regex patterns.

    Returns:
        List of (source_label, relation_type, target_label) tuples.
    """
    triples: list[tuple[str, str, str]] = []
    text_lower = text.lower()

    for pattern, rel_type in _RELATION_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            source = match.group(1).strip()
            target = match.group(2).strip()

            # Filter out stopword-only entities
            source_words = [w for w in source.split() if w not in _STOPWORDS]
            target_words = [w for w in target.split() if w not in _STOPWORDS]

            if source_words and target_words:
                source_clean = " ".join(source_words)
                target_clean = " ".join(target_words)
                if len(source_clean) > 1 and len(target_clean) > 1:
                    triples.append((source_clean, rel_type, target_clean))

    return triples


# ── Knowledge Graph ──────────────────────────────────────────────────────────


class KnowledgeGraph:
    """
    In-memory directed knowledge graph with entity and relation storage.

    Supports:
    - Add/get entities and relations
    - Neighbor queries (in/out/both)
    - Shortest path (BFS)
    - Subgraph extraction around a seed entity
    - Text ingestion via regex-based entity extraction
    """

    def __init__(self, max_entities: int = 5000):
        self.max_entities = max_entities
        self._entities: OrderedDict[str, Entity] = OrderedDict()
        self._relations: dict[str, Relation] = {}  # key = Relation.key
        self._outgoing: dict[str, list[str]] = defaultdict(list)  # entity_id → [relation_keys]
        self._incoming: dict[str, list[str]] = defaultdict(list)  # entity_id → [relation_keys]

    def _evict_lru(self) -> None:
        """Evict the oldest entities and their relations until under max_entities."""
        while len(self._entities) > self.max_entities:
            # Pop oldest item (FIFO order for LRU tracking)
            eid, _ = self._entities.popitem(last=False)

            # Remove all incoming/outgoing relations
            out_rels = self._outgoing.pop(eid, [])
            for rel_key in out_rels:
                rel = self._relations.pop(rel_key, None)
                if rel:
                    try:
                        self._incoming[rel.target_id].remove(rel_key)
                    except ValueError:
                        pass

            in_rels = self._incoming.pop(eid, [])
            for rel_key in in_rels:
                rel = self._relations.pop(rel_key, None)
                if rel:
                    try:
                        self._outgoing[rel.source_id].remove(rel_key)
                    except ValueError:
                        pass

    # ── Core operations ──────────────────────────────────────────────────

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    def add_entity(
        self,
        label: str,
        entity_type: str = "concept",
        attributes: dict[str, Any] | None = None,
    ) -> Entity:
        """Add or update an entity. Returns the entity."""
        eid = _entity_id(label)
        if eid in self._entities:
            # Merge attributes
            existing = self._entities[eid]
            if attributes:
                existing.attributes.update(attributes)
            self._entities.move_to_end(eid)  # Update LRU
            return existing

        entity = Entity(
            id=eid,
            label=label.strip().lower(),
            entity_type=entity_type,
            attributes=attributes or {},
        )
        self._entities[eid] = entity
        self._evict_lru()  # Check bounds
        return entity

    def get_entity(self, label: str) -> Entity | None:
        """Get entity by label."""
        eid = _entity_id(label)
        if eid in self._entities:
            self._entities.move_to_end(eid)  # Update LRU
            return self._entities[eid]
        return None

    def add_relation(
        self,
        source_label: str,
        target_label: str,
        relation_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
        valid_from: float | None = None,
        valid_until: float | None = None,
    ) -> Relation:
        """Add a directed relation between two entities. Creates entities if needed."""
        source = self.add_entity(source_label)
        target = self.add_entity(target_label)

        rel = Relation(
            source_id=source.id,
            target_id=target.id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata or {},
            valid_from=valid_from,
            valid_until=valid_until,
        )

        # Deduplicate by key, update weight if exists
        if rel.key in self._relations:
            existing = self._relations[rel.key]
            existing.weight = max(existing.weight, weight)
            if metadata:
                existing.metadata.update(metadata)
            return existing

        self._relations[rel.key] = rel
        self._outgoing[source.id].append(rel.key)
        self._incoming[target.id].append(rel.key)
        return rel

    def add_community(
        self,
        community_label: str,
        member_labels: list[str],
        summary: str = "",
    ) -> Entity:
        """
        Phase 11: GraphRAG Hierarchical Communities.
        Add a macro-entity (Community) and link multiple existing/new leaf entities
        to it via 'member_of' relationships.
        """
        community = self.add_entity(
            label=community_label,
            entity_type="community",
            attributes={"summary": summary, "is_macro_node": True},
        )

        for leaf in member_labels:
            # Point leaf node UP to the macro community node
            self.add_relation(
                source_label=leaf,
                target_label=community_label,
                relation_type="member_of",
                weight=2.0,  # Macro ties are strong
                metadata={"auto_generated": "graphrag"},
            )

        return community

    # ── Query operations ─────────────────────────────────────────────────

    def neighbors(
        self,
        label: str,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get neighbors of an entity.

        Args:
            label: Entity label to query.
            direction: "out", "in", or "both".
            relation_type: Filter by relation type (optional).

        Returns:
            List of dicts with neighbor entity info and relation.
        """
        eid = _entity_id(label)
        if eid not in self._entities:
            return []

        results = []

        if direction in ("out", "both"):
            for rk in self._outgoing.get(eid, []):
                rel = self._relations[rk]
                if relation_type and rel.relation_type != relation_type:
                    continue
                target = self._entities.get(rel.target_id)
                if target:
                    results.append(
                        {
                            "entity": target.label,
                            "entity_type": target.entity_type,
                            "relation": rel.relation_type,
                            "direction": "out",
                            "weight": rel.weight,
                        }
                    )

        if direction in ("in", "both"):
            for rk in self._incoming.get(eid, []):
                rel = self._relations[rk]
                if relation_type and rel.relation_type != relation_type:
                    continue
                source = self._entities.get(rel.source_id)
                if source:
                    results.append(
                        {
                            "entity": source.label,
                            "entity_type": source.entity_type,
                            "relation": rel.relation_type,
                            "direction": "in",
                            "weight": rel.weight,
                        }
                    )

        return results

    def shortest_path(
        self,
        from_label: str,
        to_label: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """
        BFS shortest path between two entities.

        Returns:
            List of entity labels on the path, or None if no path exists.
        """
        start = _entity_id(from_label)
        end = _entity_id(to_label)

        if start not in self._entities or end not in self._entities:
            return None

        if start == end:
            return [self._entities[start].label]

        # BFS using deque for O(1) popleft (list.pop(0) is O(n))
        visited = {start}
        queue = deque([(start, [self._entities[start].label])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            # Expand outgoing edges
            for rk in self._outgoing.get(current, []):
                rel = self._relations[rk]
                next_id = rel.target_id
                if next_id not in visited:
                    new_path = path + [self._entities[next_id].label]
                    if next_id == end:
                        return new_path
                    visited.add(next_id)
                    queue.append((next_id, new_path))

            # Also expand incoming edges (undirected traversal)
            for rk in self._incoming.get(current, []):
                rel = self._relations[rk]
                next_id = rel.source_id
                if next_id not in visited:
                    new_path = path + [self._entities[next_id].label]
                    if next_id == end:
                        return new_path
                    visited.add(next_id)
                    queue.append((next_id, new_path))

        return None

    def subgraph(self, label: str, depth: int = 2) -> dict[str, Any]:
        """
        Extract a subgraph around an entity up to a given depth.

        Returns:
            Dict with "entities" and "relations" lists.
        """
        seed = _entity_id(label)
        if seed not in self._entities:
            return {"entities": [], "relations": []}

        visited_entities: set[str] = {seed}
        visited_relations: set[str] = set()
        frontier = {seed}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for eid in frontier:
                for rk in self._outgoing.get(eid, []):
                    rel = self._relations[rk]
                    visited_relations.add(rk)
                    if rel.target_id not in visited_entities:
                        visited_entities.add(rel.target_id)
                        next_frontier.add(rel.target_id)

                for rk in self._incoming.get(eid, []):
                    rel = self._relations[rk]
                    visited_relations.add(rk)
                    if rel.source_id not in visited_entities:
                        visited_entities.add(rel.source_id)
                        next_frontier.add(rel.source_id)

            frontier = next_frontier

        entities = [
            {"id": e.id, "label": e.label, "type": e.entity_type, "attributes": e.attributes}
            for eid in visited_entities
            if (e := self._entities.get(eid))
        ]
        relations = [
            {
                "source": self._entities[r.source_id].label,
                "target": self._entities[r.target_id].label,
                "type": r.relation_type,
                "weight": r.weight,
            }
            for rk in visited_relations
            if (r := self._relations.get(rk))
            and r.source_id in self._entities
            and r.target_id in self._entities
        ]

        return {"entities": entities, "relations": relations}

    # ── Text ingestion ───────────────────────────────────────────────────

    def ingest_text(self, text: str, source: str = "unknown") -> int:
        """
        Extract entities and relations from text and add to the graph.

        Returns:
            Number of relations added.
        """
        triples = extract_entities_from_text(text)
        added = 0

        for source_label, rel_type, target_label in triples:
            self.add_relation(
                source_label=source_label,
                target_label=target_label,
                relation_type=rel_type,
                metadata={"source": source},
            )
            added += 1

        if added:
            logger.debug(
                "[KnowledgeGraph] Ingested %d relations from text (total: %d entities, %d relations)",
                added,
                self.entity_count,
                self.relation_count,
            )

        return added

    def clear(self) -> None:
        """Clear all entities and relations."""
        self._entities.clear()
        self._relations.clear()
        self._outgoing.clear()
        self._incoming.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a dict."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "label": e.label,
                    "type": e.entity_type,
                    "attributes": e.attributes,
                    "created_at": e.created_at,
                }
                for e in self._entities.values()
            ],
            "relations": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.relation_type,
                    "weight": r.weight,
                    "metadata": r.metadata,
                    "created_at": r.created_at,
                }
                for r in self._relations.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Reconstruct a KnowledgeGraph from a dict."""
        graph = cls()
        for e_data in data.get("entities", []):
            entity = Entity(
                id=e_data["id"],
                label=e_data["label"],
                entity_type=e_data.get("type", "concept"),
                attributes=e_data.get("attributes", {}),
                created_at=e_data.get("created_at", time.time()),
            )
            graph._entities[entity.id] = entity

        for r_data in data.get("relations", []):
            rel = Relation(
                source_id=r_data["source_id"],
                target_id=r_data["target_id"],
                relation_type=r_data["type"],
                weight=r_data.get("weight", 1.0),
                metadata=r_data.get("metadata", {}),
                created_at=r_data.get("created_at", time.time()),
            )
            graph._relations[rel.key] = rel
            graph._outgoing[rel.source_id].append(rel.key)
            graph._incoming[rel.target_id].append(rel.key)

        return graph

    # ── Entity Disambiguation ────────────────────────────────────────────

    def disambiguate_entities(self, similarity_threshold: float = 0.8) -> int:
        """
        Merge similar entities based on label similarity.

        Uses character-level Jaccard similarity to find near-duplicates
        and merges them by redirecting all relations to the canonical entity.

        Returns:
            Number of entities merged.
        """
        merged = 0
        entities = list(self._entities.values())
        merged_ids: set[str] = set()

        for i, e1 in enumerate(entities):
            if e1.id in merged_ids:
                continue
            for e2 in entities[i + 1 :]:
                if e2.id in merged_ids:
                    continue
                if e1.entity_type != e2.entity_type:
                    continue

                sim = self._label_similarity(e1.label, e2.label)
                if sim >= similarity_threshold:
                    # Merge e2 into e1 (canonical)
                    self._merge_entities(e1.id, e2.id)
                    merged_ids.add(e2.id)
                    merged += 1

        if merged:
            logger.info("Disambiguated %d entities (threshold=%.2f)", merged, similarity_threshold)
        return merged

    def _label_similarity(self, a: str, b: str) -> float:
        """Character-level Jaccard similarity between two labels."""
        if a == b:
            return 1.0
        if _HAS_RUST_KG:
            from hbllm_knowledge_graph import entity_similarity  # type: ignore[import-not-found]

            return entity_similarity(a, b)
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _merge_entities(self, keep_id: str, remove_id: str) -> None:
        """Merge remove_id entity into keep_id, redirecting all relations."""
        if remove_id not in self._entities:
            return

        removed = self._entities[remove_id]
        keeper = self._entities.get(keep_id)
        if keeper:
            keeper.attributes.update(removed.attributes)

        # Redirect outgoing relations
        for rk in list(self._outgoing.get(remove_id, [])):
            rel = self._relations.get(rk)
            if rel:
                # Remove old relation
                del self._relations[rk]
                try:
                    self._incoming[rel.target_id].remove(rk)
                except ValueError:
                    pass

                # Create redirected relation
                rel.source_id = keep_id
                new_key = rel.key
                if new_key not in self._relations:
                    self._relations[new_key] = rel
                    self._outgoing[keep_id].append(new_key)
                    self._incoming[rel.target_id].append(new_key)

        # Redirect incoming relations
        for rk in list(self._incoming.get(remove_id, [])):
            rel = self._relations.get(rk)
            if rel:
                del self._relations[rk]
                try:
                    self._outgoing[rel.source_id].remove(rk)
                except ValueError:
                    pass

                rel.target_id = keep_id
                new_key = rel.key
                if new_key not in self._relations:
                    self._relations[new_key] = rel
                    self._outgoing[rel.source_id].append(new_key)
                    self._incoming[keep_id].append(new_key)

        # Remove merged entity
        self._outgoing.pop(remove_id, None)
        self._incoming.pop(remove_id, None)
        del self._entities[remove_id]

    # ── Community Detection ──────────────────────────────────────────────

    def community_detection(self) -> dict[str, list[str]]:
        """
        Louvain-inspired greedy modularity community detection.

        Returns:
            Dict mapping community_id → list of entity labels.
        """
        # Initialize: each entity in its own community
        entity_ids = list(self._entities.keys())
        community_of: dict[str, str] = {eid: eid for eid in entity_ids}

        # Build adjacency with weights
        adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_weight = 0.0
        for rel in self._relations.values():
            if rel.source_id in self._entities and rel.target_id in self._entities:
                adj[rel.source_id][rel.target_id] += rel.weight
                adj[rel.target_id][rel.source_id] += rel.weight
                total_weight += rel.weight

        if total_weight == 0:
            return {}

        # Greedy modularity: iterate until no improvement
        improved = True
        max_iterations = 20
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for eid in entity_ids:
                current_comm = community_of[eid]

                # Find neighboring communities
                neighbor_comms: dict[str, float] = defaultdict(float)
                for neighbor_id, weight in adj.get(eid, {}).items():
                    nc = community_of[neighbor_id]
                    neighbor_comms[nc] += weight

                # Try moving to best neighboring community
                best_comm = current_comm
                best_gain = 0.0
                for comm, weight_sum in neighbor_comms.items():
                    if comm != current_comm and weight_sum > best_gain:
                        best_gain = weight_sum
                        best_comm = comm

                if best_comm != current_comm:
                    community_of[eid] = best_comm
                    improved = True

        # Group by community
        communities: dict[str, list[str]] = defaultdict(list)
        for eid, comm in community_of.items():
            entity = self._entities.get(eid)
            if entity:
                communities[comm].append(entity.label)

        # Re-key with clean IDs
        result = {}
        for i, (_, members) in enumerate(sorted(communities.items(), key=lambda x: -len(x[1]))):
            result[f"community_{i}"] = members

        logger.debug(
            "Community detection found %d communities from %d entities",
            len(result),
            len(entity_ids),
        )
        return result

    # ── PageRank ─────────────────────────────────────────────────────────

    def pagerank(self, damping: float = 0.85, iterations: int = 20) -> dict[str, float]:
        """
        Compute PageRank importance scores for all entities.

        Args:
            damping: Damping factor (0.85 standard).
            iterations: Number of power iterations.

        Returns:
            Dict mapping entity label → PageRank score.
        """
        entity_ids = list(self._entities.keys())
        n = len(entity_ids)
        if n == 0:
            return {}

        # Initialize uniform
        scores: dict[str, float] = {eid: 1.0 / n for eid in entity_ids}

        for _ in range(iterations):
            new_scores: dict[str, float] = {}
            for eid in entity_ids:
                # Sum contributions from incoming neighbors
                incoming_sum = 0.0
                for rk in self._incoming.get(eid, []):
                    rel = self._relations.get(rk)
                    if rel and rel.source_id in scores:
                        # Out-degree of source
                        out_degree = len(self._outgoing.get(rel.source_id, []))
                        if out_degree > 0:
                            incoming_sum += scores[rel.source_id] / out_degree

                new_scores[eid] = (1 - damping) / n + damping * incoming_sum
            scores = new_scores

        # Map to labels
        return {
            self._entities[eid].label: score
            for eid, score in sorted(scores.items(), key=lambda x: -x[1])
            if eid in self._entities
        }

    def save_to_disk(self, path: str | Path) -> None:
        """Save the knowledge graph to a JSON file."""
        import json
        from pathlib import Path as _Path

        save_path = _Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f)
        logger.info(
            "KnowledgeGraph saved to %s (%d entities, %d relations)",
            save_path,
            self.entity_count,
            self.relation_count,
        )

    @classmethod
    def load_from_disk(cls, path: str | Path) -> KnowledgeGraph:
        """Load a knowledge graph from a JSON file."""
        import json
        from pathlib import Path as _Path

        load_path = _Path(path)
        if not load_path.exists():
            logger.info("No KnowledgeGraph file at %s, starting empty", load_path)
            return cls()
        with open(load_path) as f:
            data = json.load(f)
        graph = cls.from_dict(data)
        logger.info(
            "KnowledgeGraph loaded from %s (%d entities, %d relations)",
            load_path,
            graph.entity_count,
            graph.relation_count,
        )
        return graph
