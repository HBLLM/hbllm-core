"""
Knowledge Graph Builder — constructs a knowledge graph during training.

As the model processes training documents, this module extracts entities,
relationships, and topic clusters, building a structured graph that can
be used at inference time by the Brain for retrieval and reasoning.

Output: knowledge_graph.json with entities, edges, and topic clusters.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A named entity in the knowledge graph."""
    name: str
    entity_type: str  # person, org, tech, concept, place
    frequency: int = 0
    first_seen_step: int = 0
    contexts: list[str] = field(default_factory=list)  # sample contexts


@dataclass
class Relationship:
    """An edge between two entities."""
    source: str
    target: str
    relation_type: str  # co-occurs, part-of, related-to
    weight: float = 1.0


class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from training data.

    During training, call add_from_text() with each batch's raw text.
    The builder extracts entities and co-occurrence relationships,
    building a graph that grows over the training run.

    At checkpoint time, call save() to persist the graph.
    """

    # Common technical terms to track as entities
    TECH_PATTERNS = [
        r'\b(?:Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|Ruby|PHP|Swift|Kotlin)\b',
        r'\b(?:React|Vue|Angular|Django|Flask|Laravel|Rails|Spring|Express|FastAPI)\b',
        r'\b(?:Docker|Kubernetes|AWS|Azure|GCP|Linux|macOS|Windows|Git|GitHub)\b',
        r'\b(?:PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|SQLite|DynamoDB)\b',
        r'\b(?:TensorFlow|PyTorch|Transformers|CUDA|ONNX|LLVM|WebAssembly)\b',
        r'\b(?:HTTP|REST|GraphQL|gRPC|WebSocket|OAuth|JWT|SSL|TLS|DNS)\b',
        r'\b(?:machine learning|deep learning|neural network|natural language)\b',
        r'\b(?:API|SDK|CLI|GUI|UI|UX|CI|CD|DevOps|MLOps)\b',
    ]

    # Patterns for concept-type entities
    CONCEPT_PATTERNS = [
        r'\b(?:algorithm|architecture|pattern|framework|protocol|paradigm)\b',
        r'\b(?:optimization|inference|training|embedding|tokenization)\b',
        r'\b(?:classification|regression|clustering|segmentation)\b',
    ]

    def __init__(self, max_entities: int = 10000, max_contexts_per_entity: int = 5):
        self.entities: dict[str, Entity] = {}
        self.edges: list[Relationship] = []
        self._cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        self._topic_keywords: Counter = Counter()
        self._doc_count = 0
        self._max_entities = max_entities
        self._max_contexts = max_contexts_per_entity

        # Compile regex patterns
        self._tech_re = re.compile('|'.join(self.TECH_PATTERNS), re.IGNORECASE)
        self._concept_re = re.compile('|'.join(self.CONCEPT_PATTERNS), re.IGNORECASE)

    def add_from_text(self, text: str, step: int = 0) -> list[str]:
        """
        Extract entities from text and add to the knowledge graph.

        Args:
            text: Raw document text from training batch
            step: Current training step number

        Returns:
            List of entity names found
        """
        self._doc_count += 1
        found_entities: list[str] = []

        # Extract tech entities
        for match in self._tech_re.finditer(text):
            name = match.group().strip()
            found_entities.append(name)
            self._add_entity(name, "tech", step, text[:200])

        # Extract concept entities
        for match in self._concept_re.finditer(text):
            name = match.group().strip().lower()
            found_entities.append(name)
            self._add_entity(name, "concept", step, text[:200])

        # Track co-occurrences for relationship building
        unique = list(set(found_entities))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = tuple(sorted([unique[i], unique[j]]))
                self._cooccurrence[pair] += 1

        # Track topic keywords
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        self._topic_keywords.update(words)

        return found_entities

    def add_from_batch(self, texts: list[str], step: int = 0) -> int:
        """Process a batch of texts. Returns total entities found."""
        total = 0
        for text in texts:
            entities = self.add_from_text(text, step)
            total += len(entities)
        return total

    def _add_entity(self, name: str, entity_type: str, step: int, context: str) -> None:
        """Add or update an entity."""
        key = name.lower()
        if key not in self.entities:
            if len(self.entities) >= self._max_entities:
                return
            self.entities[key] = Entity(
                name=name,
                entity_type=entity_type,
                frequency=0,
                first_seen_step=step,
            )
        ent = self.entities[key]
        ent.frequency += 1
        if len(ent.contexts) < self._max_contexts:
            ent.contexts.append(context[:100])

    def build_edges(self, min_cooccurrence: int = 3) -> list[Relationship]:
        """Build relationship edges from co-occurrence counts."""
        self.edges = []
        for (src, tgt), count in self._cooccurrence.items():
            if count >= min_cooccurrence:
                self.edges.append(Relationship(
                    source=src,
                    target=tgt,
                    relation_type="co-occurs",
                    weight=float(count),
                ))
        self.edges.sort(key=lambda e: e.weight, reverse=True)
        return self.edges

    def get_topic_clusters(self, top_n: int = 20) -> list[dict]:
        """Get top topic clusters from keyword frequency."""
        stopwords = {
            "that", "this", "with", "from", "have", "been", "were", "will",
            "would", "could", "should", "their", "which", "about", "there",
            "other", "some", "when", "them", "then", "than", "also", "into",
            "more", "very", "just", "each", "only", "between", "such",
        }
        filtered = [
            (word, count) for word, count in self._topic_keywords.most_common(top_n * 3)
            if word not in stopwords
        ]
        return [{"keyword": w, "count": c} for w, c in filtered[:top_n]]

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        type_counts = Counter(e.entity_type for e in self.entities.values())
        return {
            "total_entities": len(self.entities),
            "total_edges": len(self.edges),
            "documents_processed": self._doc_count,
            "entity_types": dict(type_counts),
            "top_entities": [
                {"name": e.name, "type": e.entity_type, "freq": e.frequency}
                for e in sorted(self.entities.values(), key=lambda x: x.frequency, reverse=True)[:10]
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save knowledge graph to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.build_edges()

        data = {
            "metadata": {
                "created_at": time.time(),
                "documents_processed": self._doc_count,
                "total_entities": len(self.entities),
                "total_edges": len(self.edges),
            },
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "frequency": e.frequency,
                    "first_seen_step": e.first_seen_step,
                    "contexts": e.contexts,
                }
                for e in sorted(self.entities.values(), key=lambda x: x.frequency, reverse=True)
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation_type,
                    "weight": e.weight,
                }
                for e in self.edges[:5000]  # cap edges
            ],
            "topic_clusters": self.get_topic_clusters(50),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "Knowledge graph saved: %d entities, %d edges → %s",
            len(self.entities), len(self.edges), path,
        )

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraphBuilder:
        """Load a previously saved knowledge graph."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        builder = cls()
        builder._doc_count = data["metadata"]["documents_processed"]

        for ent_data in data["entities"]:
            key = ent_data["name"].lower()
            builder.entities[key] = Entity(
                name=ent_data["name"],
                entity_type=ent_data["type"],
                frequency=ent_data["frequency"],
                first_seen_step=ent_data.get("first_seen_step", 0),
                contexts=ent_data.get("contexts", []),
            )

        for edge_data in data["edges"]:
            builder.edges.append(Relationship(
                source=edge_data["source"],
                target=edge_data["target"],
                relation_type=edge_data["relation"],
                weight=edge_data["weight"],
            ))

        logger.info("Loaded knowledge graph: %d entities, %d edges", len(builder.entities), len(builder.edges))
        return builder
