"""Concept Formation Engine — discovers abstractions from causal models.

Runs during sleep to find recurring causal structures across different
concepts and abstract them into generalized concepts.

Example:
    SQL Injection, XSS, LDAP Injection all share the causal structure:
        Unsanitized Input → Parser Manipulation → Unauthorized Execution

    The engine discovers this shared structure and creates:
        AbstractConcept("Injection Attack Pattern")

This is how intelligence compresses knowledge — the same mechanism
of concept formation that humans use unconsciously.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class AbstractConcept:
    """A generalized concept discovered from recurring causal patterns."""

    concept_id: str = field(
        default_factory=lambda: f"abs_{uuid.uuid4().hex[:12]}"
    )
    label: str = ""  # e.g. "Injection Attack Pattern"
    description: str = ""
    generalized_steps: list[str] = field(default_factory=list)
    generalized_assumptions: list[str] = field(default_factory=list)
    instances: list[str] = field(default_factory=list)  # Specific concept names
    instance_model_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    domain: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "description": self.description,
            "generalized_steps": self.generalized_steps,
            "generalized_assumptions": self.generalized_assumptions,
            "instances": self.instances,
            "instance_model_ids": self.instance_model_ids,
            "confidence": self.confidence,
            "domain": self.domain,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AbstractConcept:
        return cls(
            concept_id=d.get("concept_id", f"abs_{uuid.uuid4().hex[:12]}"),
            label=d.get("label", ""),
            description=d.get("description", ""),
            generalized_steps=d.get("generalized_steps", []),
            generalized_assumptions=d.get("generalized_assumptions", []),
            instances=d.get("instances", []),
            instance_model_ids=d.get("instance_model_ids", []),
            confidence=d.get("confidence", 0.5),
            domain=d.get("domain", ""),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class CrossDomainAnalogy:
    """A discovered analogy between concepts in different domains."""

    analogy_id: str = field(
        default_factory=lambda: f"ana_{uuid.uuid4().hex[:10]}"
    )
    domain_a: str = ""
    domain_b: str = ""
    concept_a: str = ""
    concept_b: str = ""
    shared_structure: list[str] = field(default_factory=list)  # Shared steps
    similarity_score: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analogy_id": self.analogy_id,
            "domain_a": self.domain_a,
            "domain_b": self.domain_b,
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "shared_structure": self.shared_structure,
            "similarity_score": self.similarity_score,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrossDomainAnalogy:
        return cls(
            analogy_id=d.get("analogy_id", f"ana_{uuid.uuid4().hex[:10]}"),
            domain_a=d.get("domain_a", ""),
            domain_b=d.get("domain_b", ""),
            concept_a=d.get("concept_a", ""),
            concept_b=d.get("concept_b", ""),
            shared_structure=d.get("shared_structure", []),
            similarity_score=d.get("similarity_score", 0.0),
            created_at=d.get("created_at", time.time()),
        )


# ── LLM Prompt ───────────────────────────────────────────────────────────────

_ABSTRACTION_PROMPT = """\
Given these concepts that share a similar causal structure:

{concepts_description}

Shared mechanism steps:
{shared_steps}

Questions:
1. What is the generalized pattern that these concepts share?
2. Give it a concise label (e.g. "Injection Attack Pattern").
3. Describe the abstract mechanism in general terms.
4. What are the key assumptions that must hold for this pattern?

Return a JSON object:
{{
  "label": "Concise name for the abstract concept",
  "description": "General description of the pattern",
  "generalized_steps": ["abstract step 1", "abstract step 2", ...],
  "generalized_assumptions": ["assumption 1", ...]
}}

Return ONLY valid JSON, no markdown."""


# ── Concept Formation Engine ─────────────────────────────────────────────────


class ConceptFormationEngine:
    """Discovers abstract concepts from recurring causal patterns.

    Runs during sleep to:
    1. Find causal models with similar mechanism structures
    2. Abstract shared patterns into generalized concepts
    3. Discover cross-domain analogies
    4. Compress knowledge by linking specific instances to abstractions

    This is where intelligence compresses knowledge — the same mechanism
    of concept formation that humans use during sleep.
    """

    def __init__(
        self,
        llm: Any | None = None,
        causal_model_builder: Any | None = None,
        knowledge_graph: Any | None = None,
        data_dir: str | Path = "data",
    ) -> None:
        self.llm = llm
        self.causal_model_builder = causal_model_builder
        self.knowledge_graph = knowledge_graph
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # SQLite storage
        self._db_path = self.data_dir / "concept_formation.db"
        self._init_db()

        # In-memory caches
        self._abstract_concepts: dict[str, AbstractConcept] = {}
        self._analogies: list[CrossDomainAnalogy] = []
        self._load_from_db()

        # Telemetry
        self._concepts_formed = 0
        self._analogies_found = 0

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS abstract_concepts (
                    concept_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    data TEXT NOT NULL,
                    domain TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.5,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_domain_analogies (
                    analogy_id TEXT PRIMARY KEY,
                    domain_a TEXT NOT NULL,
                    domain_b TEXT NOT NULL,
                    data TEXT NOT NULL,
                    similarity_score REAL DEFAULT 0.0,
                    created_at REAL NOT NULL
                )
            """)

    def _load_from_db(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                for row in conn.execute("SELECT * FROM abstract_concepts"):
                    concept = AbstractConcept.from_dict(json.loads(row["data"]))
                    self._abstract_concepts[concept.label.lower()] = concept
                for row in conn.execute("SELECT * FROM cross_domain_analogies"):
                    analogy = CrossDomainAnalogy.from_dict(json.loads(row["data"]))
                    self._analogies.append(analogy)
        except Exception as e:
            logger.debug("Failed to load concept formation data: %s", e)

    def _persist_concept(self, concept: AbstractConcept) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO abstract_concepts
                       (concept_id, label, data, domain, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        concept.concept_id,
                        concept.label,
                        json.dumps(concept.to_dict()),
                        concept.domain,
                        concept.confidence,
                        concept.created_at,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist abstract concept: %s", e)

    def _persist_analogy(self, analogy: CrossDomainAnalogy) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO cross_domain_analogies
                       (analogy_id, domain_a, domain_b, data,
                        similarity_score, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        analogy.analogy_id,
                        analogy.domain_a,
                        analogy.domain_b,
                        json.dumps(analogy.to_dict()),
                        analogy.similarity_score,
                        analogy.created_at,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist analogy: %s", e)

    # ── Core API ─────────────────────────────────────────────────────────

    async def discover_abstractions(
        self,
        domain: str | None = None,
    ) -> list[AbstractConcept]:
        """Find recurring causal structures and abstract them into concepts.

        This is the main entry point, typically called during sleep.
        """
        if self.causal_model_builder is None:
            logger.debug("No CausalModelBuilder available — skipping abstraction")
            return []

        models = self.causal_model_builder.get_all_models(domain=domain)
        if len(models) < 2:
            return []

        # Group models by structural similarity of their mechanisms
        clusters = self._cluster_by_mechanism(models)
        abstractions: list[AbstractConcept] = []

        for cluster_key, cluster_models in clusters.items():
            if len(cluster_models) < 2:
                continue

            # Extract shared mechanism steps
            shared_steps = self._find_shared_steps(cluster_models)
            if not shared_steps:
                continue

            # Use LLM to generate abstract concept label and description
            abstract = await self._generate_abstraction(
                cluster_models, shared_steps, domain or ""
            )
            if abstract is not None:
                abstractions.append(abstract)
                self._abstract_concepts[abstract.label.lower()] = abstract
                self._persist_concept(abstract)
                self._concepts_formed += 1

                # Store in KnowledgeGraph if available
                if self.knowledge_graph is not None:
                    self._store_in_knowledge_graph(abstract)

        logger.info(
            "Concept formation: discovered %d abstractions from %d models",
            len(abstractions),
            len(models),
        )
        return abstractions

    async def discover_cross_domain_analogies(self) -> list[CrossDomainAnalogy]:
        """Find same causal structure across different domains.

        Example:
            cybersecurity: "infection → propagation → containment"
            biology: "infection → spread → immune response"
        """
        if self.causal_model_builder is None:
            return []

        models = self.causal_model_builder.get_all_models()
        if len(models) < 2:
            return []

        # Group models by domain
        by_domain: dict[str, list[Any]] = defaultdict(list)
        for model in models:
            if model.domain:
                by_domain[model.domain].append(model)

        domains = list(by_domain.keys())
        if len(domains) < 2:
            return []

        analogies: list[CrossDomainAnalogy] = []

        # Compare models across domains
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain_a = domains[i]
                domain_b = domains[j]

                for model_a in by_domain[domain_a]:
                    for model_b in by_domain[domain_b]:
                        similarity = self._compute_structural_similarity(
                            model_a, model_b
                        )
                        if similarity > 0.4:
                            shared = self._extract_shared_structure(model_a, model_b)
                            analogy = CrossDomainAnalogy(
                                domain_a=domain_a,
                                domain_b=domain_b,
                                concept_a=model_a.concept,
                                concept_b=model_b.concept,
                                shared_structure=shared,
                                similarity_score=similarity,
                            )
                            analogies.append(analogy)
                            self._analogies.append(analogy)
                            self._persist_analogy(analogy)
                            self._analogies_found += 1

        if analogies:
            logger.info(
                "Cross-domain analogy discovery: found %d analogies across %d domains",
                len(analogies),
                len(domains),
            )
        return analogies

    async def compress_knowledge(self) -> int:
        """Link specific instances to their abstract concepts.

        Returns number of instances linked.
        """
        linked = 0
        if self.knowledge_graph is None:
            return linked

        for concept in self._abstract_concepts.values():
            for instance in concept.instances:
                try:
                    self.knowledge_graph.add_relation(
                        source_label=instance,
                        target_label=concept.label,
                        relation_type="instance_of",
                        weight=concept.confidence,
                        metadata={
                            "abstract_concept_id": concept.concept_id,
                            "auto_discovered": True,
                        },
                    )
                    linked += 1
                except Exception:
                    pass

        if linked:
            logger.info("Compressed knowledge: linked %d instances to abstractions", linked)
        return linked

    def get_abstract_concepts(self, domain: str | None = None) -> list[AbstractConcept]:
        """Get all abstract concepts, optionally filtered by domain."""
        concepts = list(self._abstract_concepts.values())
        if domain:
            concepts = [c for c in concepts if c.domain == domain]
        return concepts

    def get_analogies(self) -> list[CrossDomainAnalogy]:
        """Get all discovered cross-domain analogies."""
        return list(self._analogies)

    def stats(self) -> dict[str, Any]:
        return {
            "abstract_concepts": len(self._abstract_concepts),
            "cross_domain_analogies": len(self._analogies),
            "concepts_formed": self._concepts_formed,
            "analogies_found": self._analogies_found,
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _cluster_by_mechanism(
        self,
        models: list[Any],
    ) -> dict[str, list[Any]]:
        """Group models by structural similarity of their mechanisms."""
        clusters: dict[str, list[Any]] = defaultdict(list)

        for model in models:
            # Create a fingerprint from mechanism step patterns
            step_fingerprint = self._mechanism_fingerprint(model)
            clusters[step_fingerprint].append(model)

        return dict(clusters)

    def _mechanism_fingerprint(self, model: Any) -> str:
        """Create a structural fingerprint from a model's mechanism steps."""
        all_steps: list[str] = []
        for edge in getattr(model, "edges", []):
            mechanism = getattr(edge, "mechanism", None)
            if mechanism:
                steps = getattr(mechanism, "steps", [])
                # Normalize: extract key verbs/nouns
                for step in steps:
                    words = step.lower().split()
                    key_words = [w for w in words if len(w) > 3][:3]
                    all_steps.extend(key_words)

        # Sort to make order-independent
        return "|".join(sorted(set(all_steps)))

    def _find_shared_steps(self, models: list[Any]) -> list[str]:
        """Find mechanism steps that appear across multiple models."""
        step_counts: dict[str, int] = defaultdict(int)

        for model in models:
            model_steps: set[str] = set()
            for edge in getattr(model, "edges", []):
                mechanism = getattr(edge, "mechanism", None)
                if mechanism:
                    for step in getattr(mechanism, "steps", []):
                        normalized = step.lower().strip()
                        if normalized:
                            model_steps.add(normalized)

            for step in model_steps:
                step_counts[step] += 1

        # Return steps that appear in at least 2 models
        min_models = min(2, len(models))
        shared = [
            step for step, count in step_counts.items() if count >= min_models
        ]
        return sorted(shared)

    async def _generate_abstraction(
        self,
        models: list[Any],
        shared_steps: list[str],
        domain: str,
    ) -> AbstractConcept | None:
        """Use LLM to generate an abstract concept from shared patterns."""
        if self.llm is None:
            # Fallback: generate from shared steps without LLM
            return AbstractConcept(
                label=f"Shared Pattern: {shared_steps[0]}" if shared_steps else "Unknown",
                description="Auto-discovered shared causal pattern",
                generalized_steps=shared_steps,
                instances=[getattr(m, "concept", "") for m in models],
                instance_model_ids=[getattr(m, "model_id", "") for m in models],
                confidence=0.3,
                domain=domain,
            )

        concepts_desc = "\n".join(
            f"- {getattr(m, 'concept', 'unknown')}: "
            f"{', '.join(n.label for n in getattr(m, 'nodes', []))}"
            for m in models[:5]
        )

        prompt = _ABSTRACTION_PROMPT.format(
            concepts_description=concepts_desc,
            shared_steps="\n".join(f"- {s}" for s in shared_steps),
        )

        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_json(content)

            return AbstractConcept(
                label=parsed.get("label", f"Pattern from {len(models)} concepts"),
                description=parsed.get("description", ""),
                generalized_steps=parsed.get("generalized_steps", shared_steps),
                generalized_assumptions=parsed.get("generalized_assumptions", []),
                instances=[getattr(m, "concept", "") for m in models],
                instance_model_ids=[getattr(m, "model_id", "") for m in models],
                confidence=0.5,
                domain=domain,
            )
        except Exception as e:
            logger.warning("LLM abstraction generation failed: %s", e)
            return None

    def _compute_structural_similarity(
        self,
        model_a: Any,
        model_b: Any,
    ) -> float:
        """Compute structural similarity between two causal models."""
        steps_a: set[str] = set()
        steps_b: set[str] = set()

        for edge in getattr(model_a, "edges", []):
            mechanism = getattr(edge, "mechanism", None)
            if mechanism:
                for step in getattr(mechanism, "steps", []):
                    # Extract key words for comparison
                    words = set(step.lower().split())
                    steps_a.update(w for w in words if len(w) > 3)

        for edge in getattr(model_b, "edges", []):
            mechanism = getattr(edge, "mechanism", None)
            if mechanism:
                for step in getattr(mechanism, "steps", []):
                    words = set(step.lower().split())
                    steps_b.update(w for w in words if len(w) > 3)

        if not steps_a or not steps_b:
            return 0.0

        intersection = steps_a & steps_b
        union = steps_a | steps_b
        return len(intersection) / len(union) if union else 0.0

    def _extract_shared_structure(
        self,
        model_a: Any,
        model_b: Any,
    ) -> list[str]:
        """Extract shared structural elements between two models."""
        steps_a: list[str] = []
        steps_b: list[str] = []

        for edge in getattr(model_a, "edges", []):
            mechanism = getattr(edge, "mechanism", None)
            if mechanism:
                steps_a.extend(getattr(mechanism, "steps", []))

        for edge in getattr(model_b, "edges", []):
            mechanism = getattr(edge, "mechanism", None)
            if mechanism:
                steps_b.extend(getattr(mechanism, "steps", []))

        # Find word-level overlap
        words_a = {s.lower().strip() for s in steps_a}
        words_b = {s.lower().strip() for s in steps_b}
        return sorted(words_a & words_b)

    def _store_in_knowledge_graph(self, concept: AbstractConcept) -> None:
        """Store abstract concept in KnowledgeGraph."""
        try:
            self.knowledge_graph.add_entity(
                label=concept.label,
                entity_type="abstract_concept",
                attributes={
                    "concept_id": concept.concept_id,
                    "description": concept.description,
                    "instances": concept.instances,
                    "confidence": concept.confidence,
                    "domain": concept.domain,
                },
            )

            # Link instances to the abstract concept
            for instance in concept.instances:
                self.knowledge_graph.add_relation(
                    source_label=instance,
                    target_label=concept.label,
                    relation_type="instance_of",
                    weight=concept.confidence,
                )
        except Exception as e:
            logger.debug("Failed to store abstract concept in KG: %s", e)

    def _parse_json(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {}
