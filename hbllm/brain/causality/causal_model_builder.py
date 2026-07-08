"""Causal Model Builder — transforms flat facts into cause→effect reasoning chains.

Builds structured causal models from learned knowledge using LLM to extract
cause-effect relationships. Models are stored in the existing CausalGraph
and cross-referenced with the KnowledgeGraph.

Key concepts:
    - Mechanism: A first-class reusable reasoning primitive (not plain text).
    - CausalNode: A step in a causal chain (precondition, process, outcome).
    - CausalEdge: A directed link with a Mechanism explaining HOW.
    - CausalModel: A complete causal model for a domain concept.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class Mechanism:
    """First-class reasoning primitive — reusable across causal models.

    Instead of storing mechanism as plain text like:
        "unsanitized input alters query"
    we store structured process steps, assumptions, and confidence.
    Mechanisms become reusable across models (e.g. "injection" mechanism
    is shared by SQL injection, XSS, LDAP injection).
    """

    mechanism_id: str = field(default_factory=lambda: f"mech_{uuid.uuid4().hex[:12]}")
    description: str = ""
    steps: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    reuse_count: int = 0
    domain: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "description": self.description,
            "steps": self.steps,
            "assumptions": self.assumptions,
            "confidence": self.confidence,
            "reuse_count": self.reuse_count,
            "domain": self.domain,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Mechanism:
        return cls(
            mechanism_id=d.get("mechanism_id", f"mech_{uuid.uuid4().hex[:12]}"),
            description=d.get("description", ""),
            steps=d.get("steps", []),
            assumptions=d.get("assumptions", []),
            confidence=d.get("confidence", 0.5),
            reuse_count=d.get("reuse_count", 0),
            domain=d.get("domain", ""),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class CausalNode:
    """A step in a causal chain."""

    node_id: str = field(default_factory=lambda: f"cn_{uuid.uuid4().hex[:10]}")
    label: str = ""
    node_type: str = "process"  # "precondition" | "process" | "outcome"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalNode:
        return cls(
            node_id=d.get("node_id", f"cn_{uuid.uuid4().hex[:10]}"),
            label=d.get("label", ""),
            node_type=d.get("node_type", "process"),
            confidence=d.get("confidence", 0.5),
        )


@dataclass
class CausalEdge:
    """A directed causal link with a first-class Mechanism."""

    source_id: str = ""
    target_id: str = ""
    mechanism: Mechanism = field(default_factory=Mechanism)
    probability: float = 0.5
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "mechanism": self.mechanism.to_dict(),
            "probability": self.probability,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalEdge:
        return cls(
            source_id=d.get("source_id", ""),
            target_id=d.get("target_id", ""),
            mechanism=Mechanism.from_dict(d.get("mechanism", {})),
            probability=d.get("probability", 0.5),
            evidence=d.get("evidence", []),
        )


@dataclass
class CausalModel:
    """A complete causal model for a domain concept."""

    model_id: str = field(default_factory=lambda: f"cm_{uuid.uuid4().hex[:12]}")
    concept: str = ""
    domain: str = ""
    nodes: list[CausalNode] = field(default_factory=list)
    edges: list[CausalEdge] = field(default_factory=list)
    confidence: float = 0.5
    evidence_count: int = 0
    verified: bool = False
    created_at: float = field(default_factory=time.time)
    last_verified_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "concept": self.concept,
            "domain": self.domain,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "verified": self.verified,
            "created_at": self.created_at,
            "last_verified_at": self.last_verified_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalModel:
        return cls(
            model_id=d.get("model_id", f"cm_{uuid.uuid4().hex[:12]}"),
            concept=d.get("concept", ""),
            domain=d.get("domain", ""),
            nodes=[CausalNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[CausalEdge.from_dict(e) for e in d.get("edges", [])],
            confidence=d.get("confidence", 0.5),
            evidence_count=d.get("evidence_count", 0),
            verified=d.get("verified", False),
            created_at=d.get("created_at", time.time()),
            last_verified_at=d.get("last_verified_at"),
        )


# ── LLM Prompt Templates ────────────────────────────────────────────────────

_CAUSAL_DECOMPOSITION_PROMPT = """\
Decompose the concept "{concept}" into a causal chain.

{context}

Return a JSON object with:
{{
  "nodes": [
    {{"label": "...", "node_type": "precondition|process|outcome", "confidence": 0.0-1.0}}
  ],
  "edges": [
    {{
      "source_label": "...",
      "target_label": "...",
      "mechanism": {{
        "description": "HOW the cause produces the effect",
        "steps": ["step1", "step2", ...],
        "assumptions": ["what must be true for this to work"]
      }},
      "probability": 0.0-1.0
    }}
  ],
  "domain": "the knowledge domain this belongs to"
}}

Be precise and specific. Each edge must explain the mechanism — not just that
A causes B, but HOW A causes B.

Return ONLY valid JSON, no markdown."""


# ── Causal Model Builder ─────────────────────────────────────────────────────


class CausalModelBuilder:
    """Extracts and builds causal models from knowledge.

    Uses LLM to decompose concepts into causal chains with first-class
    Mechanisms, then stores them in the CausalGraph for reasoning.
    """

    def __init__(
        self,
        llm: Any,
        causal_graph: Any | None = None,
        knowledge_graph: Any | None = None,
        mechanism_store: Any | None = None,
        data_dir: str | Path = "data",
    ) -> None:
        self.llm = llm
        self.causal_graph = causal_graph
        self.knowledge_graph = knowledge_graph
        self.mechanism_store = mechanism_store
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # SQLite storage for models and mechanisms
        self._db_path = self.data_dir / "causal_models.db"
        self._init_db()

        # In-memory caches
        self._models: dict[str, CausalModel] = {}
        self._mechanisms: dict[str, Mechanism] = {}

        # Telemetry
        self._models_built = 0
        self._mechanisms_created = 0

        # Load persisted models
        self._load_from_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_models (
                    model_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    domain TEXT DEFAULT '',
                    data TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    evidence_count INTEGER DEFAULT 0,
                    verified INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_verified_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cm_concept ON causal_models(concept)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mechanisms (
                    mechanism_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    data TEXT NOT NULL,
                    domain TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.5,
                    reuse_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)

    def _load_from_db(self) -> None:
        """Load persisted models and mechanisms into memory."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                for row in conn.execute("SELECT * FROM causal_models"):
                    model = CausalModel.from_dict(json.loads(row["data"]))
                    self._models[model.concept.lower()] = model
                for row in conn.execute("SELECT * FROM mechanisms"):
                    mech = Mechanism.from_dict(json.loads(row["data"]))
                    self._mechanisms[mech.mechanism_id] = mech
        except Exception as e:
            logger.debug("Failed to load causal models from DB: %s", e)

    def _persist_model(self, model: CausalModel) -> None:
        """Save model to SQLite."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO causal_models
                       (model_id, concept, domain, data, confidence,
                        evidence_count, verified, created_at, last_verified_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model.model_id,
                        model.concept,
                        model.domain,
                        json.dumps(model.to_dict()),
                        model.confidence,
                        model.evidence_count,
                        1 if model.verified else 0,
                        model.created_at,
                        model.last_verified_at,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist causal model: %s", e)

    def _persist_mechanism(self, mech: Mechanism) -> None:
        """Save mechanism to SQLite."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO mechanisms
                       (mechanism_id, description, data, domain,
                        confidence, reuse_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        mech.mechanism_id,
                        mech.description,
                        json.dumps(mech.to_dict()),
                        mech.domain,
                        mech.confidence,
                        mech.reuse_count,
                        mech.created_at,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist mechanism: %s", e)

    # ── Core API ─────────────────────────────────────────────────────────

    async def build_model(
        self,
        concept: str,
        context: str = "",
        domain: str = "",
    ) -> CausalModel:
        """Use LLM to decompose a concept into a causal chain.

        Args:
            concept: The concept to model (e.g. "SQL Injection").
            context: Additional context for the LLM.
            domain: Knowledge domain (e.g. "cybersecurity").

        Returns:
            A structured CausalModel with nodes, edges, and mechanisms.
        """
        # Check if we already have a model
        existing = self.get_model(concept)
        if existing is not None:
            logger.debug("Reusing existing causal model for '%s'", concept)
            return existing

        # Build prompt
        ctx = f"Additional context: {context}" if context else ""
        prompt = _CAUSAL_DECOMPOSITION_PROMPT.format(concept=concept, context=ctx)

        # Call LLM
        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)

            # Parse JSON from response
            parsed = self._parse_llm_json(content)
            model = self._build_from_parsed(parsed, concept, domain)
        except Exception as e:
            logger.warning("LLM causal decomposition failed for '%s': %s", concept, e)
            # Return minimal stub model
            model = CausalModel(
                concept=concept,
                domain=domain,
                nodes=[CausalNode(label=concept, node_type="process")],
                confidence=0.1,
            )

        # Store
        self._models[concept.lower()] = model
        self._persist_model(model)
        self._models_built += 1

        # Cross-reference with CausalGraph if available
        if self.causal_graph is not None:
            self._store_in_causal_graph(model)

        # Cross-reference with KnowledgeGraph if available
        if self.knowledge_graph is not None:
            self._store_in_knowledge_graph(model)

        logger.info(
            "Built causal model for '%s': %d nodes, %d edges, confidence=%.2f",
            concept,
            len(model.nodes),
            len(model.edges),
            model.confidence,
        )
        return model

    async def extend_model(
        self,
        model: CausalModel,
        new_evidence: str,
    ) -> CausalModel:
        """Add new evidence to an existing causal model.

        Uses LLM to check if the evidence adds new nodes/edges or
        updates confidence of existing ones.
        """
        prompt = (
            f"Given the causal model for '{model.concept}':\n"
            f"{json.dumps(model.to_dict(), indent=2)}\n\n"
            f"New evidence:\n{new_evidence}\n\n"
            f"Return a JSON object with any NEW nodes and edges to add, "
            f"and any confidence updates. Same format as before.\n"
            f"Return ONLY valid JSON, no markdown."
        )

        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_llm_json(content)

            # Add new nodes
            for node_data in parsed.get("nodes", []):
                label = node_data.get("label", "")
                if label and not any(n.label == label for n in model.nodes):
                    model.nodes.append(CausalNode.from_dict(node_data))

            # Add or update edges
            for edge_data in parsed.get("edges", []):
                mech = self._create_mechanism(edge_data.get("mechanism", {}), model.domain)
                src = edge_data.get("source_label", "")
                tgt = edge_data.get("target_label", "")
                prob = edge_data.get("probability", 0.5)

                existing_edge = None
                for e in model.edges:
                    if e.source_id.lower() == src.lower() and e.target_id.lower() == tgt.lower():
                        existing_edge = e
                        break

                if existing_edge:
                    existing_edge.probability = (existing_edge.probability + prob) / 2.0
                    evidence_str = new_evidence[:200]
                    if evidence_str not in existing_edge.evidence:
                        existing_edge.evidence.append(evidence_str)
                    if mech.confidence > existing_edge.mechanism.confidence:
                        existing_edge.mechanism = mech
                else:
                    edge = CausalEdge(
                        source_id=src,
                        target_id=tgt,
                        mechanism=mech,
                        probability=prob,
                        evidence=[new_evidence[:200]],
                    )
                    model.edges.append(edge)

            model.evidence_count += 1
            # Confidence increases with more evidence (diminishing returns)
            model.confidence = min(1.0, model.confidence + 0.05)
        except Exception as e:
            logger.warning("Failed to extend model for '%s': %s", model.concept, e)

        self._models[model.concept.lower()] = model
        self._persist_model(model)
        return model

    async def merge_models(
        self,
        a: CausalModel,
        b: CausalModel,
    ) -> CausalModel:
        """Merge overlapping causal models into a unified model."""
        merged = CausalModel(
            concept=f"{a.concept} + {b.concept}",
            domain=a.domain or b.domain,
            nodes=list(a.nodes),
            edges=list(a.edges),
            confidence=(a.confidence + b.confidence) / 2.0,
            evidence_count=a.evidence_count + b.evidence_count,
        )

        # Add non-duplicate nodes from b
        existing_labels = {n.label for n in merged.nodes}
        for node in b.nodes:
            if node.label not in existing_labels:
                merged.nodes.append(node)
                existing_labels.add(node.label)

        # Add edges from b, merging duplicates
        for edge_b in b.edges:
            existing_edge = None
            for edge_a in merged.edges:
                if edge_a.source_id.lower() == edge_b.source_id.lower() and edge_a.target_id.lower() == edge_b.target_id.lower():
                    existing_edge = edge_a
                    break

            if existing_edge:
                existing_edge.probability = (existing_edge.probability + edge_b.probability) / 2.0
                combined_evidence = set(existing_edge.evidence + edge_b.evidence)
                existing_edge.evidence = list(combined_evidence)
                if edge_b.mechanism.confidence > existing_edge.mechanism.confidence:
                    existing_edge.mechanism = edge_b.mechanism
            else:
                merged.edges.append(edge_b)

        self._models[merged.concept.lower()] = merged
        self._persist_model(merged)
        return merged

    async def find_shared_mechanisms(
        self,
        models: list[CausalModel] | None = None,
    ) -> list[Mechanism]:
        """Find mechanisms that are reused across multiple models.

        This is the foundation for concept formation — shared mechanisms
        indicate shared causal structure across different concepts.
        """
        if models is None:
            models = list(self._models.values())

        # Count mechanism descriptions across models
        desc_to_mechs: dict[str, list[Mechanism]] = {}
        for model in models:
            for edge in model.edges:
                desc = edge.mechanism.description.lower().strip()
                if desc:
                    if desc not in desc_to_mechs:
                        desc_to_mechs[desc] = []
                    desc_to_mechs[desc].append(edge.mechanism)

        # Return mechanisms that appear in multiple models
        shared = []
        for desc, mechs in desc_to_mechs.items():
            if len(mechs) > 1:
                # Use the highest-confidence version
                best = max(mechs, key=lambda m: m.confidence)
                best.reuse_count = len(mechs)
                shared.append(best)

        return shared

    def get_model(self, concept: str) -> CausalModel | None:
        """Retrieve a stored causal model by concept name."""
        return self._models.get(concept.lower())

    def get_all_models(self, domain: str | None = None) -> list[CausalModel]:
        """Get all stored causal models, optionally filtered by domain."""
        models = list(self._models.values())
        if domain:
            models = [m for m in models if m.domain == domain]
        return models

    def get_mechanism(self, mechanism_id: str) -> Mechanism | None:
        """Retrieve a mechanism by ID."""
        return self._mechanisms.get(mechanism_id)

    def get_all_mechanisms(self, domain: str | None = None) -> list[Mechanism]:
        """Get all stored mechanisms, optionally filtered by domain."""
        mechs = list(self._mechanisms.values())
        if domain:
            mechs = [m for m in mechs if m.domain == domain]
        return mechs

    def query_chain(
        self,
        cause: str,
        effect: str,
    ) -> list[list[CausalNode]]:
        """Find all causal paths from cause to effect across all models.

        Returns list of paths, where each path is a list of CausalNodes.
        """
        paths: list[list[CausalNode]] = []

        for model in self._models.values():
            # Build adjacency list for this model
            adj: dict[str, list[str]] = {}
            node_map: dict[str, CausalNode] = {}
            for node in model.nodes:
                node_map[node.label.lower()] = node
            for edge in model.edges:
                src = edge.source_id.lower()
                tgt = edge.target_id.lower()
                if src not in adj:
                    adj[src] = []
                adj[src].append(tgt)

            # BFS from cause to effect
            cause_l = cause.lower()
            effect_l = effect.lower()
            if cause_l not in node_map or effect_l not in node_map:
                continue

            queue: list[list[str]] = [[cause_l]]
            visited: set[str] = set()

            while queue:
                path = queue.pop(0)
                current = path[-1]
                if current == effect_l:
                    paths.append([node_map[label] for label in path if label in node_map])
                    continue
                if current in visited or len(path) > 10:
                    continue
                visited.add(current)
                for neighbor in adj.get(current, []):
                    queue.append(path + [neighbor])

        return paths

    def update_model_confidence(
        self,
        concept: str,
        delta: float,
        verified: bool = False,
    ) -> CausalModel | None:
        """Update confidence of a causal model based on new evidence."""
        model = self.get_model(concept)
        if model is None:
            return None
        model.confidence = max(0.0, min(1.0, model.confidence + delta))
        if verified:
            model.verified = True
            model.last_verified_at = time.time()
        self._persist_model(model)
        return model

    def stats(self) -> dict[str, Any]:
        """Return builder statistics."""
        return {
            "models_count": len(self._models),
            "mechanisms_count": len(self._mechanisms),
            "models_built": self._models_built,
            "mechanisms_created": self._mechanisms_created,
            "avg_confidence": (
                sum(m.confidence for m in self._models.values()) / len(self._models)
                if self._models
                else 0.0
            ),
            "verified_count": sum(1 for m in self._models.values() if m.verified),
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _parse_llm_json(self, content: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = content.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse LLM JSON response")
            return {"nodes": [], "edges": [], "domain": ""}

    def _build_from_parsed(
        self,
        parsed: dict[str, Any],
        concept: str,
        domain: str,
    ) -> CausalModel:
        """Build a CausalModel from parsed LLM output."""
        model_domain = domain or parsed.get("domain", "")

        nodes: list[CausalNode] = []

        for node_data in parsed.get("nodes", []):
            node = CausalNode(
                label=node_data.get("label", ""),
                node_type=node_data.get("node_type", "process"),
                confidence=node_data.get("confidence", 0.5),
            )
            nodes.append(node)

        edges: list[CausalEdge] = []
        for edge_data in parsed.get("edges", []):
            mech_data = edge_data.get("mechanism", {})
            mechanism = self._create_mechanism(mech_data, model_domain)

            src_label = edge_data.get("source_label", "")
            tgt_label = edge_data.get("target_label", "")

            edge = CausalEdge(
                source_id=src_label,
                target_id=tgt_label,
                mechanism=mechanism,
                probability=edge_data.get("probability", 0.5),
                evidence=[],
            )
            edges.append(edge)

        # Calculate overall confidence as average of edge probabilities
        avg_prob = sum(e.probability for e in edges) / len(edges) if edges else 0.3

        return CausalModel(
            concept=concept,
            domain=model_domain,
            nodes=nodes,
            edges=edges,
            confidence=avg_prob,
            evidence_count=1,
        )

    def _create_mechanism(
        self,
        mech_data: dict[str, Any],
        domain: str,
    ) -> Mechanism:
        """Create or reuse a Mechanism.

        Also registers the mechanism in the shared MechanismStore
        if available, bridging causal model mechanisms to the
        cognitive primitive store.
        """
        if isinstance(mech_data, str):
            mech_data = {"description": mech_data}

        desc = mech_data.get("description", "").lower().strip()
        # Reuse existing mechanism if descriptions match
        for existing in self._mechanisms.values():
            if existing.description.lower().strip() == desc:
                existing.reuse_count += 1
                self._persist_mechanism(existing)
                return existing

        mech = Mechanism(
            description=mech_data.get("description", ""),
            steps=mech_data.get("steps", []),
            assumptions=mech_data.get("assumptions", []),
            confidence=mech_data.get("confidence", 0.5),
            domain=domain,
        )
        self._mechanisms[mech.mechanism_id] = mech
        self._persist_mechanism(mech)
        self._mechanisms_created += 1

        # Register in shared MechanismStore (cognitive primitive store)
        if self.mechanism_store is not None:
            try:
                self.mechanism_store.create(
                    description=mech.description,
                    preconditions=mech.assumptions,
                    process_steps=mech.steps,
                    expected_outcomes=[],
                    domain=domain,
                    abstraction_level=0,
                )
            except Exception:
                logger.debug(
                    "Failed to register mechanism in MechanismStore",
                    exc_info=True,
                )

        return mech

    def _store_in_causal_graph(self, model: CausalModel) -> None:
        """Store model edges in the existing CausalGraph."""
        try:
            from hbllm.brain.causality.causal_graph import CausalLink

            for edge in model.edges:
                link = CausalLink(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    probability=edge.probability,
                    metadata={
                        "mechanism_id": edge.mechanism.mechanism_id,
                        "model_id": model.model_id,
                        "concept": model.concept,
                    },
                )
                self.causal_graph._insert(link)
        except Exception as e:
            logger.debug("Failed to store in CausalGraph: %s", e)

    def _store_in_knowledge_graph(self, model: CausalModel) -> None:
        """Cross-reference model nodes as KnowledgeGraph entities."""
        try:
            for node in model.nodes:
                self.knowledge_graph.add_entity(
                    label=node.label,
                    entity_type=f"causal_{node.node_type}",
                    attributes={
                        "causal_model_id": model.model_id,
                        "confidence": node.confidence,
                        "domain": model.domain,
                    },
                )

            # Add causal relations
            for edge in model.edges:
                self.knowledge_graph.add_relation(
                    source_label=edge.source_id,
                    target_label=edge.target_id,
                    relation_type="causes",
                    weight=edge.probability,
                    metadata={
                        "mechanism_id": edge.mechanism.mechanism_id,
                        "mechanism_desc": edge.mechanism.description,
                    },
                )
        except Exception as e:
            logger.debug("Failed to store in KnowledgeGraph: %s", e)
