"""Contradiction Detector & Belief Revision Engine.

Detects when new knowledge contradicts existing causal models and
maintains competing beliefs instead of immediately resolving to
binary truth. Humans store "probably true" — not "true/false".

Architecture:
    ContradictionDetector: Scans for logical contradictions in CausalGraph.
    BeliefRevisionEngine: Maintains competing hypotheses with evidence,
        allowing gradual convergence rather than flip-flopping.
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
class BeliefHypothesis:
    """A single hypothesis within a belief state."""

    hypothesis_id: str = field(
        default_factory=lambda: f"bh_{uuid.uuid4().hex[:10]}"
    )
    claim: str = ""
    confidence: float = 0.5
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    source: str = ""  # Where this hypothesis came from
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "confidence": self.confidence,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeliefHypothesis:
        return cls(
            hypothesis_id=d.get("hypothesis_id", f"bh_{uuid.uuid4().hex[:10]}"),
            claim=d.get("claim", ""),
            confidence=d.get("confidence", 0.5),
            evidence_for=d.get("evidence_for", []),
            evidence_against=d.get("evidence_against", []),
            source=d.get("source", ""),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class BeliefState:
    """Maintains competing hypotheses for a concept — not binary truth."""

    concept: str = ""
    hypotheses: list[BeliefHypothesis] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def dominant_belief(self) -> BeliefHypothesis | None:
        """Return the highest-confidence hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.confidence)

    @property
    def is_contested(self) -> bool:
        """True if top two hypotheses are within 0.15 confidence of each other."""
        if len(self.hypotheses) < 2:
            return False
        sorted_h = sorted(self.hypotheses, key=lambda h: h.confidence, reverse=True)
        return abs(sorted_h[0].confidence - sorted_h[1].confidence) < 0.15

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeliefState:
        return cls(
            concept=d.get("concept", ""),
            hypotheses=[BeliefHypothesis.from_dict(h) for h in d.get("hypotheses", [])],
            created_at=d.get("created_at", time.time()),
            last_updated=d.get("last_updated", time.time()),
        )


@dataclass
class Contradiction:
    """A detected contradiction between existing and new knowledge."""

    contradiction_id: str = field(
        default_factory=lambda: f"ctr_{uuid.uuid4().hex[:10]}"
    )
    existing_claim: str = ""
    new_claim: str = ""
    concept: str = ""
    severity: float = 0.5  # 0.0 (minor) to 1.0 (fundamental)
    existing_confidence: float = 0.5
    new_confidence: float = 0.5
    resolution: str | None = None
    resolved: bool = False
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "existing_claim": self.existing_claim,
            "new_claim": self.new_claim,
            "concept": self.concept,
            "severity": self.severity,
            "existing_confidence": self.existing_confidence,
            "new_confidence": self.new_confidence,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "detected_at": self.detected_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Contradiction:
        return cls(
            contradiction_id=d.get("contradiction_id", f"ctr_{uuid.uuid4().hex[:10]}"),
            existing_claim=d.get("existing_claim", ""),
            new_claim=d.get("new_claim", ""),
            concept=d.get("concept", ""),
            severity=d.get("severity", 0.5),
            existing_confidence=d.get("existing_confidence", 0.5),
            new_confidence=d.get("new_confidence", 0.5),
            resolution=d.get("resolution"),
            resolved=d.get("resolved", False),
            detected_at=d.get("detected_at", time.time()),
        )


# ── Contradiction Detector ───────────────────────────────────────────────────

_CONTRADICTION_CHECK_PROMPT = """\
Given existing knowledge about "{concept}":
{existing_claims}

New claim:
"{new_claim}"

Questions:
1. Does the new claim contradict any existing knowledge?
2. If yes, what is the severity? (0.0 = minor nuance, 1.0 = fundamental contradiction)
3. Which specific existing claim does it contradict?

Return a JSON object:
{{
  "is_contradiction": true/false,
  "severity": 0.0-1.0,
  "contradicted_claim": "the specific existing claim that conflicts",
  "reasoning": "why this is or isn't a contradiction"
}}

Return ONLY valid JSON, no markdown."""


class ContradictionDetector:
    """Detects contradictions in learned knowledge.

    Scans CausalGraph and KnowledgeGraph for logical contradictions
    and routes them to the BeliefRevisionEngine for resolution.
    """

    def __init__(
        self,
        llm: Any,
        causal_graph: Any | None = None,
        knowledge_graph: Any | None = None,
    ) -> None:
        self.llm = llm
        self.causal_graph = causal_graph
        self.knowledge_graph = knowledge_graph

        # Telemetry
        self._scans_run = 0
        self._contradictions_found = 0

    async def check_contradiction(
        self,
        new_claim: str,
        concept: str,
    ) -> Contradiction | None:
        """Check if new knowledge contradicts existing models.

        Returns a Contradiction if found, None otherwise.
        """
        # Gather existing claims about this concept
        existing_claims = self._gather_existing_claims(concept)
        if not existing_claims:
            return None

        prompt = _CONTRADICTION_CHECK_PROMPT.format(
            concept=concept,
            existing_claims="\n".join(f"- {c}" for c in existing_claims[:10]),
            new_claim=new_claim,
        )

        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_json(content)

            if not parsed.get("is_contradiction", False):
                return None

            contradiction = Contradiction(
                existing_claim=parsed.get("contradicted_claim", existing_claims[0]),
                new_claim=new_claim,
                concept=concept,
                severity=parsed.get("severity", 0.5),
                existing_confidence=0.5,  # Will be updated from CausalGraph
                new_confidence=0.5,
            )

            self._contradictions_found += 1
            logger.info(
                "Contradiction detected for '%s': severity=%.2f — '%s' vs '%s'",
                concept,
                contradiction.severity,
                contradiction.existing_claim[:50],
                contradiction.new_claim[:50],
            )
            return contradiction
        except Exception as e:
            logger.warning("Contradiction check failed for '%s': %s", concept, e)
            return None

    async def scan_graph(self) -> list[Contradiction]:
        """Scan the CausalGraph for internal contradictions.

        Looks for opposing causal links:
            A causes B (prob=0.7)
            A prevents B (prob=0.6)
        """
        contradictions: list[Contradiction] = []

        if self.causal_graph is None:
            return contradictions

        self._scans_run += 1

        try:
            with sqlite3.connect(self.causal_graph.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Find all unique source_ids
                sources = conn.execute(
                    "SELECT DISTINCT source_id FROM causal_links"
                ).fetchall()

                for row in sources:
                    source_id = row["source_id"]
                    effects = self.causal_graph.get_effects(source_id)

                    # Group by target_id to find conflicting effects
                    target_groups: dict[str, list[Any]] = {}
                    for effect in effects:
                        tid = effect.target_id
                        if tid not in target_groups:
                            target_groups[tid] = []
                        target_groups[tid].append(effect)

                    # Check for conflicting metadata (e.g. opposing mechanisms)
                    for target_id, links in target_groups.items():
                        if len(links) < 2:
                            continue
                        # If same source→target has very different probabilities,
                        # that might indicate a contradiction
                        probs = [l.probability for l in links]
                        if max(probs) - min(probs) > 0.4:
                            c = Contradiction(
                                existing_claim=f"{source_id} → {target_id} (p={max(probs):.2f})",
                                new_claim=f"{source_id} → {target_id} (p={min(probs):.2f})",
                                concept=source_id,
                                severity=max(probs) - min(probs),
                                existing_confidence=max(probs),
                                new_confidence=min(probs),
                            )
                            contradictions.append(c)
        except Exception as e:
            logger.warning("Graph scan failed: %s", e)

        if contradictions:
            self._contradictions_found += len(contradictions)
            logger.info(
                "Graph scan found %d contradictions",
                len(contradictions),
            )

        return contradictions

    def _gather_existing_claims(self, concept: str) -> list[str]:
        """Gather existing knowledge claims about a concept."""
        claims: list[str] = []

        # From CausalGraph
        if self.causal_graph is not None:
            try:
                effects = self.causal_graph.get_effects(concept)
                for e in effects[:5]:
                    meta = e.metadata or {}
                    desc = meta.get("mechanism_desc", "")
                    claims.append(
                        f"{concept} causes {e.target_id}"
                        f" (p={e.probability:.2f})"
                        f"{f' via {desc}' if desc else ''}"
                    )
                causes = self.causal_graph.get_causes(concept)
                for c in causes[:5]:
                    claims.append(
                        f"{c.source_id} causes {concept}"
                        f" (p={c.probability:.2f})"
                    )
            except Exception:
                pass

        # From KnowledgeGraph
        if self.knowledge_graph is not None:
            try:
                neighbors = self.knowledge_graph.neighbors(concept, direction="both")
                for n in neighbors[:5]:
                    claims.append(
                        f"{concept} {n.get('relation', 'relates_to')} "
                        f"{n.get('entity', 'unknown')}"
                    )
            except Exception:
                pass

        return claims

    def stats(self) -> dict[str, Any]:
        return {
            "scans_run": self._scans_run,
            "contradictions_found": self._contradictions_found,
        }

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


# ── Belief Revision Engine ───────────────────────────────────────────────────


class BeliefRevisionEngine:
    """Maintains competing beliefs instead of binary truth.

    When a contradiction is detected, instead of immediately discarding
    one claim, both are maintained as competing hypotheses with their
    own evidence and confidence scores. Over time, evidence accumulates
    and the weaker belief's confidence decays naturally.

    This mirrors human cognition: we hold "probably true" beliefs,
    not certain facts.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.data_dir / "belief_states.db"
        self._init_db()

        # In-memory cache
        self._beliefs: dict[str, BeliefState] = {}
        self._load_from_db()

        # Telemetry
        self._revisions = 0
        self._pruned = 0

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS belief_states (
                    concept TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contradictions (
                    contradiction_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    data TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    detected_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_belief_concept "
                "ON contradictions(concept)"
            )

    def _load_from_db(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                for row in conn.execute("SELECT * FROM belief_states"):
                    state = BeliefState.from_dict(json.loads(row["data"]))
                    self._beliefs[state.concept.lower()] = state
        except Exception as e:
            logger.debug("Failed to load belief states: %s", e)

    def _persist(self, state: BeliefState) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO belief_states
                       (concept, data, created_at, last_updated)
                       VALUES (?, ?, ?, ?)""",
                    (
                        state.concept,
                        json.dumps(state.to_dict()),
                        state.created_at,
                        state.last_updated,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist belief state: %s", e)

    def _persist_contradiction(self, contradiction: Contradiction) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO contradictions
                       (contradiction_id, concept, data, resolved, detected_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        contradiction.contradiction_id,
                        contradiction.concept,
                        json.dumps(contradiction.to_dict()),
                        1 if contradiction.resolved else 0,
                        contradiction.detected_at,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist contradiction: %s", e)

    # ── Core API ─────────────────────────────────────────────────────────

    def get_belief_state(self, concept: str) -> BeliefState:
        """Get or create belief state for a concept."""
        key = concept.lower()
        if key not in self._beliefs:
            self._beliefs[key] = BeliefState(concept=concept)
        return self._beliefs[key]

    async def integrate_evidence(
        self,
        concept: str,
        claim: str,
        confidence: float,
        evidence: str,
        source: str = "",
    ) -> BeliefState:
        """Integrate new evidence into a belief state.

        If the claim matches an existing hypothesis, reinforce it.
        If it's new, add it as a competing hypothesis.
        """
        state = self.get_belief_state(concept)

        # Check if this matches an existing hypothesis
        matched = False
        for hyp in state.hypotheses:
            if self._claims_similar(hyp.claim, claim):
                hyp.evidence_for.append(evidence[:200])
                # Bayesian-like update: move toward the new confidence
                hyp.confidence = hyp.confidence * 0.7 + confidence * 0.3
                matched = True
                break

        if not matched:
            # Add as new competing hypothesis
            state.hypotheses.append(
                BeliefHypothesis(
                    claim=claim,
                    confidence=confidence,
                    evidence_for=[evidence[:200]],
                    source=source,
                )
            )

        state.last_updated = time.time()
        self._revisions += 1
        self._persist(state)
        return state

    async def handle_contradiction(
        self,
        contradiction: Contradiction,
    ) -> BeliefState:
        """Handle a detected contradiction by maintaining competing beliefs."""
        state = self.get_belief_state(contradiction.concept)

        # Ensure both claims exist as hypotheses
        has_existing = any(
            self._claims_similar(h.claim, contradiction.existing_claim)
            for h in state.hypotheses
        )
        has_new = any(
            self._claims_similar(h.claim, contradiction.new_claim)
            for h in state.hypotheses
        )

        if not has_existing:
            state.hypotheses.append(
                BeliefHypothesis(
                    claim=contradiction.existing_claim,
                    confidence=contradiction.existing_confidence,
                    source="existing_knowledge",
                )
            )

        if not has_new:
            state.hypotheses.append(
                BeliefHypothesis(
                    claim=contradiction.new_claim,
                    confidence=contradiction.new_confidence,
                    source="new_evidence",
                )
            )

        # Add cross-evidence
        for hyp in state.hypotheses:
            if self._claims_similar(hyp.claim, contradiction.existing_claim):
                hyp.evidence_against.append(
                    f"Contradicted by: {contradiction.new_claim[:100]}"
                )
            elif self._claims_similar(hyp.claim, contradiction.new_claim):
                hyp.evidence_against.append(
                    f"Contradicts: {contradiction.existing_claim[:100]}"
                )

        state.last_updated = time.time()
        self._persist(state)
        self._persist_contradiction(contradiction)

        logger.info(
            "Belief state for '%s': %d competing hypotheses, contested=%s",
            contradiction.concept,
            len(state.hypotheses),
            state.is_contested,
        )
        return state

    async def prune_weak_beliefs(self, threshold: float = 0.1) -> int:
        """Remove beliefs with confidence below threshold.

        Called during sleep to clean up noise.
        """
        pruned = 0
        for concept, state in self._beliefs.items():
            before = len(state.hypotheses)
            state.hypotheses = [
                h for h in state.hypotheses if h.confidence >= threshold
            ]
            removed = before - len(state.hypotheses)
            if removed > 0:
                pruned += removed
                state.last_updated = time.time()
                self._persist(state)

        self._pruned += pruned
        if pruned > 0:
            logger.info("Pruned %d weak beliefs (threshold=%.2f)", pruned, threshold)
        return pruned

    def decay_all_beliefs(self, decay_rate: float = 0.01) -> int:
        """Apply confidence decay to all beliefs.

        Knowledge confidence gradually drops unless reinforced.
        Called during sleep to simulate biological memory decay.
        """
        decayed = 0
        for state in self._beliefs.values():
            for hyp in state.hypotheses:
                old = hyp.confidence
                hyp.confidence = max(0.0, hyp.confidence - decay_rate)
                if old != hyp.confidence:
                    decayed += 1
            self._persist(state)
        return decayed

    def get_contested_beliefs(self) -> list[BeliefState]:
        """Get all belief states where hypotheses are closely contested."""
        return [s for s in self._beliefs.values() if s.is_contested]

    def get_all_contradictions(self, resolved: bool | None = None) -> list[Contradiction]:
        """Get stored contradictions, optionally filtered by resolution status."""
        contradictions: list[Contradiction] = []
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                if resolved is not None:
                    rows = conn.execute(
                        "SELECT * FROM contradictions WHERE resolved = ?",
                        (1 if resolved else 0,),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM contradictions").fetchall()
                for row in rows:
                    contradictions.append(
                        Contradiction.from_dict(json.loads(row["data"]))
                    )
        except Exception as e:
            logger.debug("Failed to load contradictions: %s", e)
        return contradictions

    def stats(self) -> dict[str, Any]:
        total_hypotheses = sum(len(s.hypotheses) for s in self._beliefs.values())
        return {
            "belief_states": len(self._beliefs),
            "total_hypotheses": total_hypotheses,
            "contested_beliefs": len(self.get_contested_beliefs()),
            "revisions": self._revisions,
            "pruned": self._pruned,
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _claims_similar(self, a: str, b: str) -> bool:
        """Simple similarity check between two claims."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return False
        intersection = a_words & b_words
        union = a_words | b_words
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard > 0.5
