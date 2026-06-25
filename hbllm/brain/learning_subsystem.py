"""CognitiveGraph — 3-layer cognitive query infrastructure.

Evolves LearningSubsystem into a queryable cognition substrate.

Architecture:
    Layer 3: Synthesis  — knowledge_about(), detect_conflicts()
    Layer 2: Integrator — CognitiveIntegrator (dispatch + normalize + rank)
    Layer 1: Adapters   — CognitiveBeliefs, CognitiveMechanisms, CognitiveConcepts

Design rules:
    1. Views are pure adapters — delegate, reshape, unify naming
    2. Integrator does dispatch + normalization + ranking, NOT synthesis
    3. Synthesis is explicitly expensive (mode="fast|deep")
    4. GoalManager is explicit — has_goals + goals property
    5. Conflict detection is first-class

This is the seed of what eventually becomes attention-weighted
cognitive routing when integrated with SNN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from hbllm.brain.belief_store import BeliefStore
    from hbllm.brain.causality.causal_model_builder import CausalModelBuilder
    from hbllm.brain.concept_formation import ConceptFormationEngine
    from hbllm.brain.contradiction_detector import (
        BeliefRevisionEngine,
        ContradictionDetector,
    )
    from hbllm.brain.failure_analyzer import FailureAnalyzer
    from hbllm.brain.goal_manager import GoalManager
    from hbllm.brain.mechanism_store import MechanismStore
    from hbllm.brain.meta_learner import MetaLearner

logger = logging.getLogger(__name__)


# ── Cognitive IR (Intermediate Representation) ──────────────────────────────

@dataclass
class CognitiveQueryResult:
    """Cross-store contract for unified query results.

    This is the cognitive IR — all stores return compatible objects
    through this schema, enabling consistent ranking, deduplication,
    and synthesis across heterogeneous memory systems.
    """

    source: Literal["beliefs", "mechanisms", "concepts", "goals"]
    entity_id: str
    content: str  # Human-readable summary
    relevance: float = 0.0  # 0.0–1.0, how relevant to the query
    confidence: float = 0.0  # 0.0–1.0, system confidence in this entity
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Layer 1: Typed View Adapters ────────────────────────────────────────────


class CognitiveBeliefs:
    """Read-only lens over BeliefStore.

    Pure adapter — delegates to BeliefStore, never duplicates logic.
    """

    def __init__(self, belief_store: BeliefStore | None) -> None:
        self._store = belief_store

    @property
    def available(self) -> bool:
        return self._store is not None

    def active(self, domain: str | None = None) -> list[Any]:
        """Get active beliefs, optionally filtered by domain."""
        if self._store is None:
            return []
        if domain:
            return self._store.get_beliefs_by_domain(domain)
        from hbllm.brain.belief_store import BeliefStatus
        return self._store.get_beliefs_by_status(BeliefStatus.ACTIVE)

    def contested(self) -> list[Any]:
        """Get beliefs with unresolved contradictions."""
        if self._store is None:
            return []
        return self._store.get_contested_beliefs()

    def for_concept(self, concept: str) -> list[Any]:
        """Get beliefs about a specific concept."""
        if self._store is None:
            return []
        return self._store.get_beliefs_for_concept(concept)

    def strongest(self, n: int = 10) -> list[Any]:
        """Get the highest-confidence beliefs."""
        if self._store is None:
            return []
        return self._store.get_strongest(n)

    def search(self, topic: str) -> list[CognitiveQueryResult]:
        """Search beliefs for a topic. Returns cognitive IR."""
        if self._store is None:
            return []
        results = []
        # Search by concept match
        beliefs = self._store.get_beliefs_for_concept(topic)
        # Also search by domain
        beliefs.extend(self._store.get_beliefs_by_domain(topic))
        seen: set[str] = set()
        for b in beliefs:
            if b.belief_id in seen:
                continue
            seen.add(b.belief_id)
            results.append(CognitiveQueryResult(
                source="beliefs",
                entity_id=b.belief_id,
                content=b.claim,
                relevance=_text_relevance(topic, b.claim),
                confidence=b.confidence,
                metadata={
                    "belief_type": b.belief_type.value if hasattr(b.belief_type, "value") else str(b.belief_type),
                    "status": b.status.value if hasattr(b.status, "value") else str(b.status),
                    "concept": b.concept,
                    "domain": b.domain,
                },
            ))
        return results

    def stats(self) -> dict[str, Any]:
        if self._store is None:
            return {}
        return self._store.stats()


class CognitiveMechanisms:
    """Read-only lens over MechanismStore.

    Pure adapter — delegates to MechanismStore, never duplicates logic.
    """

    def __init__(self, mechanism_store: MechanismStore | None) -> None:
        self._store = mechanism_store

    @property
    def available(self) -> bool:
        return self._store is not None

    def core(self) -> list[Any]:
        """Get promoted core mechanisms (cognitive primitives)."""
        if self._store is None:
            return []
        return self._store.get_core_mechanisms()

    def for_domain(self, domain: str) -> list[Any]:
        """Get mechanisms in a specific domain."""
        if self._store is None:
            return []
        return self._store.find_by_domain(domain)

    def weak(self, threshold: float = 0.3) -> list[Any]:
        """Get low-confidence mechanisms needing review."""
        if self._store is None:
            return []
        return self._store.get_weak(threshold)

    def promotable(self) -> list[Any]:
        """Get mechanisms ready for promotion to core."""
        if self._store is None:
            return []
        return self._store.find_promotable()

    def search(self, topic: str) -> list[CognitiveQueryResult]:
        """Search mechanisms for a topic. Returns cognitive IR."""
        if self._store is None:
            return []
        results = []
        mechanisms = self._store.find_by_domain(topic)
        # Also search by precondition keywords
        mechanisms.extend(self._store.find_by_preconditions([topic]))
        seen: set[str] = set()
        for m in mechanisms:
            if m.id in seen:
                continue
            seen.add(m.id)
            results.append(CognitiveQueryResult(
                source="mechanisms",
                entity_id=m.id,
                content=m.description,
                relevance=_text_relevance(topic, m.description),
                confidence=m.confidence,
                metadata={
                    "domain": m.domain,
                    "abstraction_level": m.abstraction_level,
                    "is_core": m.is_core,
                    "usage_count": m.usage_count,
                    "success_rate": m.success_rate,
                },
            ))
        return results

    def stats(self) -> dict[str, Any]:
        if self._store is None:
            return {}
        return self._store.stats()


class CognitiveConcepts:
    """Read-only lens over ConceptFormationEngine.

    Pure adapter — delegates to ConceptFormationEngine, never duplicates logic.
    """

    def __init__(self, concept_engine: ConceptFormationEngine | None) -> None:
        self._engine = concept_engine

    @property
    def available(self) -> bool:
        return self._engine is not None

    def abstractions(self, domain: str | None = None) -> list[Any]:
        """Get abstract concepts, optionally filtered by domain."""
        if self._engine is None:
            return []
        return self._engine.get_abstract_concepts(domain)

    def analogies(self) -> list[Any]:
        """Get discovered cross-domain analogies."""
        if self._engine is None:
            return []
        return self._engine.get_analogies()

    def search(self, topic: str) -> list[CognitiveQueryResult]:
        """Search concepts for a topic. Returns cognitive IR."""
        if self._engine is None:
            return []
        results = []
        concepts = self._engine.get_abstract_concepts(domain=topic)
        for c in concepts:
            results.append(CognitiveQueryResult(
                source="concepts",
                entity_id=c.concept_id,
                content=f"{c.label}: {c.description}",
                relevance=_text_relevance(topic, f"{c.label} {c.description}"),
                confidence=c.confidence,
                metadata={
                    "domain": c.domain,
                    "instances": c.instances,
                    "generalized_steps": c.generalized_steps,
                },
            ))
        return results

    def stats(self) -> dict[str, Any]:
        if self._engine is None:
            return {}
        return self._engine.stats()


class CognitiveGoals:
    """Read-only lens over GoalManager.

    Pure adapter — delegates to GoalManager, never duplicates logic.
    """

    def __init__(self, goal_manager: GoalManager | None) -> None:
        self._manager = goal_manager

    @property
    def available(self) -> bool:
        return self._manager is not None

    def active(self) -> list[Any]:
        """Get active goals."""
        if self._manager is None:
            return []
        return self._manager.get_active_goals()

    def learning(self) -> list[Any]:
        """Get active learning-type goals."""
        if self._manager is None:
            return []
        return self._manager.get_learning_goals()

    def next(self) -> Any | None:
        """Get next goal to work on."""
        if self._manager is None:
            return None
        return self._manager.next_goal()

    def search(self, topic: str) -> list[CognitiveQueryResult]:
        """Search goals for a topic. Returns cognitive IR."""
        if self._manager is None:
            return []
        results = []
        for g in self._manager.get_active_goals():
            relevance = _text_relevance(topic, f"{g.name} {g.description}")
            if relevance > 0.0:
                results.append(CognitiveQueryResult(
                    source="goals",
                    entity_id=g.goal_id,
                    content=f"{g.name}: {g.description}",
                    relevance=relevance,
                    confidence=g.progress,
                    metadata={
                        "goal_type": g.goal_type,
                        "priority": g.priority.value if hasattr(g.priority, "value") else str(g.priority),
                        "status": g.status.value if hasattr(g.status, "value") else str(g.status),
                        "progress": g.progress,
                    },
                ))
        return results

    def stats(self) -> dict[str, Any]:
        if self._manager is None:
            return {}
        return self._manager.stats()


# ── Layer 2: CognitiveIntegrator ────────────────────────────────────────────


class CognitiveIntegrator:
    """Dispatch + normalize + deduplicate + rank.

    This is NOT synthesis. It does not resolve conflicts,
    merge beliefs, or produce new knowledge. It only:
    1. Dispatches queries to scoped views
    2. Normalizes results into CognitiveQueryResult
    3. Deduplicates by entity_id
    4. Ranks by relevance × confidence

    Synthesis belongs in Layer 3 (CognitiveGraph methods).
    """

    def __init__(
        self,
        beliefs: CognitiveBeliefs,
        mechanisms: CognitiveMechanisms,
        concepts: CognitiveConcepts,
        goals: CognitiveGoals,
    ) -> None:
        self._views = {
            "beliefs": beliefs,
            "mechanisms": mechanisms,
            "concepts": concepts,
            "goals": goals,
        }

    def query(
        self,
        topic: str,
        scope: list[str] | None = None,
    ) -> list[CognitiveQueryResult]:
        """Query cognitive stores with optional scoping.

        Args:
            topic: What to search for
            scope: Which stores to search (default: all available)

        Returns:
            Deduplicated, ranked list of CognitiveQueryResult
        """
        targets = scope or ["beliefs", "mechanisms", "concepts", "goals"]
        all_results: list[CognitiveQueryResult] = []

        for target in targets:
            view = self._views.get(target)
            if view is None or not view.available:
                continue
            try:
                results = view.search(topic)
                all_results.extend(results)
            except Exception as e:
                logger.warning("Query to %s failed: %s", target, e)

        # Deduplicate by (source, entity_id)
        seen: set[tuple[str, str]] = set()
        deduped: list[CognitiveQueryResult] = []
        for r in all_results:
            key = (r.source, r.entity_id)
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        # Rank: relevance × confidence (both weighted equally)
        deduped.sort(
            key=lambda r: r.relevance * 0.6 + r.confidence * 0.4,
            reverse=True,
        )

        return deduped


# ── Layer 3 + Container: CognitiveGraph ─────────────────────────────────────


@dataclass
class CognitiveGraph:
    """3-layer cognitive query infrastructure.

    Evolves from LearningSubsystem. Backward compatible via alias.

    Layer 1: Typed views (pure adapters over stores)
    Layer 2: CognitiveIntegrator (dispatch + rank)
    Layer 3: Synthesis methods on this class (knowledge_about, detect_conflicts)

    Both AutonomousLearner (research-time) and LearningEventHandler
    (experience-time) receive the same instance.  This ensures:
    - No late binding (everything injected at construction)
    - No dependency drift (single source of truth)
    - Each evidence source updates beliefs exactly once

    The raw store fields remain for direct access by consumers
    that need write access (e.g., BeliefRevisionEngine).
    """

    # ── Raw stores (write access for consumers) ─────────────────────

    # Always available (created with SkillEngine)
    mechanism_store: MechanismStore | None = None
    failure_analyzer: FailureAnalyzer | None = None

    # Available when autonomous learning is injected
    belief_engine: BeliefRevisionEngine | None = None
    contradiction_detector: ContradictionDetector | None = None
    meta_learner: MetaLearner | None = None
    causal_model_builder: CausalModelBuilder | None = None

    # Persistent stores
    belief_store: BeliefStore | None = None
    concept_engine: ConceptFormationEngine | None = None

    # Goal system (optional, explicit)
    goal_manager: GoalManager | None = None

    # ── Cached views (created lazily) ───────────────────────────────

    _beliefs_view: CognitiveBeliefs | None = field(default=None, repr=False)
    _mechanisms_view: CognitiveMechanisms | None = field(default=None, repr=False)
    _concepts_view: CognitiveConcepts | None = field(default=None, repr=False)
    _goals_view: CognitiveGoals | None = field(default=None, repr=False)
    _integrator: CognitiveIntegrator | None = field(default=None, repr=False)

    # ── Capability surface ──────────────────────────────────────────

    @property
    def capabilities(self) -> set[str]:
        """Explicit capability signaling.

        Consumers check capabilities instead of repetitive None checks.
        """
        caps: set[str] = set()
        if self.belief_store is not None:
            caps.add("beliefs")
        if self.mechanism_store is not None:
            caps.add("mechanisms")
        if self.concept_engine is not None:
            caps.add("concepts")
        if self.goal_manager is not None:
            caps.add("goals")
        if self.has_belief_infrastructure:
            caps.add("belief_revision")
            caps.add("conflict_detection")
        if self.has_research_infrastructure:
            caps.add("research")
            caps.add("synthesis")
        return caps

    @property
    def has_goals(self) -> bool:
        """Explicit goal availability check."""
        return self.goal_manager is not None

    @property
    def has_belief_infrastructure(self) -> bool:
        """True if belief revision pipeline is available."""
        return self.belief_engine is not None and self.contradiction_detector is not None

    @property
    def has_research_infrastructure(self) -> bool:
        """True if full research pipeline is available."""
        return (
            self.causal_model_builder is not None
            and self.meta_learner is not None
            and self.has_belief_infrastructure
        )

    # ── Layer 1: Typed Views (pure adapters) ────────────────────────

    @property
    def beliefs(self) -> CognitiveBeliefs:
        """Typed view over BeliefStore."""
        if self._beliefs_view is None:
            self._beliefs_view = CognitiveBeliefs(self.belief_store)
        return self._beliefs_view

    @property
    def mechanisms(self) -> CognitiveMechanisms:
        """Typed view over MechanismStore."""
        if self._mechanisms_view is None:
            self._mechanisms_view = CognitiveMechanisms(self.mechanism_store)
        return self._mechanisms_view

    @property
    def concepts(self) -> CognitiveConcepts:
        """Typed view over ConceptFormationEngine."""
        if self._concepts_view is None:
            self._concepts_view = CognitiveConcepts(self.concept_engine)
        return self._concepts_view

    @property
    def goals(self) -> CognitiveGoals:
        """Typed view over GoalManager.

        Always returns a CognitiveGoals instance (check .available).
        """
        if self._goals_view is None:
            self._goals_view = CognitiveGoals(self.goal_manager)
        return self._goals_view

    # ── Layer 2: CognitiveIntegrator ────────────────────────────────

    @property
    def integrator(self) -> CognitiveIntegrator:
        """The query integrator (dispatch + normalize + rank)."""
        if self._integrator is None:
            self._integrator = CognitiveIntegrator(
                beliefs=self.beliefs,
                mechanisms=self.mechanisms,
                concepts=self.concepts,
                goals=self.goals,
            )
        return self._integrator

    def query(
        self,
        topic: str,
        scope: list[str] | None = None,
    ) -> list[CognitiveQueryResult]:
        """Query cognitive stores. Delegates to integrator.

        This is the main entry point for "what do I know about X?"
        """
        return self.integrator.query(topic, scope=scope)

    # ── Layer 3: Synthesis (explicitly expensive) ───────────────────

    def knowledge_about(
        self,
        domain: str,
        mode: str = "fast",
    ) -> dict[str, Any]:
        """Structured merge of everything known about a domain.

        Args:
            domain: The domain to query
            mode: "fast" (parallel search only) or "deep" (+ conflict detection)

        Returns structured dict, NOT a summary score.
        """
        result: dict[str, Any] = {
            "domain": domain,
            "beliefs": [],
            "mechanisms": [],
            "concepts": [],
            "goals": [],
            "conflicts": [],
        }

        # Layer 1: Collect from each view
        if self.beliefs.available:
            result["beliefs"] = [
                {"claim": b.claim, "confidence": b.confidence, "status": str(b.status)}
                for b in self.beliefs.active(domain=domain)
            ]

        if self.mechanisms.available:
            result["mechanisms"] = [
                {"description": m.description, "confidence": m.confidence, "is_core": m.is_core}
                for m in self.mechanisms.for_domain(domain)
            ]

        if self.concepts.available:
            result["concepts"] = [
                {"label": c.label, "confidence": c.confidence, "instances": c.instances}
                for c in self.concepts.abstractions(domain=domain)
            ]

        if self.goals.available:
            for g in self.goals.active():
                if domain.lower() in g.name.lower() or domain.lower() in g.description.lower():
                    result["goals"].append({
                        "name": g.name,
                        "progress": g.progress,
                        "priority": str(g.priority),
                    })

        # Layer 3: Synthesis (deep mode only)
        if mode == "deep":
            result["conflicts"] = self.detect_conflicts(domain)

        return result

    def detect_conflicts(self, topic: str) -> list[dict[str, Any]]:
        """Cross-store conflict detection.

        This is where CognitiveGraph becomes "cognition infrastructure"
        instead of just "search layer."

        Detects:
        1. Belief contradictions (from BeliefStore)
        2. Mechanism-belief conflicts (mechanism says X, belief says ~X)
        3. Weak mechanisms contradicting strong beliefs

        Returns structured conflict descriptors.
        """
        conflicts: list[dict[str, Any]] = []

        # 1. Belief-level contradictions (from persistent store)
        if self.belief_store is not None:
            unresolved = self.belief_store.get_unresolved_contradictions()
            for ctr in unresolved:
                concept = ctr.get("concept", "")
                if topic.lower() in concept.lower() or not topic:
                    conflicts.append({
                        "type": "belief_contradiction",
                        "source_a": "beliefs",
                        "source_b": "beliefs",
                        "claim_a": ctr.get("claim_a", ""),
                        "claim_b": ctr.get("claim_b", ""),
                        "severity": ctr.get("severity", 0.5),
                        "concept": concept,
                    })

        # 2. Mechanism vs belief conflicts
        if self.mechanism_store is not None and self.belief_store is not None:
            mechanisms = self.mechanism_store.find_by_domain(topic)
            beliefs = self.belief_store.get_beliefs_by_domain(topic)

            for mech in mechanisms:
                for belief in beliefs:
                    # Weak mechanism contradicting strong belief = conflict
                    if (
                        mech.confidence < 0.3
                        and belief.confidence > 0.7
                        and _text_overlap(mech.description, belief.claim) > 0.2
                    ):
                        conflicts.append({
                            "type": "mechanism_belief_tension",
                            "source_a": "mechanisms",
                            "source_b": "beliefs",
                            "claim_a": mech.description,
                            "claim_b": belief.claim,
                            "severity": belief.confidence - mech.confidence,
                            "concept": topic,
                        })

        return conflicts

    # ── Backward-compatible API ─────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a summary of available components.

        Backward compatible with LearningSubsystem.summary().
        """
        return {
            "mechanism_store": self.mechanism_store is not None,
            "failure_analyzer": self.failure_analyzer is not None,
            "belief_engine": self.belief_engine is not None,
            "belief_store": self.belief_store is not None,
            "contradiction_detector": self.contradiction_detector is not None,
            "meta_learner": self.meta_learner is not None,
            "causal_model_builder": self.causal_model_builder is not None,
            "concept_engine": self.concept_engine is not None,
            "goal_manager": self.goal_manager is not None,
            "has_belief_infrastructure": self.has_belief_infrastructure,
            "has_research_infrastructure": self.has_research_infrastructure,
            "capabilities": sorted(self.capabilities),
        }

    def cognitive_state(self) -> dict[str, Any]:
        """Structured cognitive state — NOT a summary score.

        Returns per-store stats without artificial aggregation.
        Heterogeneous systems should not be normalized into
        a single scalar.
        """
        state: dict[str, Any] = {
            "capabilities": sorted(self.capabilities),
        }
        if self.beliefs.available:
            state["beliefs"] = self.beliefs.stats()
        if self.mechanisms.available:
            state["mechanisms"] = self.mechanisms.stats()
        if self.concepts.available:
            state["concepts"] = self.concepts.stats()
        if self.goals.available:
            state["goals"] = self.goals.stats()
        return state


# ── Backward compatibility ──────────────────────────────────────────────────

LearningSubsystem = CognitiveGraph


# ── Helpers (module-private) ────────────────────────────────────────────────

def _text_relevance(query: str, text: str) -> float:
    """Simple word-overlap relevance score."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words or not t_words:
        return 0.0
    overlap = len(q_words & t_words)
    return min(1.0, overlap / len(q_words)) if q_words else 0.0


def _text_overlap(a: str, b: str) -> float:
    """Jaccard similarity between two texts."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)
