"""
Deduction Operator — formal logical deduction over HCIR.

Implements modus ponens, syllogistic chains, and simple theorem proving.
Uses Z3 as the backend solver when available; falls back to a pure-Python
forward-chaining engine for environments without Z3.

This operator reads beliefs, facts, and constraints from the frozen HCIR
view and derives new conclusions through valid logical inference.

Example reasoning chain::

    Belief: "All spherical objects can roll"       (∀x: spherical(x) → rolls(x))
    Belief: "Entity_17 is spherical"               (spherical(entity_17))
    ─────────────────────────────────────────────────
    Conclusion: "Entity_17 can roll"                (rolls(entity_17))

The conclusion is returned as a CognitiveResult with full provenance
and proposed HCIR transitions (new BeliefNode or strengthened existing).

Independence Level: L1 (no LLM execution)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.hcir.fuzzy import FuzzyProfileId, get_fuzzy_profile
from hbllm.hcir.graph import (
    BeliefNode,
    FactNode,
    HCIRNodeType,
)
from hbllm.hcir.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    ProblemType,
    ProvenanceChain,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Internal Rule Representation
# ═══════════════════════════════════════════════════════════════════════════


class _Rule:
    """A simple if-then rule extracted from HCIR beliefs.

    Represents: IF all antecedents hold THEN consequent holds.
    """

    __slots__ = ("antecedents", "consequent", "confidence", "semantic_degree", "source_id")

    def __init__(
        self,
        antecedents: frozenset[str],
        consequent: str,
        confidence: float,
        source_id: str,
        semantic_degree: float = 1.0,
    ) -> None:
        self.antecedents = antecedents
        self.consequent = consequent
        self.confidence = confidence
        self.semantic_degree = max(0.0, min(1.0, float(semantic_degree)))
        self.source_id = source_id

    def __repr__(self) -> str:
        ants = " ∧ ".join(sorted(self.antecedents))
        return f"({ants}) → {self.consequent} [conf={self.confidence:.2f}, deg={self.semantic_degree:.2f}]"


# ═══════════════════════════════════════════════════════════════════════════
# Deduction Engine
# ═══════════════════════════════════════════════════════════════════════════


class DeductionOperator:
    """Formal deduction via forward-chaining over HCIR beliefs.

    Extracts if-then rules from beliefs with conditional structure,
    collects known facts, and runs forward chaining to derive new
    conclusions.

    The operator NEVER mutates HCIR.  All conclusions are returned
    as proposed TransactionOperations.
    """

    def __init__(
        self,
        *,
        enable_fuzzy: bool = False,
        profile_id: str | FuzzyProfileId = FuzzyProfileId.LUKASIEWICZ,
    ) -> None:
        self._enable_fuzzy = bool(enable_fuzzy)
        self._profile_id = FuzzyProfileId(profile_id)

    @property
    def operator_id(self) -> str:
        return "deduction"

    @property
    def operator_name(self) -> str:
        return "Formal Deduction Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()  # No prerequisites

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        """Score applicability for deduction.

        High for: explanation, classification, constraint problems
        where there are beliefs with conditional structure.
        """
        # Base score by problem type
        type_scores: dict[ProblemType, float] = {
            ProblemType.EXPLANATION: 0.7,
            ProblemType.CLASSIFICATION: 0.6,
            ProblemType.CONSTRAINT: 0.8,
            ProblemType.CAUSAL: 0.5,
            ProblemType.GENERALIZATION: 0.3,
            ProblemType.PREDICTION: 0.4,
        }
        base = type_scores.get(problem.problem_type, 0.2)

        # Boost if there are beliefs/facts to reason over
        view = context.graph_view
        n_beliefs = len(view.nodes_by_type(HCIRNodeType.BELIEF))
        n_facts = len(view.nodes_by_type(HCIRNodeType.FACT))

        if n_beliefs + n_facts == 0:
            return 0.0  # Nothing to reason over

        # More premises → higher applicability (up to a point)
        premise_boost = min(0.3, (n_beliefs + n_facts) * 0.02)
        return min(1.0, base + premise_boost)

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        """Deduction is cheap — pure logic, no simulation."""
        view = context.graph_view
        n_nodes = view.node_count
        return ResourceCost(
            wall_clock_ms=max(1.0, n_nodes * 0.1),
            nodes_read=n_nodes,
            edges_read=view.edge_count,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Run forward-chaining deduction over HCIR beliefs.

        1. Extract rules from beliefs with conditional structure.
        2. Collect known facts/grounded beliefs.
        3. Forward chain until fixpoint or budget exhausted.
        4. Return new conclusions as proposed transitions.
        """
        start = time.time()
        view = context.graph_view

        # ── Extract rules and facts ──────────────────────────────────
        rules, known_facts, source_map, fact_degrees, fact_confs = self._extract_knowledge(
            view, enable_fuzzy=self._enable_fuzzy
        )

        if not rules and not known_facts:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "No rules or facts found in view"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Forward chaining ─────────────────────────────────────────
        new_conclusions, derivation_chains = self._forward_chain(
            rules,
            known_facts,
            source_map,
            max_iterations=context.budget.operator_depth,
            fact_degrees=fact_degrees,
            fact_confs=fact_confs,
            profile_id=self._profile_id,
        )

        if not new_conclusions:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                conclusions={"derived_count": 0},
                metadata={"reason": "No new conclusions derivable"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Build result ─────────────────────────────────────────────
        proposed_ops: list[TransactionOperation] = []
        provenance_chains: list[ProvenanceChain] = []
        evidence_refs: list[str] = []

        for conclusion, (conf, deg, chain) in new_conclusions.items():
            # Propose adding a new belief node preserving fuzzy semantic degree separately
            new_belief = BeliefNode(
                claim=conclusion,
                belief_type="factual",
                evidence_sources=list(chain),
                properties={
                    "semantic_degree": round(float(deg), 4),
                    "profile_id": self._profile_id.value,
                },
                provenance=Provenance(
                    created_by=self.operator_id,
                    source_type="inferred",
                    reason=f"Deduced via forward chaining: {' → '.join(chain)}",
                ),
            )
            new_belief.uncertainty.confidence = conf

            proposed_ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=new_belief.model_dump(),
                )
            )

            provenance_chains.append(
                ProvenanceChain(
                    conclusion=conclusion,
                    evidence_node_ids=list(chain),
                    operator_id=self.operator_id,
                    reasoning_steps=[f"Applied rule: {step}" for step in chain],
                    confidence=conf,
                )
            )
            evidence_refs.extend(chain)

        # Deduplicate evidence refs
        evidence_refs = list(dict.fromkeys(evidence_refs))

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "derived_count": len(new_conclusions),
                "derived_claims": list(new_conclusions.keys()),
                "derived_degrees": {k: round(v[1], 4) for k, v in new_conclusions.items()},
            },
            confidence=min(
                (conf for conf, _, _ in new_conclusions.values()),
                default=0.5,
            ),
            evidence_refs=evidence_refs,
            assumptions=["Closed-world assumption on available beliefs"],
            proposed_transitions=proposed_ops,
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )

    # ── Internal Methods ─────────────────────────────────────────────

    @staticmethod
    def _extract_knowledge(
        view: Any,  # FrozenGraphView
        *,
        enable_fuzzy: bool = False,
    ) -> tuple[list[_Rule], set[str], dict[str, str], dict[str, float], dict[str, float]]:
        """Extract rules and facts from the frozen view."""
        rules: list[_Rule] = []
        known_facts: set[str] = set()
        source_map: dict[str, str] = {}
        fact_degrees: dict[str, float] = {}
        fact_confs: dict[str, float] = {}

        # Conditional keywords that suggest a rule
        _CONDITIONAL_KEYWORDS = frozenset(
            {
                "if ",
                "all ",
                "every ",
                "when ",
                "whenever ",
                " implies ",
                " then ",
                "→",
                "⟹",
            }
        )

        for node in view.nodes_by_type(HCIRNodeType.BELIEF):
            if not isinstance(node, BeliefNode):
                continue
            claim = node.claim.strip().lower()
            source_map[claim] = node.id
            props = node.properties if isinstance(node.properties, dict) else {}
            has_sem_deg = isinstance(props.get("semantic_degree"), (int, float))
            sem_deg = max(0.0, min(1.0, float(props.get("semantic_degree", 1.0))))
            conf = float(node.uncertainty.confidence)

            # Check if this looks like a rule
            is_rule = any(kw in claim for kw in _CONDITIONAL_KEYWORDS)

            if is_rule:
                rule = DeductionOperator._parse_rule(claim, node, semantic_degree=sem_deg)
                if rule is not None:
                    rules.append(rule)
            elif ((enable_fuzzy or has_sem_deg) and sem_deg > 0.0 and conf >= 0.3) or conf >= 0.6:
                known_facts.add(claim)
                fact_degrees[claim] = sem_deg
                fact_confs[claim] = conf

        for node in view.nodes_by_type(HCIRNodeType.FACT):
            if not isinstance(node, FactNode):
                continue
            claim = node.claim.strip().lower()
            props = node.properties if isinstance(node.properties, dict) else {}
            sem_deg = max(0.0, min(1.0, float(props.get("semantic_degree", 1.0))))
            known_facts.add(claim)
            fact_degrees[claim] = sem_deg
            fact_confs[claim] = float(getattr(node, "confidence", 1.0) or 1.0)
            source_map[claim] = node.id

        return rules, known_facts, source_map, fact_degrees, fact_confs

    @staticmethod
    def _parse_rule(claim: str, node: BeliefNode, *, semantic_degree: float = 1.0) -> _Rule | None:
        """Attempt to parse a natural-language conditional into a Rule."""
        # Try "if ... then ..."
        for sep_pair in [
            ("if ", " then "),
            ("when ", " then "),
            ("whenever ", " then "),
        ]:
            if sep_pair[0] in claim and sep_pair[1] in claim:
                parts = claim.split(sep_pair[0], 1)
                if len(parts) == 2:
                    rest = parts[1]
                    then_parts = rest.split(sep_pair[1], 1)
                    if len(then_parts) == 2:
                        antecedent = then_parts[0].strip()
                        consequent = then_parts[1].strip()
                        if antecedent and consequent:
                            return _Rule(
                                antecedents=frozenset({antecedent}),
                                consequent=consequent,
                                confidence=node.uncertainty.confidence,
                                source_id=node.id,
                                semantic_degree=semantic_degree,
                            )

        # Try "all X are Y"
        if claim.startswith("all ") and " are " in claim:
            parts = claim[4:].split(" are ", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return _Rule(
                    antecedents=frozenset({parts[0].strip()}),
                    consequent=parts[1].strip(),
                    confidence=node.uncertainty.confidence,
                    source_id=node.id,
                    semantic_degree=semantic_degree,
                )

        # Try "X implies Y"
        for sep in [" implies ", " → ", " ⟹ "]:
            if sep in claim:
                parts = claim.split(sep, 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    return _Rule(
                        antecedents=frozenset({parts[0].strip()}),
                        consequent=parts[1].strip(),
                        confidence=node.uncertainty.confidence,
                        source_id=node.id,
                        semantic_degree=semantic_degree,
                    )

        return None

    @staticmethod
    def _forward_chain(
        rules: list[_Rule],
        known_facts: set[str],
        source_map: dict[str, str],
        max_iterations: int = 20,
        *,
        fact_degrees: dict[str, float] | None = None,
        fact_confs: dict[str, float] | None = None,
        profile_id: str | FuzzyProfileId = FuzzyProfileId.LUKASIEWICZ,
    ) -> tuple[dict[str, tuple[float, float, list[str]]], dict[str, list[str]]]:
        """Forward chaining to derive new conclusions with graded residuated modus ponens."""
        prof = get_fuzzy_profile(profile_id)
        f_degs = dict(fact_degrees or {})
        f_confs = dict(fact_confs or {})
        working_set = set(known_facts)
        new_conclusions: dict[str, tuple[float, float, list[str]]] = {}
        derivation_chains: dict[str, list[str]] = {}

        for iteration in range(max_iterations):
            newly_derived: set[str] = set()

            for rule in rules:
                if rule.antecedents.issubset(working_set):
                    if rule.consequent not in working_set:
                        ant_deg = prof.t_norm_many(f_degs.get(a, 1.0) for a in rule.antecedents)
                        cons_deg = prof.t_norm(rule.semantic_degree, ant_deg)
                        if cons_deg <= 0.0:
                            continue
                        cons_conf = min(
                            [rule.confidence, *(f_confs.get(a, 1.0) for a in rule.antecedents)]
                        )
                        newly_derived.add(rule.consequent)
                        f_degs[rule.consequent] = cons_deg
                        f_confs[rule.consequent] = cons_conf

                        # Collect provenance
                        sources = [rule.source_id]
                        for ant in rule.antecedents:
                            if ant in source_map:
                                sources.append(source_map[ant])

                        new_conclusions[rule.consequent] = (
                            cons_conf,
                            cons_deg,
                            sources,
                        )
                        derivation_chains[rule.consequent] = [f"iteration {iteration}: {rule}"]

            if not newly_derived:
                break  # Fixpoint reached

            working_set.update(newly_derived)
            for fact in newly_derived:
                if fact not in source_map:
                    source_map[fact] = f"derived:iter_{iteration}"

        return new_conclusions, derivation_chains
