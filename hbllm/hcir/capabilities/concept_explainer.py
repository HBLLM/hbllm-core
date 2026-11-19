"""Multi-Hop Knowledge Graph Traversal & Structured Discourse Composition.

Implements non-LLM multi-sentence explanation and paragraph/essay synthesis by
traversing across 4 knowledge dimensions:
  1. Lexical definition & synonyms (LexicalOntologyStore / ConceptNode)
  2. Taxonomy & classification (hypernym_edges / GENERALIZES edges)
  3. Causal dynamics & triggers (CAUSES edges / causal beliefs)
  4. Functional dependencies & requirements (REQUIRES edges / dependency beliefs)
  5. Empirical evidence & context (BeliefNode / EpisodeNode in TieredWorkspace)

Upholds all AGENTS.md invariants: 100% deterministic, zero LLM dependencies,
strictly task-agnostic, and read-only over immutable graph snapshots.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.nodes import HCIREdgeType

logger = logging.getLogger(__name__)


def _indefinite_article(word: str) -> str:
    """Return 'A' or 'An' for the given word (title-cased)."""
    if word and word[0].lower() in "aeiou":
        return "An"
    return "A"


def _clean_concept_name(name: str) -> str:
    """Normalize concept name (strip prefixes like 't4_concept_')."""
    clean = re.sub(r"^t4_concept_", "", name.strip())
    clean = clean.replace("_", " ").strip()
    return clean


@dataclass
class ConceptKnowledgeSnapshot:
    """Structured multi-dimensional knowledge bundle for a concept."""

    concept: str
    definition: str = ""
    hypernyms: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    caused_by: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    required_by: list[str] = field(default_factory=list)
    beliefs: list[str] = field(default_factory=list)
    domain: str = "general"
    provenance: str = ""

    @property
    def has_knowledge(self) -> bool:
        return bool(
            self.definition or self.hypernyms or self.causes or self.requires or self.beliefs
        )


class ConceptExplainer:
    """Traverses HCIR knowledge stores and synthesizes multi-sentence discourse.

    Usage::

        explainer = ConceptExplainer(tiered_workspace, ontology_store)
        snapshot = explainer.gather_knowledge("microscope")
        essay = explainer.synthesize_discourse(snapshot)
    """

    def __init__(
        self,
        tiered_workspace: Any | None = None,
        ontology_store: Any | None = None,
    ) -> None:
        self._tiered_workspace = tiered_workspace
        self._ontology_store = ontology_store

    def gather_knowledge(self, concept_query: str) -> ConceptKnowledgeSnapshot:
        """Traverse ontology and tiered workspace to gather multi-dimensional knowledge."""
        concept = concept_query.strip().lower()
        # Strip common leading words
        for prefix in ("a ", "an ", "the ", "about "):
            if concept.startswith(prefix):
                concept = concept[len(prefix) :].strip()

        snapshot = ConceptKnowledgeSnapshot(concept=concept)

        # 1. Lexical Ontology Store (Definitions, Hypernyms, Synonyms)
        if self._ontology_store is not None:
            try:
                entries = self._ontology_store.lookup(concept)
                for entry in entries:
                    if entry.definition and not snapshot.definition:
                        snapshot.definition = entry.definition
                        snapshot.domain = entry.domain
                        snapshot.provenance = f"ontology:{entry.source or 'lexical_store'}"
                    if entry.synonyms and not snapshot.synonyms:
                        snapshot.synonyms = [
                            s.replace("_", " ") for s in entry.synonyms if s.lower() != concept
                        ]

                # Hypernyms / Taxonomy
                hyps = self._ontology_store.get_hypernyms(concept)
                if hyps:
                    clean_hyps = [
                        h.replace("_", " ") for h in hyps if h.lower() != concept and len(h) > 1
                    ]
                    snapshot.hypernyms = sorted(set(clean_hyps))[:6]
            except Exception as exc:
                logger.debug("ConceptExplainer: ontology gather error for %r: %s", concept, exc)

        # 2. Tiered Workspace Graph Traversal (Concepts, Edges, Beliefs)
        if self._tiered_workspace is not None:
            try:
                self._traverse_workspace_graph(concept, snapshot)
            except Exception as exc:
                logger.debug("ConceptExplainer: workspace traversal error for %r: %s", concept, exc)

        return snapshot

    def _traverse_workspace_graph(
        self,
        concept: str,
        snapshot: ConceptKnowledgeSnapshot,
    ) -> None:
        """Traverse ConceptNodes, BeliefNodes, and typed edges across workspace tiers."""
        from hbllm.hcir.graph import BeliefNode, ConceptNode
        from hbllm.hcir.query import GraphQuery
        from hbllm.hcir.workspace_tiers import WorkspaceTier

        # Query persistent and brain tiers
        graph_query = GraphQuery(text_contains=concept, limit=25)
        result = self._tiered_workspace.query_across_tiers(
            graph_query,
            tiers=[WorkspaceTier.PERSISTENT, WorkspaceTier.BRAIN, WorkspaceTier.WORKING],
        )

        for node in result.nodes:
            # ConceptNode definition fallback
            if isinstance(node, ConceptNode):
                if node.label.lower() == concept or node.id.lower() in (
                    f"t4_concept_{concept}",
                    f"concept_{concept}",
                ):
                    if not snapshot.definition and node.definition:
                        snapshot.definition = node.definition

            # BeliefNode extraction
            elif isinstance(node, BeliefNode):
                claim_text = (node.claim or node.statement or "").strip()
                if not claim_text:
                    continue

                # Check for causal or dependency patterns
                claim_lower = claim_text.lower()
                if (
                    " causes " in claim_lower
                    or " triggers " in claim_lower
                    or " leads to " in claim_lower
                ):
                    parts = re.split(r"\b(?:causes|triggers|leads to)\b", claim_lower)
                    if len(parts) == 2:
                        subj_part, obj_part = parts[0].strip(), parts[1].strip()
                        if concept in subj_part:
                            snapshot.causes.append(_clean_concept_name(obj_part))
                        elif concept in obj_part:
                            snapshot.caused_by.append(_clean_concept_name(subj_part))
                elif (
                    " requires " in claim_lower
                    or " depends on " in claim_lower
                    or " needs " in claim_lower
                ):
                    parts = re.split(r"\b(?:requires|depends on|needs)\b", claim_lower)
                    if len(parts) == 2:
                        subj_part, obj_part = parts[0].strip(), parts[1].strip()
                        if concept in subj_part:
                            snapshot.requires.append(_clean_concept_name(obj_part))
                        elif concept in obj_part:
                            snapshot.required_by.append(_clean_concept_name(subj_part))
                else:
                    if claim_text not in snapshot.beliefs and len(claim_text) > 10:
                        snapshot.beliefs.append(claim_text)

        # Inspect graph edges from persistent tier for typed relational traversal
        p_graph = getattr(getattr(self._tiered_workspace, "persistent", None), "graph", None)
        if p_graph is not None:
            candidate_node_ids = {
                concept,
                f"t4_concept_{concept}",
                f"concept_{concept}",
                f"sem_{concept}",
            }
            for edge in p_graph.edges.values():
                src_matches = any(
                    cid in s.lower() for s in edge.sources for cid in candidate_node_ids
                )
                dst_matches = any(
                    cid in t.lower() for t in edge.targets for cid in candidate_node_ids
                )

                if src_matches:
                    for target_id in edge.targets:
                        target_clean = _clean_concept_name(target_id)
                        if edge.edge_type in (HCIREdgeType.CAUSES,):
                            snapshot.causes.append(target_clean)
                        elif edge.edge_type in (HCIREdgeType.REQUIRES,):
                            snapshot.requires.append(target_clean)
                        elif edge.edge_type in (HCIREdgeType.GENERALIZES,):
                            if target_clean not in snapshot.hypernyms:
                                snapshot.hypernyms.append(target_clean)

                if dst_matches:
                    for source_id in edge.sources:
                        source_clean = _clean_concept_name(source_id)
                        if edge.edge_type in (HCIREdgeType.CAUSES,):
                            snapshot.caused_by.append(source_clean)
                        elif edge.edge_type in (HCIREdgeType.REQUIRES,):
                            snapshot.required_by.append(source_clean)

        # De-duplicate lists
        snapshot.causes = list(dict.fromkeys(snapshot.causes))[:5]
        snapshot.caused_by = list(dict.fromkeys(snapshot.caused_by))[:5]
        snapshot.requires = list(dict.fromkeys(snapshot.requires))[:5]
        snapshot.required_by = list(dict.fromkeys(snapshot.required_by))[:5]
        snapshot.beliefs = list(dict.fromkeys(snapshot.beliefs))[:5]

    def synthesize_discourse(
        self,
        snapshot: ConceptKnowledgeSnapshot,
        style: str = "paragraph",
    ) -> str:
        """Compose a structured, multi-sentence discourse from knowledge snapshot.

        Generates 2 to 5 calibrated English sentences:
          Sentence 1: Definition & core concept meaning
          Sentence 2: Taxonomic classification & hypernyms
          Sentence 3: Functional requirements & dependencies
          Sentence 4: Causal mechanisms, triggers & effects
          Sentence 5: Grounded contextual beliefs & observations
        """
        if not snapshot.has_knowledge:
            return f"I do not have sufficient recorded knowledge about '{snapshot.concept}' to provide an explanation."

        sentences: list[str] = []
        name = snapshot.concept
        article = _indefinite_article(name)

        # 1. Definition Sentence
        if snapshot.definition:
            defn = snapshot.definition.rstrip(".")
            # Ensure definition begins smoothly
            if defn.lower().startswith(f"a {name} is") or defn.lower().startswith(f"an {name} is"):
                sentences.append(f"{defn}.")
            elif defn.lower().startswith("is ") or defn.lower().startswith("are "):
                sentences.append(f"{article} {name} {defn}.")
            else:
                sentences.append(f"{article} {name} is {defn}.")
        else:
            sentences.append(f"{article} {name} is a recognized conceptual entity.")

        # 2. Taxonomy / Supercategory Sentence
        if snapshot.hypernyms:
            hyps = snapshot.hypernyms[:4]
            if len(hyps) == 1:
                cat_str = f"a type of {hyps[0]}"
            elif len(hyps) == 2:
                cat_str = f"a type of {hyps[0]} and {hyps[1]}"
            else:
                cat_str = f"a type of {', '.join(hyps[:-1])}, and {hyps[-1]}"
            sentences.append(f"Taxonomically, it is classified as {cat_str}.")

        # 3. Functional Requirements & Dependencies
        if snapshot.requires:
            reqs = snapshot.requires[:3]
            if len(reqs) == 1:
                req_str = reqs[0]
            elif len(reqs) == 2:
                req_str = f"{reqs[0]} and {reqs[1]}"
            else:
                req_str = f"{', '.join(reqs[:-1])}, and {reqs[-1]}"
            sentences.append(f"Functionally, it requires {req_str} to operate.")

        # 4. Causal Dynamics (Outcomes & Triggers)
        causal_parts: list[str] = []
        if snapshot.causes:
            effects = snapshot.causes[:3]
            eff_str = (
                effects[0]
                if len(effects) == 1
                else (
                    f"{effects[0]} and {effects[1]}"
                    if len(effects) == 2
                    else f"{', '.join(effects[:-1])}, and {effects[-1]}"
                )
            )
            causal_parts.append(f"causes {eff_str}")
        if snapshot.caused_by:
            causes = snapshot.caused_by[:2]
            c_str = f"{causes[0]} and {causes[1]}" if len(causes) == 2 else causes[0]
            causal_parts.append(f"is triggered or produced by {c_str}")

        if causal_parts:
            sentences.append(f"Operationally, it {' and '.join(causal_parts)}.")

        # 5. Contextual Knowledge & Stored Beliefs
        if snapshot.beliefs:
            first_belief = snapshot.beliefs[0].rstrip(".")
            sentences.append(f"Recorded knowledge notes that {first_belief}.")

        return " ".join(sentences)

    def explain(self, concept_query: str) -> str:
        """End-to-end convenience method: gather knowledge and synthesize explanation."""
        snapshot = self.gather_knowledge(concept_query)
        return self.synthesize_discourse(snapshot)
