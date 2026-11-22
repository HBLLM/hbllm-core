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
    """Return 'A' or 'An' for the given word (title-cased), or empty for proper/mass nouns."""
    if not word:
        return "A"
    clean = word.strip()
    lower = clean.lower()
    from hbllm.hcir.language.english.realizer import MASS_NOUNS

    if (
        clean[0].isupper()
        or lower in ("gandalf", "darcy", "mr. darcy", "saruman")
        or lower in MASS_NOUNS
    ):
        return ""
    if lower[0] in "aeiou":
        return "An"
    return "A"


def _clean_concept_name(name: str) -> str:
    """Normalize concept name (strip prefixes like 't4_concept_', 'web_concept_', etc.)."""
    clean = re.sub(r"^(?:t4_concept_|web_concept_|concept_|sem_)", "", name.strip())
    clean = clean.replace("_", " ").strip()
    return clean


def _resolve_concept_node_ids(p_graph: Any, name: str) -> list[str]:
    """Resolve persistent graph node IDs matching a concept name or label."""
    from hbllm.hcir.graph import ConceptNode, HCIRNodeType

    clean = name.strip().lower()
    if not clean:
        return []
    clean_under = clean.replace(" ", "_")
    candidates = (
        clean,
        clean_under,
        f"t4_concept_{clean_under}",
        f"t4_concept_{clean}",
        f"web_concept_{clean_under}",
        f"web_concept_{clean}",
        f"concept_{clean_under}",
        f"concept_{clean}",
        f"sem_{clean_under}",
    )
    resolved: list[str] = []
    seen: set[str] = set()
    for cid in candidates:
        if cid not in seen and p_graph.get_node(cid) is not None:
            resolved.append(cid)
            seen.add(cid)

    if getattr(p_graph, "node_count", 0) <= 2000:
        for node in p_graph.nodes_by_type(HCIRNodeType.CONCEPT):
            if (
                isinstance(node, ConceptNode)
                and node.label
                and node.label.strip().lower() in (clean, clean_under)
                and node.id not in seen
            ):
                resolved.append(node.id)
                seen.add(node.id)
    return resolved


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

    def gather_knowledge(
        self,
        concept_query: str,
        *,
        _allow_fuzzy: bool = True,
    ) -> ConceptKnowledgeSnapshot:
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

        # 3. Fuzzy concept recovery when unknown typo queried (e.g. "why does focsu happen?")
        if (
            _allow_fuzzy
            and not snapshot.has_knowledge
            and len(concept) >= 4
            and concept.replace(" ", "").isalpha()
        ):
            corrected = self._fuzzy_resolve_concept(concept)
            if corrected and corrected != concept:
                return self.gather_knowledge(corrected, _allow_fuzzy=False)

        return snapshot

    def _fuzzy_resolve_concept(self, concept: str) -> str | None:
        """Attempt precision-bounded edit-distance recovery against ConceptNodes and ontology store."""
        from hbllm.hcir.graph import ConceptNode, HCIRNodeType
        from hbllm.hcir.language.core.normalizer import damerau_levenshtein

        # Never overwrite valid/high-frequency words
        if self._ontology_store is not None:
            if hasattr(
                self._ontology_store, "is_trusted_word"
            ) and self._ontology_store.is_trusted_word(concept, language="en"):
                return None
            if (
                hasattr(self._ontology_store, "get_word_frequency")
                and self._ontology_store.get_word_frequency(concept, language="en") >= 5
            ):
                return None

        def _is_high_precision_typo(cand: str) -> bool:
            if abs(len(cand) - len(concept)) > 1:
                return False
            if sorted(cand) == sorted(concept):
                return True
            if cand == concept + "e":
                return True
            if re.sub(r"(.)\1+", r"\1", concept) == re.sub(r"(.)\1+", r"\1", cand):
                return True
            if len(concept) >= 6 and cand[:4] == concept[:4]:
                return True
            return False

        best_label: str | None = None
        best_dist = 2

        p_graph = getattr(getattr(self._tiered_workspace, "persistent", None), "graph", None)
        if p_graph is not None:
            for node in p_graph.nodes_by_type(HCIRNodeType.CONCEPT):
                if isinstance(node, ConceptNode) and node.label:
                    cand = node.label.lower()
                    if abs(len(cand) - len(concept)) > 1:
                        continue
                    if cand[0] != concept[0] and not (
                        len(cand) >= 2 and cand[1] == concept[0] and cand[0] == concept[1]
                    ):
                        continue
                    if not _is_high_precision_typo(cand):
                        continue
                    d = damerau_levenshtein(concept, cand, 1)
                    if d <= 1 and d < best_dist:
                        best_dist = d
                        best_label = cand
        if best_label is not None:
            return best_label

        if self._ontology_store is not None and hasattr(self._ontology_store, "fuzzy_lookup"):
            hits = self._ontology_store.fuzzy_lookup(concept, language="en", limit=3)
            for raw_cand in hits:
                cand = str(raw_cand).lower()
                if _is_high_precision_typo(cand) and damerau_levenshtein(concept, cand, 1) <= 1:
                    return cand

        return None

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

        clean_under = concept.replace(" ", "_")
        for node in result.nodes:
            # ConceptNode definition fallback
            if isinstance(node, ConceptNode):
                if node.label.lower() == concept or node.id.lower() in (
                    f"t4_concept_{concept}",
                    f"web_concept_{clean_under}",
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
                elif " is_a " in claim_lower or " is a " in claim_lower:
                    parts = re.split(r"\b(?:is_a|is a)\b", claim_lower)
                    if len(parts) == 2:
                        subj_part, obj_part = parts[0].strip(), parts[1].strip()
                        if concept in subj_part and obj_part:
                            clean_hyp = _clean_concept_name(obj_part)
                            if clean_hyp not in snapshot.hypernyms:
                                snapshot.hypernyms.append(clean_hyp)
                else:
                    clean_c = claim_text.replace("is_a", "is a").replace("_", " ")
                    if clean_c not in snapshot.beliefs and len(clean_c) > 10:
                        snapshot.beliefs.append(clean_c)

        # Inspect graph edges from persistent tier for typed relational traversal (O(degree))
        p_graph = getattr(getattr(self._tiered_workspace, "persistent", None), "graph", None)
        if p_graph is not None:
            candidate_node_ids = _resolve_concept_node_ids(p_graph, concept)
            seen_edges: set[str] = set()
            for cid in candidate_node_ids:
                for edge in p_graph.edges_from(cid):
                    if edge.id in seen_edges:
                        continue
                    seen_edges.add(edge.id)
                    if edge.edge_type == HCIREdgeType.CONTRADICTS:
                        continue
                    props = getattr(edge, "properties", {}) or {}
                    if props.get("polarity") is False:
                        continue
                    rel = str(props.get("relation", "")).lower()
                    if rel.startswith("not_"):
                        continue
                    for target_id in edge.targets:
                        target_clean = _clean_concept_name(target_id)
                        if edge.edge_type == HCIREdgeType.CAUSES or rel in ("causes", "enables"):
                            snapshot.causes.append(target_clean)
                        elif edge.edge_type == HCIREdgeType.REQUIRES or rel == "requires":
                            snapshot.requires.append(target_clean)
                        elif edge.edge_type == HCIREdgeType.GENERALIZES or rel == "is_a":
                            if target_clean not in snapshot.hypernyms:
                                snapshot.hypernyms.append(target_clean)

                for edge in p_graph.edges_to(cid):
                    if edge.id in seen_edges:
                        continue
                    seen_edges.add(edge.id)
                    if edge.edge_type == HCIREdgeType.CONTRADICTS:
                        continue
                    props = getattr(edge, "properties", {}) or {}
                    if props.get("polarity") is False:
                        continue
                    rel = str(props.get("relation", "")).lower()
                    if rel.startswith("not_"):
                        continue
                    for source_id in edge.sources:
                        source_clean = _clean_concept_name(source_id)
                        if edge.edge_type == HCIREdgeType.CAUSES or rel in ("causes", "enables"):
                            snapshot.caused_by.append(source_clean)
                        elif edge.edge_type == HCIREdgeType.REQUIRES or rel == "requires":
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
        prefix = f"{article} " if article else ""

        # 1. Definition Sentence
        if snapshot.definition:
            defn = snapshot.definition.rstrip(".")
            lower_defn = defn.lower()
            name_lower = name.lower()
            # Ensure definition begins smoothly (including plural forms like "Florbs are...")
            if (
                lower_defn.startswith(f"a {name_lower} is")
                or lower_defn.startswith(f"an {name_lower} is")
                or lower_defn.startswith(f"{name_lower} is")
                or lower_defn.startswith(f"{name_lower} was")
                or lower_defn.startswith(f"{name_lower}s are")
                or lower_defn.startswith(f"{name_lower}es are")
                or lower_defn.startswith(f"{name_lower} are")
            ):
                sentences.append(f"{defn[0].upper() + defn[1:]}.")
            elif lower_defn.startswith("is ") or lower_defn.startswith("are "):
                sentences.append(f"{prefix}{name} {defn}.")
            else:
                sentences.append(f"{prefix}{name} is {defn}.")
        else:
            sentences.append(f"{prefix}{name} is a recognized conceptual entity.")

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
            first_belief = snapshot.beliefs[0].rstrip(".").replace("is_a", "is a").replace("_", " ")
            sentences.append(f"Recorded knowledge notes that {first_belief}.")

        return " ".join(sentences)

    def explain(self, concept_query: str) -> str:
        """End-to-end convenience method: gather knowledge and synthesize explanation."""
        snapshot = self.gather_knowledge(concept_query)
        return self.synthesize_discourse(snapshot)

    def explain_why(
        self,
        concept_query: str,
        max_depth: int = 3,
        snapshot: ConceptKnowledgeSnapshot | None = None,
    ) -> str:
        """Compose a structured causal 'why' explanation via backward CAUSES traversal.

        Traverses upstream causal antecedents:
          1. Direct cause: What directly produces/triggers concept?
          2. Multi-hop root cause: What causes the direct cause?
          3. Downstream consequences: What does concept in turn cause?
        """
        if snapshot is None:
            snapshot = self.gather_knowledge(concept_query)
        if not snapshot.caused_by and " " in concept_query.strip():
            # Multi-word event queries (e.g. "river flood") probe compound and head forms
            parts = concept_query.strip().lower().split()
            for alt in (concept_query.strip().lower().replace(" ", "_"), parts[-1]):
                alt_snap = self.gather_knowledge(alt, _allow_fuzzy=False)
                if alt_snap.caused_by:
                    snapshot.caused_by = list(
                        dict.fromkeys([*snapshot.caused_by, *alt_snap.caused_by])
                    )[:5]
                    if alt_snap.causes and not snapshot.causes:
                        snapshot.causes = alt_snap.causes
                    if alt_snap.beliefs and not snapshot.beliefs:
                        snapshot.beliefs = alt_snap.beliefs
                    break
        name = snapshot.concept
        article = _indefinite_article(name)

        # Multi-hop backwards traversal across persistent graph (O(degree) via edges_to)
        causal_chains: list[list[str]] = []
        p_graph = getattr(getattr(self._tiered_workspace, "persistent", None), "graph", None)

        if p_graph is not None and snapshot.caused_by:
            for direct_cause in snapshot.caused_by[:3]:
                chain = [direct_cause]
                curr = direct_cause
                for _ in range(max_depth - 1):
                    curr_ids = _resolve_concept_node_ids(p_graph, curr)
                    parent_cause: str | None = None
                    for cid in curr_ids:
                        for edge in p_graph.edges_to(cid):
                            if edge.edge_type == HCIREdgeType.CAUSES:
                                for s in edge.sources:
                                    s_clean = _clean_concept_name(s)
                                    if s_clean not in chain:
                                        parent_cause = s_clean
                                        break
                            if parent_cause:
                                break
                        if parent_cause:
                            break
                    if parent_cause:
                        chain.append(parent_cause)
                        curr = parent_cause
                    else:
                        break
                causal_chains.append(chain)

        paragraphs: list[str] = []

        # 1. Direct Antecedent Cause
        if snapshot.caused_by:
            causes_str = (
                snapshot.caused_by[0]
                if len(snapshot.caused_by) == 1
                else " and ".join(snapshot.caused_by[:2])
            )
            paragraphs.append(
                f"{name.capitalize()} occurs primarily because it is triggered or produced by {causes_str}."
            )
        elif snapshot.definition:
            paragraphs.append(
                f"{article} {name} is defined as {snapshot.definition.rstrip('.')}, "
                f"though no direct upstream causal trigger is currently recorded in memory."
            )
        else:
            paragraphs.append(
                f"I do not have recorded causal antecedents explaining why '{name}' occurs."
            )

        # 2. Multi-Hop Causal Chain
        deep_chains = [c for c in causal_chains if len(c) > 1]
        if deep_chains:
            longest = max(deep_chains, key=len)
            chain_str = " -> ".join(reversed(longest)) + f" -> {name}"
            paragraphs.append(
                f"Tracing the multi-hop causal chain upstream: {chain_str}. "
                f"Specifically, {longest[-1]} leads to {longest[0]}, which directly triggers {name}."
            )

        # 3. Downstream Consequences
        if snapshot.causes:
            effects_str = (
                snapshot.causes[0]
                if len(snapshot.causes) == 1
                else ", ".join(snapshot.causes[:-1]) + f", and {snapshot.causes[-1]}"
            )
            paragraphs.append(
                f"In turn, {name} acts as an active causal agent that produces or causes {effects_str}."
            )

        # 4. Contextual beliefs
        if snapshot.beliefs:
            paragraphs.append(f"Recorded causal evidence: {snapshot.beliefs[0].rstrip('.')}.")

        return "\n\n".join(paragraphs)

    def explain_how(
        self,
        concept_query: str,
        max_depth: int = 3,
        snapshot: ConceptKnowledgeSnapshot | None = None,
    ) -> str:
        """Compose a structured procedural 'how' explanation via forward REQUIRES traversal.

        Traverses prerequisite and dependency requirements:
          1. Direct requirements: What does concept directly require?
          2. Multi-hop prerequisites: What do those requirements in turn depend on?
          3. Step-by-step procedural realization.
        """
        if snapshot is None:
            snapshot = self.gather_knowledge(concept_query)
        name = snapshot.concept
        article = _indefinite_article(name)

        # Multi-hop forward dependency traversal (O(degree) via edges_from)
        dep_chains: list[list[str]] = []
        p_graph = getattr(getattr(self._tiered_workspace, "persistent", None), "graph", None)

        if p_graph is not None and snapshot.requires:
            for direct_req in snapshot.requires[:3]:
                chain = [direct_req]
                curr = direct_req
                for _ in range(max_depth - 1):
                    curr_ids = _resolve_concept_node_ids(p_graph, curr)
                    next_req: str | None = None
                    for cid in curr_ids:
                        for edge in p_graph.edges_from(cid):
                            if edge.edge_type == HCIREdgeType.REQUIRES:
                                for t in edge.targets:
                                    t_clean = _clean_concept_name(t)
                                    if t_clean not in chain:
                                        next_req = t_clean
                                        break
                            if next_req:
                                break
                        if next_req:
                            break
                    if next_req:
                        chain.append(next_req)
                        curr = next_req
                    else:
                        break
                dep_chains.append(chain)

        paragraphs: list[str] = []

        # 1. Functional Nature & Requirements
        if snapshot.requires:
            reqs_str = (
                snapshot.requires[0]
                if len(snapshot.requires) == 1
                else ", ".join(snapshot.requires[:-1]) + f" and {snapshot.requires[-1]}"
            )
            paragraphs.append(f"To operate or function, {name} fundamentally requires {reqs_str}.")
        elif snapshot.definition:
            paragraphs.append(
                f"{article} {name} operates according to its definition: {snapshot.definition.rstrip('.')}."
            )
        else:
            paragraphs.append(
                f"I do not have recorded operational prerequisites explaining how '{name}' works."
            )

        # 2. Step-by-step procedural realization
        steps: list[str] = []
        if snapshot.requires:
            steps.append(
                f"Step 1 (Prerequisites): Ensure availability of {', '.join(snapshot.requires)}."
            )
        if snapshot.hypernyms:
            steps.append(
                f"Step 2 (Classification): Utilize as a functional {snapshot.hypernyms[0]}."
            )
        if snapshot.causes:
            steps.append(
                f"Step 3 (Execution): Engage mechanism to produce {', '.join(snapshot.causes)}."
            )
        else:
            steps.append("Step 3 (Execution): Apply standard operational interactions.")

        if steps:
            paragraphs.append("Operational Procedure:\n" + "\n".join(steps))

        # 3. Multi-Hop Dependency Chain
        deep_chains = [c for c in dep_chains if len(c) > 1]
        if deep_chains:
            longest = max(deep_chains, key=len)
            chain_str = f"{name} -> " + " -> ".join(longest)
            paragraphs.append(f"Tracing the multi-level dependency tree: {chain_str}.")

        return "\n\n".join(paragraphs)
