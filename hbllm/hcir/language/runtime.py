"""Multilingual Non-LLM Language Runtime for A16.

The unified orchestrator connecting natural language utterances to HCIR cognition:
1. Pluggable Language Engines (English, Sinhala, etc.)
2. Language-Neutral SemanticFrames
3. Reference & Grounding Resolver (A13/A15)
4. HCIR Gateway (Evidence, Queries, Goals)
5. Calibrated Surface Realization
6. Interlingual Semantic Transfer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    ConceptNode,
    ConstructionNode,
    GroundedConceptNode,
    HCIRNodeType,
    PhysicalEntityNode,
    SkillNode,
)
from hbllm.hcir.language.core.discourse import DiscourseSegmenter
from hbllm.hcir.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
)
from hbllm.hcir.language.core.gateway import HCIRGateway
from hbllm.hcir.language.core.grounding import GroundingResolver
from hbllm.hcir.language.core.profile import LanguageProfile
from hbllm.hcir.language.core.protocol import LanguageParser, LanguageRealizer
from hbllm.hcir.language.core.reference import ReferenceResolver
from hbllm.hcir.language.core.semantic_frame import (
    FrameType,
    GroundedSemanticFrame,
    LanguageErrorType,
    SemanticFrame,
)
from hbllm.hcir.language.core.universal_engine import (
    UniversalLanguageParser,
    UniversalLanguageRealizer,
)
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.language.english.realizer import EnglishRealizer
from hbllm.hcir.language.sinhala.parser import SinhalaParser
from hbllm.hcir.language.sinhala.realizer import SinhalaRealizer
from hbllm.hcir.language.tamil.parser import TamilParser
from hbllm.hcir.language.tamil.realizer import TamilRealizer

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR: Path = Path(__file__).resolve().parent / "profiles"


@dataclass
class LanguageTurnResult:
    """Complete result of processing a language turn."""

    raw_input: str
    language: str
    semantic_frame: SemanticFrame
    grounded_frame: GroundedSemanticFrame | None = None
    epistemic_state: CognitiveEpistemicState | None = None
    response_text: str = ""
    is_success: bool = True
    error_type: LanguageErrorType | None = None
    error_detail: str = ""
    hcir_node_id: str | None = (
        None  # EvidenceNode ID (for assertions) or GoalNode ID (for commands)
    )
    sub_results: list[LanguageTurnResult] = field(default_factory=list)


class MultilingualLanguageRuntime:
    """Orchestrates non-LLM multilingual parsing, grounding, and realization.

    Usage::

        runtime = MultilingualLanguageRuntime(graph)

        # 1. Assertion
        res = runtime.process_utterance("The red ball is on the table.")

        # 2. Query
        res = runtime.process_utterance("Where is the red ball?")
        # res.response_text -> "The ball is on the table."

        # 3. Multilingual Query in Sinhala
        res = runtime.process_utterance("බෝලය කොහෙද?", language="si")
        # res.response_text -> "බෝලය මේසය මත තියෙනවා."

        # 4. Interlingual Semantic Transfer (English -> Sinhala)
        sin_text = runtime.translate("The red ball is on the table.", source_lang="en", target_lang="si")
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        reference_resolver: ReferenceResolver | None = None,
        epistemic_policy: EpistemicRealizationPolicy | None = None,
        workspace: Any | None = None,
        ontology_store: Any | None = None,
    ) -> None:
        self._graph = graph
        self._workspace = workspace
        self._ref_resolver = reference_resolver or ReferenceResolver()
        self._epistemic_policy = epistemic_policy or EpistemicRealizationPolicy()
        self._gateway = HCIRGateway(
            graph,
            workspace=workspace,
            ontology_store=None,  # Set after ontology_store init below
            tiered_workspace=getattr(workspace, "_tiered_workspace", None),
        )
        self._dyn_vocab_sig: tuple[int, int] = (-1, -1)

        if ontology_store is None:
            from hbllm.hcir.language.ontology_store import (
                DEFAULT_ONTOLOGY_DB,
                LexicalOntologyStore,
            )

            try:
                ontology_store = LexicalOntologyStore(DEFAULT_ONTOLOGY_DB)
            except Exception as exc:
                logger.debug("Failed to initialize LexicalOntologyStore: %s", exc)
        self._ontology_store = ontology_store

        # Wire ontology_store into gateway for epistemic prior lookups
        self._gateway._ontology_store = ontology_store

        # Register language engines
        en_parser = EnglishParser()
        si_parser = SinhalaParser()
        ta_parser = TamilParser()
        if ontology_store is not None:
            en_parser._lexicon.attach_ontology_store(ontology_store)
            si_parser._lexicon.attach_ontology_store(ontology_store)
            ta_parser._lexicon.attach_ontology_store(ontology_store)
        if graph.node_count > 0:
            from hbllm.hcir.language.ontology_bootstrap import OntologyBootstrapper

            OntologyBootstrapper.hydrate_lexicon(graph, en_parser._lexicon, language="en")

        self._grounding_resolver = GroundingResolver(
            graph,
            self._ref_resolver,
            workspace=workspace,
            lexicon=en_parser._lexicon,
        )

        self._parsers: dict[str, LanguageParser] = {
            "en": en_parser,
            "si": si_parser,
            "ta": ta_parser,
        }
        self._realizers: dict[str, LanguageRealizer] = {
            "en": EnglishRealizer(self._epistemic_policy),
            "si": SinhalaRealizer(self._epistemic_policy),
            "ta": TamilRealizer(self._epistemic_policy),
        }

        # Auto-discover declarative LanguageProfiles (Universal Cognitive Grammar)
        self._profiles: dict[str, LanguageProfile] = {}
        self._load_default_profiles()

    def attach_ontology_store(self, ontology_store: Any) -> None:
        """Attach or update the LexicalOntologyStore across parsers and gateway."""
        self._ontology_store = ontology_store
        self._gateway._ontology_store = ontology_store
        for lang, parser in self._parsers.items():
            if hasattr(parser, "_lexicon") and hasattr(parser._lexicon, "attach_ontology_store"):
                parser._lexicon.attach_ontology_store(ontology_store)

    def _load_default_profiles(self) -> None:
        """Auto-discover and load declarative LanguageProfiles from profiles/."""
        if not DEFAULT_PROFILES_DIR.exists():
            return
        for json_file in sorted(DEFAULT_PROFILES_DIR.glob("*.json")):
            try:
                prof = LanguageProfile.from_file(json_file)
                self.register_profile(prof)
            except Exception as exc:
                logger.debug("Failed to load language profile %s: %s", json_file, exc)

    def register_profile(self, profile: LanguageProfile) -> None:
        """Register a declarative LanguageProfile and instantiate its universal parser & realizer."""
        self._profiles[profile.iso_code] = profile
        if profile.iso_code not in self._parsers:
            self._parsers[profile.iso_code] = UniversalLanguageParser(
                profile, ontology_store=self._ontology_store
            )
        if profile.iso_code not in self._realizers:
            self._realizers[profile.iso_code] = UniversalLanguageRealizer(
                profile, epistemic_policy=self._epistemic_policy
            )

    def load_profile(self, path_or_code: str | Path) -> LanguageProfile:
        """Load and register a LanguageProfile from a file path or known ISO code."""
        p = Path(path_or_code)
        if not p.exists() and isinstance(path_or_code, str):
            p = DEFAULT_PROFILES_DIR / f"{path_or_code}.json"
        if not p.exists():
            raise FileNotFoundError(f"Language profile not found: {path_or_code}")
        prof = LanguageProfile.from_file(p)
        self.register_profile(prof)
        return prof

    def get_profile(self, lang_code: str) -> LanguageProfile | None:
        """Get the registered LanguageProfile for *lang_code* if available."""
        return self._profiles.get(lang_code)

    @property
    def supported_languages(self) -> list[str]:
        """Return list of all supported ISO language codes."""
        return sorted(self._parsers.keys())

    def register_language(
        self,
        lang_code: str,
        parser: LanguageParser,
        realizer: LanguageRealizer,
    ) -> None:
        """Register a new language engine."""
        self._parsers[lang_code] = parser
        self._realizers[lang_code] = realizer

    # ── Utterance Processing Pipeline ─────────────────────────────────

    def _sync_dynamic_vocabulary(self, parser: LanguageParser) -> None:
        """Keep parser's Tier-1 typo normalizer aware of active world & persistent concepts."""
        if not hasattr(parser, "set_dynamic_vocabulary"):
            return
        tiered_ws = self._gateway._tiered_workspace or getattr(
            self._workspace, "_tiered_workspace", None
        )
        p_graph = getattr(getattr(tiered_ws, "persistent", None), "graph", None)
        p_count = p_graph.node_count if p_graph is not None else 0
        sig = (self._graph.node_count, p_count)
        if sig == self._dyn_vocab_sig:
            return
        self._dyn_vocab_sig = sig

        vocab: list[str] = []
        learned_verbs: set[str] = set()
        for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if isinstance(node, PhysicalEntityNode) and node.entity_type:
                vocab.append(node.entity_type)
        for node in self._graph.nodes_by_type(HCIRNodeType.GROUNDED_CONCEPT):
            if isinstance(node, GroundedConceptNode) and node.concept_name:
                vocab.append(node.concept_name)
        for g_src in (self._graph, p_graph):
            if g_src is None:
                continue
            for node in g_src.nodes_by_type(HCIRNodeType.CONCEPT):
                if isinstance(node, ConceptNode) and node.label:
                    vocab.append(node.label)
            for node in g_src.nodes_by_type(HCIRNodeType.CONSTRUCTION):
                if isinstance(node, ConstructionNode):
                    learned_verbs.update(node.trigger_tokens)
            for node in g_src.nodes_by_type(HCIRNodeType.SKILL):
                if isinstance(node, SkillNode) and node.skill_name:
                    learned_verbs.add(node.skill_name)
        vocab.extend(sorted(learned_verbs))
        parser.set_dynamic_vocabulary(vocab)
        if learned_verbs and isinstance(parser, (EnglishParser, UniversalLanguageParser)):
            parser.register_command_verbs(learned_verbs)

    def process_discourse(
        self,
        text: str,
        language: str = "en",
        speaker: str = "human",
    ) -> list[LanguageTurnResult]:
        """Process multi-sentence / multi-clause discourse sequentially within one turn."""
        segments = DiscourseSegmenter.segment(text) if language == "en" else [text]
        if not segments:
            segments = [text]
        self._ref_resolver.next_turn()
        return [
            self.process_utterance(seg, language=language, speaker=speaker, advance_turn=False)
            for seg in segments
        ]

    def process_utterance(
        self,
        text: str,
        language: str = "en",
        speaker: str = "human",
        *,
        advance_turn: bool = True,
    ) -> LanguageTurnResult:
        """Process a natural language utterance through the complete A16 pipeline."""
        parser = self._parsers.get(language)
        realizer = self._realizers.get(language)

        if not parser or not realizer:
            err_frame = SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNSUPPORTED_LANGUAGE,
                error_detail=f"Language '{language}' is not supported.",
            )
            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=err_frame,
                is_success=False,
                error_type=LanguageErrorType.UNSUPPORTED_LANGUAGE,
                error_detail=f"Language '{language}' is not supported.",
            )

        # 0. Advance conversational discourse turn once per user message
        if advance_turn:
            self._ref_resolver.next_turn()
        self._sync_dynamic_vocabulary(parser)

        # 1. Parse into language-neutral SemanticFrame
        frame = parser.parse(text)
        if frame.is_error:
            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=frame,
                is_success=False,
                error_type=frame.error_type,
                error_detail=frame.error_detail,
                response_text="I did not understand that sentence.",
            )

        # 2. Grounding & Reference Resolution against A13/A15 HCIR state
        grounded = self._grounding_resolver.ground_frame(frame)
        # Register concrete conceptual topics for multi-turn coreference
        # (only when not already grounded to a physical entity in the world graph)
        self._register_frame_topics(frame, grounded)
        if not grounded.grounding_success:
            # COMMAND frames with tool verbs bypass NOT-FOUND grounding failures —
            # the CommandDispatcher interprets raw concept names as tool params.
            # However, AMBIGUOUS_REFERENCE must still be reported: the user must
            # disambiguate before we can commit a GoalNode.
            if (
                frame.frame_type == FrameType.COMMAND
                and grounded.grounding_error != LanguageErrorType.AMBIGUOUS_REFERENCE
            ):
                goal_node = self._gateway.process_command(grounded)
                from hbllm.hcir.language.core.epistemic_policy import CognitiveEpistemicState

                return LanguageTurnResult(
                    raw_input=text,
                    language=language,
                    semantic_frame=frame,
                    grounded_frame=grounded,
                    epistemic_state=CognitiveEpistemicState(
                        target_predicate=frame.predicate or "command",
                        target_subject=frame.predicate or "command",
                        confidence=0.96,
                        uncertainty=0.04,
                        support_count=1,
                        is_known=True,
                    ),
                    is_success=True,
                    hcir_node_id=goal_node.id,
                    response_text="Goal accepted.",
                )

            err_type = grounded.grounding_error or LanguageErrorType.GROUNDING_FAILED
            err_detail = grounded.grounding_error_detail
            resp = "I could not identify the entity you are referring to."
            if err_type == LanguageErrorType.AMBIGUOUS_REFERENCE:
                # Build a helpful disambiguation response listing candidates
                candidates_info = ""
                if grounded.candidate_entities:
                    for role, ids in grounded.candidate_entities.items():
                        if len(ids) > 1:
                            names = []
                            for eid in ids:
                                node = self._graph.get_node(eid)
                                if node is not None:
                                    props = getattr(node, "properties", {})
                                    label = props.get("color", "") or props.get("name", "")
                                    etype = getattr(node, "entity_type", "object")
                                    names.append(f"{label} {etype}".strip() if label else etype)
                                else:
                                    names.append(eid)
                            candidates_info = " (candidates: " + ", ".join(names) + ")"
                            break
                resp = (
                    f"There are multiple objects matching your description"
                    f"{candidates_info}; please be more specific."
                )

            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=frame,
                grounded_frame=grounded,
                is_success=False,
                error_type=err_type,
                error_detail=err_detail,
                response_text=resp,
            )

        # 3. HCIR Operations based on FrameType
        # A. ASSERTION -> Ingest as EvidenceNode
        if frame.frame_type == FrameType.ASSERTION:
            evidence_node = self._gateway.process_assertion(grounded, speaker=speaker)
            from hbllm.hcir.language.core.epistemic_policy import CognitiveEpistemicState
            from hbllm.hcir.language.core.semantic_frame import ThematicRole

            target_subj = ""
            ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
            if ref is not None and ref.concept_name:
                target_subj = str(ref.concept_name)
            epistemic_state = CognitiveEpistemicState(
                target_predicate=frame.predicate or "assert",
                target_subject=target_subj,
                confidence=0.95,
                uncertainty=0.05,
                support_count=1,
                is_known=True,
            )
            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=frame,
                grounded_frame=grounded,
                epistemic_state=epistemic_state,
                is_success=True,
                hcir_node_id=evidence_node.id,
                response_text="Understood.",
            )

        # B. COMMAND -> Create GoalNode for planner
        elif frame.frame_type == FrameType.COMMAND:
            goal_node = self._gateway.process_command(grounded)
            from hbllm.hcir.language.core.epistemic_policy import CognitiveEpistemicState

            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=frame,
                grounded_frame=grounded,
                epistemic_state=CognitiveEpistemicState(
                    target_predicate=frame.predicate or "command",
                    target_subject=frame.predicate or "command",
                    confidence=0.96,
                    uncertainty=0.04,
                    support_count=1,
                    is_known=True,
                ),
                is_success=True,
                hcir_node_id=goal_node.id,
                response_text="Goal accepted.",
            )

        # C. QUERY -> Query HCIR and verbalize calibrated response
        elif frame.frame_type == FrameType.QUERY:
            epistemic_state = self._gateway.process_query(grounded)
            response_text = realizer.realize(epistemic_state, original_frame=frame)
            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=frame,
                grounded_frame=grounded,
                epistemic_state=epistemic_state,
                is_success=True,
                response_text=response_text,
            )

        return LanguageTurnResult(
            raw_input=text,
            language=language,
            semantic_frame=frame,
            grounded_frame=grounded,
            is_success=True,
        )

    def _register_frame_topics(self, frame: SemanticFrame, grounded: Any | None = None) -> None:
        """Register concrete conceptual topics from a parsed frame into ReferenceResolver."""
        from hbllm.hcir.language.core.reference import is_anaphoric_term
        from hbllm.hcir.language.core.semantic_frame import ThematicRole

        grounded_ents = getattr(grounded, "grounded_entities", {}) if grounded is not None else {}
        for role in (ThematicRole.THEME, ThematicRole.PATIENT, ThematicRole.AGENT):
            if role in grounded_ents:
                # Physical entity mention is already registered with its graph entity_id
                return
            ref = frame.get_role(role)
            if ref is not None and ref.concept_name and not is_anaphoric_term(ref.concept_name):
                self._ref_resolver.register_topic(ref.concept_name)
                return

    # ── Interlingual Semantic Transfer ────────────────────────────────

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "si",
    ) -> str:
        """Perform interlingual semantic transfer (Source Text -> SemanticFrame -> Target Text)."""
        src_parser = self._parsers.get(source_lang)
        tgt_realizer = self._realizers.get(target_lang)

        if not src_parser or not tgt_realizer:
            return ""

        frame = src_parser.parse(text)
        if frame.is_error:
            return ""

        return tgt_realizer.realize_frame(frame)
