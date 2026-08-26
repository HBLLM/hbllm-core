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
from dataclasses import dataclass

from hbllm.brain.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
)
from hbllm.brain.language.core.gateway import HCIRGateway
from hbllm.brain.language.core.grounding import GroundingResolver
from hbllm.brain.language.core.protocol import LanguageParser, LanguageRealizer
from hbllm.brain.language.core.reference import ReferenceResolver
from hbllm.brain.language.core.semantic_frame import (
    FrameType,
    GroundedSemanticFrame,
    LanguageErrorType,
    SemanticFrame,
)
from hbllm.brain.language.english.parser import EnglishParser
from hbllm.brain.language.english.realizer import EnglishRealizer
from hbllm.brain.language.sinhala.parser import SinhalaParser
from hbllm.brain.language.sinhala.realizer import SinhalaRealizer
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._graph = graph
        self._ref_resolver = reference_resolver or ReferenceResolver()
        self._grounding_resolver = GroundingResolver(graph, self._ref_resolver)
        self._gateway = HCIRGateway(graph)
        self._epistemic_policy = epistemic_policy or EpistemicRealizationPolicy()

        # Register language engines
        self._parsers: dict[str, LanguageParser] = {
            "en": EnglishParser(),
            "si": SinhalaParser(),
        }
        self._realizers: dict[str, LanguageRealizer] = {
            "en": EnglishRealizer(self._epistemic_policy),
            "si": SinhalaRealizer(self._epistemic_policy),
        }

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

    def process_utterance(
        self,
        text: str,
        language: str = "en",
        speaker: str = "human",
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
        if not grounded.grounding_success:
            err_type = grounded.grounding_error or LanguageErrorType.GROUNDING_FAILED
            err_detail = grounded.grounding_error_detail
            resp = "I could not identify the entity you are referring to."
            if err_type == LanguageErrorType.AMBIGUOUS_REFERENCE:
                resp = (
                    "There are multiple objects matching your description; please be more specific."
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
            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=frame,
                grounded_frame=grounded,
                is_success=True,
                hcir_node_id=evidence_node.id,
                response_text="Understood.",
            )

        # B. COMMAND -> Create GoalNode for planner
        elif frame.frame_type == FrameType.COMMAND:
            goal_node = self._gateway.process_command(grounded)
            return LanguageTurnResult(
                raw_input=text,
                language=language,
                semantic_frame=frame,
                grounded_frame=grounded,
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
