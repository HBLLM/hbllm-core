"""Tests for robust Pure-HCIR language processing without LLM fallbacks:

1. Long multi-sentence paragraphs & expository essays (with follow-up Q&A)
2. Multi-step instruction sets (semicolons, numbered lists, 'and then', mixed acts)
3. Orthographic typo normalization & fuzzy entity/concept recovery
4. Ungrammatical, telegraphic, and polite/conversational framing recovery
5. Strict epistemic boundaries (word-salad rejection, ambiguity preservation, no false corrections)
"""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.graph import CognitiveGraph, HCIREdgeType, PhysicalEntityNode
from hbllm.hcir.language.core.discourse import DiscourseSegmenter
from hbllm.hcir.language.core.semantic_frame import FrameType, LanguageErrorType, ThematicRole
from hbllm.hcir.language.english.lexicon import EnglishLexicon
from hbllm.hcir.language.english.normalizer import (
    OrthographicNormalizer,
    strip_conversational_wrappers,
)
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.language.runtime import MultilingualLanguageRuntime
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def _make_system(tmp_path: Path):
    return build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_robust_lang",
            memory_dir=tmp_path / "mem",
            enable_grounding=False,
            enable_brain_llm=False,
            enable_autonomous_curiosity=False,
        )
    )


class TestLongParagraphsAndEssays:
    """Verify Pure-HCIR comprehension over multi-sentence paragraphs & essays."""

    def test_multi_sentence_spatial_scene_and_inline_queries(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)
        sys.process_dialogue("The red ball is on the table.")

        res = sys.process_dialogue(
            "The blue cup is on the shelf. The green box is under the table. Where is the blue cup?"
        )
        assert res.is_success is True
        assert "shelf" in res.response_text.lower()
        assert len(res.sub_results) == 3

        res2 = sys.process_dialogue("Where is the green box?")
        assert res2.is_success is True
        assert "table" in res2.response_text.lower()
        assert "under" in res2.response_text.lower()

    def test_causal_paragraph_commits_edges_and_answers_followup_why(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)

        paragraph = (
            "Focus is a cognitive process. "
            "Focus requires attention. "
            "Attention causes learning. "
            "Therefore, people who practice focus improve their memory over time."
        )
        res = sys.process_dialogue(paragraph)
        assert res.is_success is True
        assert "extracted and stored" in res.response_text.lower()

        p_edges = list(sys.tiered_workspace.persistent.graph.edges.values())
        assert any(e.edge_type == HCIREdgeType.CAUSES for e in p_edges)
        assert any(e.edge_type == HCIREdgeType.REQUIRES for e in p_edges)
        assert any(e.edge_type == HCIREdgeType.GENERALIZES for e in p_edges)

        followup = sys.process_dialogue("Why does learning happen?")
        assert followup.is_success is True
        assert "attention" in followup.response_text.lower()

    def test_technical_essay_extracts_exemplars_and_answers_what_is(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)

        essay = (
            "In modern computer science, distributed systems face many challenges such as "
            "network partitions, latency, and clock drift. Because nodes can fail independently, "
            "engineers design consensus algorithms like Raft and Paxos to maintain consistency "
            "across replicas."
        )
        res = sys.process_dialogue(essay)
        assert res.is_success is True
        assert "extracted and stored" in res.response_text.lower()

        q_raft = sys.process_dialogue("What is Raft?")
        assert q_raft.is_success is True
        assert (
            "consensus" in q_raft.response_text.lower()
            or "algorithm" in q_raft.response_text.lower()
        )


class TestMultiStepInstructionSets:
    """Verify sequential execution across mixed & multi-command instruction sets."""

    def test_discourse_segmenter_preserves_filenames_and_splits_steps(self) -> None:
        segs = DiscourseSegmenter.segment("Read README.md; explain focus; list files.")
        assert len(segs) == 3
        assert "README.md" in segs[0]

        numbered = DiscourseSegmenter.segment(
            "1. Remember the server is in the cluster. 2. Explain focus. 3. List files."
        )
        assert numbered == [
            "Remember the server is in the cluster.",
            "Explain focus.",
            "List files.",
        ]

    def test_sequential_and_then_instructions(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)
        res = sys.process_dialogue("List files and then explain focus.")
        assert res.is_success is True
        assert "[Step 1]" in res.response_text
        assert "[Step 2]" in res.response_text
        assert "focus" in res.response_text.lower()

    def test_semicolon_and_numbered_instruction_sets(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)

        res_semi = sys.process_dialogue("Read README.md; explain focus; list files.")
        assert res_semi.is_success is True
        assert "[Step 1]" in res_semi.response_text
        assert "[Step 3]" in res_semi.response_text

        res_num = sys.process_dialogue(
            "1. Remember the server is in the cluster. 2. Explain focus. 3. List files."
        )
        assert res_num.is_success is True
        assert "focus" in res_num.response_text.lower()

    def test_physical_world_multi_action_instructions(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)
        sys.process_dialogue("The red ball is on the table.")
        sys.process_dialogue("The blue cup is on the shelf.")

        res = sys.process_dialogue(
            "Push the red ball to the shelf and then move the blue cup to the table."
        )
        assert res.is_success is True
        assert res.response_text.count("Goal accepted") == 2


class TestOrthographicTypoRecovery:
    """Verify deterministic Damerau-Levenshtein typo recovery & false-positive guards."""

    def test_does_not_overwrite_valid_words(self) -> None:
        lexicon = EnglishLexicon()
        norm = OrthographicNormalizer(lexicon=lexicon)
        assert norm.normalize_token("cup") == "cup"
        assert norm.normalize_token("box") == "box"
        assert norm.normalize_token("red") == "red"
        assert norm.normalize_token("table") == "table"
        assert norm.normalize_token("README.md") == "README.md"

    def test_misspelled_verbs_wh_and_entities_recover(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)
        sys.process_dialogue("The red ball is on the table.")

        for cmd in ("explian focus", "explane focus"):
            res = sys.process_dialogue(cmd)
            assert res.is_success is True
            assert "focus" in res.response_text.lower()

        res_list = sys.process_dialogue("lisst files")
        assert res_list.is_success is True
        assert len(res_list.response_text) > 0

        for q in ("wher is the red ball?", "Where is the red blal?"):
            res_q = sys.process_dialogue(q)
            assert res_q.is_success is True
            assert "table" in res_q.response_text.lower()

        res_why = sys.process_dialogue("why does focsu happen?")
        assert res_why.is_success is True
        assert "focus" in res_why.response_text.lower()

        # Typo in assertion grounds to existing 'table' rather than duplicating 'tabel'
        before_tables = [
            n
            for n in sys.workspace.graph.nodes.values()
            if getattr(n, "entity_type", "") == "tabel"
        ]
        assert len(before_tables) == 0
        res_ass = sys.process_dialogue("Teh red ball is on teh tabel.")
        assert res_ass.is_success is True
        after_tabel = [
            n
            for n in sys.workspace.graph.nodes.values()
            if getattr(n, "entity_type", "") == "tabel"
        ]
        assert len(after_tabel) == 0


class TestUngrammaticalAndInformalRecovery:
    """Verify conversational wrapper stripping & grammar recovery without false positives."""

    def test_conversational_wrappers_and_contractions(self) -> None:
        cleaned, changed = strip_conversational_wrappers("please can you explain focus?")
        assert changed is True
        assert cleaned.lower().startswith("explain focus")

        cleaned2, _ = strip_conversational_wrappers("Where's the red ball?")
        assert cleaned2.lower() == "where is the red ball?"

    def test_informal_and_ungrammatical_dialogue(self, tmp_path: Path) -> None:
        sys = _make_system(tmp_path)
        sys.process_dialogue("The red ball is on the table.")

        for utter in (
            "please can you explain focus?",
            "can you tell me what is focus?",
            "I want you to explain focus",
            "me want know about focus",
        ):
            res = sys.process_dialogue(utter)
            assert res.is_success is True, f"Failed on: {utter}"
            assert "focus" in res.response_text.lower()

        res_list = sys.process_dialogue("list me the files please")
        assert res_list.is_success is True
        assert len(res_list.response_text) > 0

        res_why = sys.process_dialogue("why focus is happen?")
        assert res_why.is_success is True
        assert "focus" in res_why.response_text.lower()

        res_wh = sys.process_dialogue("where red ball is?")
        assert res_wh.is_success is True
        assert "table" in res_wh.response_text.lower()

        # Telegraphic missing-copula assertions on fresh entities should auto-ground cleanly
        for tele in ("ball red on table", "the red ball on the table", "cup green on desk"):
            r_tele = sys.process_dialogue(tele)
            assert r_tele.is_success is True, f"Failed telegraphic: {tele}"
        r_cup_q = sys.process_dialogue("Where is the green cup?")
        assert r_cup_q.is_success is True
        assert "desk" in r_cup_q.response_text.lower()

    def test_epistemic_safety_guards_preserved(self) -> None:
        # 1. Word-salad must remain UNRESOLVED_LANGUAGE
        parser = EnglishParser()
        frame = parser.parse("Table ball on the is quickly blue.")
        assert frame.frame_type == FrameType.ERROR
        assert frame.error_type == LanguageErrorType.UNRESOLVED_LANGUAGE

        # 2. Ambiguous definite references must remain AMBIGUOUS_REFERENCE
        graph = CognitiveGraph()
        graph.add_node(PhysicalEntityNode(entity_type="ball", properties={"color": "red"}))
        graph.add_node(PhysicalEntityNode(entity_type="ball", properties={"color": "red"}))
        rt = MultilingualLanguageRuntime(graph)
        res = rt.process_utterance("Where is the red ball?")
        assert res.is_success is False
        assert res.error_type == LanguageErrorType.AMBIGUOUS_REFERENCE
        assert res.grounded_frame is not None
        assert len(res.grounded_frame.candidate_entities[ThematicRole.THEME]) == 2
