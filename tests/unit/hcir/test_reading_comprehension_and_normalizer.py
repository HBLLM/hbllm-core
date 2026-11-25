"""Regression tests for reading comprehension, SymSpell normalizer, stemmer, command routing, and Gutenberg offline reporting."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from hbllm.hcir.language.english.lexicon import EnglishLexicon, EnglishPOS
from hbllm.hcir.language.english.morphology import EnglishMorphology, singularize_noun
from hbllm.hcir.language.english.normalizer import OrthographicNormalizer
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.language.reading_curriculum import extract_propositions_from_text
from hbllm.hcir.stores.ontology_store import LexicalOntologyStore
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_symspell_normalizer_speed_and_precision_guards(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    lex = EnglishLexicon(ontology_store=store)
    norm = OrthographicNormalizer(lexicon=lex, ontology_store=store)

    # Populate 12,000 synthetic vocabulary words to simulate post-book training scale
    large_vocab = [f"lexword{i:05d}" for i in range(12_000)]
    norm.set_dynamic_vocabulary(large_vocab)

    # 1. Sub-20ms O(L) lookup across 100 tokens even at 12k vocab scale
    probe_tokens = [
        "plants",
        "leaves",
        "flood",
        "Mira",
        "drell",
        "explian",
        "blal",
        "zint",
    ] * 12 + [
        "focsu",
        "tabel",
        "lisst",
        "wher",
    ]
    t0 = time.perf_counter()
    results = [
        norm.normalize_token(tok, is_sentence_start=(i == 0)) for i, tok in enumerate(probe_tokens)
    ]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 25.0, f"Fuzzy normalizer too slow at 12k vocab: {elapsed_ms:.2f}ms"
    assert len(results) == len(probe_tokens)

    # 2. Never falsely "correct" proper names, high-frequency words, or valid stems
    assert norm.normalize_token("Mira", is_sentence_start=False) == "Mira"
    assert norm.normalize_token("Mira", is_sentence_start=True) == "Mira"
    assert norm.normalize_token("Flood", is_sentence_start=True) == "Flood"
    assert norm.normalize_token("flood", is_sentence_start=False) == "flood"
    assert norm.normalize_token("plants", is_sentence_start=False) == "plants"
    assert norm.normalize_token("zint", is_sentence_start=False) == "zint"
    assert norm.normalize_token("drell", is_sentence_start=False) == "drell"

    # Frequency threshold protects repeated unknown domain tokens
    norm.set_word_frequencies({"glorp": 5})
    assert norm.normalize_token("glorp", is_sentence_start=False) == "glorp"

    # 3. Genuine typos still recover accurately
    assert norm.normalize_token("explian", is_sentence_start=True) == "explain"
    assert norm.normalize_token("explane", is_sentence_start=True) == "explain"
    assert norm.normalize_token("lisst", is_sentence_start=True) == "list"
    assert norm.normalize_token("blal", is_sentence_start=False) == "ball"
    assert norm.normalize_token("focsu", is_sentence_start=False) == "focus"
    assert norm.normalize_token("tabel", is_sentence_start=False) == "table"
    store.close()


def test_stemmer_and_morphology_preserve_stems_and_hyphen_adjs() -> None:
    assert singularize_noun("plants") == "plant"
    assert singularize_noun("leaves") == "leaf"
    assert singularize_noun("photosynthesis") == "photosynthesis"
    assert singularize_noun("species") == "species"
    assert singularize_noun("boxes") == "box"
    assert singularize_noun("berries") == "berry"

    morph = EnglishMorphology(EnglishLexicon())
    assert morph.analyze("plants", is_sentence_start=True).lemma == "plant"
    assert morph.analyze("leaves", is_sentence_start=False).lemma == "leaf"
    assert morph.analyze("photosynthesis", is_sentence_start=True).lemma == "photosynthesis"
    assert morph.analyze("flood", is_sentence_start=False).lemma == "flood"
    warm = morph.analyze("warm-blooded", is_sentence_start=False)
    assert warm.pos == EnglishPOS.ADJ
    assert warm.lemma == "warm-blooded"


def test_no_command_misrouting_and_zero_dict_or_enum_leaks() -> None:
    sys = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_no_leak",
            memory_dir=None,
            enable_grounding=False,
            enable_brain_llm=False,
            enable_autonomous_curiosity=False,
        )
    )
    para = (
        "Plants absorb sunlight through their leaves. "
        "Photosynthesis converts sunlight and water into glucose and releases oxygen."
    )
    turn = sys.process_dialogue(para)
    assert turn.is_success is True
    assert turn.semantic_frame.frame_type.value == "assertion"
    assert "HCIRNodeType" not in turn.response_text
    assert "planning." not in turn.response_text
    assert "{'status'" not in turn.response_text

    # Direct planning command must also format cleanly without raw dict/enum output
    plan_turn = sys.process_dialogue("Plan the route.")
    assert plan_turn.is_success is True
    assert "HCIRNodeType" not in plan_turn.response_text
    assert "'candidates'" not in plan_turn.response_text


def test_parser_driven_svo_polarity_and_modifiers_extraction() -> None:
    parser = EnglishParser()
    props = extract_propositions_from_text(
        "Florbs are small aquatic creatures. Florbs live in caves. "
        "Florbs do not eat meat. Gravity is a force that attracts objects with mass toward each other.",
        include_extended=True,
        parser=parser,
    )
    triples = {(p.subject, p.relation, p.obj, p.polarity) for p in props}
    assert ("florb", "is_a", "creature", True) in triples
    assert ("florb", "live", "cave", True) in triples
    assert ("florb", "not_eat", "meat", False) in triples
    assert ("gravity", "is_a", "force", True) in triples
    assert ("gravity", "attract", "object", True) in triples
    assert all(p.obj != "other" for p in props)

    florb_isa = next(p for p in props if p.subject == "florb" and p.relation == "is_a")
    assert ("adj", "aquatic") in florb_isa.modifiers
    florb_live = next(p for p in props if p.subject == "florb" and p.relation == "live")
    assert ("in", "cave") in florb_live.modifiers


def test_gutenberg_offline_fallback_reports_loudly(tmp_path: Path) -> None:
    from plugins.developmental_adapter.gutenberg_streamer import (
        GutenbergBookMetadata,
        GutenbergCorpusStreamer,
        KnowledgeGainTracker,
    )
    from plugins.developmental_adapter.raw_book_pipeline import AutomatedCurriculumCompiler
    from plugins.developmental_adapter.trainer import (
        ContinuousTextbookSchoolTrainer,
        TextbookSchoolConfig,
    )

    cache_dir = tmp_path / "empty_cache"
    ckpt_dir = tmp_path / "ckpts"
    meta = GutenbergBookMetadata(book_id=14474, title="Faraday Candle", author="Faraday")

    with patch(
        "plugins.developmental_adapter.raw_book_pipeline.RawBookDownloader.download_gutenberg",
        side_effect=OSError("[Errno -2] Name or service not known"),
    ):
        streamer = GutenbergCorpusStreamer(cache_dir=cache_dir, enable_prefetch=False)
        raw_text, source_mode, fetch_err = streamer.fetch_book_with_status(meta)
        streamer.close()

    assert source_mode == "offline_fallback"
    assert "Name or service not known" in fetch_err

    trainer = ContinuousTextbookSchoolTrainer(
        TextbookSchoolConfig(
            student_name="OfflineTest", checkpoint_dir=ckpt_dir, episodes_per_chapter=1
        )
    )
    from plugins.developmental_adapter.raw_book_pipeline import BookCurriculumItem, BookSourceType

    chap = AutomatedCurriculumCompiler.compile_chapter(
        raw_text,
        BookCurriculumItem(
            item_id="gutenberg_14474",
            source_type=BookSourceType.GUTENBERG,
            identifier="14474",
            title=meta.title,
            domain="chemistry",
        ),
    )
    trainer.train(chapters=[chap])
    tracker = KnowledgeGainTracker()
    tracker.record_progress(
        trainer, meta, len(raw_text), source_mode=source_mode, fetch_error=fetch_err
    )
    report = tracker.render_knowledge_report()
    assert "OFFLINE, no text fetched." in report
    assert "OFFLINE STUB" in report
