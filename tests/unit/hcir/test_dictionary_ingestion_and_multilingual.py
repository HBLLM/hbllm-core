"""Hermetic unit tests for Track T1.5 dictionary ingestion and multilingual runtime wiring."""

from __future__ import annotations

import gzip
import json
import sqlite3
import zipfile
from pathlib import Path

from hbllm.hcir.capabilities.belief_revision import BeliefConflictResolver
from hbllm.hcir.capabilities.concept_explainer import ConceptExplainer
from hbllm.hcir.graph import BeliefNode, CognitiveGraph
from hbllm.hcir.language.dictionary_ingestors import (
    evaluate_dakshina_transliteration,
    ingest_bilingual_dictionaries,
    ingest_conceptnet_gz,
    ingest_dakshina_transliterations,
    ingest_frequency_list,
    ingest_gutenberg_enrichment,
    ingest_kaikki_multilingual,
    ingest_propbank_zip,
    ingest_verbnet_zip,
)
from hbllm.hcir.language.english.lexicon import EnglishLexicon
from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord
from hbllm.hcir.language.sinhala.lexicon import SinhalaLexicon
from hbllm.hcir.language.tamil.lexicon import TamilLexicon
from hbllm.hcir.stores.ontology_store import LexicalOntologyStore


def test_ontology_store_new_tables_and_caches(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    assert (
        store.upsert_semantic_relations(
            [
                ("en", "fire", "Causes", "heat", 4.0, "cn"),
                ("en", "fire", "HasPrerequisite", "oxygen", 3.5, "cn"),
                ("en", "hot", "Antonym", "cold", 5.0, "cn"),
            ]
        )
        == 3
    )
    rels = store.get_relations("fire", language="en")
    assert rels[0] == ("Causes", "heat", 4.0)
    assert store.are_antonyms("hot", "cold") is True
    assert store.are_antonyms("cold", "hot") is True
    assert store.are_antonyms("hot", "warm") is False

    store.upsert_concept_alignments([("si", "පොත", "book", 0.9, "osdb")])
    assert store.align_to_english("පොත", language="si") == "book"
    assert store.align_from_english("book", language="si") == ["පොත"]

    store.upsert_surface_forms([("en", "went", "go", "verb", '["past"]')])
    assert ("go", "verb") in store.lemmatize_surface("went", language="en")

    store.upsert_transliterations([("si", "watura", "වතුර", 12, "dakshina:train")])
    assert store.transliterate_romanized("watura", language="si") == ["වතුර"]

    st = store.stats()
    assert st["semantic_relations"] == 3
    assert st["concept_alignments"] == 1
    assert st["surface_forms"] == 1
    assert st["transliterations"] == 1
    store.close()


def test_frequency_and_frequency_guided_kaikki_and_lemmatization(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    freq_file = tmp_path / "en_full.txt"
    freq_file.write_text("zebra 5000\naardvark 2\n", encoding="utf-8")
    fmap, frep = ingest_frequency_list(freq_file, store, language="en")
    assert frep.entries_written == 2

    kaikki_file = tmp_path / "kaikki-en.jsonl"
    lines = [
        json.dumps(
            {
                "word": "zebra",
                "pos": "noun",
                "senses": [{"glosses": ["Striped equine."], "hypernyms": [{"word": "equine"}]}],
                "forms": [{"form": "zebras", "tags": ["plural"]}],
            }
        ),
        json.dumps(
            {
                "word": "aardvark",
                "pos": "noun",
                "senses": [{"glosses": ["Burrowing mammal."]}],
                "forms": [{"form": "aardvarks", "tags": ["plural"]}],
            }
        ),
    ]
    kaikki_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Even with limit=1 and frequency guidance over top word ("zebra"), it ingests zebra!
    krep = ingest_kaikki_multilingual(
        kaikki_file, store, language="en", limit=1, freq_map={"zebra": 5000}
    )
    assert krep.entries_written >= 1
    assert store.has_word("zebra", language="en")

    en_lex = EnglishLexicon(ontology_store=store)
    assert any(e.lemma == "zebra" for e in en_lex.lookup("zebras"))
    store.close()


def test_verbnet_propbank_and_conceptnet_streaming(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many([LexicalOntologyRecord("give", "verb", "en", "Transfer")], source="seed")

    vn_zip = tmp_path / "verbnet-master.zip"
    with zipfile.ZipFile(vn_zip, "w") as zf:
        zf.writestr(
            "verbnet-master/verbnet3.4/give-13.1.xml",
            "<VNCLASS><THEMROLES><THEMROLE type='Agent'/><THEMROLE type='Recipient'/></THEMROLES>"
            "<MEMBERS><MEMBER name='give'/></MEMBERS></VNCLASS>",
        )
    pb_zip = tmp_path / "propbank-frames-main.zip"
    with zipfile.ZipFile(pb_zip, "w") as zf:
        zf.writestr(
            "propbank-frames-main/frames/open.xml",
            "<frameset><predicate lemma='open'><roleset><roles>"
            "<role n='0' f='PAG' descr='opener'/><role n='1' f='PPT' descr='thing opened'/>"
            "</roles></roleset></predicate></frameset>",
        )

    assert ingest_verbnet_zip(vn_zip, store).entries_written >= 1
    assert ingest_propbank_zip(pb_zip, store).entries_written >= 1
    give_roles = store.lookup("give", language="en")[0].thematic_roles
    assert "Agent" in give_roles and "Recipient" in give_roles
    open_roles = store.lookup("open", language="en")[0].thematic_roles
    assert "Agent" in open_roles and "Patient" in open_roles

    cn_gz = tmp_path / "conceptnet.csv.gz"
    edges = [
        '/a/1\t/r/IsA\t/c/en/oak/n\t/c/en/tree/n\t{"weight": 3.0}',
        '/a/2\t/r/Causes\t/c/en/rain/n\t/c/en/puddles/n\t{"weight": 2.5}',
        '/a/3\t/r/UsedFor\t/c/en/rain/n\t/c/en/irrigation/n\t{"weight": 2.5}',
        '/a/4\t/r/Antonym\t/c/en/dry/a\t/c/en/wet/a\t{"weight": 4.0}',
        '/a/5\t/r/Synonym\t/c/si/වැස්ස/n\t/c/en/rain/n\t{"weight": 3.0}',
        '/a/6\t/r/Causes\t/c/en/noise/n\t/c/en/static/n\t{"weight": 0.5}',
    ]
    with gzip.open(cn_gz, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(edges) + "\n")

    cn_rep = ingest_conceptnet_gz(cn_gz, store, min_weight=2.0)
    assert cn_rep.relations_written == 3
    assert cn_rep.alignments_written == 1
    assert "tree" in store.get_hypernyms("oak")
    assert store.align_to_english("වැස්ස", language="si") == "rain"
    assert store.are_antonyms("dry", "wet") is True
    assert store.get_relations("noise") == []  # Filtered by min_weight

    # ConceptExplainer enrichment
    explainer = ConceptExplainer(ontology_store=store)
    snap = explainer.gather_knowledge("rain")
    assert "puddles" in snap.causes
    assert any("irrigation" in b for b in snap.beliefs)

    # BeliefConflictResolver antonym detection
    graph = CognitiveGraph()
    b_old = BeliefNode(id="b1", statement="The surface is dry", confidence=0.70)
    graph.add_node(b_old)
    resolver = BeliefConflictResolver(ontology_store=store)
    rev = resolver.evaluate_belief(
        BeliefNode(id="b2", statement="The surface is wet", confidence=0.90),
        graph,
    )
    assert rev.has_conflicts
    assert "Antonym conflict" in rev.conflicts[0].explanation
    store.close()


def test_bilingual_dakshina_gutenberg_and_si_ta_lexicons(tmp_path: Path) -> None:
    dict_root = tmp_path / "dictionaries"
    (dict_root / "sinhala").mkdir(parents=True)
    (dict_root / "tamil").mkdir(parents=True)
    (dict_root / "dakshina" / "si" / "lexicons").mkdir(parents=True)
    (dict_root / "dakshina" / "ta" / "lexicons").mkdir(parents=True)
    (dict_root / "gutenberg").mkdir(parents=True)

    (dict_root / "sinhala" / "osdb-sn2en.json").write_text(
        json.dumps([{"word": "ගස", "definitions": ["tree", "plant"]}]), encoding="utf-8"
    )
    (dict_root / "sinhala" / "En-Si-dict-filtered-V2.tsv").write_text(
        "mountain\tකන්ද\n", encoding="utf-8"
    )
    with zipfile.ZipFile(dict_root / "sinhala" / "SEDictionary-master.zip", "w") as zf:
        zf.writestr("SEDictionary-master/ENtoSI.json", json.dumps([{"river": "ගඟ"}]))

    with zipfile.ZipFile(dict_root / "tamil" / "eng2tamildictionary-master.zip", "w") as zf:
        zf.writestr(
            "eng2tamildictionary-master/dictionary.json",
            json.dumps([{"eng": "tree", "tamil": "n. மரம்"}]),
        )
    pt_db = tmp_path / "pt.db"
    conn = sqlite3.connect(pt_db)
    conn.execute("CREATE TABLE dictionary (word TEXT, meanings TEXT)")
    conn.execute("INSERT INTO dictionary VALUES ('மலை', 'உயர்ந்த நிலப்பகுதி')")
    conn.commit()
    conn.close()
    with zipfile.ZipFile(dict_root / "tamil" / "powertamil-dictionary-master.zip", "w") as zf:
        zf.write(pt_db, "powertamil-dictionary-master/PTTamil.db")
    with zipfile.ZipFile(dict_root / "tamil" / "rprabhu-TamilDictionary-master.zip", "w") as zf:
        zf.writestr("TamilDictionary-master/json/v01.json", json.dumps({"நீர்": "தண்ணீர்"}))
    with zipfile.ZipFile(dict_root / "tamil" / "Tamilinaiyam-Dictionary-master.zip", "w") as zf:
        zf.writestr(
            "Tamilinaiyam-Dictionary-master/DictionaryExe/koppu/aa.json",
            json.dumps({"கடல்": {"h": "சமுத்திரம்¦ocean"}}),
        )

    (dict_root / "dakshina" / "si" / "lexicons" / "si.translit.sampled.train.tsv").write_text(
        "ගස\tgasa\t10\n", encoding="utf-8"
    )
    (dict_root / "dakshina" / "si" / "lexicons" / "si.translit.sampled.test.tsv").write_text(
        "ගස\tgasa\t1\n", encoding="utf-8"
    )
    (dict_root / "dakshina" / "ta" / "lexicons" / "ta.translit.sampled.train.tsv").write_text(
        "மரம்\tmaram\t9\n", encoding="utf-8"
    )
    (dict_root / "dakshina" / "ta" / "lexicons" / "ta.translit.sampled.test.tsv").write_text(
        "மரம்\tmaram\t1\n", encoding="utf-8"
    )

    (dict_root / "gutenberg" / "websters-1913.json").write_text(
        json.dumps({"mountain": "A large natural elevation."}), encoding="utf-8"
    )
    (dict_root / "gutenberg" / "moby-thesaurus.txt").write_text(
        "mountain,peak,summit,ridge\n", encoding="utf-8"
    )

    store = LexicalOntologyStore(tmp_path / "ont.db")
    b_reports = ingest_bilingual_dictionaries(dict_root, store)
    assert len(b_reports) == 7
    d_rep = ingest_dakshina_transliterations(dict_root / "dakshina", store)
    assert d_rep.transliterations_written == 2
    assert (
        evaluate_dakshina_transliteration(dict_root / "dakshina", store, language="si")["top1"]
        == 1.0
    )
    g_reports = ingest_gutenberg_enrichment(dict_root / "gutenberg", store)
    assert len(g_reports) == 2
    assert "peak" in store.lookup("mountain")[0].synonyms

    si_lex = SinhalaLexicon(ontology_store=store)
    ta_lex = TamilLexicon(ontology_store=store)
    assert si_lex.lookup("ගස")[0].semantic_predicate == "tree"
    assert si_lex.lookup("gasa")[0].semantic_predicate == "tree"
    assert si_lex.lookup("කන්ද")[0].semantic_predicate == "mountain"
    assert ta_lex.lookup("மரம்")[0].semantic_predicate == "tree"
    assert ta_lex.lookup("maram")[0].semantic_predicate == "tree"
    assert ta_lex.lookup("கடல்")[0].semantic_predicate == "ocean"
    store.close()


def test_non_override_safeguards_preserve_curated_and_workspace_knowledge(tmp_path: Path) -> None:
    from hbllm.hcir.transactions import TransactionOp, TransactionOperation
    from hbllm.hcir.workspace_tiers import TieredWorkspace

    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord(
                lemma="kernel",
                pos="noun",
                language="en",
                definition="Authoritative HCIR state owner.",
                domain="cognitive_os",
            )
        ],
        source="core_ontology",
    )
    kaikki_file = tmp_path / "kaikki-en.jsonl"
    kaikki_file.write_text(
        json.dumps(
            {
                "word": "kernel",
                "pos": "noun",
                "senses": [{"glosses": ["Edible core of a nut or seed."]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ingest_kaikki_multilingual(kaikki_file, store, language="en", limit=10)
    curated = store.lookup("kernel", language="en")[0]
    assert curated.definition == "Authoritative HCIR state owner."
    assert curated.source == "core_ontology"
    assert curated.domain == "cognitive_os"

    # Workspace BeliefNodes must take priority over ConceptNet commonsense fallback
    store.upsert_semantic_relations(
        [
            ("en", "kernel", "UsedFor", "popcorn", 3.0, "cn"),
            ("en", "kernel", "HasProperty", "small", 3.0, "cn"),
            ("en", "kernel", "AtLocation", "cob", 3.0, "cn"),
        ]
    )
    tw = TieredWorkspace()
    b1 = BeliefNode(id="b1", statement="kernel enforces transaction verification", confidence=0.95)
    b2 = BeliefNode(id="b2", statement="kernel owns persistent graph state", confidence=0.95)
    tw.persistent.commit_operations(
        [
            TransactionOperation(op=TransactionOp.UPSERT_NODE, node=b1),
            TransactionOperation(op=TransactionOp.UPSERT_NODE, node=b2),
        ],
        author="test",
    )
    explainer = ConceptExplainer(tiered_workspace=tw, ontology_store=store)
    snap = explainer.gather_knowledge("kernel")
    assert snap.beliefs[0] == "kernel enforces transaction verification"
    assert "kernel owns persistent graph state" in snap.beliefs
    assert len(snap.beliefs) <= 3
    store.close()


def test_kaikki_skips_form_of_and_misspelling_stubs(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    kaikki_file = tmp_path / "kaikki-en.jsonl"
    lines = [
        json.dumps(
            {
                "word": "distribute",
                "pos": "verb",
                "senses": [{"glosses": ["To divide and give out in shares."]}],
            }
        ),
        json.dumps(
            {
                "word": "distributed",
                "pos": "verb",
                "senses": [
                    {
                        "glosses": ["simple past and past participle of distribute"],
                        "tags": ["form-of", "participle", "past"],
                        "form_of": [{"word": "distribute"}],
                    }
                ],
            }
        ),
        json.dumps(
            {
                "word": "teh",
                "pos": "article",
                "senses": [
                    {
                        "glosses": ["Misspelling of the."],
                        "tags": ["misspelling", "humorous"],
                    }
                ],
            }
        ),
    ]
    kaikki_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ingest_kaikki_multilingual(kaikki_file, store, language="en", limit=10)
    assert store.has_word("distribute", language="en") is True
    assert store.has_word("distributed", language="en") is False
    assert store.has_word("teh", language="en") is False
    assert ("distribute", "verb") in store.lemmatize_surface("distributed", language="en")
    store.close()


def test_common_typos_override_untrusted_entries_and_gateway_noise(tmp_path: Path) -> None:
    from hbllm.hcir.language.english.morphology import EnglishMorphology

    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord(
                "wher", "noun", "en", "wher is the red ball?", domain="t4_reading"
            ),
            LexicalOntologyRecord("teh", "noun", "en", "Abbreviation stub.", domain="wiktionary"),
        ],
        source="hcir_language_gateway",
    )
    assert store.is_trusted_word("wher", language="en") is False
    assert store.is_trusted_word("teh", language="en") is False

    lex = EnglishLexicon(ontology_store=store)
    morph = EnglishMorphology(lexicon=lex)
    tok_wher = morph.analyze("wher", is_sentence_start=True)
    tok_teh = morph.analyze("Teh", is_sentence_start=True)
    assert tok_wher.lemma == "where"
    assert tok_wher.features.get("corrected_from") == "wher"
    assert tok_teh.lemma == "the"
    assert tok_teh.features.get("corrected_from") == "teh"
    store.close()


def test_wordnet_pos_precedence_and_expository_leadin_not_imperative(tmp_path: Path) -> None:
    from hbllm.hcir.language.core.semantic_frame import FrameType
    from hbllm.hcir.language.english.parser import EnglishParser

    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [LexicalOntologyRecord("distributed", "a", "en", "spread out or scattered")],
        source="open_english_wordnet",
    )
    store.upsert_many(
        [LexicalOntologyRecord("distributed", "verb", "en", "simple past of distribute")],
        source="kaikki:kaikki-en.jsonl",
        preserve_existing=True,
    )
    entries = store.lookup("distributed", language="en")
    assert entries[0].source == "open_english_wordnet"
    assert entries[0].pos == "a"

    parser = EnglishParser(lexicon=EnglishLexicon(ontology_store=store))
    frame = parser.parse(
        "In modern computer science, distributed systems face many challenges "
        "such as network partitions, latency, and clock drift."
    )
    assert frame.frame_type != FrameType.COMMAND
    store.close()
