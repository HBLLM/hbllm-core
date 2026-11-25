---
title: "API Reference — Language Runtime & Reading Comprehension"
description: "API reference for CognitiveSystem.process_dialogue, MultilingualLanguageRuntime, HCIRGateway, SemanticFrame, ExpositoryProseProcessor, ReadingCurriculumIngestor, BaseOrthographicNormalizer, and LexiconAcquisitionLoop."
---

# Language & Reading Comprehension API (`hbllm.hcir.language`)

**Package:** `hbllm.hcir.language`

---

## High-Level Entry Point: `CognitiveSystem.process_dialogue()`

**Module:** `hbllm.runtime.bootstrap.CognitiveSystem`

Processes assertions, queries, multi-sentence prose, and governed capability commands end-to-end without LLMs:

```python
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

system = build_cognitive_system(
    CognitiveSystemConfig(
        world_id="dialogue_demo",
        memory_dir=None,  # In-memory LexicalOntologyStore + workspace
        enable_grounding=False,
    )
)

# 1. Teach an assertion (committed as HCIR BeliefNode / EvidenceNode)
system.process_dialogue("Gravity pulls objects toward the Earth.")

# 2. Query knowledge with calibrated epistemic realization
reply = system.process_dialogue("What does gravity do?")
print(reply.response_text)
print(
    f"Confidence: {reply.epistemic_state.confidence:.2f}, Known: {reply.epistemic_state.is_known}"
)
```

---

## `MultilingualLanguageRuntime`

**Module:** `hbllm.hcir.language.runtime.MultilingualLanguageRuntime`

Supports English (`"en"`), Sinhala (`"si"`), and Tamil (`"ta"`), plus custom `LanguageProfile` plugins:

```python
from hbllm.hcir.graph import CognitiveGraph
from hbllm.hcir.language.runtime import MultilingualLanguageRuntime

graph = CognitiveGraph()
rt = MultilingualLanguageRuntime(graph=graph)

res = rt.process_utterance("The red ball is on the table.", language="en", speaker="user")
print(res.semantic_frame.frame_type, res.is_success, res.response_text)

# Multi-sentence discourse
turns = rt.process_discourse("A photon carries energy. What carries energy?", language="en")
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `process_utterance()` | `(text: str, language: str = 'en', speaker: str = 'human', *, advance_turn: bool = True) -> LanguageTurnResult` | Normalizes, parses, resolves coreference, commits/queries `HCIRGateway`, and realizes reply |
| `process_discourse()` | `(text: str, language: str = 'en', speaker: str = 'human') -> list[LanguageTurnResult]` | Processes multi-sentence discourse preserving cross-sentence coreference |
| `translate()` | `(text: str, source_lang: str = 'en', target_lang: str = 'si') -> str` | Translates via semantic interlingua (`SemanticFrame`) without neural seq2seq |
| `register_language()` / `register_profile()` | `...` | Adds new language parsers, realizers, or declarative `LanguageProfile`s |
| `supported_languages()` | `() -> list[str]` | Lists currently active ISO language codes (`['en', 'si', 'ta', ...]`) |

---

## `SemanticFrame` & `HCIRGateway`

**Modules:** `hbllm.hcir.language.core.semantic_frame`, `hbllm.hcir.language.core.gateway`

- **`FrameType`:** `ASSERTION`, `QUERY`, `COMMAND`, etc.
- **`ThematicRole`:** `AGENT`, `PATIENT`, `THEME`, `RECIPIENT`, `LOCATION`, `SOURCE`, `DESTINATION`, `INSTRUMENT`, `TIME`, `MANNER`, `CAUSE`, `CONDITION`, etc.
- **`HCIRGateway`:**
  - `process_assertion(grounded, speaker='human') -> EvidenceNode`
  - `process_query(grounded) -> CognitiveEpistemicState`
  - `process_command(grounded) -> GoalNode`

---

## `DiscourseSegmenter` & `ExpositoryProseProcessor`

**Module:** `hbllm.hcir.language.core.discourse`

```python
from hbllm.hcir.language.core.discourse import DiscourseSegmenter, ExpositoryProseProcessor

segments = DiscourseSegmenter.segment("Heat transfers energy. Energy drives work.")
processor = ExpositoryProseProcessor()
summary = processor.ingest_prose(
    "Heat transfers energy because molecules collide.",
    workspace=system.workspace,
    tiered_workspace=system.tiered_workspace,
)
print(summary.nodes_committed, summary.summary_text)
```

---

## `ReadingCurriculumIngestor` & `LexicalOntologyStore`

**Modules:** `hbllm.hcir.language.reading_curriculum`, `hbllm.hcir.stores.ontology_store`

```python
from hbllm.hcir.language.reading_curriculum import (
    ReadingCurriculumIngestor,
    commit_propositions,
    extract_propositions_from_text,
)
from hbllm.hcir.stores.ontology_store import LexicalOntologyStore

props = extract_propositions_from_text(
    "Photosynthesis converts light into chemical energy.",
    source_file="biology.md",
)
nodes_added, edges_added, hypernyms = commit_propositions(props, workspace=system.workspace)

store = LexicalOntologyStore(":memory:")
print(store.stats())
```

---

## `BaseOrthographicNormalizer` & `LexiconAcquisitionLoop`

**Modules:** `hbllm.hcir.language.core.normalizer`, `hbllm.hcir.language.acquisition.lexicon_acquisition_loop`

- **`BaseOrthographicNormalizer`:** SymSpell `_Delete1Index` + Damerau-Levenshtein correction (`normalize_token(token, is_sentence_start=True)`).
- **`LexiconAcquisitionLoop`:** Interactive symbol-to-word learning (`observe_scene`, `teach_ostensive`, `correct_mistake`, `understand`, `produce`).

