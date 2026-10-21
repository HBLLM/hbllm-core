"""Autonomous Language Dictionary & Grammar Foundation for Developmental Learning.

Provides an authoritative semantic dictionary store that grounds vocabulary directly
from dictionary entries, grammar primers, and reference literature. Eliminates the
need for developer-curated word heuristics by determining grammatical categories
(noun, verb, adjective, preposition), physical semantic roles, and definitions directly
from linguistic source material.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .types import LexicalCategory

logger = logging.getLogger(__name__)


class SemanticRole(str, Enum):
    """Semantic role of a lexical token within the physical/relational simulation."""

    CONTAINER = "CONTAINER"
    TOOL = "TOOL"
    BALL = "BALL"
    BLOCK = "BLOCK"
    PULL = "PULL"
    PUSH = "PUSH"
    ROLL = "ROLL"
    GRASP = "GRASP"
    NEAR = "NEAR"
    INSIDE = "INSIDE"
    NONE = ""


@dataclass
class DictionaryEntry:
    """An authoritative dictionary entry defining a word's syntax, semantics, and definition."""

    word: str
    category: LexicalCategory
    definition: str
    semantic_role: str = ""  # SemanticRole value
    parent_concept: str | None = None
    inherited_affordances: set[str] = field(default_factory=set)
    translations: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.95

    @property
    def is_container(self) -> bool:
        return (
            self.semantic_role == "CONTAINER"
            or "CONTAINER" in self.inherited_affordances
            or "HOLDS_INSIDE" in self.inherited_affordances
        )

    @property
    def is_tool(self) -> bool:
        return (
            self.semantic_role == "TOOL"
            or "TOOL" in self.inherited_affordances
            or "EXTENDS_REACH" in self.inherited_affordances
        )


class LanguageDictionary:
    """Comprehensive semantic lexicon and dictionary reference engine.

    Provides high-speed O(1) in-memory lexical lookup, morphological fallback
    for inflected forms, dictionary file ingestion, and on-demand translation.
    """

    _INSTANCE: LanguageDictionary | None = None

    def __init__(self) -> None:
        from .taxonomy import TaxonomyHierarchyEngine

        self.taxonomy = TaxonomyHierarchyEngine.get_instance()
        self.entries: dict[str, DictionaryEntry] = {}
        self._load_foundational_lexicon()

    @classmethod
    def get_instance(cls) -> LanguageDictionary:
        """Singleton accessor for shared dictionary store across pipelines."""
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def register_entry(
        self,
        word: str,
        category: LexicalCategory | str,
        definition: str,
        semantic_role: str = "",
        translations: dict[str, str] | None = None,
        parent_concept: str | None = None,
    ) -> DictionaryEntry:
        """Register or update an authoritative dictionary entry."""
        w_clean = word.strip().lower()
        if isinstance(category, str):
            category = self._parse_pos_tag(category)

        if hasattr(semantic_role, "value"):
            semantic_role = semantic_role.value
        elif isinstance(semantic_role, str) and semantic_role.startswith("SemanticRole."):
            semantic_role = semantic_role.replace("SemanticRole.", "")
        elif semantic_role is None:
            semantic_role = ""

        # Taxonomy linking and affordance inheritance
        if parent_concept and (
            w_clean not in self.taxonomy.nodes or not self.taxonomy.nodes[w_clean].parent_concept
        ):
            self.taxonomy.register_concept(name=w_clean, parent_concept=parent_concept)
        taxon_node = self.taxonomy.induce_is_a_relation(w_clean, definition)
        inherited_affords = self.taxonomy.resolve_inherited_affordances(w_clean)

        if not semantic_role:
            if "CONTAINER" in inherited_affords or "HOLDS_INSIDE" in inherited_affords:
                semantic_role = "CONTAINER"
            elif "TOOL" in inherited_affords or "EXTENDS_REACH" in inherited_affords:
                semantic_role = "TOOL"
            elif "ROLLABLE" in inherited_affords:
                semantic_role = "BALL"
            else:
                semantic_role = self._infer_semantic_role(w_clean, category, definition)

        entry = DictionaryEntry(
            word=w_clean,
            category=category,
            definition=definition.strip(),
            semantic_role=semantic_role,
            parent_concept=taxon_node.parent_concept if taxon_node else parent_concept,
            inherited_affordances=inherited_affords,
            translations=translations or {},
            confidence=0.98,
        )
        self.entries[w_clean] = entry
        return entry

    # Alias for convenience
    register_word = register_entry

    def lookup(self, word: str) -> DictionaryEntry | None:
        """Lookup word in the dictionary, applying morphological lemmatization if needed."""
        w_clean = word.strip().lower()
        if not w_clean:
            return None

        # 1. Exact match
        if w_clean in self.entries:
            return self.entries[w_clean]

        # 2. Morphological lemmatization fallbacks
        # Plural nouns: -ies -> -y, -es -> -, -s -> -
        if w_clean.endswith("ies") and len(w_clean) > 4:
            stem = w_clean[:-3] + "y"
            if stem in self.entries:
                return self.entries[stem]
        if w_clean.endswith("es") and len(w_clean) > 3:
            stem = w_clean[:-2]
            if stem in self.entries:
                return self.entries[stem]
            stem_e = w_clean[:-1]
            if stem_e in self.entries:
                return self.entries[stem_e]
        if w_clean.endswith("s") and len(w_clean) > 2 and not w_clean.endswith("ss"):
            stem = w_clean[:-1]
            if stem in self.entries:
                return self.entries[stem]

        # Past tense / participle verbs: -ed -> -
        if w_clean.endswith("ed") and len(w_clean) > 3:
            stem = w_clean[:-2]
            if stem in self.entries:
                return self.entries[stem]
            stem_e = w_clean[:-1]
            if stem_e in self.entries:
                return self.entries[stem_e]

        # Continuous verbs: -ing -> -
        if w_clean.endswith("ing") and len(w_clean) > 4:
            stem = w_clean[:-3]
            if stem in self.entries:
                return self.entries[stem]
            stem_e = w_clean[:-3] + "e"
            if stem_e in self.entries:
                return self.entries[stem_e]

        return None

    def translate(self, word: str, target_lang: str = "es") -> str | None:
        """Lookup translation of a word in target language."""
        entry = self.lookup(word)
        if entry and target_lang in entry.translations:
            return entry.translations[target_lang]
        return None

    def load_dictionary_file(self, file_path: Path | str) -> int:
        """Ingest external dictionary file (TSV/CSV or formatted definitions)."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Dictionary file not found: {path}")
            return 0

        count = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    word, pos_str, defn = parts[0], parts[1], parts[2]
                    role = parts[3] if len(parts) > 3 else ""
                    translations = {}
                    if len(parts) > 4 and parts[4]:
                        try:
                            translations = json.loads(parts[4])
                        except Exception:
                            translations = {}
                    self.register_entry(
                        word, pos_str, defn, semantic_role=role, translations=translations
                    )
                    count += 1
        logger.info(f"Loaded {count} entries from dictionary file {path.name}.")
        return count

    def export_to_tsv(self, file_path: Path | str) -> None:
        """Export dictionary entries to a standardized TSV file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("#word\tcategory\tdefinition\tsemantic_role\ttranslations\n")
            for entry in self.entries.values():
                trans_str = json.dumps(entry.translations) if entry.translations else ""
                cat_val = (
                    entry.category.value
                    if hasattr(entry.category, "value")
                    else str(entry.category)
                )
                role_val = (
                    entry.semantic_role.value
                    if hasattr(entry.semantic_role, "value")
                    else str(entry.semantic_role)
                )
                if role_val.startswith("SemanticRole."):
                    role_val = role_val.replace("SemanticRole.", "")
                f.write(f"{entry.word}\t{cat_val}\t{entry.definition}\t{role_val}\t{trans_str}\n")

    def load_from_tsv(self, file_path: Path | str) -> int:
        """Load dictionary entries from TSV file."""
        return self.load_dictionary_file(file_path)

    @staticmethod
    def _parse_pos_tag(pos_str: str) -> LexicalCategory:
        """Map standard dictionary POS tag abbreviations to LexicalCategory."""
        p = pos_str.strip().lower()
        if p in ("n", "n.", "noun", "nouns"):
            return LexicalCategory.NOUN
        if p in ("v", "v.", "verb", "verbs", "vb", "vt", "vi"):
            return LexicalCategory.VERB
        if p in ("adj", "adj.", "adjective", "adjectives", "a."):
            return LexicalCategory.ADJECTIVE
        if p in ("prep", "prep.", "preposition", "prepositions"):
            return LexicalCategory.PREPOSITION
        return LexicalCategory.ADJECTIVE

    @staticmethod
    def _infer_semantic_role(word: str, category: LexicalCategory, definition: str) -> str:
        """Infer functional physical semantic role directly from dictionary definition text."""
        def_low = definition.lower()
        word_low = word.lower()

        if category == LexicalCategory.NOUN:
            if any(
                k in def_low or k in word_low
                for k in (
                    "container",
                    "box",
                    "receptacle",
                    "vessel",
                    "enclosure",
                    "bin",
                    "chamber",
                    "hopper",
                    "cavity",
                    "compartment",
                    "vault",
                    "crate",
                    "chest",
                    "basket",
                    "storage",
                    "store",
                )
            ):
                return "CONTAINER"
            if any(
                k in def_low or k in word_low
                for k in (
                    "tool",
                    "lever",
                    "instrument",
                    "implement",
                    "stick",
                    "rod",
                    "bar",
                    "handle",
                    "device",
                    "machine",
                )
            ):
                return "TOOL"
            return "BLOCK"

        if category == LexicalCategory.VERB:
            if any(
                k in word_low or k in def_low for k in ("pull", "drag", "haul", "draw", "attract")
            ):
                return "PULL"
            if any(
                k in word_low or k in def_low
                for k in ("push", "press", "shove", "thrust", "propel", "accelerate", "repel")
            ):
                return "PUSH"
            if any(k in word_low or k in def_low for k in ("roll", "rotate", "spin", "tumble")):
                return "ROLL"
            if any(
                k in word_low or k in def_low for k in ("grasp", "hold", "grip", "clutch", "seize")
            ):
                return "GRASP"
            return "PUSH"

        if category == LexicalCategory.PREPOSITION:
            if any(
                k in word_low
                for k in ("near", "between", "against", "beside", "adjacent", "by", "around")
            ):
                return "NEAR"
            return "INSIDE"

        return ""

    def _load_foundational_lexicon(self) -> None:
        """Bootstrap authoritative core English lexicon dynamically from curriculum data."""
        tsv_path = Path(__file__).parent / "curriculum_data" / "foundational_lexicon.tsv"
        if tsv_path.exists():
            try:
                with open(tsv_path, encoding="utf-8") as f:
                    lines = f.read().strip().splitlines()
                if lines:
                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            w = parts[0].strip()
                            cat = parts[1].strip()
                            d = parts[2].strip()
                            s_role = parts[3].strip() if len(parts) > 3 else ""
                            p_concept = parts[4].strip() if len(parts) > 4 else None
                            self.register_entry(
                                word=w,
                                category=cat,
                                definition=d,
                                semantic_role=s_role,
                                parent_concept=p_concept or None,
                            )
                return
            except Exception as e:
                logger.warning(f"Failed to load foundational lexicon from TSV: {e}")

        # Minimal bootstrapping fallback if TSV missing
        for w, d, r in [
            ("box", "A rigid container or receptacle.", "CONTAINER"),
            ("stick", "A slender piece of rigid wood used to extend reach.", "TOOL"),
            ("ball", "A spherical physical body capable of rolling across surfaces.", "BALL"),
            ("block", "A large solid piece of hard material with flat surfaces.", "BLOCK"),
        ]:
            self.register_entry(w, LexicalCategory.NOUN, d, semantic_role=r)
