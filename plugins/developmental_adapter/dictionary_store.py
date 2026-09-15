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
        if parent_concept:
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
        """Bootstrap authoritative core English lexicon spanning foundational domains."""
        # 1. Foundational Containers (NOUN -> CONTAINER)
        containers = [
            (
                "box",
                "A rigid container or receptacle, typically rectangular, with a lid or open top.",
            ),
            ("container", "An object that can be used to hold or transport something."),
            (
                "vessel",
                "A hollow container, especially one used to hold liquid or granular matter.",
            ),
            ("receptacle", "An object or space used to contain something."),
            ("chamber", "An enclosed physical space or compartment in an apparatus."),
            ("hopper", "A funnel-shaped container for delivering bulk materials into a machine."),
            ("enclosure", "An area or receptacle sealed off with a barrier."),
            ("bin", "A box or receptacle for storing materials."),
            ("crate", "A slatted wooden case or box used for transporting or storing goods."),
            ("chest", "A large strong box, typically made of wood, used for storage."),
            ("basket", "A container used to hold or carry things."),
            ("vault", "A secure room or compartment used to store and protect items."),
        ]
        for w, d in containers:
            self.register_entry(w, LexicalCategory.NOUN, d, semantic_role="CONTAINER")

        # 2. Foundational Tools (NOUN -> TOOL)
        tools = [
            ("stick", "A slender piece of rigid wood or material used to extend reach."),
            ("tool", "A device or implement used to carry out a particular physical function."),
            (
                "lever",
                "A rigid bar resting on a pivot, used to help move a heavy or firmly fixed load.",
            ),
            ("rod", "A thin straight bar, especially of wood or metal."),
            ("bar", "A rigid piece of metal or wood, broader than a rod, used as a lever."),
            ("handle", "The part by which an object is held, carried, or manipulated."),
            ("instrument", "A tool or implement, especially one for delicate or scientific work."),
            (
                "implement",
                "A tool, utensil, or other piece of equipment, used for a practical purpose.",
            ),
        ]
        for w, d in tools:
            self.register_entry(w, LexicalCategory.NOUN, d, semantic_role="TOOL")

        # 3. Foundational Spheres & Geometric Objects (NOUN -> BLOCK / BALL)
        spheres = [
            ("ball", "A spherical physical body capable of rolling across surfaces."),
            (
                "sphere",
                "A round solid figure in which every point on the surface is equidistant from center.",
            ),
            ("globe", "A spherical body or celestial object."),
            ("orb", "A spherical body or globe."),
        ]
        for w, d in spheres:
            self.register_entry(w, LexicalCategory.NOUN, d, semantic_role="BALL")

        # 4. Standard Physical Objects & Entities (NOUN -> BLOCK)
        blocks_and_entities = [
            ("block", "A large solid piece of hard material with flat surfaces on each side."),
            (
                "cube",
                "A symmetrical three-dimensional shape, either solid or hollow, contained by six equal squares.",
            ),
            (
                "prism",
                "A solid geometric figure whose two end faces are similar, equal, and parallel rectilinear figures.",
            ),
            (
                "lens",
                "A piece of glass or other transparent substance with curved sides for concentrating or dispersing light.",
            ),
            (
                "mirror",
                "A reflective surface, now typically of glass coated with a metal amalgam, that reflects a clear image.",
            ),
            (
                "candle",
                "A cylinder of wax with a central wick that is lit to produce light as it burns.",
            ),
            (
                "wax",
                "A sticky, yellowish, moldable substance secreted by bees or synthesized from petroleum.",
            ),
            (
                "matter",
                "Physical substance in general, as distinct from mind and spirit; that which occupies space.",
            ),
            (
                "mass",
                "A coherent, typically large body of matter with no definite shape, or quantity of matter in a body.",
            ),
            ("lead", "A heavy, bluish-gray, soft, ductile metal."),
            ("iron", "A strong, hard magnetic silvery-gray metal."),
            (
                "copper",
                "A red-brown metal, the chemical element of atomic number 29; an excellent electrical conductor.",
            ),
            ("gold", "A yellow precious metal, the chemical element of atomic number 79."),
            (
                "atom",
                "The basic unit of a chemical element, consisting of a nucleus surrounded by electrons.",
            ),
            (
                "molecule",
                "A group of atoms bonded together, representing the smallest fundamental unit of a chemical compound.",
            ),
            (
                "cell",
                "The smallest structural and functional unit of an organism, typically microscopic.",
            ),
            (
                "membrane",
                "A pliable sheetlike structure acting as a boundary, lining, or partition in an organism.",
            ),
            ("organism", "An individual animal, plant, or single-celled life form."),
            (
                "plant",
                "A living organism of the kind exemplified by trees, shrubs, herbs, grasses, ferns, and mosses.",
            ),
            (
                "animal",
                "A living organism that feeds on organic matter, typically having specialized sense organs.",
            ),
            (
                "species",
                "A group of living organisms consisting of similar individuals capable of exchanging genes.",
            ),
            (
                "star",
                "A fixed luminous point in the night sky that is a large, remote incandescent body.",
            ),
            ("planet", "A celestial body moving in an elliptical orbit around a star."),
            ("moon", "The natural satellite of the earth, or celestial body orbiting a planet."),
            (
                "orbit",
                "The curved path of a celestial object or spacecraft around a star, planet, or moon.",
            ),
            (
                "fluid",
                "A substance that has no fixed shape and yields easily to external pressure; a gas or a liquid.",
            ),
            ("solid", "Firm and stable in shape; not liquid or fluid."),
            (
                "gas",
                "An air-like fluid substance which expands freely to fill any space available.",
            ),
            (
                "liquid",
                "A substance that flows freely but is of constant volume, having a consistency like that of water.",
            ),
            ("ray", "A narrow beam of light or other radiation."),
            (
                "beam",
                "A ray or shaft of light or other radiation, or a long, sturdy piece of squared timber or metal.",
            ),
            ("wave", "A periodic disturbance that transfers energy through space or a medium."),
            (
                "charge",
                "A physical property of matter that causes it to experience a force when placed in an electromagnetic field.",
            ),
            (
                "field",
                "A region in which each point has a physical quantity such as force or electric potential.",
            ),
            ("engine", "A machine with moving parts that converts power into motion."),
            (
                "acid",
                "A chemical substance that neutralizes alkalis, dissolves some metals, and turns litmus red.",
            ),
            ("substance", "A particular kind of matter with uniform properties."),
            (
                "solution",
                "A liquid mixture in which the minor component is uniformly distributed within the major component.",
            ),
            (
                "tissue",
                "Any of the distinct types of material of which animals or plants are made.",
            ),
            (
                "nucleus",
                "The central and most important part of an object, movement, or group, forming the basis for its activity.",
            ),
            ("core", "The central or innermost part of an object or celestial body."),
            ("solute", "The minor component in a solution, dissolved in the solvent."),
            (
                "solvent",
                "Having assets in excess of liabilities; or the liquid in which a solute is dissolved.",
            ),
            (
                "magnet",
                "A piece of iron or other material which has its component atoms so ordered that that material exhibits magnetism.",
            ),
            (
                "piston",
                "A disc or short cylinder fitting closely within a tube in which it moves up and down against a fluid.",
            ),
            (
                "wheel",
                "A circular object that revolves on an axle and is fixed below a vehicle or other object to enable it to move easily.",
            ),
            (
                "gravity",
                "The force that attracts a body toward the center of the earth, or toward any other physical body having mass.",
            ),
            (
                "inertia",
                "A property of matter by which it continues in its existing state of rest or uniform motion in a straight line unless changed by an external force.",
            ),
            (
                "friction",
                "The resistance that one surface or object encounters when moving over another.",
            ),
            (
                "force",
                "Strength or energy as an attribute of physical action or movement; an influence that changes the motion of a body.",
            ),
            ("velocity", "The speed of something in a given direction."),
            ("acceleration", "The rate of change of velocity per unit of time."),
            ("temperature", "The degree or intensity of heat present in a substance or object."),
            (
                "heat",
                "Energy that is transferred from one body to another as the result of a difference in temperature.",
            ),
            (
                "energy",
                "The property of matter and radiation that is manifest as a capacity to perform work.",
            ),
            (
                "work",
                "Activity involving mental or physical effort done in order to achieve a purpose; force times displacement.",
            ),
            (
                "power",
                "The rate of doing work or transferring energy, or the capacity or ability to direct or influence the behavior of others.",
            ),
            ("equilibrium", "A state in which opposing forces or influences are balanced."),
            (
                "spectrum",
                "A band of colors, as seen in a rainbow, produced by separation of the components of light by their different degrees of refraction.",
            ),
            (
                "dispersion",
                "The action or process of distributing things or people over a wide area; or separation of light into color components.",
            ),
            (
                "focus",
                "The center of interest or activity; or the point at which rays or waves meet after reflection or refraction.",
            ),
            (
                "trajectory",
                "The curved path followed by a projectile flying or an object moving under the action of given forces.",
            ),
            (
                "satellite",
                "An artificial body placed in orbit round the earth or moon or another planet in order to collect information or for communication.",
            ),
        ]
        for w, d in blocks_and_entities:
            self.register_entry(w, LexicalCategory.NOUN, d, semantic_role="BLOCK")

        # 5. Grammatical -ing Nouns (NOUN -> BLOCK)
        ing_nouns = [
            ("morning", "The period of time between midnight or sunrise and noon."),
            (
                "evening",
                "The period of time at the end of the day, usually from about 6 p.m. to bedtime.",
            ),
            (
                "spring",
                "The season after winter and before summer; or a resilient elastic coil storing mechanical energy.",
            ),
            (
                "building",
                "A structure with a roof and walls, such as a house, school, store, or factory.",
            ),
            ("ceiling", "The upper interior surface of a room or other similar compartment."),
            (
                "lightning",
                "The occurrence of a natural electrical discharge of very short duration and high voltage between a cloud and the ground or within a cloud.",
            ),
            (
                "string",
                "Material consisting of threads of cotton, hemp, or other material twisted together to form a thin length.",
            ),
            ("meaning", "What is meant by a word, text, concept, or action."),
            ("feeling", "An emotional state or reaction, or the perception of tactile sensations."),
            (
                "painting",
                "The action or skill of using paint, or a picture or design executed in paints.",
            ),
            (
                "drawing",
                "A picture or diagram made with a pencil, pen, or crayon rather than paint.",
            ),
            ("clothing", "Clothes collectively; items worn on the body."),
            ("offspring", "A person's child or children, or an animal's young."),
            ("pudding", "A cooked sweet dish, typically served hot as a dessert."),
            (
                "sibling",
                "Each of two or more children or offspring having one or both parents in common.",
            ),
            ("shilling", "A former British coin and monetary unit equal to twelve pence."),
            ("darling", "A dear or beloved person."),
            ("thing", "An object to which one need not or cannot give a specific name."),
            ("something", "A thing that is unspecified or unknown."),
            ("anything", "A thing of any kind; used to refer to a thing, no matter what."),
            ("nothing", "Not anything; no single thing."),
            ("everything", "All things; all the things of a group or class."),
            (
                "wing",
                "A modified forelimb for flight, as of a bird or bat, or an airfoil that develops lift.",
            ),
            (
                "ring",
                "A small circular band, typically of precious metal and often set with one or more gemstones.",
            ),
            (
                "king",
                "The male ruler of an independent state, especially one who inherits the position by right of birth.",
            ),
            ("being", "Existence; or a real or imaginary living creature or entity."),
            ("living", "An income sufficient to live on, or the state of being alive."),
        ]
        for w, d in ing_nouns:
            self.register_entry(w, LexicalCategory.NOUN, d, semantic_role="BLOCK")

        # 6. Action Verbs (VERB -> PULL, PUSH, ROLL, GRASP)
        pull_verbs = [
            (
                "pull",
                "Exert force upon an object so as to cause it to move toward oneself or the origin of the force.",
            ),
            ("drag", "Pull an object along forcefully, roughly, or with difficulty."),
            ("haul", "Pull or drag with effort or force."),
            (
                "draw",
                "Produce a picture or diagram; or pull or drag a load or object toward oneself.",
            ),
            (
                "attract",
                "Exert a force on an object that tends to pull it toward the attracting body.",
            ),
        ]
        for w, d in pull_verbs:
            self.register_entry(w, LexicalCategory.VERB, d, semantic_role="PULL")

        push_verbs = [
            (
                "push",
                "Exert force on someone or something in order to move them away from oneself or origin.",
            ),
            ("press", "Apply continuous physical force against something."),
            ("shove", "Push someone or something roughly or with force."),
            ("thrust", "Push suddenly or violently in a specified direction."),
            ("propel", "Drive or push something forwards."),
            ("accelerate", "Increase velocity or rate of motion over time."),
            ("repel", "Drive or force an object back or away."),
            ("move", "Go in a specified direction or manner; change position."),
            ("lift", "Raise to a higher position or level."),
            ("drop", "Let or make something fall vertically."),
            ("transfer", "Move from one place to another."),
            ("conduct", "Transmit heat, electricity, or other energy through a substance."),
            ("diffuse", "Spread over a wide area or between substances through random motion."),
            ("flow", "Move steadily and continuously in a current or stream."),
            ("combust", "Consume or be consumed by fire or burn chemically."),
            ("ignite", "Catch fire or cause to catch fire."),
            ("absorb", "Take in or soak up energy, liquid, or other substances."),
            ("emit", "Produce and discharge radiation, sound, or gas."),
            ("radiate", "Emit energy, especially light or heat, in the form of rays or waves."),
            ("deflect", "Cause something to change direction by interposing an object."),
            (
                "displace",
                "Take over the place, position, or role of; or move from proper position.",
            ),
            ("expand", "Become or make larger or more extensive."),
            ("contract", "Decrease in size, number, or range."),
            ("heat", "Make or become hot or warm."),
            ("cool", "Become or make less hot."),
            ("melt", "Make or become liquefied by heat."),
            ("freeze", "Be turned into ice or another solid as a result of extreme cold."),
            ("vaporize", "Convert or be converted into vapor."),
            ("condense", "Make something denser or more concentrated; change from gas to liquid."),
            ("refract", "Make a ray of light change direction when it enters at an angle."),
            ("reflect", "Throw back heat, light, or sound without absorbing it."),
            ("dissolve", "Become incorporated into a liquid so as to form a solution."),
            ("precipitate", "Cause a substance to be deposited in solid form from a solution."),
            ("oscillate", "Move or swing back and forth at a regular speed."),
            ("vibrate", "Move or cause to move continuously and rapidly to and fro."),
            ("collide", "Hit with force when moving."),
            (
                "strike",
                "Hit forcibly and deliberately with one's hand or a weapon or other implement.",
            ),
        ]
        for w, d in push_verbs:
            self.register_entry(w, LexicalCategory.VERB, d, semantic_role="PUSH")

        roll_verbs = [
            ("roll", "Move in a particular direction by turning over and over on an axis."),
            ("rotate", "Move or cause to move in a circle around an axis or center."),
            ("spin", "Turn or cause to turn or whirl round rapidly."),
        ]
        for w, d in roll_verbs:
            self.register_entry(w, LexicalCategory.VERB, d, semantic_role="ROLL")

        grasp_verbs = [
            ("grasp", "Seize and hold firmly with the hand or fingers."),
            ("hold", "Grasp, carry, or support with one's hands or in one's arms."),
            ("grip", "Take and keep a firm hold of."),
            ("seize", "Take hold of suddenly and forcibly."),
            ("release", "Allow or enable to escape from confinement; set free."),
            ("place", "Put in a particular position."),
            ("stop", "Come to an end; finish; prevent from happening or moving."),
        ]
        for w, d in grasp_verbs:
            self.register_entry(w, LexicalCategory.VERB, d, semantic_role="GRASP")

        # 7. Spatial Prepositions (PREPOSITION -> INSIDE, NEAR)
        near_preps = [
            ("near", "At or to a short distance away; nearby."),
            ("between", "At, into, or across the space separating two objects or points."),
            ("against", "In close proximity to and touching someone or something."),
            ("beside", "At the side of; next to."),
            ("by", "Identifying the agent performing an action; or near to; beside."),
            ("around", "Located or moving on every side; about."),
            ("along", "Moving in a constant direction on a more or less horizontal surface."),
            ("across", "From one side to the other of something with clear limits."),
        ]
        for w, d in near_preps:
            self.register_entry(w, LexicalCategory.PREPOSITION, d, semantic_role="NEAR")

        inside_preps = [
            ("inside", "The inner side or surface of something; within the bounds."),
            (
                "in",
                "Expressing the situation of something that is or appears to be enclosed or surrounded by something else.",
            ),
            (
                "into",
                "Expressing movement or action with the result that someone or something becomes enclosed or surrounded by something else.",
            ),
            ("within", "Inside something."),
            ("on", "Physically in contact with and supported by a surface."),
            ("upon", "More formal term for on, especially in abstract senses."),
            ("above", "At a higher level or layer than."),
            ("under", "Extending or directly below something."),
            ("below", "At a lower level or layer than."),
            (
                "through",
                "Moving in one side and out of the other side of an opening, channel, or location.",
            ),
            ("outside", "The external side or surface of something."),
            ("behind", "At the back of or in the wake of."),
        ]
        for w, d in inside_preps:
            self.register_entry(w, LexicalCategory.PREPOSITION, d, semantic_role="INSIDE")

        # 8. Physical Properties & Qualitative Descriptors (ADJECTIVE)
        adjectives = [
            (
                "red",
                "Of a color at the end of the spectrum next to orange, resembling that of blood.",
            ),
            (
                "blue",
                "Of a color intermediate between green and violet, as of the sky or sea on a sunny day.",
            ),
            ("green", "Of the color between blue and yellow, resembling growing foliage."),
            (
                "yellow",
                "Of the color between green and orange in the spectrum, a primary subtractive color.",
            ),
            ("heavy", "Of great weight; difficult to lift or move."),
            (
                "light",
                "Having little weight; not heavy; or the natural agent that stimulates sight and makes things visible.",
            ),
            ("hot", "Having a high degree of heat or a high temperature."),
            ("cold", "Of or at a low or relatively low temperature."),
            ("hard", "Solid, firm, and rigid; not easily broken, bent, or pierced."),
            ("soft", "Easy to mold, cut, compress, or fold; not hard or firm to the touch."),
            (
                "smooth",
                "Having an even and regular surface or consistency; free from perceptible projections, lumps, or indentations.",
            ),
            ("rough", "Having an uneven or irregular surface; not smooth or level."),
            ("opaque", "Not able to be seen through; not transparent."),
            (
                "transparent",
                "Allowing light to pass through so that objects behind can be distinctly seen.",
            ),
            ("dense", "Closely compacted in substance."),
            ("elastic", "Able to resume its normal shape after being stretched or compressed."),
            (
                "viscous",
                "Having a thick, sticky consistency between solid and liquid; having a high viscosity.",
            ),
            ("magnetic", "Exhibiting or relating to magnetism."),
            ("electric", "Of, worked by, or producing electricity."),
            (
                "optical",
                "Operating in or relating to the visible part of the spectrum; relating to sight.",
            ),
            (
                "chemical",
                "Relating to chemistry, or the interactions of substances as studied in chemistry.",
            ),
            ("kinetic", "Relating to or resulting from motion."),
            (
                "potential",
                "Latent qualities or abilities that may be developed; or energy held by an object because of its position.",
            ),
        ]
        for w, d in adjectives:
            self.register_entry(w, LexicalCategory.ADJECTIVE, d, semantic_role="")
