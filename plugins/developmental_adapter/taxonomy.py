"""Taxonomic Concept Hierarchy & Property Inheritance Engine (Milestone A24).

Extracts and maintains a directed acyclic taxonomic graph (is-a relations)
grounded in physical sensory categories. Propagates functional affordances,
spatial constraints, and physical properties top-down, allowing novel entities
(e.g., crucible, prybar) to inherit behavioral affordances zero-shot without
direct physical trial-and-error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .types import BabyObjectType

logger = logging.getLogger(__name__)


@dataclass
class TaxonNode:
    """A conceptual category node in the taxonomic inheritance hierarchy."""

    name: str
    parent_concept: str | None = None
    level: int = 0
    direct_affordances: set[str] = field(default_factory=set)
    inherited_affordances: set[str] = field(default_factory=set)
    canonical_properties: dict[str, Any] = field(default_factory=dict)
    exemplars: list[str] = field(default_factory=list)
    confidence: float = 0.95

    @property
    def all_affordances(self) -> set[str]:
        """Union of direct and inherited affordances."""
        return self.direct_affordances | self.inherited_affordances

    @property
    def is_container(self) -> bool:
        return "CONTAINER" in self.all_affordances or "HOLDS_INSIDE" in self.all_affordances

    @property
    def is_tool(self) -> bool:
        return "TOOL" in self.all_affordances or "EXTENDS_REACH" in self.all_affordances

    @property
    def is_rollable(self) -> bool:
        return "ROLLABLE" in self.all_affordances or bool(self.canonical_properties.get("rollable"))


class TaxonomyHierarchyEngine:
    """Maintains the taxonomic ontology and resolves zero-shot inheritance."""

    _INSTANCE: TaxonomyHierarchyEngine | None = None

    def __init__(self) -> None:
        self.nodes: dict[str, TaxonNode] = {}
        self._bootstrap_root_taxonomy()

    @classmethod
    def get_instance(cls) -> TaxonomyHierarchyEngine:
        """Singleton accessor for shared taxonomy store across pipelines."""
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def _bootstrap_root_taxonomy(self) -> None:
        """Seed core physical ontology rooted in BabyWorld sensory-motor primitives."""
        # Level 0: Root
        self.register_concept(
            name="physical_entity",
            parent_concept=None,
            level=0,
            direct_affordances={"EXISTENT"},
            properties={"is_physical": True},
        )

        # Level 1: Core functional categories
        self.register_concept(
            name="container",
            parent_concept="physical_entity",
            level=1,
            direct_affordances={"CONTAINER", "HOLDS_INSIDE", "GRASPABLE"},
            properties={"is_container": True, "open_boundary": True},
        )
        self.register_concept(
            name="tool",
            parent_concept="physical_entity",
            level=1,
            direct_affordances={"TOOL", "EXTENDS_REACH", "GRASPABLE", "SLIDABLE"},
            properties={"is_tool": True, "rigid": True},
        )
        self.register_concept(
            name="spherical_body",
            parent_concept="physical_entity",
            level=1,
            direct_affordances={"ROLLABLE", "GRASPABLE", "SLIDABLE"},
            properties={"shape": "ball", "rollable": True},
        )
        self.register_concept(
            name="block_body",
            parent_concept="physical_entity",
            level=1,
            direct_affordances={"GRASPABLE", "SLIDABLE"},
            properties={"shape": "block", "rollable": False},
        )
        self.register_concept(
            name="spatial_barrier",
            parent_concept="physical_entity",
            level=1,
            direct_affordances={"STATIC", "OBSTACLE"},
            properties={"immovable": True, "mass": 10.0},
        )

        # Level 2: Sub-categories
        # Receptacles
        self.register_concept(
            name="receptacle",
            parent_concept="container",
            level=2,
            direct_affordances={"ENCLOSURE"},
            properties={"is_container": True},
        )
        for c in (
            "box",
            "crate",
            "bin",
            "chest",
            "basket",
            "vault",
            "hopper",
            "vessel",
            "crucible",
        ):
            self.register_concept(
                name=c,
                parent_concept="receptacle",
                level=3,
                direct_affordances=set(),
                properties={"entity_type": BabyObjectType.BOX},
            )

        # Lever tools
        self.register_concept(
            name="lever_instrument",
            parent_concept="tool",
            level=2,
            direct_affordances={"LEVERAGE", "PULL_TARGET"},
            properties={"is_tool": True},
        )
        for t in ("stick", "rod", "bar", "lever", "handle", "prybar", "crowbar", "wrench"):
            self.register_concept(
                name=t,
                parent_concept="lever_instrument",
                level=3,
                direct_affordances=set(),
                properties={"entity_type": BabyObjectType.TOOL},
            )

        # Spheres
        for s in ("ball", "sphere", "orb", "globe", "marble"):
            self.register_concept(
                name=s,
                parent_concept="spherical_body",
                level=2,
                direct_affordances=set(),
                properties={"entity_type": BabyObjectType.BALL, "rollable": True},
            )

        # Blocks
        for b in ("block", "cube", "brick", "slab", "stone", "rock"):
            self.register_concept(
                name=b,
                parent_concept="block_body",
                level=2,
                direct_affordances=set(),
                properties={"entity_type": BabyObjectType.BLOCK, "rollable": False},
            )

    def register_concept(
        self,
        name: str,
        parent_concept: str | None = None,
        level: int = 1,
        direct_affordances: set[str] | None = None,
        properties: dict[str, Any] | None = None,
        confidence: float = 0.95,
    ) -> TaxonNode:
        """Register or update a concept in the taxonomy, automatically inheriting ancestor traits."""
        n_clean = name.strip().lower()
        p_clean = parent_concept.strip().lower() if parent_concept else None

        existing = self.nodes.get(n_clean)
        affords = (
            direct_affordances
            if direct_affordances is not None
            else (set(existing.direct_affordances) if existing else set())
        )
        props = dict(existing.canonical_properties) if existing else {}
        if properties:
            props.update(properties)

        node = TaxonNode(
            name=n_clean,
            parent_concept=p_clean if p_clean else (existing.parent_concept if existing else None),
            level=level if level > 1 else (existing.level if existing else level),
            direct_affordances=affords,
            canonical_properties=props,
            confidence=confidence,
        )

        # Compute inherited affordances from ancestors
        if p_clean and p_clean in self.nodes:
            parent = self.nodes[p_clean]
            node.inherited_affordances = set(parent.all_affordances)
            # Inherit parent properties if not overridden
            for k, v in parent.canonical_properties.items():
                if k not in node.canonical_properties:
                    node.canonical_properties[k] = v
            node.level = max(level, parent.level + 1)

        self.nodes[n_clean] = node

        # Propagate changes downward to any existing children
        self._propagate_downward(n_clean)
        return node

    def _propagate_downward(self, parent_name: str) -> None:
        """Recursively update inherited affordances for all descendants of parent_name."""
        parent = self.nodes.get(parent_name)
        if not parent:
            return

        for child_name, child_node in self.nodes.items():
            if child_node.parent_concept == parent_name:
                child_node.inherited_affordances = set(parent.all_affordances)
                for k, v in parent.canonical_properties.items():
                    if k not in child_node.canonical_properties:
                        child_node.canonical_properties[k] = v
                child_node.level = parent.level + 1
                self._propagate_downward(child_name)

    def induce_is_a_relation(
        self, child_term: str, definition: str, fallback_parent: str = "physical_entity"
    ) -> TaxonNode:
        """Parse natural language definitional text to deduce parent concept and establish is-a link.

        Recognizes patterns:
        - "X is a [type of | kind of | form of] Y"
        - "X is a [rigid|large|small] Y [used to|for]..."
        """
        c_clean = child_term.strip().lower()
        def_low = definition.lower()

        # Check existing
        if c_clean in self.nodes and self.nodes[c_clean].parent_concept:
            return self.nodes[c_clean]

        parent_found: str | None = None

        # Pattern 1: explicit hypernym phrases
        patterns = [
            r"\b(?:is|are)\s+(?:a|an)\s+(?:type|kind|form|variety|category)\s+of\s+([a-z\-]+)",
            r"\b(?:is|are)\s+(?:a|an)\s+(?:large|small|rigid|slender|flexible|heavy|light|hollow)?\s*([a-z\-]+)\s+(?:used|designed|intended|serving)\s+(?:to|for)\b",
            r"\b(?:is|are)\s+(?:a|an)\s+([a-z\-]+)\b",
        ]

        for pat in patterns:
            m = re.search(pat, def_low)
            if m:
                cand = m.group(1).strip().lower()
                # Check if candidate is a known concept
                if cand in self.nodes and cand != c_clean:
                    parent_found = cand
                    break

        # Fallback heuristic by keyword presence if regex didn't hit a known taxon
        if not parent_found:
            for known_parent in (
                "container",
                "tool",
                "receptacle",
                "lever_instrument",
                "spherical_body",
                "block_body",
            ):
                if known_parent in def_low:
                    parent_found = known_parent
                    break

        if not parent_found:
            parent_found = fallback_parent

        return self.register_concept(name=c_clean, parent_concept=parent_found, level=3)

    def resolve_inherited_affordances(self, term: str) -> set[str]:
        """Return the complete set of effective direct and inherited affordances for a term."""
        t_clean = term.strip().lower()
        if t_clean in self.nodes:
            return self.nodes[t_clean].all_affordances

        # Check ancestors via morphological fallback
        if t_clean.endswith("s") and t_clean[:-1] in self.nodes:
            return self.nodes[t_clean[:-1]].all_affordances

        return set()

    def resolve_inherited_properties(self, term: str) -> dict[str, Any]:
        """Return inherited property dictionary for a term."""
        t_clean = term.strip().lower()
        if t_clean in self.nodes:
            return dict(self.nodes[t_clean].canonical_properties)
        return {}

    def get_ancestor_chain(self, term: str) -> list[str]:
        """Return list of ancestor concept names from term up to the root."""
        chain: list[str] = []
        curr = term.strip().lower()

        visited = set()
        while curr and curr in self.nodes and curr not in visited:
            visited.add(curr)
            chain.append(curr)
            curr = self.nodes[curr].parent_concept or ""

        return chain
