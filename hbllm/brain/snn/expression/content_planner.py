"""
Content Planner — SNN-driven content decision making.

Converts a ``RenderingContext`` (from v3 ShallowRenderer) into explicit
``ContentNode`` objects that specify exactly what to say, in what order,
with what emphasis.  The LLM no longer decides content — the SNN does.

This is the reasoning core of v4:

    v3: SNN produces concepts + associations → LLM decides what to say
    v4: SNN produces ContentNodes (what to say) → LLM just renders text

Components:
    ContentNode        — one atomic content decision
    ContentPlanNetwork — 4-layer SNN for content type selection
    ContentPlanner     — orchestrator that maps context → ContentNodes

Usage::

    from hbllm.brain.snn.expression.content_planner import ContentPlanner

    planner = ContentPlanner()
    nodes = planner.plan(rendering_context)
    # nodes = [ContentNode(type='assertion', key_points=[...]), ...]
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.snn.lif import LIFConfig
from hbllm.brain.snn.network import NeuronLayer, SpikingNetwork

logger = logging.getLogger(__name__)

# Content type constants
CONTENT_TYPES = [
    "assertion",    # Direct statement of fact/answer
    "explanation",  # Why/how something works
    "example",      # Concrete illustration
    "transition",   # Connecting between sections
    "caveat",       # Qualification or limitation
]


@dataclass
class ContentNode:
    """One atomic content decision — what to say in one sentence/clause.

    The SNN decides all content properties; the LLM only handles
    grammar and fluency.

    Attributes:
        node_id: Unique identifier.
        content_type: One of: assertion, explanation, example, transition, caveat.
        key_points: Strings the rendered sentence MUST contain.
        source_concept: Originating comprehension concept text.
        causal_basis: Causal chain summary justifying this node (if any).
        tone: One of: neutral, emphatic, cautionary.
        position: Order index in the response (0 = first).
        confidence: SNN's confidence in this content decision [0, 1].
        goal_id: ThoughtGoal that spawned this node.
    """

    node_id: str = ""
    content_type: str = "assertion"
    key_points: list[str] = field(default_factory=list)
    source_concept: str = ""
    causal_basis: str = ""
    tone: str = "neutral"
    position: int = 0
    confidence: float = 0.0
    goal_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content_type": self.content_type,
            "key_points": self.key_points,
            "source_concept": self.source_concept,
            "causal_basis": self.causal_basis,
            "tone": self.tone,
            "position": self.position,
            "confidence": round(self.confidence, 4),
        }


class ContentPlanNetwork:
    """4-layer SNN for content type selection.

    For each concept, decides what type of content to produce
    and how to emphasize it.

    Layers:
        input (8 neurons):
            Encodes concept/context features:
            - concept_salience, domain_strength, association_count,
              causal_confidence, memory_density, constraint_strength,
              concept_novelty, query_specificity

        planning (12 neurons):
            Learns which feature combinations warrant which content types.

        selection (6 neurons):
            Content type selectors:
            - assertion (0), explanation (1), example (2),
              transition (3), caveat (4), skip (5)

        output (3 neurons):
            - include: fire = add this content node
            - emphasize: fire = mark high-priority / emphatic tone
            - caveat: fire = add qualification

    Args:
        stdp_rule: Optional STDP rule for learning.
        settle_steps: Simulation steps per decision.
    """

    def __init__(
        self,
        stdp_rule: Any | None = None,
        settle_steps: int = 3,
    ) -> None:
        self._settle_steps = settle_steps
        self._network = SpikingNetwork(name="content_plan")

        # Input layer: 8 feature neurons
        self._network.add_layer(
            NeuronLayer(
                name="input",
                neuron_count=8,
                config=LIFConfig(
                    threshold=0.25,
                    decay_half_life=0.3,
                    reset_potential=0.0,
                    refractory_period=0.01,
                ),
            )
        )

        # Planning layer: 12 neurons
        self._network.add_layer(
            NeuronLayer(
                name="planning",
                neuron_count=12,
                config=LIFConfig(
                    threshold=0.35,
                    decay_half_life=0.5,
                    reset_potential=0.0,
                    refractory_period=0.02,
                ),
            )
        )

        # Selection layer: 6 content type neurons
        self._network.add_layer(
            NeuronLayer(
                name="selection",
                neuron_count=6,
                config=LIFConfig(
                    threshold=0.4,
                    decay_half_life=0.4,
                    reset_potential=0.0,
                    refractory_period=0.02,
                ),
            )
        )

        # Output layer: 3 decision neurons
        self._network.add_layer(
            NeuronLayer(
                name="output",
                neuron_count=3,
                config=LIFConfig(
                    threshold=0.35,
                    decay_half_life=0.4,
                    reset_potential=0.0,
                    refractory_period=0.01,
                ),
            )
        )

        # Input → Planning projection
        # salience: high salience → assertion + explanation
        # domain: strong domain → assertion
        # associations: many → explanation + example
        # causal: high → explanation + caveat
        # memory: dense → example
        # constraint: strong → caveat
        # novelty: high → explanation
        # specificity: high → assertion
        input_to_planning = [
            # salience → broad activation
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1],
            # domain → assertion-biased
            [0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1],
            # association_count → explanation
            [0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.2],
            # causal_confidence → explanation + caveat
            [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2],
            # memory_density → example
            [0.1, 0.1, 0.2, 0.2, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1],
            # constraint_strength → caveat
            [0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, 0.4, 0.2],
            # novelty → explanation
            [0.2, 0.3, 0.3, 0.4, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1],
            # specificity → assertion
            [0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1],
        ]

        self._network.connect(
            "input", "planning",
            initial_weights=input_to_planning,
            stdp_rule=stdp_rule,
        )

        # Planning → Selection projection
        # planning neurons 0-3 → assertion (strong)
        # planning neurons 2-5 → explanation
        # planning neurons 4-7 → example
        # planning neurons 6-8 → transition
        # planning neurons 8-10 → caveat
        # planning neurons 10-11 → skip
        planning_to_selection = [
            [0.5, 0.2, 0.1, 0.1, 0.0, 0.0],  # → assertion
            [0.4, 0.3, 0.1, 0.0, 0.1, 0.0],
            [0.3, 0.4, 0.2, 0.0, 0.0, 0.0],
            [0.1, 0.4, 0.2, 0.1, 0.1, 0.0],
            [0.1, 0.2, 0.4, 0.1, 0.1, 0.0],  # → example
            [0.0, 0.2, 0.3, 0.2, 0.2, 0.1],
            [0.0, 0.1, 0.2, 0.3, 0.3, 0.1],  # → transition/caveat
            [0.1, 0.3, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.2, 0.4, 0.1],  # → caveat
            [0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.0, 0.1, 0.1, 0.1, 0.4, 0.2],
            [0.0, 0.0, 0.1, 0.1, 0.1, 0.4],  # → skip
        ]

        self._network.connect(
            "planning", "selection",
            initial_weights=planning_to_selection,
            stdp_rule=stdp_rule,
        )

        # Selection → Output projection
        # All content types (except skip) → include
        # High-confidence types → emphasize
        # Caveat → caveat output
        selection_to_output = [
            [0.5, 0.3, 0.0],  # assertion → include + emphasize
            [0.4, 0.2, 0.0],  # explanation → include
            [0.3, 0.1, 0.0],  # example → include
            [0.3, 0.0, 0.0],  # transition → include
            [0.4, 0.1, 0.5],  # caveat → include + caveat
            [0.0, 0.0, 0.0],  # skip → nothing
        ]

        self._network.connect(
            "selection", "output",
            initial_weights=selection_to_output,
            stdp_rule=stdp_rule,
        )

    def decide(self, features: dict[str, float]) -> dict[str, Any]:
        """Run the SNN to decide content type and emphasis.

        Args:
            features: Dict with 8 input features.

        Returns:
            Dict with 'content_type', 'include', 'emphasize', 'caveat',
            'confidence' keys.
        """
        self._network.reset()

        input_currents = [
            features.get("concept_salience", 0.0),
            features.get("domain_strength", 0.0),
            features.get("association_count", 0.0),
            features.get("causal_confidence", 0.0),
            features.get("memory_density", 0.0),
            features.get("constraint_strength", 0.0),
            features.get("concept_novelty", 0.0),
            features.get("query_specificity", 0.0),
        ]

        t = time.time()
        last_result: dict[str, list] = {}

        for step in range(self._settle_steps):
            last_result = self._network.step(
                {"input": input_currents},
                t + step * 0.001,
                learn=True,
            )

        # Determine winning content type from selection layer
        content_type = "assertion"  # default
        if "selection" in last_result:
            sel_spikes = last_result["selection"]
            # Find strongest firing neuron
            best_idx = 0
            best_strength = 0.0
            for i, spike in enumerate(sel_spikes):
                strength = spike.strength if spike.fired else 0.0
                if strength > best_strength:
                    best_strength = strength
                    best_idx = i

            # Fallback: use membrane potentials if nothing fired
            if best_strength == 0.0:
                potentials = self._network.get_layer("selection").get_potential_vector()
                best_idx = max(range(len(potentials)), key=lambda i: potentials[i]) if potentials else 0

            if best_idx < len(CONTENT_TYPES):
                content_type = CONTENT_TYPES[best_idx]
            else:
                content_type = "skip"

        # Determine output decisions
        # If the selection layer chose a real content type, inclusion is
        # implied — the output layer can upgrade (emphasize/caveat) or
        # override (skip).
        include = content_type != "skip"
        emphasize = False
        add_caveat = False

        if "output" in last_result:
            out = last_result["output"]
            # Output can override inclusion only if skip was chosen
            if out[0].fired:
                include = True
            emphasize = out[1].fired
            add_caveat = out[2].fired

        # Confidence from selection strength
        confidence = best_strength if best_strength > 0 else 0.3

        return {
            "content_type": content_type,
            "include": include,
            "emphasize": emphasize,
            "caveat": add_caveat,
            "confidence": min(1.0, confidence),
        }

    @property
    def network(self) -> SpikingNetwork:
        return self._network


class ContentPlanner:
    """Orchestrates SNN-driven content planning.

    For each concept in the RenderingContext, runs the ContentPlanNetwork
    to decide:
    1. What content type to produce
    2. What key points to include (derived from concept + associations)
    3. Order and emphasis

    Args:
        plan_network: Optional pre-configured ContentPlanNetwork.
        stdp_rule: Optional STDP rule for the planning network.
        min_confidence: Minimum confidence to include a node.
    """

    def __init__(
        self,
        plan_network: ContentPlanNetwork | None = None,
        stdp_rule: Any | None = None,
        min_confidence: float = 0.2,
    ) -> None:
        self._network = plan_network or ContentPlanNetwork(stdp_rule=stdp_rule)
        self._min_confidence = min_confidence

    def plan(self, context: Any) -> list[ContentNode]:
        """Produce ordered ContentNodes from a RenderingContext.

        Args:
            context: RenderingContext with pre-reasoned data.

        Returns:
            Ordered list of ContentNodes ready for BrocaEncoder.
        """
        nodes: list[ContentNode] = []
        position = 0

        concepts = getattr(context, "concepts", [])
        goals = getattr(context, "goals", [])
        associations = getattr(context, "associations", [])
        causal_chains = getattr(context, "causal_chains", [])
        memory_hints = getattr(context, "memory_hints", [])
        constraints = getattr(context, "constraints", {})
        domain_context = getattr(context, "domain_context", {})

        for i, concept in enumerate(concepts):
            # Extract features for this concept
            features = self._extract_concept_features(
                concept=concept,
                concept_idx=i,
                total_concepts=len(concepts),
                associations=associations,
                causal_chains=causal_chains,
                memory_hints=memory_hints,
                constraints=constraints,
                domain_context=domain_context,
            )

            # Run SNN decision
            decision = self._network.decide(features)

            if not decision["include"]:
                continue

            if decision["content_type"] == "skip":
                continue

            if decision["confidence"] < self._min_confidence:
                continue

            # Build key points from concept + related data
            key_points = self._build_key_points(
                concept, associations, causal_chains, memory_hints
            )

            # Determine tone
            tone = "neutral"
            if decision["emphasize"]:
                tone = "emphatic"
            elif decision["caveat"]:
                tone = "cautionary"

            # Find causal basis
            causal_basis = ""
            for chain in causal_chains:
                src = chain.get("source_concept", "")
                if concept.lower() in src.lower() or src.lower() in concept.lower():
                    conclusion = chain.get("conclusion", "")
                    causal_basis = f"{src} → {conclusion}"
                    break

            # Find matching goal
            goal_id = ""
            for goal in goals:
                if concept in (goal.source_concept_text, goal.text):
                    goal_id = goal.id
                    break

            node = ContentNode(
                node_id=self._make_id(concept, position),
                content_type=decision["content_type"],
                key_points=key_points,
                source_concept=concept,
                causal_basis=causal_basis,
                tone=tone,
                position=position,
                confidence=decision["confidence"],
                goal_id=goal_id,
            )
            nodes.append(node)
            position += 1

            # If caveat decision fired, add a caveat node
            if decision["caveat"] and decision["content_type"] != "caveat":
                caveat_points = [f"Note: {concept} has limitations"]
                if constraints:
                    for k in constraints:
                        caveat_points.append(f"Constraint: {k}")

                caveat_node = ContentNode(
                    node_id=self._make_id(concept, position, "cav"),
                    content_type="caveat",
                    key_points=caveat_points,
                    source_concept=concept,
                    tone="cautionary",
                    position=position,
                    confidence=decision["confidence"] * 0.8,
                    goal_id=goal_id,
                )
                nodes.append(caveat_node)
                position += 1

        # Add transition nodes between major sections if > 2 nodes
        if len(nodes) > 2:
            nodes = self._inject_transitions(nodes)

        logger.debug(
            "ContentPlanner produced %d nodes from %d concepts",
            len(nodes),
            len(concepts),
        )

        return nodes

    def _extract_concept_features(
        self,
        concept: str,
        concept_idx: int,
        total_concepts: int,
        associations: list[dict],
        causal_chains: list[dict],
        memory_hints: list[str],
        constraints: dict[str, float],
        domain_context: dict[str, float],
    ) -> dict[str, float]:
        """Extract 8 SNN input features for a concept."""

        # concept_salience: position-weighted (earlier = higher)
        salience = max(0.1, 1.0 - (concept_idx / max(1, total_concepts)))

        # domain_strength: max domain activation
        domain_strength = max(domain_context.values()) if domain_context else 0.5

        # association_count: normalized
        concept_lower = concept.lower()
        related = sum(
            1 for a in associations
            if concept_lower in a.get("source_text", "").lower()
            or concept_lower in a.get("target_text", "").lower()
        )
        association_count = min(1.0, related / 3.0)

        # causal_confidence: best chain confidence for this concept
        causal_confidence = 0.0
        for chain in causal_chains:
            src = chain.get("source_concept", "").lower()
            if concept_lower in src or src in concept_lower:
                conf = chain.get("snn_confidence", 0.0)
                causal_confidence = max(causal_confidence, conf)

        # memory_density: how many memory hints mention this concept
        memory_matches = sum(
            1 for h in memory_hints if concept_lower in h.lower()
        )
        memory_density = min(1.0, memory_matches / 2.0)

        # constraint_strength: max constraint
        constraint_strength = max(constraints.values()) if constraints else 0.0

        # concept_novelty: inverse of how common the words are
        # (simple heuristic: longer concepts are more specific)
        concept_novelty = min(1.0, len(concept.split()) / 5.0)

        # query_specificity: concept length as proxy
        query_specificity = min(1.0, len(concept) / 30.0)

        return {
            "concept_salience": salience,
            "domain_strength": domain_strength,
            "association_count": association_count,
            "causal_confidence": causal_confidence,
            "memory_density": memory_density,
            "constraint_strength": constraint_strength,
            "concept_novelty": concept_novelty,
            "query_specificity": query_specificity,
        }

    def _build_key_points(
        self,
        concept: str,
        associations: list[dict],
        causal_chains: list[dict],
        memory_hints: list[str],
    ) -> list[str]:
        """Build key points the rendered sentence must contain."""
        points = [concept]

        concept_lower = concept.lower()

        # Add relevant association targets
        for assoc in associations:
            if concept_lower in assoc.get("source_text", "").lower():
                target = assoc.get("target_text", "")
                a_type = assoc.get("association_type", "related")
                if target:
                    points.append(f"{a_type}: {target}")

        # Add causal conclusions
        for chain in causal_chains:
            src = chain.get("source_concept", "").lower()
            if concept_lower in src or src in concept_lower:
                conclusion = chain.get("conclusion", "")
                if conclusion:
                    points.append(f"causes: {conclusion}")

        # Add memory hints
        for hint in memory_hints:
            if concept_lower in hint.lower():
                points.append(hint[:100])
                break  # one hint per concept

        return points[:5]  # Cap at 5 key points

    def _inject_transitions(
        self, nodes: list[ContentNode]
    ) -> list[ContentNode]:
        """Insert transition nodes between major content shifts."""
        result: list[ContentNode] = []

        for i, node in enumerate(nodes):
            if i > 0 and node.content_type != "transition":
                prev = nodes[i - 1]
                # Add transition when content type changes
                if prev.content_type != node.content_type and prev.content_type != "transition":
                    trans = ContentNode(
                        node_id=self._make_id("transition", node.position, "tr"),
                        content_type="transition",
                        key_points=[prev.source_concept, node.source_concept],
                        source_concept="",
                        tone="neutral",
                        position=node.position,
                        confidence=0.5,
                    )
                    result.append(trans)
            result.append(node)

        # Reindex positions
        for i, node in enumerate(result):
            node.position = i

        return result

    @staticmethod
    def _make_id(text: str, position: int, suffix: str = "") -> str:
        h = hashlib.md5(
            f"{text}_{position}".encode(), usedforsecurity=False
        ).hexdigest()[:8]
        return f"cn_{h}_{suffix}" if suffix else f"cn_{h}"

    @property
    def plan_network(self) -> ContentPlanNetwork:
        return self._network
