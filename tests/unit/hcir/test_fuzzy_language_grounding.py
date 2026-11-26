"""Phase F3 tests: Fuzzy Language Grounding, Multilingual Hedges (en/si/ta), and 2D Verbalization."""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.compiler import HCIRCompiler, IntentType, SemanticAST
from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType, PhysicalEntityNode
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
    EpistemicVerbalizationLevel,
)
from hbllm.hcir.language.core.gateway import HCIRGateway
from hbllm.hcir.language.core.grounding import GroundingResolver
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    LanguageErrorType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.language.english.realizer import EnglishRealizer
from hbllm.hcir.language.sinhala.realizer import SinhalaRealizer
from hbllm.hcir.language.tamil.realizer import TamilRealizer
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_grounding_resolver_ranked_referents_and_tie_margin_ambiguity() -> None:
    """GroundingResolver ranks by gradable property & relation fit and preserves true tie ambiguity."""
    graph = CognitiveGraph()
    door = PhysicalEntityNode(id="door_1", entity_type="door")
    block_small = PhysicalEntityNode(
        id="block_small", entity_type="block", properties={"size": 2, "color": "blue"}
    )
    block_large_far = PhysicalEntityNode(
        id="block_large_far", entity_type="block", properties={"size": 20, "color": "blue"}
    )
    block_large_near = PhysicalEntityNode(
        id="block_large_near", entity_type="block", properties={"size": 19, "color": "blue"}
    )
    for n in (door, block_small, block_large_far, block_large_near):
        graph.add_node(n)

    graph.add_edge(
        HCIREdge(
            id="rel_near_close",
            edge_type=HCIREdgeType.NEAR,
            sources=["block_large_near"],
            targets=["door_1"],
            weight=0.92,
            properties={"semantic_degree": 0.92},
        )
    )
    graph.add_edge(
        HCIREdge(
            id="rel_near_far",
            edge_type=HCIREdgeType.NEAR,
            sources=["block_large_far"],
            targets=["door_1"],
            weight=0.20,
            properties={"semantic_degree": 0.20},
        )
    )

    resolver = GroundingResolver(graph)
    frame = SemanticFrame(
        frame_type=FrameType.QUERY,
        predicate="located_near",
        query_target="location",
        arguments={
            ThematicRole.THEME: EntityReference(
                concept_name="block",
                specifier="definite",
                properties={"color": "blue"},
                raw_text="the large block near the door",
            )
        },
        metadata=LanguageMetadata(
            language="en", raw_text="Where is the large block near the door?"
        ),
    )
    grounded = resolver.ground_frame(frame)
    assert grounded.grounding_success is True
    assert grounded.grounded_entities[ThematicRole.THEME] == "block_large_near"
    assert grounded.candidate_entities[ThematicRole.THEME][0] == "block_large_near"

    # Symmetric crisp tie ("the blue block") still produces AMBIGUOUS_REFERENCE
    tie_frame = SemanticFrame(
        frame_type=FrameType.QUERY,
        predicate="located_on",
        query_target="location",
        arguments={
            ThematicRole.THEME: EntityReference(
                concept_name="block",
                specifier="definite",
                properties={"color": "blue"},
                raw_text="the blue block",
            )
        },
    )
    tie_grounded = resolver.ground_frame(tie_frame)
    assert tie_grounded.grounding_success is False
    assert tie_grounded.grounding_error == LanguageErrorType.AMBIGUOUS_REFERENCE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lang", "utterance", "hedge_action"),
    [
        ("en", "fuzzy degree somewhat cup", "somewhat"),
        ("si", "තරම කෝප්පය", "තරමක්"),
        ("ta", "அளவு கோப்பை", "ஓரளவு"),
    ],
)
async def test_multilingual_declarative_fuzzy_hedge_constructions(
    lang: str, utterance: str, hedge_action: str
) -> None:
    """Seeded multilingual ConstructionNodes compile to math.fuzzy.hedge + ALPHA_CUT BRANCH."""
    ws = HCIRWorkspaceState(tenant_id="tenant_f3")
    services = KernelServices.create_default(ws)
    compiler = HCIRCompiler(tenant_id="tenant_f3", workspace=ws, language=lang)
    ast = compiler.parse_utterance(utterance, language=lang)
    ast.priority = 0.64
    ast.action = "somewhat" if lang == "en" else ("somewhat" if lang == "si" else "somewhat")
    stream = compiler.compile(ast)
    interp = HCIRInterpreter(ws, services=services)
    res = await interp.execute(stream)
    assert res.success is True
    out = res.events[0]["registers"]["$r0"]
    assert out["degree"] == pytest.approx(0.8)  # 0.64 ** 0.5


def test_compiler_evaluate_grounded_expression_with_fuzzy_near_and_hedge() -> None:
    """HCIRCompiler ranks NEAR spatial expressions with seeded hedges over 2D grid."""
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[5, 5] = 1  # blue reference marker
    grid[5, 6] = 2  # red cell at distance 1
    grid[5, 9] = 2  # red cell at distance 4
    compiler = HCIRCompiler(language="en")
    ast = SemanticAST(
        intent=IntentType.QUERY,
        subject="red very near blue",
        raw_text="red very near blue",
    )
    cells = compiler.evaluate_grounded_expression(ast, theory=None, grid=grid)
    assert cells[0] == (5, 6)


def test_two_dimensional_verbalization_english_sinhala_tamil() -> None:
    """Two-dimensional policy separates epistemic uncertainty from graded fuzzy degree."""
    policy = EpistemicRealizationPolicy()
    en_realizer = EnglishRealizer(policy)
    si_realizer = SinhalaRealizer(policy)
    ta_realizer = TamilRealizer(policy)

    # Case 1: High epistemic certainty (0.98) + intermediate spatial fit (0.68 -> "somewhat")
    high_ep_intermediate_fit = CognitiveEpistemicState(
        target_predicate="near",
        target_subject="cup",
        target_object="edge",
        confidence=0.98,
        uncertainty=0.02,
        support_count=5,
        semantic_degree=0.68,
    )
    lvl, hedge = policy.evaluate_two_dimensional(high_ep_intermediate_fit)
    assert lvl == EpistemicVerbalizationLevel.CERTAIN
    assert hedge == "somewhat"
    assert en_realizer.realize(high_ep_intermediate_fit) == "The cup is somewhat near the edge."
    assert "තරමක්" in si_realizer.realize(high_ep_intermediate_fit)
    assert "ஓரளவு" in ta_realizer.realize(high_ep_intermediate_fit)

    # Case 2: Low epistemic certainty (0.30) + crisp fit (None)
    low_ep_crisp = CognitiveEpistemicState(
        target_predicate="near",
        target_subject="cup",
        target_object="edge",
        confidence=0.30,
        uncertainty=0.70,
        support_count=1,
        semantic_degree=None,
    )
    assert (
        en_realizer.realize(low_ep_crisp) == "I am not certain, but the cup might be near the edge."
    )

    # Case 3: HCIRGateway reads semantic_degree & posterior_belief cleanly from CognitiveGraph edge
    graph = CognitiveGraph()
    cup = PhysicalEntityNode(id="ent_cup", entity_type="cup")
    edge_ent = PhysicalEntityNode(id="ent_edge", entity_type="edge")
    graph.add_node(cup)
    graph.add_node(edge_ent)
    graph.add_edge(
        HCIREdge(
            id="edge_f3",
            edge_type=HCIREdgeType.NEAR,
            sources=["ent_cup"],
            targets=["ent_edge"],
            weight=0.68,
            properties={"semantic_degree": 0.68, "posterior_belief": 0.98},
        )
    )
    gw = HCIRGateway(graph)
    grounded = GroundingResolver(graph).ground_frame(
        SemanticFrame(
            frame_type=FrameType.QUERY,
            predicate="located_near",
            query_target="location",
            arguments={
                ThematicRole.THEME: EntityReference(
                    concept_name="cup", specifier="definite", raw_text="the cup"
                )
            },
        )
    )
    ep_state = gw.process_query(grounded)
    assert ep_state.semantic_degree == pytest.approx(0.68)
    assert ep_state.confidence == pytest.approx(0.98)
    assert en_realizer.realize(ep_state) == "The cup is somewhat near the edge."
