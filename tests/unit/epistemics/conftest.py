"""Shared fixtures for epistemics tests."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from typing import Any

import pytest

from hbllm.brain.epistemics.calibration import EpistemicCalibrationEngine
from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner
from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    HypothesisNode,
)
from hbllm.hcir.types import BeliefConfidence, EvidenceStrength


@pytest.fixture
def graph() -> CognitiveGraph:
    """Fresh cognitive graph."""
    return CognitiveGraph()


@pytest.fixture
def tmp_dir() -> Generator[str, None, None]:
    """Temporary directory for persistent data."""
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def workspace(tmp_dir: str, graph: CognitiveGraph) -> DiscoveryWorkspace:
    """Workspace with a research program."""
    return DiscoveryWorkspace(data_dir=tmp_dir, graph=graph)


@pytest.fixture
def memory(tmp_dir: str) -> Generator[EpistemicMemory, None, None]:
    """Epistemic memory with cleanup."""
    mem = EpistemicMemory(data_dir=tmp_dir)
    yield mem
    mem.close()


@pytest.fixture
def calibrator(memory: EpistemicMemory) -> EpistemicCalibrationEngine:
    """Calibration engine wired to memory."""
    return EpistemicCalibrationEngine(memory=memory)


@pytest.fixture
def counterfactual(graph: CognitiveGraph) -> CounterfactualReasoner:
    """Counterfactual reasoner."""
    return CounterfactualReasoner(graph=graph)


@pytest.fixture
def populated_graph(graph: CognitiveGraph) -> dict[str, Any]:
    """Graph with evidence → hypothesis → belief chain.

    Returns dict with keys: graph, evidence_ids, hypothesis_id, belief_id.
    """
    ev1 = EvidenceNode(
        evidence_type=EvidenceStrength.EXPERIMENTAL,
        methodology="RCT n=200",
        sample_size=200,
        reproducible=True,
    )
    ev2 = EvidenceNode(
        evidence_type=EvidenceStrength.OBSERVATIONAL,
        methodology="Survey n=50",
        sample_size=50,
        reproducible=False,
    )
    graph.upsert_node(ev1)
    graph.upsert_node(ev2)

    hyp = HypothesisNode(claim="Z causes X")
    graph.upsert_node(hyp)

    belief = BeliefNode(
        claim="X is caused by mechanism Z",
        belief_confidence=BeliefConfidence(
            evidence_quality=0.8,
            evidence_quantity=0.5,
            reproducibility=0.7,
            prediction_accuracy=0.6,
        ),
    )
    graph.upsert_node(belief)

    # Wire edges
    graph.add_edge(HCIREdge(sources=[ev1.id], targets=[hyp.id], edge_type=HCIREdgeType.SUPPORTS))
    graph.add_edge(HCIREdge(sources=[ev2.id], targets=[hyp.id], edge_type=HCIREdgeType.SUPPORTS))
    graph.add_edge(HCIREdge(sources=[ev1.id], targets=[belief.id], edge_type=HCIREdgeType.SUPPORTS))
    graph.add_edge(HCIREdge(sources=[ev2.id], targets=[belief.id], edge_type=HCIREdgeType.SUPPORTS))
    graph.add_edge(HCIREdge(sources=[hyp.id], targets=[belief.id], edge_type=HCIREdgeType.SUPPORTS))

    return {
        "graph": graph,
        "evidence_ids": [ev1.id, ev2.id],
        "hypothesis_id": hyp.id,
        "belief_id": belief.id,
    }
