"""Unit tests for Multimodal Scientific Diagram Grounding & Cross-Modal Verifier."""

from __future__ import annotations

from typing import Any

import pytest

from plugins.developmental_adapter.diagram_grounding import (
    CrossModalVerifier,
    DiagramPrimitive,
    DiagramPrimitiveType,
    ScientificDiagramParser,
)


def test_diagram_primitive_containment_and_distance() -> None:
    """Verify axis-aligned and polygon ray-casting containment tests."""
    box = DiagramPrimitive(
        primitive_id="box_1",
        primitive_type=DiagramPrimitiveType.CONTAINER_BOX,
        label="Beaker",
        bounds=(0.0, 0.0, 10.0, 10.0),
        centroid=(5.0, 5.0),
        points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        is_closed=True,
    )

    assert box.contains_point((5.0, 5.0)) is True
    assert box.contains_point((1.0, 1.0)) is True
    assert box.contains_point((15.0, 5.0)) is False

    particle = DiagramPrimitive(
        primitive_id="part_1",
        primitive_type=DiagramPrimitiveType.CIRCLE,
        label="Solute Particle",
        bounds=(4.0, 4.0, 6.0, 6.0),
        centroid=(5.0, 5.0),
    )
    assert box.distance_to(particle) == pytest.approx(0.0)


def test_scientific_diagram_parser_to_cognitive_graph() -> None:
    """Verify lifting primitives to CognitiveGraph with ENCLOSES, SUPPORTS, and DIVIDES edges."""
    elements: list[dict[str, Any]] = [
        {
            "id": "container_1",
            "type": "container_box",
            "label": "Cell Membrane",
            "bounds": [0.0, 0.0, 10.0, 10.0],
            "is_closed": True,
        },
        {
            "id": "organelle_1",
            "type": "circle",
            "label": "Mitochondria",
            "bounds": [4.0, 4.0, 6.0, 6.0],
        },
        {
            "id": "fulcrum_1",
            "type": "fulcrum",
            "label": "Fulcrum",
            "bounds": [4.5, 0.0, 5.5, 1.0],
        },
        {
            "id": "beam_1",
            "type": "rectangle",
            "label": "Lever Beam",
            "bounds": [1.0, 1.0, 9.0, 1.2],
        },
    ]

    prims = ScientificDiagramParser.parse_primitives_dict(elements)
    assert len(prims) == 4

    graph = ScientificDiagramParser.to_cognitive_graph(prims)
    assert len(graph._nodes) == 4

    # Verify ENCLOSES edge: container_1 -> organelle_1
    encloses_edges = [
        e for e in graph._edges.values() if e.properties.get("relation") == "ENCLOSES"
    ]
    assert len(encloses_edges) >= 1
    assert encloses_edges[0].targets[0] == "container_1"
    assert encloses_edges[0].sources[0] == "organelle_1"

    # Verify SUPPORTS edge: fulcrum_1 -> beam_1
    supports_edges = [
        e for e in graph._edges.values() if e.properties.get("relation") == "SUPPORTS"
    ]
    assert len(supports_edges) >= 1
    assert supports_edges[0].sources[0] == "fulcrum_1"
    assert supports_edges[0].targets[0] == "beam_1"


def test_cross_modal_verifier_consistent_scientific_claims() -> None:
    """Verify valid scientific statements against lever, containment, and membrane diagrams."""
    verifier = CrossModalVerifier()

    # 1. Lever diagram: Load at x=1.0, Fulcrum at x=5.0, Effort at x=9.0 (First-Class Lever)
    prims_lever = [
        DiagramPrimitive(
            primitive_id="load_1",
            primitive_type=DiagramPrimitiveType.RECTANGLE,
            label="Load",
            bounds=(0.5, 1.2, 1.5, 2.2),
            centroid=(1.0, 1.7),
        ),
        DiagramPrimitive(
            primitive_id="fulcrum_1",
            primitive_type=DiagramPrimitiveType.FULCRUM,
            label="Fulcrum",
            bounds=(4.5, 0.0, 5.5, 1.0),
            centroid=(5.0, 0.5),
        ),
        DiagramPrimitive(
            primitive_id="effort_1",
            primitive_type=DiagramPrimitiveType.ARROW,
            label="Applied Effort",
            bounds=(8.5, 1.2, 9.5, 2.2),
            centroid=(9.0, 1.7),
        ),
    ]

    report_lever = verifier.verify_claim(
        prims_lever,
        "The fulcrum is positioned between the load and the effort in this first-class lever.",
    )
    assert report_lever.is_consistent is True
    assert report_lever.brier_score < 0.01
    assert any("FIRST_CLASS" in r for r in report_lever.grounded_relations)

    # 2. Membrane diagram: Membrane at x=5.0 partitions chamber into Left and Right
    prims_membrane = [
        DiagramPrimitive(
            primitive_id="membrane_1",
            primitive_type=DiagramPrimitiveType.MEMBRANE,
            label="Semi-Permeable Membrane",
            bounds=(4.9, 0.0, 5.1, 10.0),
            centroid=(5.0, 5.0),
        ),
        DiagramPrimitive(
            primitive_id="solute_left",
            primitive_type=DiagramPrimitiveType.CIRCLE,
            label="Solute Left",
            bounds=(1.0, 4.0, 2.0, 5.0),
            centroid=(1.5, 4.5),
        ),
        DiagramPrimitive(
            primitive_id="solute_right",
            primitive_type=DiagramPrimitiveType.CIRCLE,
            label="Solute Right",
            bounds=(7.0, 4.0, 8.0, 5.0),
            centroid=(7.5, 4.5),
        ),
    ]

    report_membrane = verifier.verify_claim(
        prims_membrane,
        "The semi-permeable membrane divides the chamber into two compartments.",
    )
    assert report_membrane.is_consistent is True
    assert any("BIPARTITION" in r for r in report_membrane.grounded_relations)


def test_cross_modal_verifier_detects_contradictions() -> None:
    """Verify detection of counterfactual contradictions between prose and diagram."""
    verifier = CrossModalVerifier()

    # Diagram: Beaker (bounds 0..10) encloses particle (at 5.0, 5.0)
    prims = [
        DiagramPrimitive(
            primitive_id="beaker_1",
            primitive_type=DiagramPrimitiveType.CONTAINER_BOX,
            label="Beaker",
            bounds=(0.0, 0.0, 10.0, 10.0),
            centroid=(5.0, 5.0),
            is_closed=True,
        ),
        DiagramPrimitive(
            primitive_id="part_1",
            primitive_type=DiagramPrimitiveType.CIRCLE,
            label="Molecule",
            bounds=(4.0, 4.0, 6.0, 6.0),
            centroid=(5.0, 5.0),
        ),
    ]

    # False assertion: text asserts molecule is outside the beaker
    false_claim = "The molecule is situated outside the beaker."
    report = verifier.verify_claim(prims, false_claim)

    assert report.is_consistent is False
    assert len(report.contradictions_detected) >= 1
    assert "ENCLOSED" in report.contradictions_detected[0]

    # Check Markdown report formatting
    md = report.format_markdown()
    assert "CONTRADICTION DETECTED" in md
    assert "Beaker" in md
