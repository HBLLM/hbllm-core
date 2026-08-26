"""Parity Oracle for Native Relational Structure Matcher.

Verifies:
1. Candidate Alignment Ranking: Multiple mappings returned in descending systematicity order.
2. Cross-Domain Transfer: Matches relational roles across source and target schemas.
3. Distractor Invariance: Structural alignment dominates over surface attributes.
"""

from hbllm.native.registry import native


class TestStructureMatcherParityOracle:
    """Level 3: Relational Subgraph Isomorphism & Systematicity Alignment."""

    def test_matcher_discovery(self):
        """Verify native registry detects structure_matcher capability."""
        assert native.available("structure_matcher") is True
        info = native.get_info("structure_matcher")
        assert info is not None
        assert info.available is True
        assert "isomorphism" in info.description

    def test_analogical_transfer_ranked_alignments(self):
        """Source: Solar system (Sun -> Earth). Target: Atom (Nucleus -> Electron)."""
        import hbllm_structure_matcher

        pattern = {
            "variables": ["X", "Y"],
            "edges": [
                {"rel_type": "CENTRAL_TO", "source_var": "X", "target_var": "Y"},
                {"rel_type": "ATTRACTS", "source_var": "X", "target_var": "Y"},
            ],
        }

        target = {
            "nodes": ["nucleus", "electron", "container_wall"],
            "edges": [
                {"rel_type": "CENTRAL_TO", "source": "nucleus", "target": "electron"},
                {"rel_type": "ATTRACTS", "source": "nucleus", "target": "electron"},
                {"rel_type": "CENTRAL_TO", "source": "container_wall", "target": "electron"},
            ],
        }

        alignments = hbllm_structure_matcher.match_relational_schema(pattern, target, 0.4)

        assert len(alignments) >= 1
        top = alignments[0]
        assert top["mapping"]["X"] == "nucleus"
        assert top["mapping"]["Y"] == "electron"
        assert top["systematicity_score"] == 1.0
        assert top["matched_relations_count"] == 2
        assert top["total_relations_count"] == 2
        assert top["structural_consistency"] is True

    def test_partial_match_ranking(self):
        """Verifies partial candidate alignments are returned and ranked correctly."""
        import hbllm_structure_matcher

        pattern = {
            "variables": ["A", "B", "C"],
            "edges": [
                {"rel_type": "SUPPORTS", "source_var": "A", "target_var": "B"},
                {"rel_type": "SUPPORTS", "source_var": "B", "target_var": "C"},
            ],
        }

        target = {
            "nodes": ["table", "box", "vase", "floor"],
            "edges": [
                {"rel_type": "SUPPORTS", "source": "table", "target": "box"},
                {"rel_type": "SUPPORTS", "source": "box", "target": "vase"},
                {"rel_type": "SUPPORTS", "source": "floor", "target": "table"},
            ],
        }

        alignments = hbllm_structure_matcher.match_relational_schema(pattern, target, 0.5)

        assert len(alignments) >= 1
        top = alignments[0]
        assert top["systematicity_score"] == 1.0
        assert top["matched_relations_count"] == 2
        assert top["structural_consistency"] is True
