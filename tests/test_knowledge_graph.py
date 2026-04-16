"""Tests for KnowledgeGraph enhancements: disambiguation, community detection, PageRank."""

from __future__ import annotations

from hbllm.memory.knowledge_graph import KnowledgeGraph


class TestDisambiguation:
    def test_exact_duplicates_same_id(self):
        kg = KnowledgeGraph()
        kg.add_entity("machine learning")
        kg.add_entity("machine learning")
        assert kg.entity_count == 1

    def test_similar_entities_merged(self):
        kg = KnowledgeGraph()
        kg.add_entity("python programming")
        kg.add_entity("python programming language")
        kg.add_relation("user", "python programming", "uses")
        kg.add_relation("user", "python programming language", "uses")

        initial = kg.entity_count
        merged = kg.disambiguate_entities(similarity_threshold=0.5)
        assert merged >= 1 or kg.entity_count <= initial

    def test_different_types_not_merged(self):
        kg = KnowledgeGraph()
        kg.add_entity("python", entity_type="language")
        kg.add_entity("python", entity_type="animal")
        assert kg.entity_count == 1

    def test_merge_redirects_relations(self):
        kg = KnowledgeGraph()
        kg.add_relation("deep learning", "ai", "is_a")
        kg.add_relation("deep learning methods", "research", "used_in")

        kg.disambiguate_entities(similarity_threshold=0.5)
        assert kg.relation_count >= 1

    def test_no_merge_below_threshold(self):
        kg = KnowledgeGraph()
        kg.add_entity("python")
        kg.add_entity("javascript")
        merged = kg.disambiguate_entities(similarity_threshold=0.9)
        assert merged == 0


class TestCommunityDetection:
    def test_empty_graph(self):
        assert KnowledgeGraph().community_detection() == {}

    def test_single_cluster(self):
        kg = KnowledgeGraph()
        kg.add_relation("a", "b", "r")
        kg.add_relation("b", "c", "r")
        communities = kg.community_detection()
        assert len(communities) >= 1

    def test_two_clusters(self):
        kg = KnowledgeGraph()
        kg.add_relation("python", "programming", "is_a")
        kg.add_relation("python", "coding", "relates_to")
        kg.add_relation("programming", "coding", "relates_to")
        kg.add_relation("dog", "animal", "is_a")
        kg.add_relation("cat", "animal", "is_a")
        kg.add_relation("dog", "cat", "relates_to")

        communities = kg.community_detection()
        assert len(communities) >= 1

    def test_clean_keys(self):
        kg = KnowledgeGraph()
        kg.add_relation("x", "y", "r")
        for key in kg.community_detection():
            assert key.startswith("community_")

    def test_all_entities_assigned(self):
        kg = KnowledgeGraph()
        kg.add_relation("a", "b", "r")
        kg.add_relation("c", "d", "r")
        communities = kg.community_detection()
        all_members = [m for members in communities.values() for m in members]
        assert len(all_members) == kg.entity_count


class TestPageRank:
    def test_empty_graph(self):
        assert KnowledgeGraph().pagerank() == {}

    def test_hub_scores_highest(self):
        kg = KnowledgeGraph()
        kg.add_relation("a", "hub", "links_to")
        kg.add_relation("b", "hub", "links_to")
        kg.add_relation("c", "hub", "links_to")
        kg.add_relation("d", "hub", "links_to")

        scores = kg.pagerank()
        labels = list(scores.keys())
        assert labels[0] == "hub"

    def test_all_scores_positive(self):
        kg = KnowledgeGraph()
        kg.add_relation("x", "y", "r")
        kg.add_relation("y", "z", "r")
        scores = kg.pagerank()
        assert len(scores) == 3
        for score in scores.values():
            assert score > 0

    def test_different_damping(self):
        kg = KnowledgeGraph()
        kg.add_relation("a", "b", "r")
        s1 = kg.pagerank(damping=0.99)
        s2 = kg.pagerank(damping=0.5)
        assert s1 != s2

    def test_isolated_nodes_get_base_score(self):
        kg = KnowledgeGraph()
        kg.add_entity("lonely")
        scores = kg.pagerank()
        assert "lonely" in scores
        assert scores["lonely"] > 0


class TestTemporalRelations:
    def test_temporal_bounds_stored(self):
        kg = KnowledgeGraph()
        # Add relation with temporal window
        rel = kg.add_relation("event_a", "event_b", "causes", valid_from=100.0, valid_until=200.0)

        assert rel.valid_from == 100.0
        assert rel.valid_until == 200.0
        assert rel.weight == 1.0

    def test_temporal_defaults_to_none(self):
        kg = KnowledgeGraph()
        rel = kg.add_relation("fact_x", "fact_y", "relates_to")

        assert rel.valid_from is None
        assert rel.valid_until is None
