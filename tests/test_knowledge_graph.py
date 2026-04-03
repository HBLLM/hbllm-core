"""
Tests for KnowledgeGraph — entity/relation storage, query, and text ingestion.
"""

from hbllm.memory.knowledge_graph import (
    KnowledgeGraph,
    extract_entities_from_text,
)

# ── Entity & Relation CRUD ───────────────────────────────────────────────────


class TestEntityCRUD:
    def test_add_entity(self):
        kg = KnowledgeGraph()
        e = kg.add_entity("Python", entity_type="language")
        assert e.label == "python"
        assert e.entity_type == "language"
        assert kg.entity_count == 1

    def test_add_duplicate_entity_merges(self):
        kg = KnowledgeGraph()
        kg.add_entity("Python", attributes={"version": "3.11"})
        kg.add_entity("Python", attributes={"creator": "Guido"})
        assert kg.entity_count == 1
        e = kg.get_entity("Python")
        assert e.attributes["version"] == "3.11"
        assert e.attributes["creator"] == "Guido"

    def test_get_entity_returns_none_for_missing(self):
        kg = KnowledgeGraph()
        assert kg.get_entity("nonexistent") is None

    def test_add_relation_creates_entities(self):
        kg = KnowledgeGraph()
        rel = kg.add_relation("Python", "programming", "is_a")
        assert kg.entity_count == 2
        assert kg.relation_count == 1
        assert rel.relation_type == "is_a"

    def test_duplicate_relation_updates_weight(self):
        kg = KnowledgeGraph()
        kg.add_relation("A", "B", "uses", weight=0.5)
        kg.add_relation("A", "B", "uses", weight=0.9)
        assert kg.relation_count == 1
        # Weight should be max(0.5, 0.9) = 0.9
        rel = list(kg._relations.values())[0]
        assert rel.weight == 0.9

    def test_clear(self):
        kg = KnowledgeGraph()
        kg.add_relation("A", "B", "uses")
        kg.add_relation("C", "D", "has")
        kg.clear()
        assert kg.entity_count == 0
        assert kg.relation_count == 0


# ── Neighbor Queries ─────────────────────────────────────────────────────────


class TestNeighborQueries:
    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.kg.add_relation("python", "programming", "is_a")
        self.kg.add_relation("python", "guido", "created_by")
        self.kg.add_relation("javascript", "programming", "is_a")
        self.kg.add_relation("python", "flask", "uses")

    def test_outgoing_neighbors(self):
        results = self.kg.neighbors("python", direction="out")
        labels = [r["entity"] for r in results]
        assert "programming" in labels
        assert "guido" in labels
        assert "flask" in labels
        assert len(results) == 3

    def test_incoming_neighbors(self):
        results = self.kg.neighbors("programming", direction="in")
        labels = [r["entity"] for r in results]
        assert "python" in labels
        assert "javascript" in labels

    def test_filter_by_relation_type(self):
        results = self.kg.neighbors("python", relation_type="is_a")
        assert len(results) >= 1
        assert all(r["relation"] == "is_a" for r in results)

    def test_neighbors_of_nonexistent(self):
        results = self.kg.neighbors("nonexistent")
        assert results == []


# ── Shortest Path ────────────────────────────────────────────────────────────


class TestShortestPath:
    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.kg.add_relation("A", "B", "connects")
        self.kg.add_relation("B", "C", "connects")
        self.kg.add_relation("C", "D", "connects")
        self.kg.add_relation("A", "E", "connects")
        self.kg.add_relation("E", "D", "connects")

    def test_direct_path(self):
        path = self.kg.shortest_path("A", "B")
        assert path == ["a", "b"]

    def test_shortest_path_prefers_shorter(self):
        # A→E→D is shorter than A→B→C→D
        path = self.kg.shortest_path("A", "D")
        assert path is not None
        assert len(path) <= 3  # A, E, D

    def test_same_node_path(self):
        path = self.kg.shortest_path("A", "A")
        assert path == ["a"]

    def test_no_path_returns_none(self):
        kg = KnowledgeGraph()
        kg.add_entity("X")
        kg.add_entity("Y")
        path = kg.shortest_path("X", "Y")
        assert path is None

    def test_nonexistent_node_returns_none(self):
        path = self.kg.shortest_path("A", "Z")
        assert path is None


# ── Subgraph Extraction ──────────────────────────────────────────────────────


class TestSubgraph:
    def test_subgraph_depth_1(self):
        kg = KnowledgeGraph()
        kg.add_relation("center", "north", "connects")
        kg.add_relation("center", "south", "connects")
        kg.add_relation("north", "far_north", "connects")

        sg = kg.subgraph("center", depth=1)
        labels = {e["label"] for e in sg["entities"]}
        assert "center" in labels
        assert "north" in labels
        assert "south" in labels
        assert "far_north" not in labels  # depth 1 shouldn't reach

    def test_subgraph_depth_2(self):
        kg = KnowledgeGraph()
        kg.add_relation("center", "north", "connects")
        kg.add_relation("north", "far_north", "connects")

        sg = kg.subgraph("center", depth=2)
        labels = {e["label"] for e in sg["entities"]}
        assert "far_north" in labels

    def test_subgraph_nonexistent(self):
        kg = KnowledgeGraph()
        sg = kg.subgraph("nothing")
        assert sg == {"entities": [], "relations": []}


# ── Text Ingestion ───────────────────────────────────────────────────────────


class TestTextIngestion:
    def test_extract_is_a_relation(self):
        triples = extract_entities_from_text("Python is a programming language")
        assert len(triples) >= 1
        source, rel, target = triples[0]
        assert rel == "is_a"
        assert "python" in source.lower()
        assert "programming" in target.lower()

    def test_extract_has_relation(self):
        triples = extract_entities_from_text("The system has multiple nodes")
        assert any(r == "has" for _, r, _ in triples)

    def test_extract_causes_relation(self):
        triples = extract_entities_from_text("High memory usage causes system slowdown")
        assert any(r == "causes" for _, r, _ in triples)

    def test_extract_uses_relation(self):
        triples = extract_entities_from_text("The router uses message passing")
        assert any(r == "uses" for _, r, _ in triples)

    def test_ingest_text_adds_to_graph(self):
        kg = KnowledgeGraph()
        count = kg.ingest_text("Python is a programming language. JavaScript uses event loops.")
        assert count >= 2
        assert kg.entity_count >= 4

    def test_empty_text_returns_zero(self):
        kg = KnowledgeGraph()
        assert kg.ingest_text("") == 0

    def test_stopword_only_entities_filtered(self):
        triples = extract_entities_from_text("the is a the")
        assert len(triples) == 0


# ── Serialization ────────────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self):
        kg = KnowledgeGraph()
        kg.add_relation("Python", "language", "is_a")
        d = kg.to_dict()
        assert "entities" in d
        assert "relations" in d
        assert len(d["entities"]) == 2
        assert len(d["relations"]) == 1
        assert d["relations"][0]["type"] == "is_a"
