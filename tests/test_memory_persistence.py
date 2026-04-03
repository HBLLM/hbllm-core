"""
Memory Persistence Tests — verifies SemanticMemory and KnowledgeGraph
survive save/load round-trips.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from hbllm.memory.knowledge_graph import KnowledgeGraph
from hbllm.memory.semantic import SemanticMemory


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── KnowledgeGraph Persistence ───────────────────────────────────────────────

class TestKnowledgeGraphPersistence:
    def test_save_load_round_trip(self, tmp_dir):
        """KG entities and relations survive save/load."""
        kg = KnowledgeGraph()
        kg.add_relation("python", "programming language", "is_a")
        kg.add_relation("python", "indentation", "uses")
        kg.add_entity("machine learning", entity_type="field")

        save_path = Path(tmp_dir) / "kg.json"
        kg.save_to_disk(save_path)

        # Reload
        kg2 = KnowledgeGraph.load_from_disk(save_path)

        assert kg2.entity_count == kg.entity_count
        assert kg2.relation_count == kg.relation_count

        # Verify entities accessible
        e = kg2.get_entity("python")
        assert e is not None
        assert e.label == "python"

        e_ml = kg2.get_entity("machine learning")
        assert e_ml is not None
        assert e_ml.entity_type == "field"

    def test_neighbors_after_reload(self, tmp_dir):
        """Neighbor queries work correctly after reload."""
        kg = KnowledgeGraph()
        kg.add_relation("neural network", "machine learning", "is_a")
        kg.add_relation("neural network", "backpropagation", "uses")

        save_path = Path(tmp_dir) / "kg.json"
        kg.save_to_disk(save_path)

        kg2 = KnowledgeGraph.load_from_disk(save_path)
        neighbors = kg2.neighbors("neural network", direction="out")
        assert len(neighbors) == 2
        labels = {n["entity"] for n in neighbors}
        assert "machine learning" in labels
        assert "backpropagation" in labels

    def test_shortest_path_after_reload(self, tmp_dir):
        """Shortest path works after reload."""
        kg = KnowledgeGraph()
        kg.add_relation("A", "B", "connects")
        kg.add_relation("B", "C", "connects")
        kg.add_relation("C", "D", "connects")

        save_path = Path(tmp_dir) / "kg.json"
        kg.save_to_disk(save_path)

        kg2 = KnowledgeGraph.load_from_disk(save_path)
        path = kg2.shortest_path("A", "D")
        assert path is not None
        assert len(path) == 4

    def test_from_dict_roundtrip(self):
        """from_dict correctly reconstructs a KG from to_dict output."""
        kg = KnowledgeGraph()
        kg.add_relation("sun", "star", "is_a")
        kg.add_entity("moon", entity_type="satellite", attributes={"orbits": "earth"})

        data = kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)

        assert kg2.entity_count == kg.entity_count
        assert kg2.relation_count == kg.relation_count

        moon = kg2.get_entity("moon")
        assert moon is not None
        assert moon.attributes["orbits"] == "earth"

    def test_load_nonexistent_path(self, tmp_dir):
        """Loading from nonexistent path returns empty KG."""
        kg = KnowledgeGraph.load_from_disk(Path(tmp_dir) / "nonexistent.json")
        assert kg.entity_count == 0
        assert kg.relation_count == 0

    def test_empty_graph_save_load(self, tmp_dir):
        """Empty KG can be saved and loaded."""
        kg = KnowledgeGraph()
        save_path = Path(tmp_dir) / "kg.json"
        kg.save_to_disk(save_path)

        kg2 = KnowledgeGraph.load_from_disk(save_path)
        assert kg2.entity_count == 0
        assert kg2.relation_count == 0


# ── SemanticMemory Persistence ───────────────────────────────────────────────

class TestSemanticMemoryPersistence:
    def test_save_load_round_trip(self, tmp_dir):
        """Stored documents survive save/load."""
        mem = SemanticMemory()
        mem.store("Python is a great programming language", {"domain": "tech"})
        mem.store("Machine learning uses neural networks", {"domain": "ai"})
        mem.store("The weather today is sunny", {"domain": "general"})

        save_dir = Path(tmp_dir) / "semantic"
        mem.save_to_disk(save_dir)

        # Verify files exist
        assert (save_dir / "documents.json").exists()
        assert (save_dir / "hashes.json").exists()

        # Reload
        mem2 = SemanticMemory.load_from_disk(save_dir)
        assert mem2.count == 3

        docs = mem2.get_all()
        assert len(docs) == 3
        assert any("Python" in d.get("content", "") for d in docs)

    def test_dedup_after_reload(self, tmp_dir):
        """Content hash dedup works after reload."""
        mem = SemanticMemory()
        mem.store("Hello world")
        idx2 = mem.store("Hello world")  # Duplicate
        assert idx2 is None  # Skipped

        save_dir = Path(tmp_dir) / "semantic"
        mem.save_to_disk(save_dir)

        mem2 = SemanticMemory.load_from_disk(save_dir)
        assert mem2.count == 1
        idx3 = mem2.store("Hello world")
        assert idx3 is None  # Still dedup'd

    def test_search_after_reload(self, tmp_dir):
        """Documents and vectors survive reload — search uses available vectors."""
        mem = SemanticMemory()
        mem.store("Python programming tutorial")
        mem.store("JavaScript web development")
        mem.store("Machine learning algorithms")

        save_dir = Path(tmp_dir) / "semantic"
        mem.save_to_disk(save_dir)

        mem2 = SemanticMemory.load_from_disk(save_dir)
        # Documents and vectors are loaded
        assert mem2.count == 3
        assert len(mem2._vector_list) == 3
        # get_all works
        docs = mem2.get_all()
        assert any("Python" in d.get("content", "") for d in docs)

    def test_load_nonexistent_path(self, tmp_dir):
        """Loading from nonexistent path returns empty memory."""
        mem = SemanticMemory.load_from_disk(Path(tmp_dir) / "nonexistent")
        assert mem.count == 0

    def test_empty_memory_save_load(self, tmp_dir):
        """Empty semantic memory can be saved and loaded."""
        mem = SemanticMemory()
        save_dir = Path(tmp_dir) / "semantic"
        mem.save_to_disk(save_dir)

        mem2 = SemanticMemory.load_from_disk(save_dir)
        assert mem2.count == 0

    def test_vectors_preserved(self, tmp_dir):
        """Dense vectors are preserved across save/load."""
        mem = SemanticMemory()
        mem.store("Test document one")
        mem.store("Test document two")

        save_dir = Path(tmp_dir) / "semantic"
        mem.save_to_disk(save_dir)

        # Verify vector file exists
        assert (save_dir / "dense_vectors.npy").exists()

        mem2 = SemanticMemory.load_from_disk(save_dir)
        assert len(mem2._vector_list) == 2
        # Vectors should be numpy arrays
        assert hasattr(mem2._vector_list[0], 'shape')
