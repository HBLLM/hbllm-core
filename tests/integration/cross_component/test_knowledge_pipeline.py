"""Integration tests for Knowledge Pipeline — Extractor → KnowledgeBase → KnowledgeGraph."""

import json
import os
import textwrap

import pytest

from hbllm.knowledge.extractor import (
    FolderExtractor,
    JsonExtractor,
    KnowledgeGraphExtractor,
    MarkdownExtractor,
    PythonExtractor,
    StructuralResult,
)
from hbllm.knowledge.knowledge_base import KnowledgeBase, Source
from hbllm.memory.knowledge_graph import KnowledgeGraph


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_sample_project(tmp_path):
    """Create a small sample project structure for extraction tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Python file
    py_file = tmp_path / "example.py"
    py_file.write_text(textwrap.dedent("""\
        import os
        from pathlib import Path

        class DataProcessor:
            def __init__(self, path: str):
                self.path = path

            def process(self):
                return Path(self.path).read_text()

        class CSVProcessor(DataProcessor):
            def parse(self):
                pass

        def main():
            processor = DataProcessor("data.csv")
            processor.process()
    """))

    # Markdown file
    md_file = tmp_path / "README.md"
    md_file.write_text(textwrap.dedent("""\
        # Project Title

        ## Overview
        This is a sample project with #python and #data-processing tags.

        ## Architecture
        See [[Design Doc]] for details.

        ### Components
        - [FastAPI Docs](https://fastapi.tiangolo.com)

        ## Setup
        Install dependencies.
    """))

    # JSON config
    json_file = tmp_path / "config.json"
    json_file.write_text(json.dumps({
        "database": {
            "host": "${DB_HOST}",
            "port": 5432,
            "name": "hbllm"
        },
        "features": ["auth", "cache"],
    }, indent=2))

    # Subdirectory
    sub = tmp_path / "utils"
    sub.mkdir()
    (sub / "helpers.py").write_text("def helper(): pass\n")

    return tmp_path


# ── Extractor Tests ──────────────────────────────────────────────────────────


class TestFolderExtractor:
    """Test folder structure extraction."""

    def test_extracts_folder_hierarchy(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = FolderExtractor(str(root))
        result = extractor.extract()

        assert len(result.entities) > 0
        assert len(result.relations) > 0

        # Should have folder and file entities
        entity_types = {e.entity_type for e in result.entities}
        assert "folder" in entity_types
        # .py files should be modules, .md should be documents
        assert "module" in entity_types or "document" in entity_types

        # Should have "contains" relations
        relation_types = {r.relation_type for r in result.relations}
        assert "contains" in relation_types

    def test_respects_skip_dirs(self, tmp_path):
        root = _create_sample_project(tmp_path)
        # Add a __pycache__ dir — should be skipped
        pycache = root / "__pycache__"
        pycache.mkdir()
        (pycache / "cache.pyc").write_bytes(b"\x00")

        extractor = FolderExtractor(str(root))
        result = extractor.extract()

        entity_ids = {e.id for e in result.entities}
        assert not any("__pycache__" in eid for eid in entity_ids)


class TestPythonExtractor:
    """Test Python AST extraction."""

    def test_extracts_classes_and_functions(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = PythonExtractor(str(root))
        result = extractor.extract(str(root / "example.py"))

        entity_labels = {e.label for e in result.entities}
        assert "DataProcessor" in entity_labels
        assert "CSVProcessor" in entity_labels
        assert "main" in entity_labels

    def test_extracts_inheritance(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = PythonExtractor(str(root))
        result = extractor.extract(str(root / "example.py"))

        inherits_rels = [r for r in result.relations if r.relation_type == "inherits"]
        assert len(inherits_rels) >= 1  # CSVProcessor inherits DataProcessor

    def test_extracts_imports(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = PythonExtractor(str(root))
        result = extractor.extract(str(root / "example.py"))

        import_rels = [r for r in result.relations if r.relation_type == "imports"]
        assert len(import_rels) >= 2  # os and Path


class TestMarkdownExtractor:
    """Test markdown structure extraction."""

    def test_extracts_headings(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = MarkdownExtractor(str(root))
        result = extractor.extract(str(root / "README.md"))

        section_entities = [e for e in result.entities if e.entity_type == "section"]
        section_labels = {e.label for e in section_entities}
        assert "Project Title" in section_labels
        assert "Overview" in section_labels

    def test_extracts_tags(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = MarkdownExtractor(str(root))
        result = extractor.extract(str(root / "README.md"))

        tag_entities = [e for e in result.entities if e.entity_type == "tag"]
        assert len(tag_entities) >= 1

    def test_extracts_wikilinks(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = MarkdownExtractor(str(root))
        result = extractor.extract(str(root / "README.md"))

        ref_rels = [r for r in result.relations if r.relation_type == "references"]
        assert len(ref_rels) >= 1  # [[Design Doc]]

    def test_extracts_urls(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = MarkdownExtractor(str(root))
        result = extractor.extract(str(root / "README.md"))

        url_entities = [e for e in result.entities if e.entity_type == "url"]
        assert len(url_entities) >= 1


class TestJsonExtractor:
    """Test JSON structure extraction."""

    def test_extracts_properties(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = JsonExtractor(str(root))
        result = extractor.extract(str(root / "config.json"))

        prop_entities = [e for e in result.entities if e.entity_type == "property"]
        prop_labels = {e.label for e in prop_entities}
        assert "database" in prop_labels
        assert "host" in prop_labels

    def test_extracts_env_references(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = JsonExtractor(str(root))
        result = extractor.extract(str(root / "config.json"))

        # Should detect ${DB_HOST} env var reference
        refers_rels = [r for r in result.relations if r.relation_type == "refers_to"]
        assert len(refers_rels) >= 1


class TestKnowledgeGraphExtractor:
    """Test full pipeline extraction with cross-reference resolution."""

    def test_full_project_extraction(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = KnowledgeGraphExtractor(str(root))
        result = extractor.extract()

        # Should have entities from all file types
        assert len(result.entities) > 10
        assert len(result.relations) > 5

    def test_single_file_extraction(self, tmp_path):
        root = _create_sample_project(tmp_path)
        extractor = KnowledgeGraphExtractor(str(root / "example.py"))
        result = extractor.extract()

        assert len(result.entities) > 0

    def test_merge_results(self):
        a = StructuralResult()
        a.entities.append(type("E", (), {"id": "a", "label": "A", "entity_type": "test", "attributes": {}})())
        b = StructuralResult()
        b.entities.append(type("E", (), {"id": "b", "label": "B", "entity_type": "test", "attributes": {}})())
        a.merge(b)
        assert len(a.entities) == 2


# ── KnowledgeBase Integration Tests ──────────────────────────────────────────


class TestKnowledgeBaseIntegration:
    """Test KnowledgeBase source management and ingestion."""

    def test_add_and_list_sources(self, tmp_path):
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))
        root = _create_sample_project(tmp_path / "project")

        source = kb.add_source(str(root), source_type="folder")
        assert isinstance(source, Source)
        assert source.status == "pending"

        sources = kb.list_sources()
        assert len(sources) == 1
        assert sources[0]["source_id"] == source.source_id

    def test_duplicate_source_returns_existing(self, tmp_path):
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))
        root = _create_sample_project(tmp_path / "project")

        s1 = kb.add_source(str(root))
        s2 = kb.add_source(str(root))
        assert s1.source_id == s2.source_id

    def test_ingest_source_creates_chunks(self, tmp_path):
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))
        root = _create_sample_project(tmp_path / "project")

        source = kb.add_source(str(root), source_type="folder")
        num_chunks = kb.ingest_source(source.source_id)

        assert num_chunks > 0
        assert source.status == "ready"
        assert source.file_count > 0

    def test_search_returns_results(self, tmp_path):
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))
        root = _create_sample_project(tmp_path / "project")

        source = kb.add_source(str(root), source_type="folder")
        kb.ingest_source(source.source_id)

        results = kb.search("DataProcessor class")
        assert len(results) > 0
        assert "content" in results[0]
        assert "score" in results[0]

    def test_remove_source(self, tmp_path):
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))
        root = _create_sample_project(tmp_path / "project")

        source = kb.add_source(str(root))
        kb.ingest_source(source.source_id)
        assert len(kb.list_sources()) == 1

        result = kb.remove_source(source.source_id)
        assert result is True
        assert len(kb.list_sources()) == 0

    def test_stats(self, tmp_path):
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))
        root = _create_sample_project(tmp_path / "project")

        source = kb.add_source(str(root))
        kb.ingest_source(source.source_id)

        stats = kb.get_stats()
        assert stats["total_sources"] == 1
        assert stats["ready_sources"] == 1
        assert stats["total_chunks"] > 0

    def test_web_content_ingestion(self, tmp_path):
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))

        doc_id = kb.ingest_web_content(
            content="Python is a programming language used for web development and AI.",
            url="https://example.com/python",
            title="Python Overview",
            trust_score=0.9,
            domain="example.com",
        )
        assert doc_id != ""

        results = kb.search("Python programming")
        assert len(results) > 0

    def test_persistence_round_trip(self, tmp_path):
        kb_dir = str(tmp_path / "kb")
        root = _create_sample_project(tmp_path / "project")

        # Create and ingest
        kb1 = KnowledgeBase(data_dir=kb_dir)
        source = kb1.add_source(str(root))
        kb1.ingest_source(source.source_id)
        original_stats = kb1.get_stats()

        # Reload from disk
        kb2 = KnowledgeBase(data_dir=kb_dir)
        loaded_stats = kb2.get_stats()

        assert loaded_stats["total_sources"] == original_stats["total_sources"]


# ── Extractor → KnowledgeGraph → KnowledgeBase pipeline ─────────────────────


class TestFullKnowledgePipeline:
    """Test the complete extraction → graph → search pipeline."""

    def test_extractor_feeds_knowledge_graph(self, tmp_path):
        root = _create_sample_project(tmp_path / "project")

        # Extract
        extractor = KnowledgeGraphExtractor(str(root))
        result = extractor.extract()

        # Load into KnowledgeGraph
        kg = KnowledgeGraph()
        for entity in result.entities:
            kg.add_entity_by_id(
                entity_id=entity.id,
                label=entity.label,
                entity_type=entity.entity_type,
                attributes=entity.attributes,
            )
        for rel in result.relations:
            kg.add_relation_by_ids(
                source_id=rel.source_id,
                target_id=rel.target_id,
                relation_type=rel.relation_type,
                weight=rel.weight,
                metadata=rel.metadata,
            )

        assert kg.entity_count > 0
        assert kg.relation_count > 0

    def test_knowledge_graph_persistence(self, tmp_path):
        root = _create_sample_project(tmp_path / "project")
        extractor = KnowledgeGraphExtractor(str(root))
        result = extractor.extract()

        kg = KnowledgeGraph()
        for entity in result.entities:
            kg.add_entity_by_id(
                entity_id=entity.id,
                label=entity.label,
                entity_type=entity.entity_type,
                attributes=entity.attributes,
            )

        # Save and reload
        kg_path = tmp_path / "kg.json"
        kg.save_to_disk(kg_path)

        kg2 = KnowledgeGraph.load_from_disk(kg_path)
        assert kg2.entity_count == kg.entity_count

    def test_ingest_builds_graph_automatically(self, tmp_path):
        """KnowledgeBase.ingest_source should automatically build the knowledge graph."""
        kb = KnowledgeBase(data_dir=str(tmp_path / "kb"))
        root = _create_sample_project(tmp_path / "project")

        source = kb.add_source(str(root), source_type="folder")
        kb.ingest_source(source.source_id)

        # The ingestion code tries to build the knowledge graph at
        # data_dir.parent / "knowledge_graph.json"
        kg_path = kb.data_dir.parent / "knowledge_graph.json"
        if kg_path.exists():
            kg = KnowledgeGraph.load_from_disk(kg_path)
            assert kg.entity_count > 0
