"""
Tests for Unified Knowledge Graph Extractor and Ingestion Pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hbllm.knowledge.extractor import (
    FolderExtractor,
    JsonExtractor,
    KnowledgeGraphExtractor,
    MarkdownExtractor,
    PythonExtractor,
)
from hbllm.knowledge.knowledge_base import KnowledgeBase
from hbllm.memory.knowledge_graph import KnowledgeGraph


@pytest.fixture
def temp_project():
    """Sets up a temporary directory structure mimicking a project with python, md, and json files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create directory structure
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # 2. Create Python file
        py_content = """
import os
from collections import defaultdict
from hbllm.memory.knowledge_graph import KnowledgeGraph

class CustomBase:
    pass

class ExtractorSystem(CustomBase):
    def __init__(self, val):
        self.val = val

    def process(self):
        return self.val

def helper_func():
    return 42
"""
        with open(src_dir / "extractor.py", "w", encoding="utf-8") as f:
            f.write(py_content)

        # 3. Create Markdown file
        md_content = """
# Getting Started

This is documentation for the Knowledge Graph.

## Ingestion System

Here we detail the [[FolderExtractor]] and [[MarkdownExtractor]].
See [Google](https://google.com) for details.

#tag_one #tag_two
"""
        with open(docs_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        # 4. Create JSON config file
        json_content = {
            "app": {
                "name": "HBLLM Studio",
                "port": "${APP_PORT}",
                "database": {"host": "localhost", "user": "root"},
            }
        }
        with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(json_content, f)

        yield tmp_path


class TestKnowledgeGraphExtractor:
    def test_folder_structure_extraction(self, temp_project):
        """Verifies FolderExtractor traverses structure and generates contains relations with stable IDs."""
        extractor = FolderExtractor(str(temp_project))
        result = extractor.extract()

        entities_map = {e.id: e for e in result.entities}

        # Verify folder nodes exist
        assert "folder::." in entities_map
        assert "folder::src" in entities_map
        assert "folder::docs" in entities_map

        # Verify file nodes exist
        assert "file::src/extractor.py" in entities_map
        assert "file::docs/index.md" in entities_map
        assert "file::config.json" in entities_map

        # Verify category and types
        assert entities_map["folder::src"].entity_type == "folder"
        assert entities_map["folder::src"].attributes["category"] == "structure"
        assert entities_map["file::src/extractor.py"].entity_type == "module"
        assert entities_map["file::docs/index.md"].entity_type == "document"

        # Verify contains relationships
        contains_rels = [
            (r.source_id, r.target_id) for r in result.relations if r.relation_type == "contains"
        ]
        assert ("folder::.", "folder::src") in contains_rels
        assert ("folder::src", "file::src/extractor.py") in contains_rels
        assert ("folder::.", "file::config.json") in contains_rels

    def test_markdown_extraction(self, temp_project):
        """Verifies MarkdownExtractor parses nested headers, tags, wikilinks, and external links."""
        md_file = temp_project / "docs" / "index.md"
        extractor = MarkdownExtractor(str(temp_project))
        result = extractor.extract(str(md_file))

        entities_map = {e.id: e for e in result.entities}

        # Verify nested section headers (stable structural IDs)
        assert "section::docs/index.md#h1_1" in entities_map
        assert "section::docs/index.md#h2_1" in entities_map

        assert entities_map["section::docs/index.md#h1_1"].label == "Getting Started"
        assert entities_map["section::docs/index.md#h1_1"].entity_type == "section"
        assert entities_map["section::docs/index.md#h1_1"].attributes["category"] == "knowledge"

        # Verify contains links from document -> section -> subsection
        contains_rels = [
            (r.source_id, r.target_id) for r in result.relations if r.relation_type == "contains"
        ]
        assert ("file::docs/index.md", "section::docs/index.md#h1_1") in contains_rels
        assert ("section::docs/index.md#h1_1", "section::docs/index.md#h2_1") in contains_rels

        # Verify tags
        assert "tag::tag_one" in entities_map
        assert "tag::tag_two" in entities_map

        # Verify wikilink placeholder creation
        assert "file::folderextractor.md" in entities_map
        assert entities_map["file::folderextractor.md"].attributes.get("placeholder") is True

        # Verify external URL node
        assert "url::https://google.com" in entities_map
        assert entities_map["url::https://google.com"].attributes["name"] == "Google"
        assert entities_map["url::https://google.com"].attributes["category"] == "external"

    def test_json_property_extraction(self, temp_project):
        """Verifies JsonExtractor parses dictionary key paths recursively and captures env variables."""
        json_file = temp_project / "config.json"
        extractor = JsonExtractor(str(temp_project))
        result = extractor.extract(str(json_file))

        entities_map = {e.id: e for e in result.entities}

        # Verify nested properties exist
        assert "property::config.json#app.name" in entities_map
        assert "property::config.json#app.port" in entities_map
        assert "property::config.json#app.database.host" in entities_map

        # Verify contains links
        contains_rels = [
            (r.source_id, r.target_id) for r in result.relations if r.relation_type == "contains"
        ]
        assert ("file::config.json", "property::config.json#app") in contains_rels
        assert ("property::config.json#app", "property::config.json#app.name") in contains_rels

        # Verify scalar value stored as attribute
        assert entities_map["property::config.json#app.name"].attributes["value"] == "HBLLM Studio"
        assert entities_map["property::config.json#app.name"].entity_type == "property"
        assert entities_map["property::config.json#app.name"].attributes["category"] == "code"

        # Verify environment variable reference detection
        assert "env::APP_PORT" in entities_map
        refers_to_rels = [
            (r.source_id, r.target_id) for r in result.relations if r.relation_type == "refers_to"
        ]
        assert ("property::config.json#app.port", "env::APP_PORT") in refers_to_rels

    def test_python_ast_extraction(self, temp_project):
        """Verifies PythonExtractor parses modules, classes, inheritance bases, functions, and import references."""
        py_file = temp_project / "src" / "extractor.py"
        extractor = PythonExtractor(str(temp_project))
        result = extractor.extract(str(py_file))

        entities_map = {e.id: e for e in result.entities}

        # Verify classes and methods extracted
        assert "class::src/extractor.py#CustomBase" in entities_map
        assert "class::src/extractor.py#ExtractorSystem" in entities_map
        assert "function::src/extractor.py#ExtractorSystem.__init__" in entities_map
        assert "function::src/extractor.py#ExtractorSystem.process" in entities_map

        # Verify module-level functions
        assert "function::src/extractor.py#helper_func" in entities_map

        # Verify class inheritance link (placeholder node constructed)
        assert "class::placeholder#CustomBase" in entities_map
        inherits_rels = [
            (r.source_id, r.target_id) for r in result.relations if r.relation_type == "inherits"
        ]
        assert (
            "class::src/extractor.py#ExtractorSystem",
            "class::placeholder#CustomBase",
        ) in inherits_rels

        # Verify imported symbols (with detailed symbol info encoded in placeholder target ID)
        assert "code::import#hbllm.memory.knowledge_graph.KnowledgeGraph" in entities_map
        imports_rels = [
            (r.source_id, r.target_id) for r in result.relations if r.relation_type == "imports"
        ]
        assert (
            "file::src/extractor.py",
            "code::import#hbllm.memory.knowledge_graph.KnowledgeGraph",
        ) in imports_rels

    def test_two_pass_reference_resolution(self, temp_project):
        """Verifies KnowledgeGraphExtractor correctly resolves placeholders to actual stable IDs in the second pass."""
        orchestrator = KnowledgeGraphExtractor(str(temp_project))
        result = orchestrator.extract()

        entities_map = {e.id: e for e in result.entities}

        # CustomBase should be resolved from class::placeholder#CustomBase -> class::src/extractor.py#CustomBase
        assert "class::placeholder#CustomBase" not in entities_map

        inherits_rels = [
            (r.source_id, r.target_id) for r in result.relations if r.relation_type == "inherits"
        ]
        assert (
            "class::src/extractor.py#ExtractorSystem",
            "class::src/extractor.py#CustomBase",
        ) in inherits_rels

    def test_incremental_pruning(self, temp_project):
        """Verifies KnowledgeGraph.remove_by_source cleans up prior source-associated nodes and relations."""
        kg = KnowledgeGraph()

        # Add legacy nodes with source ID 'src_1'
        kg.add_entity_by_id(
            "file::legacy.py",
            "legacy.py",
            "module",
            {"source_id": "src_1", "category": "structure"},
        )
        kg.add_entity_by_id(
            "class::legacy.py#OldClass",
            "OldClass",
            "class",
            {"source_id": "src_1", "category": "code"},
        )
        kg.add_relation_by_ids(
            "file::legacy.py",
            "class::legacy.py#OldClass",
            "contains",
            metadata={"source_id": "src_1"},
        )

        # Add unrelated node with source ID 'src_2'
        kg.add_entity_by_id(
            "file::keep.py", "keep.py", "module", {"source_id": "src_2", "category": "structure"}
        )

        assert kg.entity_count == 3
        assert kg.relation_count == 1

        # Prune source 'src_1'
        kg.remove_by_source("src_1")

        assert kg.entity_count == 1
        assert kg.relation_count == 0
        assert "file::keep.py" in kg._entities
        assert "file::legacy.py" not in kg._entities

    def test_knowledge_base_ingestion_integration(self, temp_project):
        """Verifies whole pipeline end-to-end integration via KnowledgeBase.ingest_source."""
        # Setup KnowledgeBase with temp folder
        kb = KnowledgeBase(data_dir=str(temp_project / "kb_data"))

        # Add source folder
        source = kb.add_source(str(temp_project), "folder")

        # Ingest
        kb.ingest_source(source.source_id)

        # Check that knowledge_graph.json was written to disk and has nodes
        kg_path = kb.data_dir.parent / "knowledge_graph.json"
        assert kg_path.exists()

        # Load from disk
        kg = KnowledgeGraph.load_from_disk(kg_path)
        assert kg.entity_count > 0
        assert kg.relation_count > 0

        # Verify stable IDs are queried correctly
        assert kg._resolve_entity_id("src/extractor.py") == "file::src/extractor.py"
        neighbors = kg.neighbors("file::src/extractor.py")
        assert len(neighbors) > 0
