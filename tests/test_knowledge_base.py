"""Tests for core KnowledgeBase."""

import os
import tempfile

import pytest

from hbllm.knowledge import KnowledgeBase, Source


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def kb(tmp_dir):
    return KnowledgeBase(data_dir=tmp_dir)


class TestKnowledgeBaseInit:
    def test_creates_data_dir(self, tmp_dir):
        kb = KnowledgeBase(data_dir=os.path.join(tmp_dir, "sub", "kb"))
        assert os.path.isdir(kb.data_dir)

    def test_empty_on_start(self, kb):
        assert len(kb.list_sources()) == 0
        stats = kb.get_stats()
        assert stats["total_sources"] == 0


class TestSourceManagement:
    def test_add_file_source(self, kb, tmp_dir):
        # Create a test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello, world!")

        source = kb.add_source(test_file, "file")
        assert source.source_type == "file"
        assert source.status == "pending"
        assert source.name == "test.txt"
        assert source.file_type == "text"

    def test_add_folder_source(self, kb, tmp_dir):
        folder = os.path.join(tmp_dir, "project")
        os.makedirs(folder)
        with open(os.path.join(folder, "main.py"), "w") as f:
            f.write("print('hello')")

        source = kb.add_source(folder, "folder")
        assert source.source_type == "folder"

    def test_auto_detect_type(self, kb, tmp_dir):
        test_file = os.path.join(tmp_dir, "readme.md")
        with open(test_file, "w") as f:
            f.write("# Readme")

        source = kb.add_source(test_file)  # auto
        assert source.source_type == "file"

    def test_duplicate_source_returns_existing(self, kb, tmp_dir):
        test_file = os.path.join(tmp_dir, "dup.txt")
        with open(test_file, "w") as f:
            f.write("data")

        s1 = kb.add_source(test_file)
        s2 = kb.add_source(test_file)
        assert s1.source_id == s2.source_id

    def test_remove_source(self, kb, tmp_dir):
        test_file = os.path.join(tmp_dir, "rem.txt")
        with open(test_file, "w") as f:
            f.write("remove me")

        source = kb.add_source(test_file)
        assert kb.remove_source(source.source_id)
        assert len(kb.list_sources()) == 0

    def test_remove_nonexistent_returns_false(self, kb):
        assert kb.remove_source("nonexistent") is False

    def test_add_nonexistent_raises(self, kb):
        with pytest.raises(FileNotFoundError):
            kb.add_source("/no/such/path")


class TestIngestion:
    def test_ingest_file(self, kb, tmp_dir):
        test_file = os.path.join(tmp_dir, "ingest.txt")
        with open(test_file, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 10)

        source = kb.add_source(test_file)
        chunks = kb.ingest_source(source.source_id)
        assert chunks > 0
        assert source.status == "ready"

    def test_ingest_folder(self, kb, tmp_dir):
        folder = os.path.join(tmp_dir, "proj")
        os.makedirs(folder)

        for i in range(3):
            with open(os.path.join(folder, f"f{i}.py"), "w") as f:
                f.write(f"def func_{i}(): pass\n" * 20)

        source = kb.add_source(folder)
        chunks = kb.ingest_source(source.source_id)
        assert chunks >= 3
        assert source.file_count == 3

    def test_ingest_nonexistent_raises(self, kb):
        with pytest.raises(ValueError):
            kb.ingest_source("bad_id")


class TestSearch:
    def test_search_returns_results(self, kb, tmp_dir):
        test_file = os.path.join(tmp_dir, "search.txt")
        with open(test_file, "w") as f:
            f.write("Machine learning is a subset of artificial intelligence. " * 5)

        source = kb.add_source(test_file)
        kb.ingest_source(source.source_id)

        results = kb.search("machine learning")
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_search_empty_returns_empty(self, kb):
        assert kb.search("") == []


class TestPersistence:
    def test_manifest_persists(self, tmp_dir):
        test_file = os.path.join(tmp_dir, "persist.txt")
        with open(test_file, "w") as f:
            f.write("persistence test")

        kb1 = KnowledgeBase(data_dir=tmp_dir)
        kb1.add_source(test_file)

        # Reload
        kb2 = KnowledgeBase(data_dir=tmp_dir)
        assert len(kb2.list_sources()) == 1


class TestChunking:
    def test_text_chunking(self, kb):
        header = "[File: test.txt]\n"
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = kb._chunk_text(header, text, chunk_size=100)
        assert len(chunks) > 1
        assert all(c.startswith(header) for c in chunks)

    def test_code_chunking(self, kb):
        header = "[File: test.py]\n"
        code = "\n".join([f"line_{i} = {i}" for i in range(200)])
        chunks = kb._chunk_code(header, code, max_lines=50)
        assert len(chunks) > 1


class TestSourceModel:
    def test_source_roundtrip(self):
        s = Source(
            source_id="abc",
            path="/tmp/test",
            source_type="file",
            name="test",
            file_type="text",
        )
        d = s.to_dict()
        s2 = Source.from_dict(d)
        assert s2.source_id == s.source_id
        assert s2.path == s.path
