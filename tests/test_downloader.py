"""
Tests for hbllm.data.downloader — dataset registry, text extraction, streaming.
"""
from __future__ import annotations

import json

import pytest

from hbllm.data.downloader import (
    DATASET_DOMAINS,
    PREDEFINED_SOURCES,
    DatasetDownloader,
    DatasetSource,
    iter_jsonl,
    iter_jsonl_dir,
)


class TestDatasetSource:
    """Tests for DatasetSource dataclass."""

    def test_basic_construction(self):
        src = DatasetSource("test", "org/dataset", text_column="text")
        assert src.name == "test"
        assert src.dataset_id == "org/dataset"
        assert src.text_column == "text"
        assert src.split == "train"
        assert src.streaming is True

    def test_with_config(self):
        src = DatasetSource("wiki", "wikimedia/wikipedia", config="20231101.en")
        assert src.config == "20231101.en"

    def test_max_samples(self):
        src = DatasetSource("test", "org/dataset", max_samples=500)
        assert src.max_samples == 500

    def test_repr(self):
        src = DatasetSource("myds", "myorg/myds")
        r = repr(src)
        assert "myds" in r
        assert "myorg/myds" in r


class TestPredefinedSources:
    """Tests for the PREDEFINED_SOURCES registry."""

    def test_all_10_datasets_registered(self):
        expected = {
            "fineweb", "wikipedia", "the_stack_v2", "starcoderdata",
            "codeparrot", "openwebmath", "metamath", "pes2o",
            "openhermes", "slimorca",
        }
        assert set(PREDEFINED_SOURCES.keys()) == expected

    def test_each_source_has_valid_fields(self):
        for name, src in PREDEFINED_SOURCES.items():
            assert src.name == name, f"{name}: name mismatch"
            assert src.dataset_id, f"{name}: missing dataset_id"
            assert src.text_column, f"{name}: missing text_column"
            assert src.streaming is True, f"{name}: should be streaming"

    def test_code_datasets_use_content_column(self):
        for name in ("the_stack_v2", "starcoderdata", "codeparrot"):
            assert PREDEFINED_SOURCES[name].text_column == "content"

    def test_conversation_datasets_use_conversations_column(self):
        for name in ("openhermes", "slimorca"):
            assert PREDEFINED_SOURCES[name].text_column == "conversations"

    def test_metamath_uses_query_column(self):
        assert PREDEFINED_SOURCES["metamath"].text_column == "query"


class TestDatasetDomains:
    """Tests for DATASET_DOMAINS mapping."""

    def test_all_sources_have_domain(self):
        for name in PREDEFINED_SOURCES:
            assert name in DATASET_DOMAINS, f"{name}: missing domain"

    def test_domain_values(self):
        assert DATASET_DOMAINS["fineweb"] == "general"
        assert DATASET_DOMAINS["wikipedia"] == "factual"
        assert DATASET_DOMAINS["starcoderdata"] == "code"
        assert DATASET_DOMAINS["openwebmath"] == "math"
        assert DATASET_DOMAINS["pes2o"] == "science"
        assert DATASET_DOMAINS["openhermes"] == "reasoning"


class TestDatasetDownloader:
    """Tests for DatasetDownloader."""

    def test_add_predefined(self, tmp_path):
        dl = DatasetDownloader(tmp_path)
        dl.add_predefined("fineweb", max_samples=100)
        assert len(dl.sources) == 1
        assert dl.sources[0].name == "fineweb"
        assert dl.sources[0].max_samples == 100

    def test_add_multiple_sources(self, tmp_path):
        dl = DatasetDownloader(tmp_path)
        dl.add_predefined("fineweb", max_samples=50)
        dl.add_predefined("starcoderdata", max_samples=50)
        dl.add_predefined("openwebmath", max_samples=50)
        assert len(dl.sources) == 3
        assert [s.name for s in dl.sources] == ["fineweb", "starcoderdata", "openwebmath"]

    def test_add_unknown_source_raises(self, tmp_path):
        dl = DatasetDownloader(tmp_path)
        with pytest.raises(ValueError, match="Unknown source"):
            dl.add_predefined("nonexistent_dataset")

    def test_add_custom_source(self, tmp_path):
        dl = DatasetDownloader(tmp_path)
        custom = DatasetSource("custom", "myorg/mydata", text_column="body")
        dl.add_source(custom)
        assert dl.sources[0].name == "custom"


class TestExtractText:
    """Tests for _extract_text special format handling."""

    def test_standard_text_column(self):
        src = DatasetSource("test", "org/ds", text_column="text")
        sample = {"text": "Hello world"}
        assert DatasetDownloader._extract_text(sample, src) == "Hello world"

    def test_content_column(self):
        src = DatasetSource("test", "org/ds", text_column="content")
        sample = {"content": "def foo(): pass"}
        assert DatasetDownloader._extract_text(sample, src) == "def foo(): pass"

    def test_conversation_format_from_value(self):
        src = DatasetSource("test", "org/ds", text_column="conversations")
        sample = {
            "conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "Artificial Intelligence is..."},
            ]
        }
        text = DatasetDownloader._extract_text(sample, src)
        assert "human: What is AI?" in text
        assert "gpt: Artificial Intelligence is..." in text

    def test_conversation_format_role_content(self):
        src = DatasetSource("test", "org/ds", text_column="conversations")
        sample = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        text = DatasetDownloader._extract_text(sample, src)
        assert "user: Hello" in text
        assert "assistant: Hi!" in text

    def test_conversation_empty_list(self):
        src = DatasetSource("test", "org/ds", text_column="conversations")
        sample = {"conversations": []}
        assert DatasetDownloader._extract_text(sample, src) == ""

    def test_metamath_query_response(self):
        src = DatasetSource("test", "org/ds", text_column="query")
        sample = {"query": "What is 2+2?", "response": "4"}
        text = DatasetDownloader._extract_text(sample, src)
        assert "Question: What is 2+2?" in text
        assert "Answer: 4" in text

    def test_metamath_empty_query(self):
        src = DatasetSource("test", "org/ds", text_column="query")
        sample = {"query": "", "response": "4"}
        assert DatasetDownloader._extract_text(sample, src) == ""

    def test_missing_column_returns_empty(self):
        src = DatasetSource("test", "org/ds", text_column="text")
        sample = {"other_column": "data"}
        assert DatasetDownloader._extract_text(sample, src) == ""


class TestIterJsonl:
    """Tests for JSONL iteration utilities."""

    def test_iter_jsonl(self, tmp_path):
        path = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"text": "doc 1"}),
            json.dumps({"text": "doc 2"}),
            json.dumps({"text": "doc 3"}),
        ]
        path.write_text("\n".join(lines) + "\n")

        docs = list(iter_jsonl(path))
        assert docs == ["doc 1", "doc 2", "doc 3"]

    def test_iter_jsonl_skips_empty(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"text": "good"}\n\n{"text": ""}\n{"text": "also good"}\n')
        docs = list(iter_jsonl(path))
        assert docs == ["good", "also good"]

    def test_iter_jsonl_skips_malformed(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"text": "ok"}\nnot-json\n{"text": "fine"}\n')
        docs = list(iter_jsonl(path))
        assert docs == ["ok", "fine"]

    def test_iter_jsonl_dir(self, tmp_path):
        for i in range(3):
            path = tmp_path / f"shard_{i:05d}.jsonl"
            path.write_text(json.dumps({"text": f"doc_{i}"}) + "\n")

        docs = list(iter_jsonl_dir(tmp_path))
        assert docs == ["doc_0", "doc_1", "doc_2"]
