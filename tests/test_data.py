"""
Comprehensive tests for the HBLLM data pipeline.

Covers: downloader (JSONL I/O), sharder (write/read/cross-shard),
and dataloader.
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from hbllm.data.downloader import DatasetSource, iter_jsonl, iter_jsonl_dir
from hbllm.data.sharder import (
    HEADER_FMT,
    HEADER_SIZE,
    SHARD_MAGIC,
    SHARD_VERSION,
    ShardReader,
    ShardWriter,
)


# ──────────────────────────────────────────────
# Downloader / JSONL tests
# ──────────────────────────────────────────────

class TestDatasetSource:
    def test_creation(self):
        src = DatasetSource(name="test", dataset_id="org/dataset", text_column="text")
        assert src.name == "test"
        assert src.streaming is True

    def test_repr(self):
        src = DatasetSource(name="wiki", dataset_id="wikimedia/wikipedia")
        assert "wiki" in repr(src)


class TestJsonlIO:
    def test_write_and_read_jsonl(self, tmp_path: Path):
        """Write JSONL manually and read it back."""
        file_path = tmp_path / "test.jsonl"
        docs = ["Hello world", "Second document", "Third one"]

        with open(file_path, "w") as f:
            for doc in docs:
                f.write(json.dumps({"text": doc}) + "\n")

        result = list(iter_jsonl(file_path))
        assert result == docs

    def test_iter_jsonl_skips_empty(self, tmp_path: Path):
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
            f.write("\n")  # Empty line
            f.write(json.dumps({"text": "world"}) + "\n")

        result = list(iter_jsonl(file_path))
        assert len(result) == 2

    def test_iter_jsonl_skips_malformed(self, tmp_path: Path):
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({"text": "valid"}) + "\n")
            f.write("not json\n")
            f.write(json.dumps({"text": "also valid"}) + "\n")

        result = list(iter_jsonl(file_path))
        assert len(result) == 2

    def test_iter_jsonl_skips_missing_text_key(self, tmp_path: Path):
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({"text": "valid"}) + "\n")
            f.write(json.dumps({"other_key": "no text"}) + "\n")

        result = list(iter_jsonl(file_path))
        assert len(result) == 1

    def test_iter_jsonl_dir(self, tmp_path: Path):
        for i in range(3):
            file_path = tmp_path / f"shard_{i:05d}.jsonl"
            with open(file_path, "w") as f:
                f.write(json.dumps({"text": f"doc {i}"}) + "\n")

        result = list(iter_jsonl_dir(tmp_path))
        assert len(result) == 3


# ──────────────────────────────────────────────
# Sharder tests
# ──────────────────────────────────────────────

class TestShardWriter:
    def test_write_single_shard(self, tmp_path: Path):
        writer = ShardWriter(tmp_path, shard_size_mb=1, sequence_length=32)
        tokens = list(range(1000))
        writer.add_tokens(tokens)
        writer.flush()

        assert len(writer.created_shards) >= 1
        assert writer.total_tokens == 1000

    def test_shard_file_header(self, tmp_path: Path):
        writer = ShardWriter(tmp_path, shard_size_mb=1, sequence_length=32)
        writer.add_tokens(list(range(100)))
        writer.flush()

        shard_path = writer.created_shards[0]
        with open(shard_path, "rb") as f:
            header = f.read(HEADER_SIZE)
            magic, version, dtype_code, seq_len, num_tokens = struct.unpack(HEADER_FMT, header)
            assert magic == SHARD_MAGIC
            assert version == SHARD_VERSION
            assert seq_len == 32
            assert num_tokens == 100

    def test_shard_data_roundtrip(self, tmp_path: Path):
        writer = ShardWriter(tmp_path, shard_size_mb=1, sequence_length=32)
        original = list(range(500))
        writer.add_tokens(original)
        writer.flush()

        shard_path = writer.created_shards[0]
        with open(shard_path, "rb") as f:
            f.read(HEADER_SIZE)  # Skip header
            data = np.frombuffer(f.read(), dtype=np.uint16)
            assert list(data) == original

    def test_auto_rotation(self, tmp_path: Path):
        """Shards rotate when size limit is reached."""
        # 1 KB shards, uint16 = 2 bytes per token → 512 tokens per shard
        writer = ShardWriter(tmp_path, shard_size_mb=0, sequence_length=32)
        # Manually set smaller shard size for testing
        writer.shard_size_bytes = 1024

        writer.add_tokens(list(range(2000)))
        writer.flush()

        # 2000 tokens × 2 bytes = 4000 bytes → should create ~4 shards
        assert len(writer.created_shards) >= 3

    def test_empty_flush(self, tmp_path: Path):
        writer = ShardWriter(tmp_path / "empty", shard_size_mb=1)
        writer.flush()  # Nothing to flush
        assert len(writer.created_shards) == 0


class TestShardReader:
    def _create_test_shards(self, tmp_path: Path, tokens: list[int], shard_size: int = 10000) -> Path:
        """Helper: create shards with known tokens."""
        writer = ShardWriter(tmp_path, shard_size_mb=1, sequence_length=32)
        writer.shard_size_bytes = shard_size * 2  # uint16 = 2 bytes
        writer.add_tokens(tokens)
        writer.flush()
        return tmp_path

    def test_read_metadata(self, tmp_path: Path):
        self._create_test_shards(tmp_path, list(range(1000)))
        reader = ShardReader(tmp_path)
        assert reader.total_tokens == 1000
        assert reader.sequence_length == 32

    def test_get_sequence(self, tmp_path: Path):
        tokens = list(range(1000))
        self._create_test_shards(tmp_path, tokens)
        reader = ShardReader(tmp_path)

        seq = reader.get_sequence(0)
        assert len(seq) == 32
        assert list(seq) == list(range(32))

        seq2 = reader.get_sequence(100)
        assert seq2[0] == 100

    def test_iter_sequences(self, tmp_path: Path):
        tokens = list(range(1000))
        self._create_test_shards(tmp_path, tokens)
        reader = ShardReader(tmp_path)

        seqs = list(reader.iter_sequences())
        assert len(seqs) > 0
        assert all(len(s) == 32 for s in seqs)

    def test_num_sequences(self, tmp_path: Path):
        tokens = list(range(1000))
        self._create_test_shards(tmp_path, tokens)
        reader = ShardReader(tmp_path)

        expected = (1000 - 32) // 32
        assert reader.num_sequences == expected

    def test_cross_shard_boundary(self, tmp_path: Path):
        """Sequence spanning two shards should work."""
        # Create 2 shards of 500 tokens each
        tokens = list(range(1000))
        self._create_test_shards(tmp_path, tokens, shard_size=500)
        reader = ShardReader(tmp_path)

        # Get sequence starting near end of first shard
        seq = reader.get_sequence(490)
        assert len(seq) == 32
        assert seq[0] == 490

    def test_no_shards_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            ShardReader(empty_dir)

    def test_out_of_range_raises(self, tmp_path: Path):
        tokens = list(range(100))
        self._create_test_shards(tmp_path, tokens)
        reader = ShardReader(tmp_path)

        with pytest.raises(IndexError):
            reader.get_sequence(99999)


# ──────────────────────────────────────────────
# DataLoader tests
# ──────────────────────────────────────────────

class TestPretrainingDataset:
    def _make_shards(self, tmp_path: Path) -> Path:
        writer = ShardWriter(tmp_path, shard_size_mb=1, sequence_length=64)
        writer.add_tokens(list(range(10000)))
        writer.flush()
        return tmp_path

    def test_dataset_creation(self, tmp_path: Path):
        from hbllm.data.dataloader import PretrainingDataset

        shard_dir = self._make_shards(tmp_path)
        ds = PretrainingDataset(shard_dir, sequence_length=64)
        assert len(ds) > 0

    def test_dataset_getitem(self, tmp_path: Path):
        from hbllm.data.dataloader import PretrainingDataset

        shard_dir = self._make_shards(tmp_path)
        ds = PretrainingDataset(shard_dir, sequence_length=64)

        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (64,)
        assert sample["input_ids"].dtype == torch.int64

    def test_dataset_different_indices(self, tmp_path: Path):
        from hbllm.data.dataloader import PretrainingDataset

        shard_dir = self._make_shards(tmp_path)
        ds = PretrainingDataset(shard_dir, sequence_length=64)

        s0 = ds[0]
        s1 = ds[1]
        # Different indices should yield different sequences
        assert not torch.equal(s0["input_ids"], s1["input_ids"])

    def test_create_dataloader(self, tmp_path: Path):
        from hbllm.data.dataloader import create_dataloader

        shard_dir = self._make_shards(tmp_path)
        dl = create_dataloader(shard_dir, sequence_length=64, batch_size=4, num_workers=0)

        batch = next(iter(dl))
        assert batch["input_ids"].shape[0] == 4
        assert batch["input_ids"].shape[1] == 64
