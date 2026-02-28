"""
Dataset downloader — streaming download from HuggingFace datasets.

Downloads large-scale datasets (FineWeb, The Stack v2, Wikipedia, etc.)
with streaming to avoid OOM, progress tracking, and automatic sharding.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Generator, Iterator

from datasets import IterableDataset, load_dataset

logger = logging.getLogger(__name__)


class DatasetSource:
    """Configuration for a single dataset source."""

    def __init__(
        self,
        name: str,
        dataset_id: str,
        split: str = "train",
        config: str | None = None,
        text_column: str = "text",
        streaming: bool = True,
        max_samples: int | None = None,
    ):
        self.name = name
        self.dataset_id = dataset_id
        self.split = split
        self.config = config
        self.text_column = text_column
        self.streaming = streaming
        self.max_samples = max_samples

    def __repr__(self) -> str:
        return f"<DatasetSource {self.name} ({self.dataset_id})>"


# Pre-configured sources for HBLLM training
PREDEFINED_SOURCES = {
    "fineweb": DatasetSource(
        name="fineweb",
        dataset_id="HuggingFaceFW/fineweb",
        text_column="text",
        streaming=True,
    ),
    "wikipedia": DatasetSource(
        name="wikipedia",
        dataset_id="wikimedia/wikipedia",
        config="20231101.en",
        text_column="text",
        streaming=True,
    ),
    "the_stack_v2": DatasetSource(
        name="the_stack_v2",
        dataset_id="bigcode/the-stack-v2-dedup",
        text_column="content",
        streaming=True,
    ),
}


class DatasetDownloader:
    """
    Downloads and iterates over HuggingFace datasets with streaming.

    Handles:
    - Streaming downloads (no need to download entire dataset first)
    - Multiple dataset sources
    - Text extraction from different column names
    - Sample counting and progress
    - Saving raw text to disk in JSONL format
    """

    def __init__(self, output_dir: str | Path, sources: list[DatasetSource] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sources = sources or []

    def add_source(self, source: DatasetSource) -> None:
        """Add a dataset source."""
        self.sources.append(source)

    def add_predefined(self, name: str, max_samples: int | None = None) -> None:
        """Add a pre-configured dataset source by name."""
        if name not in PREDEFINED_SOURCES:
            raise ValueError(
                f"Unknown source '{name}'. Available: {list(PREDEFINED_SOURCES.keys())}"
            )
        source = PREDEFINED_SOURCES[name]
        if max_samples is not None:
            source.max_samples = max_samples
        self.sources.append(source)

    def stream_texts(self, source: DatasetSource) -> Generator[str, None, None]:
        """
        Stream text documents from a single source.

        Yields one document (string) at a time, never loading the full
        dataset into memory.
        """
        logger.info("Streaming from %s (%s)", source.name, source.dataset_id)

        kwargs: dict[str, Any] = {
            "path": source.dataset_id,
            "split": source.split,
            "streaming": source.streaming,
        }
        if source.config:
            kwargs["name"] = source.config

        ds = load_dataset(**kwargs)

        count = 0
        for sample in ds:
            text = sample.get(source.text_column, "")
            if not text or not isinstance(text, str):
                continue

            yield text
            count += 1

            if count % 10000 == 0:
                logger.info("  %s: streamed %d documents", source.name, count)

            if source.max_samples and count >= source.max_samples:
                logger.info("  %s: reached max_samples=%d", source.name, source.max_samples)
                break

        logger.info("  %s: finished with %d documents", source.name, count)

    def stream_all(self) -> Generator[str, None, None]:
        """Stream texts from all configured sources."""
        for source in self.sources:
            yield from self.stream_texts(source)

    def download_to_jsonl(
        self,
        source: DatasetSource,
        max_file_size_mb: int = 256,
    ) -> list[Path]:
        """
        Download a source to JSONL files on disk.

        Splits output into multiple files once max_file_size_mb is reached.
        Returns list of created file paths.
        """
        source_dir = self.output_dir / source.name
        source_dir.mkdir(parents=True, exist_ok=True)

        created_files: list[Path] = []
        file_idx = 0
        current_size = 0
        max_bytes = max_file_size_mb * 1024 * 1024

        current_path = source_dir / f"shard_{file_idx:05d}.jsonl"
        current_file = open(current_path, "w", encoding="utf-8")
        created_files.append(current_path)

        try:
            for text in self.stream_texts(source):
                line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                line_bytes = len(line.encode("utf-8"))

                # Rotate files when size limit is reached
                if current_size + line_bytes > max_bytes:
                    current_file.close()
                    file_idx += 1
                    current_path = source_dir / f"shard_{file_idx:05d}.jsonl"
                    current_file = open(current_path, "w", encoding="utf-8")
                    created_files.append(current_path)
                    current_size = 0

                current_file.write(line)
                current_size += line_bytes
        finally:
            current_file.close()

        logger.info(
            "Downloaded %s to %d files in %s",
            source.name,
            len(created_files),
            source_dir,
        )
        return created_files


def iter_jsonl(path: Path) -> Generator[str, None, None]:
    """Iterate over text documents in a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue


def iter_jsonl_dir(dir_path: Path) -> Generator[str, None, None]:
    """Iterate over all JSONL files in a directory."""
    for path in sorted(dir_path.glob("*.jsonl")):
        yield from iter_jsonl(path)
