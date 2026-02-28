"""
Streaming data loader for pre-training.

Loads tokenized shards via memory-mapping and serves batches
of packed sequences. Supports:
- Multi-worker data loading (PyTorch DataLoader compatible)
- Automatic sequence packing across documents
- Shuffling within and across shards
- Distributed training support (rank-aware sharding)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from hbllm.data.sharder import ShardReader

logger = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    """
    Map-style dataset backed by memory-mapped shards.

    Each item is a sequence of token IDs of fixed length.
    Provides random access for shuffled training.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        sequence_length: int | None = None,
    ):
        self.reader = ShardReader(shard_dir)
        self.sequence_length = sequence_length if sequence_length else self.reader.sequence_length
        
        # Override reader seq len if needed for num_sequences
        self._len = max(0, (self.reader.total_tokens - self.sequence_length) // self.sequence_length)

        logger.info(
            "PretrainingDataset: %d tokens, %d sequences of length %d",
            self.reader.total_tokens,
            self._len,
            sequence_length,
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dict with:
            - input_ids: [sequence_length] token IDs
            - labels: [sequence_length] shifted labels (same as input_ids for CLM)
        """
        start = idx * self.sequence_length
        tokens = self.reader.get_sequence(start)
        input_ids = torch.tensor(tokens.tolist(), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class StreamingPretrainingDataset(IterableDataset):
    """
    Streaming dataset that reads shards sequentially.

    Better for very large datasets that don't fit in memory.
    Supports distributed training by splitting shards across workers.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        sequence_length: int = 2048,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.shard_dir = Path(shard_dir)
        self.sequence_length = sequence_length
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self._shard_paths = sorted(self.shard_dir.glob("shard_*.bin"))

        if not self._shard_paths:
            raise FileNotFoundError(f"No shard files in {shard_dir}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over sequences from all shards."""
        worker_info = torch.utils.data.get_worker_info()

        # Split shards across workers
        shard_paths = list(self._shard_paths)

        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(shard_paths)

        if worker_info is not None:
            # Multi-worker: each worker handles a subset of shards
            per_worker = len(shard_paths) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(shard_paths)
            shard_paths = shard_paths[start:end]

        # Stream sequences from assigned shards
        for shard_path in shard_paths:
            try:
                reader = ShardReader(shard_path.parent)
                for seq in reader.iter_sequences():
                    input_ids = torch.tensor(seq.tolist(), dtype=torch.long)
                    yield {
                        "input_ids": input_ids,
                        "labels": input_ids.clone(),
                    }
            except Exception:
                logger.exception("Error reading shard %s", shard_path)
                continue


def create_dataloader(
    shard_dir: str | Path,
    sequence_length: int | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    streaming: bool = False,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader for pre-training.

    Args:
        shard_dir: Directory containing .bin shard files
        sequence_length: Token sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        streaming: Use streaming (IterableDataset) mode
        shuffle: Shuffle data (only for map-style dataset)
        seed: Random seed

    Returns:
        PyTorch DataLoader
    """
    if streaming:
        dataset = StreamingPretrainingDataset(
            shard_dir=shard_dir,
            sequence_length=sequence_length,
            shuffle_shards=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataset = PretrainingDataset(
            shard_dir=shard_dir,
            sequence_length=sequence_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
