"""
Data sharder — tokenize text and pack into memory-mapped shards.

Takes cleaned, deduplicated text and produces binary shards ready for
the training data loader. Each shard contains packed sequences of
token IDs stored as numpy arrays in memory-mappable format.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Generator, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Shard file header: magic bytes + metadata
SHARD_MAGIC = b"HBLLM_SH"
SHARD_VERSION = 1
HEADER_FMT = "<8sIIIQ"  # magic(8) + version(4) + dtype(4) + seq_len(4) + num_tokens(8)
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class ShardWriter:
    """
    Writes tokenized sequences to binary shard files.

    Output format per shard:
    - Header (28 bytes): magic, version, dtype code, sequence length, total tokens
    - Token data: packed uint16 or uint32 array (numpy mmap-compatible)
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size_mb: int = 256,
        sequence_length: int = 2048,
        dtype: str = "uint16",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        self.sequence_length = sequence_length
        self.dtype = np.dtype(dtype)
        self.dtype_code = 0 if dtype == "uint16" else 1

        self._buffer: list[int] = []
        self._shard_idx = 0
        self._total_tokens = 0
        self._created_shards: list[Path] = []

    def add_tokens(self, token_ids: list[int]) -> None:
        """
        Add token IDs to the buffer.

        When the buffer has enough tokens for the current shard size,
        automatically flushes to disk.
        """
        self._buffer.extend(token_ids)
        self._total_tokens += len(token_ids)

        # Check if we have enough for a shard
        tokens_per_shard = self.shard_size_bytes // self.dtype.itemsize
        while len(self._buffer) >= tokens_per_shard:
            shard_tokens = self._buffer[:tokens_per_shard]
            self._buffer = self._buffer[tokens_per_shard:]
            self._write_shard(shard_tokens)

    def flush(self) -> None:
        """Flush remaining tokens in buffer to a final shard."""
        if self._buffer:
            self._write_shard(self._buffer)
            self._buffer = []

    def _write_shard(self, tokens: list[int]) -> None:
        """Write a single shard file."""
        shard_path = self.output_dir / f"shard_{self._shard_idx:05d}.bin"

        # Pack tokens into numpy array
        arr = np.array(tokens, dtype=self.dtype)

        # Write header + data
        with open(shard_path, "wb") as f:
            header = struct.pack(
                HEADER_FMT,
                SHARD_MAGIC,
                SHARD_VERSION,
                self.dtype_code,
                self.sequence_length,
                len(tokens),
            )
            f.write(header)
            f.write(arr.tobytes())

        self._created_shards.append(shard_path)
        logger.info(
            "Wrote shard %s (%d tokens, %.1f MB)",
            shard_path.name,
            len(tokens),
            shard_path.stat().st_size / 1e6,
        )
        self._shard_idx += 1

    @property
    def created_shards(self) -> list[Path]:
        """List of shard files created so far."""
        return list(self._created_shards)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens written."""
        return self._total_tokens


class ShardReader:
    """
    Reads token shards via memory-mapping for fast random access.

    Shards are loaded lazily and support random sequence extraction
    for the training data loader.
    """

    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        self._shard_paths = sorted(self.shard_dir.glob("shard_*.bin"))
        self._shards: list[np.ndarray] = []
        self._shard_sizes: list[int] = []
        self._total_tokens = 0
        self._sequence_length = 0

        if not self._shard_paths:
            raise FileNotFoundError(f"No shard files found in {shard_dir}")

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Read headers from all shard files."""
        for path in self._shard_paths:
            with open(path, "rb") as f:
                header_bytes = f.read(HEADER_SIZE)
                magic, version, dtype_code, seq_len, num_tokens = struct.unpack(
                    HEADER_FMT, header_bytes
                )

                if magic != SHARD_MAGIC:
                    raise ValueError(f"Invalid shard file: {path}")
                if version != SHARD_VERSION:
                    raise ValueError(f"Unsupported shard version {version} in {path}")

                self._sequence_length = seq_len
                self._shard_sizes.append(num_tokens)
                self._total_tokens += num_tokens

        logger.info(
            "Found %d shards with %d total tokens",
            len(self._shard_paths),
            self._total_tokens,
        )

    def _load_shard(self, idx: int) -> np.ndarray:
        """Memory-map a specific shard."""
        if idx >= len(self._shards) or self._shards[idx] is None:
            # Ensure list is large enough
            while len(self._shards) <= idx:
                self._shards.append(None)

            path = self._shard_paths[idx]
            # Memory-map the data portion (skip header)
            data = np.memmap(
                path,
                dtype=np.uint16,
                mode="r",
                offset=HEADER_SIZE,
                shape=(self._shard_sizes[idx],),
            )
            self._shards[idx] = data

        return self._shards[idx]

    def get_sequence(self, global_idx: int) -> np.ndarray:
        """
        Get a sequence of tokens by global index.

        Args:
            global_idx: Starting token position across all shards

        Returns:
            numpy array of shape [sequence_length]
        """
        # Find which shard this index falls into
        cumulative = 0
        for shard_idx, size in enumerate(self._shard_sizes):
            if global_idx < cumulative + size:
                local_idx = global_idx - cumulative
                shard = self._load_shard(shard_idx)

                # Handle cross-shard boundaries
                end_idx = local_idx + self._sequence_length
                if end_idx <= len(shard):
                    return shard[local_idx:end_idx].copy()
                else:
                    # Need tokens from next shard
                    result = np.zeros(self._sequence_length, dtype=np.uint16)
                    first_part = shard[local_idx:]
                    result[: len(first_part)] = first_part

                    remaining = self._sequence_length - len(first_part)
                    if shard_idx + 1 < len(self._shard_paths):
                        next_shard = self._load_shard(shard_idx + 1)
                        result[len(first_part) : len(first_part) + min(remaining, len(next_shard))] = (
                            next_shard[:remaining]
                        )
                    return result

            cumulative += size

        raise IndexError(f"Global index {global_idx} out of range (total={self._total_tokens})")

    def iter_sequences(self, stride: int | None = None) -> Generator[np.ndarray, None, None]:
        """
        Iterate over all sequences in the dataset.

        Args:
            stride: Step size between sequences. Defaults to sequence_length (no overlap).
        """
        if stride is None:
            stride = self._sequence_length

        max_start = self._total_tokens - self._sequence_length
        for start in range(0, max_start, stride):
            yield self.get_sequence(start)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def num_sequences(self) -> int:
        """Number of non-overlapping sequences available."""
        return max(0, (self._total_tokens - self._sequence_length) // self._sequence_length)

    @property
    def sequence_length(self) -> int:
        return self._sequence_length
