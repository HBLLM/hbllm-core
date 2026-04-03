"""
End-to-End Data Pipeline for HBLLM.

Two pipeline implementations:
- DataPipeline: uses high-performance Rust extensions (fast, production)
- PurePythonPipeline: uses tiktoken (no Rust needed, good for experiments)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from hbllm.data.downloader import DatasetDownloader, iter_jsonl_dir
from hbllm.data.scorer import QualityScorer
from hbllm.data.sharder import ShardWriter

try:
    from hbllm_data_tools_rs import Deduplicator, fast_clean_batch
    from hbllm_tokenizer_rs import Trainer, Vocab
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Rust-accelerated data preparation pipeline:
    1. Download to JSONL  2. Clean (Rust)  3. Dedup (Rust)
    4. Train BPE (Rust)  5. Shard
    """

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.raw_dir = self.work_dir / "raw"
        self.shard_dir = self.work_dir / "shards"
        self.vocab_path = self.work_dir / "vocab.json"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.shard_dir.mkdir(exist_ok=True)

        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust extensions not installed. Use PurePythonPipeline instead."
            )

    def run_all(
        self,
        dataset_name: str = "fineweb",
        max_samples: int = 100_000,
        vocab_size: int = 32768,
        sequence_length: int = 2048,
        data_weights: list[float] | None = None,
    ) -> None:
        logger.info("Starting Rust-accelerated Data Pipeline in %s", self.work_dir)
        start = time.time()
        # Parse multi-dataset names (+ separated)
        dataset_names = [d.strip() for d in dataset_name.replace(',', '+').split('+') if d.strip()]
        # Calculate per-dataset samples using weights or equal split
        if data_weights and len(data_weights) == len(dataset_names):
            total_w = sum(data_weights)
            samples_list = [int(max_samples * w / total_w) for w in data_weights]
        else:
            samples_list = [max_samples // len(dataset_names)] * len(dataset_names)
        for ds_name, ds_samples in zip(dataset_names, samples_list):
            self._step_download(ds_name, ds_samples)
        clean_docs = []
        for ds_name in dataset_names:
            clean_docs.extend(self._step_clean_and_dedup(ds_name))
        vocab = self._step_train_tokenizer(clean_docs, vocab_size)
        self._step_shard(clean_docs, vocab, sequence_length)
        logger.info("Pipeline completed in %.2fs!", time.time() - start)

    def _step_download(self, dataset_name, max_samples):
        logger.info("Step 1: Downloading %s...", dataset_name)
        dl = DatasetDownloader(self.raw_dir)
        dl.add_predefined(dataset_name, max_samples=max_samples)
        for src in dl.sources:
            dl.download_to_jsonl(src)

    def _step_clean_and_dedup(self, dataset_name):
        logger.info("Step 2-3: Cleaning and deduplicating...")
        dedup = Deduplicator(num_perm=128, threshold=0.8, shingle_size=5)
        docs = list(iter_jsonl_dir(self.raw_dir / dataset_name))
        cleaned = fast_clean_batch(docs)
        scorer = QualityScorer()
        hq = [d for d in cleaned if scorer.is_high_quality(d)]
        unique = dedup.deduplicate(hq)
        logger.info("Kept %d unique docs from %d raw", len(unique), len(docs))
        return unique

    def _step_train_tokenizer(self, docs, vocab_size):
        logger.info("Step 4: Training BPE Tokenizer...")
        trainer = Trainer(vocab_size=vocab_size, min_frequency=2)
        vocab = trainer.train_from_text(" ".join(docs))
        vocab.save(str(self.vocab_path))
        return vocab

    def _step_shard(self, docs, vocab, sequence_length):
        logger.info("Step 5: Tokenizing and sharding...")
        writer = ShardWriter(self.shard_dir, shard_size_mb=256, sequence_length=sequence_length)
        for i, text in enumerate(docs):
            if i > 0 and i % 10000 == 0:
                logger.info("  Tokenized %d docs", i)
            tokens = vocab.encode(text)
            tokens.append(0)
            writer.add_tokens(tokens)
        writer.flush()
        logger.info("Wrote %d tokens across %d shards", writer.total_tokens, len(writer.created_shards))


# --------------------------------------------------------------------------- #
#  Pure Python Pipeline (tiktoken, no Rust)                                   #
# --------------------------------------------------------------------------- #


def _clean_text(text: str) -> str:
    """Basic text cleaning without Rust extensions."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


class PurePythonPipeline:
    """
    Pure-Python data pipeline -- no Rust extensions required.

    Uses tiktoken (cl100k_base) for tokenization. Suitable for
    quick experiments and local model training.

    Pipeline:
    1. Stream docs from HuggingFace datasets library
    2. Basic text cleaning (Python regex)
    3. Tokenize with tiktoken (cl100k_base, ~100K vocab)
    4. Pack into binary shards (ShardWriter)
    """

    # tiktoken cl100k_base vocab size
    TIKTOKEN_VOCAB_SIZE = 100277

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.shard_dir = self.work_dir / "shards"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir.mkdir(exist_ok=True)

    @property
    def vocab_size(self) -> int:
        """Tiktoken cl100k_base vocab size."""
        return self.TIKTOKEN_VOCAB_SIZE

    def run_all(
        self,
        dataset_name: str = "fineweb",
        max_samples: int = 100_000,
        sequence_length: int = 2048,
        min_doc_tokens: int = 64,
        shard_size_mb: int = 64,
        data_weights: list[float] | None = None,
    ) -> dict:
        """
        Run the full pipeline: stream -> clean -> tokenize -> shard.

        Supports multi-dataset mixing via '+' separator:
            dataset_name="fineweb+starcoderdata+openwebmath"

        Samples are split by weights (if provided) or evenly.
        Returns dict with pipeline statistics.
        """
        import tiktoken

        # Parse multi-dataset names
        dataset_names = [d.strip() for d in dataset_name.replace(',', '+').split('+') if d.strip()]

        # Calculate per-dataset samples using weights or equal split
        if data_weights and len(data_weights) == len(dataset_names):
            total_w = sum(data_weights)
            samples_list = [int(max_samples * w / total_w) for w in data_weights]
        else:
            samples_list = [max_samples // len(dataset_names)] * len(dataset_names)

        logger.info("=" * 60)
        logger.info("HBLLM Pure-Python Data Pipeline")
        logger.info("  Dataset(s) : %s", ' + '.join(dataset_names))
        logger.info("  Max samples: %d (total)", max_samples)
        if data_weights:
            logger.info("  Weights    : %s", ', '.join(f'{w:.2f}' for w in data_weights))
        logger.info("  Seq length : %d", sequence_length)
        logger.info("  Shard dir  : %s", self.shard_dir)
        logger.info("=" * 60)

        start_time = time.time()
        enc = tiktoken.get_encoding("cl100k_base")

        # Use a high token ID as EOS separator between documents
        eos_id = self.TIKTOKEN_VOCAB_SIZE - 1

        writer = ShardWriter(
            self.shard_dir,
            shard_size_mb=shard_size_mb,
            sequence_length=sequence_length,
            dtype="uint32",  # tiktoken has >65535 vocab so need uint32
        )

        # Stream and tokenize
        downloader = DatasetDownloader(self.work_dir / "raw")
        for ds_name, ds_samples in zip(dataset_names, samples_list):
            try:
                downloader.add_predefined(ds_name, max_samples=ds_samples)
                logger.info("  Added dataset: %s (%d max samples)", ds_name, ds_samples)
            except ValueError as e:
                logger.warning("  Skipping unknown dataset '%s': %s", ds_name, e)

        total_docs = 0
        total_tokens = 0
        skipped = 0

        for source in downloader.sources:
            logger.info("Streaming from %s...", source.name)
            for text in downloader.stream_texts(source):
                # Clean
                text = _clean_text(text)
                if len(text) < 100:
                    skipped += 1
                    continue

                # Tokenize
                tokens = enc.encode(text, disallowed_special=())

                # Filter short docs
                if len(tokens) < min_doc_tokens:
                    skipped += 1
                    continue

                # Add EOS separator and write
                tokens.append(eos_id)
                writer.add_tokens(tokens)
                total_tokens += len(tokens)
                total_docs += 1

                if total_docs % 5000 == 0:
                    elapsed = time.time() - start_time
                    tps = total_tokens / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  %d docs | %d tokens | %.0f tok/s | %d shards",
                        total_docs, total_tokens, tps, len(writer.created_shards),
                    )

        # Flush remaining buffer
        writer.flush()

        elapsed = time.time() - start_time
        stats = {
            "total_docs": total_docs,
            "total_tokens": total_tokens,
            "skipped_docs": skipped,
            "num_shards": len(writer.created_shards),
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(total_tokens / max(1, elapsed)),
            "shard_dir": str(self.shard_dir),
            "vocab_size": self.TIKTOKEN_VOCAB_SIZE,
        }

        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info("  Documents : %d (skipped %d)", total_docs, skipped)
        logger.info("  Tokens    : %d", total_tokens)
        logger.info("  Shards    : %d", len(writer.created_shards))
        logger.info("  Time      : %.1fs (%.0f tok/s)", elapsed, stats["tokens_per_second"])
        logger.info("  Shard dir : %s", self.shard_dir)
        logger.info("=" * 60)

        return stats
