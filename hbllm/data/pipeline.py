"""
End-to-End Data Pipeline for HBLLM.

Coordinates the downloading, cleaning, deduplication, and sharding
of training data, utilizing the high-performance Rust extensions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from hbllm.data.downloader import DatasetDownloader, iter_jsonl_dir
from hbllm.data.sharder import ShardWriter
from hbllm.data.scorer import QualityScorer

try:
    from hbllm_data_tools_rs import fast_clean_batch, Deduplicator
    from hbllm_tokenizer_rs import Vocab, Trainer
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust extensions not installed. Pipeline will fail if run.")

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Coordinates the full data preparation pipeline:
    1. Download to JSONL
    2. Clean text & Filter (Rust)
    3. MinHash Deduplication (Rust)
    4. Train BPE Tokenizer (Rust)
    5. Tokenize & Pack into Binary Shards
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
                "Rust extensions (hbllm_data_tools_rs, hbllm_tokenizer_rs) "
                "must be installed to run the pipeline."
            )

    def run_all(
        self,
        dataset_name: str = "fineweb",
        max_samples: int = 100_000,
        vocab_size: int = 32768,
        sequence_length: int = 2048,
    ) -> None:
        """Run the complete pipeline end-to-end."""
        logger.info("Starting HBLLM Data Pipeline in %s", self.work_dir)
        start_time = time.time()

        # Step 1: Download
        self._step_download(dataset_name, max_samples)

        # Step 2 & 3: Clean and Deduplicate (simultaneous stream)
        clean_docs = self._step_clean_and_dedup(dataset_name)

        # Step 4: Train Tokenizer
        vocab = self._step_train_tokenizer(clean_docs, vocab_size)

        # Step 5: Shard and Pack
        self._step_shard(clean_docs, vocab, sequence_length)

        elapsed = time.time() - start_time
        logger.info("Pipeline completed successfully in %.2fs!", elapsed)

    def _step_download(self, dataset_name: str, max_samples: int) -> None:
        """Download dataset to local JSONL."""
        logger.info("Step 1: Downloading %s...", dataset_name)
        downloader = DatasetDownloader(self.raw_dir)
        downloader.add_predefined(dataset_name, max_samples=max_samples)
        
        # Download all configured sources
        for source in downloader.sources:
            downloader.download_to_jsonl(source)

    def _step_clean_and_dedup(self, dataset_name: str) -> list[str]:
        """Clean texts and remove near-duplicates."""
        logger.info("Step 2 & 3: Cleaning and deduplicating...")
        
        # Initialize Rust LSH deduplicator
        dedup = Deduplicator(num_perm=128, threshold=0.8, shingle_size=5)
        
        docs = list(iter_jsonl_dir(self.raw_dir / dataset_name))
        logger.info("Loaded %d raw documents", len(docs))

        # Batch clean
        t0 = time.time()
        cleaned = fast_clean_batch(docs)
        logger.info("Cleaned %d documents in %.2fs", len(cleaned), time.time() - t0)

        # Filter by quality
        scorer = QualityScorer()
        high_quality = [doc for doc in cleaned if scorer.is_high_quality(doc)]
        
        logger.info(
            "Quality scoring removed %d low-quality docs out of %d",
            len(cleaned) - len(high_quality),
            len(cleaned),
        )

        # Batch dedup
        t0 = time.time()
        
        unique_docs = dedup.deduplicate(high_quality)
        duplicates_found = len(high_quality) - len(unique_docs)

        logger.info(
            "Deduplication finished in %.2fs. Removed %d duplicates. Kept %d unique docs.",
            time.time() - t0,
            duplicates_found,
            len(unique_docs),
        )
        return unique_docs

    def _step_train_tokenizer(self, docs: list[str], vocab_size: int) -> Vocab:
        """Train the BPE tokenizer on the cleaned corpus."""
        logger.info("Step 4: Training BPE Tokenizer (vocab_size=%d)...", vocab_size)
        t0 = time.time()
        
        trainer = Trainer(vocab_size=vocab_size, min_frequency=2)
        
        # Combine a sample of docs for training
        # If corpus is huge, we'd sample here. For now we use the whole thing.
        train_text = " ".join(docs)
        
        vocab = trainer.train_from_text(train_text)
        vocab.save(str(self.vocab_path))
        
        logger.info(
            "Tokenizer training finished in %.2fs. Saved to %s (vocab size: %d)",
            time.time() - t0,
            self.vocab_path,
            len(vocab)
        )
        return vocab

    def _step_shard(self, docs: list[str], vocab: Vocab, sequence_length: int) -> None:
        """Tokenize documents and pack them into binary shards."""
        logger.info("Step 5: Tokenizing and compiling binary shards...")
        t0 = time.time()
        
        writer = ShardWriter(self.shard_dir, shard_size_mb=256, sequence_length=sequence_length)
        
        total_tokens = 0
        for i, text in enumerate(docs):
            if i > 0 and i % 10000 == 0:
                logger.info("  Tokenized %d docs", i)
                
            # Rust tokenizer is very fast
            tokens = vocab.encode(text)
            
            # Add EOS token (assume it's the first special token, usually at end of vocab)
            # Find exact ID later, for now we just append standard separator
            tokens.append(vocab.encode("<|eos|>")[0] if "<|eos|>" in text else 0) 
            
            writer.add_tokens(tokens)
            total_tokens += len(tokens)
            
        writer.flush()
        
        logger.info(
            "Sharding finished in %.2fs. Wrote %d tokens across %d shards.",
            time.time() - t0,
            writer.total_tokens,
            len(writer.created_shards)
        )
