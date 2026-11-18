"""
Embedding Model Fine-Tuning Pipeline.

Trains a small embedding model using contrastive learning (SimCSE/InfoNCE)
on domain-specific data, replacing dependency on external models like
HuggingFace's all-MiniLM-L6-v2.

Usage:
    from hbllm.training.embeddings import EmbeddingTrainer

    trainer = EmbeddingTrainer(embedding_dim=384)
    trainer.train(train_pairs, epochs=10)
    trainer.save("./checkpoints/embeddings/domain_v1.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MiniEmbeddingModel(nn.Module):
    """
    A lightweight embedding model for semantic search.

    Architecture: Embedding → 2-layer Transformer Encoder → Mean Pooling → Projection
    Designed to be small enough to train on a single GPU in minutes.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        embedding_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # type: ignore[no-untyped-call]

        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode input tokens into embeddings.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], 1 for real tokens, 0 for padding.

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape
        seq_len = min(seq_len, self.max_seq_len)
        input_ids = input_ids[:, :seq_len]

        positions = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        )
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Transformer encoding
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling (ignoring padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        # Project and normalize
        projected = self.proj(self.layer_norm(pooled))
        return F.normalize(projected, p=2, dim=-1)


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss.

    Treats other examples in the batch as negative pairs (in-batch negatives).

    Args:
        anchor: [batch_size, dim] — anchor embeddings
        positive: [batch_size, dim] — positive pair embeddings
        temperature: Temperature scaling (lower = sharper)

    Returns:
        Scalar loss
    """
    # Similarity matrix: [batch_size, batch_size]
    similarity = torch.matmul(anchor, positive.T) / temperature

    # Labels: diagonal entries are the positive pairs
    labels = torch.arange(anchor.size(0), device=anchor.device)

    return F.cross_entropy(similarity, labels)


class EmbeddingTrainer:
    """
    Trains a MiniEmbeddingModel using contrastive learning.

    Args:
        embedding_dim: Output embedding dimension.
        vocab_size: Vocabulary size.
        device: Training device.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        vocab_size: int = 32768,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = MiniEmbeddingModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        ).to(self.device)

        self.embedding_dim = embedding_dim

    def train(
        self,
        train_pairs: list[tuple[list[int], list[int]]],
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        temperature: float = 0.07,
    ) -> dict[str, Any]:
        """
        Train the embedding model on pairs of (anchor_ids, positive_ids).

        Args:
            train_pairs: List of (anchor_token_ids, positive_token_ids) pairs.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Learning rate.
            temperature: InfoNCE temperature.

        Returns:
            Dict with training metrics.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        total_steps = 0
        losses: list[float] = []

        logger.info(
            "Training embedding model: %d pairs, %d epochs, batch_size=%d",
            len(train_pairs),
            epochs,
            batch_size,
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle pairs each epoch
            indices = torch.randperm(len(train_pairs))

            for start in range(0, len(train_pairs), batch_size):
                batch_indices = indices[start : start + batch_size]
                if len(batch_indices) < 2:
                    continue  # Need at least 2 for contrastive learning

                # Pad and stack
                anchor_ids, positive_ids, anchor_mask, positive_mask = self._collate(
                    [train_pairs[int(i)] for i in batch_indices]
                )

                anchor_emb = self.model(anchor_ids.to(self.device), anchor_mask.to(self.device))
                positive_emb = self.model(
                    positive_ids.to(self.device), positive_mask.to(self.device)
                )

                loss: torch.Tensor = info_nce_loss(anchor_emb, positive_emb, temperature)

                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1
                total_steps += 1

            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)

            logger.info(
                "  Epoch %d/%d: loss=%.4f lr=%.2e",
                epoch + 1,
                epochs,
                avg_loss,
                scheduler.get_last_lr()[0],
            )

        logger.info(
            "Embedding training complete: %d steps, final_loss=%.4f",
            total_steps,
            losses[-1] if losses else 0,
        )
        return {
            "total_steps": total_steps,
            "epochs": epochs,
            "final_loss": round(losses[-1], 4) if losses else 0.0,
            "loss_history": [round(l, 4) for l in losses],
        }

    def save(self, path: str | Path) -> Path:
        """Save the trained embedding model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "embedding_dim": self.embedding_dim,
            },
            path,
        )
        logger.info("Embedding model saved to %s", path)
        return path

    def load(self, path: str | Path) -> None:
        """Load a previously saved embedding model."""
        state = cast(dict[str, Any], torch.load(path, map_location=self.device, weights_only=True))
        self.model.load_state_dict(state["model_state_dict"])
        logger.info("Embedding model loaded from %s", path)

    def encode(self, token_ids_batch: list[list[int]]) -> np.ndarray[Any, Any]:
        """
        Encode a batch of token ID sequences into embeddings.

        Args:
            token_ids_batch: List of token ID lists.

        Returns:
            numpy array of shape [batch_size, embedding_dim]
        """
        self.model.eval()
        ids, mask = self._pad_batch(token_ids_batch)

        with torch.no_grad():
            embeddings = self.model(ids.to(self.device), mask.to(self.device))

        return cast(np.ndarray[Any, Any], embeddings.cpu().numpy())

    @staticmethod
    def _collate(
        pairs: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad and stack pairs into tensors."""
        anchors = [p[0] for p in pairs]
        positives = [p[1] for p in pairs]

        a_ids, a_mask = EmbeddingTrainer._pad_batch(anchors)
        p_ids, p_mask = EmbeddingTrainer._pad_batch(positives)

        return a_ids, p_ids, a_mask, p_mask

    @staticmethod
    def _pad_batch(
        sequences: list[list[int]],
        max_len: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences to uniform length and create attention mask."""
        batch_max = min(max(len(s) for s in sequences), max_len)
        padded = []
        masks = []
        for seq in sequences:
            truncated = seq[:batch_max]
            pad_len = batch_max - len(truncated)
            padded.append(truncated + [0] * pad_len)
            masks.append([1] * len(truncated) + [0] * pad_len)

        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)
