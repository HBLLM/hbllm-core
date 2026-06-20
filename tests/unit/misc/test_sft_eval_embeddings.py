"""
Tests for hbllm.training.sft — Supervised Fine-Tuning pipeline.
Tests for hbllm.training.evaluator — Model evaluation benchmarks.
Tests for hbllm.training.embeddings — Embedding model & contrastive training.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch

from hbllm.training.embeddings import (
    EmbeddingTrainer,
    MiniEmbeddingModel,
    info_nce_loss,
)
from hbllm.training.evaluator import ModelEvaluator
from hbllm.training.sft import InstructionDataset, collate_sft, load_sft_data

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for SFT tests."""
    tok = MagicMock()
    tok.pad_id = 0
    tok.encode = MagicMock(side_effect=lambda text, **kw: list(range(10)))
    tok.decode = MagicMock(side_effect=lambda ids: "decoded text")
    tok.apply_chat_template = MagicMock(
        side_effect=lambda msgs, **kw: " ".join(m.get("content", "") for m in msgs)
    )
    return tok


@pytest.fixture
def alpaca_data():
    """Sample Alpaca-format data."""
    return [
        {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
        {"instruction": "Translate hello", "input": "to French", "output": "Bonjour"},
        {"instruction": "Sum 2+2", "input": "", "output": "4"},
    ]


@pytest.fixture
def sharegpt_data():
    """Sample ShareGPT-format data."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        },
    ]


# ─── SFT Tests ────────────────────────────────────────────────────────────────


class TestInstructionDataset:
    """Tests for InstructionDataset."""

    def test_alpaca_format(self, mock_tokenizer, alpaca_data):
        ds = InstructionDataset(alpaca_data, mock_tokenizer, max_length=256)
        assert len(ds) == 3

    def test_sharegpt_format(self, mock_tokenizer, sharegpt_data):
        ds = InstructionDataset(sharegpt_data, mock_tokenizer, max_length=256)
        assert len(ds) == 2

    def test_getitem_returns_tensors(self, mock_tokenizer, alpaca_data):
        ds = InstructionDataset(alpaca_data, mock_tokenizer, max_length=256)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_labels_mask_prompt(self, mock_tokenizer, alpaca_data):
        ds = InstructionDataset(alpaca_data, mock_tokenizer, max_length=256)
        item = ds[0]
        # Some labels should be -100 (masked prompt tokens)
        assert (item["labels"] == -100).any()

    def test_empty_data(self, mock_tokenizer):
        ds = InstructionDataset([], mock_tokenizer)
        assert len(ds) == 0

    def test_to_messages_alpaca(self, mock_tokenizer):
        ds = InstructionDataset([], mock_tokenizer)
        msgs = ds._to_messages({"instruction": "Do X", "input": "Y", "output": "Z"})
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert "Y" in msgs[0]["content"]

    def test_to_messages_sharegpt(self, mock_tokenizer):
        ds = InstructionDataset([], mock_tokenizer)
        msgs = ds._to_messages(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )
        assert len(msgs) == 2

    def test_to_messages_direct(self, mock_tokenizer):
        ds = InstructionDataset([], mock_tokenizer)
        direct = [{"role": "user", "content": "Hi"}]
        msgs = ds._to_messages({"messages": direct})
        assert msgs == direct

    def test_to_messages_unknown_format(self, mock_tokenizer):
        ds = InstructionDataset([], mock_tokenizer)
        msgs = ds._to_messages({"random_key": "value"})
        assert msgs == []


class TestCollateSft:
    """Tests for collate_sft padding function."""

    def test_pads_to_max_length(self):
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
        ]
        result = collate_sft(batch, pad_id=0)
        assert result["input_ids"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

    def test_padding_values(self):
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
        ]
        result = collate_sft(batch, pad_id=0)
        assert result["input_ids"][1, 2].item() == 0  # pad_id
        assert result["labels"][1, 2].item() == -100  # label pad

    def test_single_item_batch(self):
        batch = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}]
        result = collate_sft(batch)
        assert result["input_ids"].shape == (1, 3)


class TestLoadSftData:
    """Tests for load_sft_data."""

    def test_load_local_jsonl(self, tmp_path):
        path = tmp_path / "train.jsonl"
        data = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
        ]
        path.write_text("\n".join(json.dumps(d) for d in data))
        result = load_sft_data(str(path))
        assert len(result) == 2

    def test_load_local_json(self, tmp_path):
        path = tmp_path / "train.json"
        data = [{"instruction": "Q1", "output": "A1"}]
        path.write_text(json.dumps(data))
        result = load_sft_data(str(path))
        assert len(result) == 1

    def test_load_local_with_max_samples(self, tmp_path):
        path = tmp_path / "train.jsonl"
        data = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        path.write_text("\n".join(json.dumps(d) for d in data))
        result = load_sft_data(str(path), max_samples=3)
        assert len(result) == 3

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown SFT dataset"):
            load_sft_data("nonexistent_dataset_xyz")


# ─── Evaluator Tests ──────────────────────────────────────────────────────────


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    @pytest.fixture
    def dummy_model(self):
        """Create a simple model that returns loss."""
        model = MagicMock()
        model.eval = MagicMock()
        model.return_value = {"loss": torch.tensor(2.5)}
        return model

    @pytest.fixture
    def dummy_tokenizer(self):
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        tok.decode = MagicMock(return_value="generated text")
        return tok

    def test_init(self, dummy_model, dummy_tokenizer):
        evaluator = ModelEvaluator(dummy_model, dummy_tokenizer)
        assert evaluator.model is dummy_model
        assert evaluator.device == torch.device("cpu")
        dummy_model.eval.assert_called_once()

    def test_compute_perplexity(self, dummy_model, dummy_tokenizer):
        evaluator = ModelEvaluator(dummy_model, dummy_tokenizer)

        batch = {
            "input_ids": torch.randint(0, 100, (2, 32)),
            "labels": torch.randint(0, 100, (2, 32)),
        }
        dataloader = [batch]

        result = evaluator.compute_perplexity(dataloader, max_batches=1)
        assert "perplexity" in result
        assert "avg_loss" in result
        assert "total_tokens" in result
        assert result["perplexity"] > 0

    def test_evaluate_hellaswag_with_examples(self, dummy_model, dummy_tokenizer):
        evaluator = ModelEvaluator(dummy_model, dummy_tokenizer)

        examples = [
            {
                "ctx": "The cat sat on the",
                "endings": ["mat", "car", "moon", "sun"],
                "label": 0,
            },
        ]
        result = evaluator.evaluate_hellaswag(examples=examples, max_examples=1)
        assert "accuracy" in result
        assert "correct" in result
        assert "total" in result
        assert result["total"] == 1

    def test_generate_samples(self, dummy_model, dummy_tokenizer):
        dummy_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        evaluator = ModelEvaluator(dummy_model, dummy_tokenizer)

        results = evaluator.generate_samples(
            prompts=["Hello"],
            max_new_tokens=10,
        )
        assert len(results) == 1
        assert "prompt" in results[0]
        assert "generated" in results[0]


# ─── Embedding Tests ──────────────────────────────────────────────────────────


class TestMiniEmbeddingModel:
    """Tests for MiniEmbeddingModel."""

    def test_forward_shape(self):
        model = MiniEmbeddingModel(vocab_size=1000, embedding_dim=64, num_heads=2, num_layers=1)
        input_ids = torch.randint(0, 1000, (4, 32))
        output = model(input_ids)
        assert output.shape == (4, 64)

    def test_output_normalized(self):
        model = MiniEmbeddingModel(vocab_size=1000, embedding_dim=64, num_heads=2, num_layers=1)
        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids)
        norms = output.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_with_attention_mask(self):
        model = MiniEmbeddingModel(vocab_size=1000, embedding_dim=64, num_heads=2, num_layers=1)
        input_ids = torch.randint(0, 1000, (2, 16))
        mask = torch.ones(2, 16)
        mask[1, 8:] = 0  # second example is shorter
        output = model(input_ids, attention_mask=mask)
        assert output.shape == (2, 64)

    def test_different_inputs_different_embeddings(self):
        model = MiniEmbeddingModel(vocab_size=1000, embedding_dim=64, num_heads=2, num_layers=1)
        out1 = model(torch.tensor([[1, 2, 3, 4]]))
        out2 = model(torch.tensor([[5, 6, 7, 8]]))
        assert not torch.allclose(out1, out2)


class TestInfoNCELoss:
    """Tests for info_nce_loss."""

    def test_loss_is_scalar(self):
        anchor = torch.randn(8, 64)
        positive = torch.randn(8, 64)
        loss = info_nce_loss(anchor, positive)
        assert loss.dim() == 0

    def test_identical_pairs_low_loss(self):
        embeddings = torch.randn(8, 64)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        loss = info_nce_loss(embeddings, embeddings)
        # When anchor == positive, loss should be low
        assert loss.item() < 5.0  # reasonable upper bound

    def test_temperature_scaling(self):
        anchor = torch.randn(4, 32)
        positive = torch.randn(4, 32)
        loss_high_temp = info_nce_loss(anchor, positive, temperature=1.0)
        loss_low_temp = info_nce_loss(anchor, positive, temperature=0.01)
        # Lower temperature should give higher loss (sharper distribution)
        assert loss_low_temp.item() != loss_high_temp.item()


class TestEmbeddingTrainer:
    """Tests for EmbeddingTrainer."""

    def test_init(self):
        trainer = EmbeddingTrainer(embedding_dim=66, vocab_size=500, device="cpu")
        assert trainer.embedding_dim == 66
        assert trainer.device == torch.device("cpu")

    def test_encode_via_model(self):
        trainer = EmbeddingTrainer(embedding_dim=66, vocab_size=500, device="cpu")
        ids, mask = EmbeddingTrainer._pad_batch([[1, 2, 3, 4], [5, 6, 7]])
        trainer.model.eval()
        with torch.no_grad():
            emb = trainer.model(ids, mask)
        assert emb.shape == (2, 66)

    def test_output_is_normalized(self):
        trainer = EmbeddingTrainer(embedding_dim=66, vocab_size=500, device="cpu")
        ids, mask = EmbeddingTrainer._pad_batch([[1, 2, 3]])
        trainer.model.eval()
        with torch.no_grad():
            emb = trainer.model(ids, mask)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_train_short(self):
        trainer = EmbeddingTrainer(embedding_dim=36, vocab_size=100, device="cpu")
        pairs = [
            ([1, 2, 3], [4, 5, 6]),
            ([7, 8, 9], [10, 11, 12]),
            ([13, 14], [15, 16]),
            ([17, 18], [19, 20]),
        ]
        metrics = trainer.train(pairs, epochs=2, batch_size=4, lr=1e-3)
        assert "total_steps" in metrics
        assert "final_loss" in metrics
        assert metrics["epochs"] == 2

    def test_save_load_roundtrip(self, tmp_path):
        trainer = EmbeddingTrainer(embedding_dim=36, vocab_size=100, device="cpu")
        path = tmp_path / "emb.pt"
        trainer.save(path)
        assert path.exists()

        trainer2 = EmbeddingTrainer(embedding_dim=36, vocab_size=100, device="cpu")
        trainer2.load(path)

        # Same embedding output after load
        ids, mask = EmbeddingTrainer._pad_batch([[1, 2, 3]])
        trainer.model.eval()
        trainer2.model.eval()
        with torch.no_grad():
            emb1 = trainer.model(ids, mask)
            emb2 = trainer2.model(ids, mask)
        assert torch.allclose(emb1, emb2, atol=1e-5)

    def test_pad_batch(self):
        seqs = [[1, 2, 3], [4, 5]]
        ids, mask = EmbeddingTrainer._pad_batch(seqs)
        assert ids.shape == (2, 3)
        assert mask.shape == (2, 3)
        assert mask[1, 2].item() == 0  # padding position
        assert ids[1, 2].item() == 0  # pad token
