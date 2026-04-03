"""Tests for InteractionMiner — dataset flywheel."""

import pytest

from hbllm.data.interaction_miner import InteractionMiner


@pytest.fixture
def miner(tmp_path):
    return InteractionMiner(data_dir=str(tmp_path))


class TestRecording:
    def test_record_interaction(self, miner):
        miner.record_interaction(
            query="What is Python?", response="A programming language.",
            reward=0.8, tenant_id="t1",
        )
        stats = miner.stats()
        assert stats["total_interactions"] == 1

    def test_record_regenerated(self, miner):
        miner.record_interaction("q", "r1", reward=-0.3, regenerated=True)
        miner.record_interaction("q", "r2", reward=0.8, regenerated=False)
        stats = miner.stats()
        assert stats["total_interactions"] == 2
        assert stats["regeneration_rate"] == 0.5


class TestMining:
    def test_mine_sft_samples(self, miner):
        for i in range(5):
            miner.record_interaction(
                f"query {i}", f"A good response about topic {i} with enough detail.",
                reward=0.7 + 0.05 * i,
            )
        samples = miner.mine_sft_samples(min_reward=0.7)
        assert len(samples) >= 1
        assert samples[0].data_type == "sft"

    def test_mine_hard_negatives(self, miner):
        miner.record_interaction("q", "bad", reward=-0.5)
        negatives = miner.mine_hard_negatives(max_reward=-0.3)
        assert len(negatives) == 1
        assert negatives[0].data_type == "safety"

    def test_mine_knowledge(self, miner):
        miner.record_interaction("What is AI?", "AI is artificial intelligence.", reward=0.9)
        knowledge = miner.mine_knowledge(min_reward=0.7)
        assert len(knowledge) == 1


class TestExport:
    def test_export_dataset(self, miner, tmp_path):
        for i in range(3):
            miner.record_interaction(
                f"q{i}", f"response {i} with good detail and info", reward=0.8,
            )
        output = str(tmp_path / "dataset.jsonl")
        data = miner.export_dataset(output_path=output, min_reward=0.5)
        assert len(data) >= 1
        assert (tmp_path / "dataset.jsonl").exists()
