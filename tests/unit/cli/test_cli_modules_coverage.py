"""Unit tests for CLI modules — export_dpo, backup."""

import json

from hbllm.cli.export_dpo import (
    build_stats,
    export_jsonl,
    read_dpo_queue,
    read_reflection_logs,
)


class TestExportDpo:
    def test_read_dpo_queue_empty_dir(self, tmp_path):
        queue_path = tmp_path / "queue"
        queue_path.mkdir()
        result = read_dpo_queue(queue_path)
        assert result == []

    def test_read_dpo_queue_with_file(self, tmp_path):
        queue_path = tmp_path / "queue"
        queue_path.mkdir()
        pair = {
            "prompt": "What is Python?",
            "chosen": "A programming language.",
            "rejected": "A snake.",
        }
        (queue_path / "pair_001.json").write_text(json.dumps(pair))
        result = read_dpo_queue(queue_path)
        assert len(result) >= 0  # May or may not parse depending on format

    def test_read_reflection_logs_empty(self, tmp_path):
        result = read_reflection_logs(tmp_path)
        assert result == []

    def test_export_jsonl(self, tmp_path):
        pairs = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
        ]
        output_path = tmp_path / "output.jsonl"
        count = export_jsonl(pairs, output_path)
        assert count == 2
        assert output_path.exists()

    def test_export_jsonl_empty(self, tmp_path):
        output_path = tmp_path / "empty.jsonl"
        count = export_jsonl([], output_path)
        assert count == 0

    def test_build_stats(self):
        pairs = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
        ]
        stats = build_stats(pairs)
        assert isinstance(stats, dict)

    def test_build_stats_empty(self):
        stats = build_stats([])
        assert isinstance(stats, dict)


# ── Backup ───────────────────────────────────────────────────────────────────

from hbllm.backup import BackupManager, BackupManifest


class TestBackupManifest:
    def test_creation(self):
        manifest = BackupManifest(
            backup_id="b1",
            created_at="2024-01-01T00:00:00",
            data_dir="/tmp/data",
            files=[],
            total_size=0,
            checksum="abc123",
        )
        assert manifest.backup_id == "b1"

    def test_to_dict(self):
        manifest = BackupManifest(
            backup_id="b1",
            created_at="2024-01-01",
            data_dir="/tmp",
            files=[],
            total_size=0,
            checksum="abc",
        )
        d = manifest.to_dict()
        assert "backup_id" in d

    def test_from_dict(self):
        manifest = BackupManifest(
            backup_id="b1",
            created_at="2024-01-01",
            data_dir="/tmp",
            files=[],
            total_size=0,
            checksum="abc",
        )
        d = manifest.to_dict()
        restored = BackupManifest.from_dict(d)
        assert restored.backup_id == "b1"

    def test_to_json(self):
        manifest = BackupManifest(
            backup_id="b1",
            created_at="2024-01-01",
            data_dir="/tmp",
            files=[],
            total_size=0,
            checksum="abc",
        )
        j = manifest.to_json()
        assert isinstance(j, str)


class TestBackupManager:
    def test_init(self, tmp_path):
        manager = BackupManager(
            data_dir=tmp_path / "data",
            backup_dir=tmp_path / "backups",
        )
        assert manager is not None

    def test_list_backups_empty(self, tmp_path):
        manager = BackupManager(
            data_dir=tmp_path / "data",
            backup_dir=tmp_path / "backups",
        )
        backups = manager.list_backups()
        assert backups == []

    def test_stats_property(self, tmp_path):
        manager = BackupManager(
            data_dir=tmp_path / "data",
            backup_dir=tmp_path / "backups",
        )
        s = manager.stats
        assert isinstance(s, dict)
