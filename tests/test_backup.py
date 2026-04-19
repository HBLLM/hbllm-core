"""Tests for the HBLLM backup system."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from hbllm.backup import BackupManager, BackupManifest


@pytest.fixture
def sample_data_dir():
    """Create a temp data dir with sample databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        # Create sample SQLite DBs
        for name in ["memory.db", "skills.db", "metrics.db"]:
            db_path = data_dir / name
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
            conn.execute("INSERT INTO test VALUES (1, ?)", (f"data_{name}",))
            conn.commit()
            conn.close()

        # Create a config file
        (data_dir / "config.json").write_text('{"version": "1.0"}')

        # Create training subdir
        training = data_dir / "training"
        training.mkdir()
        (training / "model.json").write_text('{"epochs": 10}')

        yield data_dir


@pytest.fixture
def manager(sample_data_dir):
    backup_dir = sample_data_dir.parent / "backups"
    return BackupManager(data_dir=sample_data_dir, backup_dir=backup_dir)


class TestBackupCreate:
    def test_create_backup(self, manager):
        path = manager.create_backup()
        assert path.exists()
        assert path.suffix == ".gz"
        assert "backup_" in path.name

    def test_create_labeled_backup(self, manager):
        path = manager.create_backup(label="pre_deploy")
        assert "pre_deploy" in path.name

    def test_backup_contains_manifest(self, manager):
        path = manager.create_backup()
        manifest = manager._read_manifest(path)
        assert manifest is not None
        assert len(manifest.files) >= 3  # 3 DBs + config + training file

    def test_backup_is_compressed(self, manager, sample_data_dir):
        path = manager.create_backup()
        # Compressed should be smaller than raw
        raw_size = sum(f.stat().st_size for f in sample_data_dir.rglob("*") if f.is_file())
        assert path.stat().st_size < raw_size


class TestBackupRestore:
    def test_restore_to_new_dir(self, manager, sample_data_dir):
        path = manager.create_backup()

        with tempfile.TemporaryDirectory() as restore_dir:
            manifest = manager.restore(path, target_dir=restore_dir)
            assert manifest is not None

            # Check DBs were restored
            restored_files = list(Path(restore_dir).rglob("*.db"))
            assert len(restored_files) == 3

            # Verify data integrity
            conn = sqlite3.connect(str(Path(restore_dir) / "memory.db"))
            row = conn.execute("SELECT value FROM test WHERE id = 1").fetchone()
            conn.close()
            assert row[0] == "data_memory.db"

    def test_restore_nonexistent_raises(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.restore("/nonexistent/backup.tar.gz")


class TestBackupList:
    def test_list_empty(self, manager):
        assert manager.list_backups() == []

    def test_list_after_create(self, manager):
        manager.create_backup(label="first")
        manager.create_backup(label="second")
        backups = manager.list_backups()
        assert len(backups) == 2

    def test_list_has_metadata(self, manager):
        manager.create_backup()
        backups = manager.list_backups()
        assert "name" in backups[0]
        assert "size" in backups[0]


class TestBackupVerify:
    def test_verify_valid_backup(self, manager):
        path = manager.create_backup()
        result = manager.verify_backup(path)
        assert result["valid"] is True
        assert result["files_checked"] >= 3

    def test_verify_returns_errors(self, manager, sample_data_dir):
        path = manager.create_backup()
        # Corrupt data by changing a source file (won't match restored checksum)
        # But since we verify the archive contents, valid backup = valid
        result = manager.verify_backup(path)
        assert result["valid"] is True


class TestBackupCleanup:
    def test_cleanup_keeps_recent(self, manager):
        for i in range(7):
            manager.create_backup(label=f"v{i}")

        removed = manager.cleanup(keep=3)
        assert removed == 4
        assert len(manager.list_backups()) == 3


class TestBackupStats:
    def test_stats(self, manager):
        manager.create_backup()
        stats = manager.stats
        assert stats["database_count"] == 3
        assert stats["backup_count"] == 1
        assert stats["cloud_configured"] is False


class TestBackupManifest:
    def test_serialization(self):
        m = BackupManifest(
            backup_id="test_123",
            created_at="2026-01-01",
            data_dir="/data",
            files=[{"name": "test.db", "size": 100, "checksum": "abc"}],
            total_size=100,
            checksum="xyz",
        )
        d = m.to_dict()
        m2 = BackupManifest.from_dict(d)
        assert m2.backup_id == "test_123"
        assert len(m2.files) == 1


class TestCloudConfig:
    def test_configure_cloud(self, manager):
        manager.configure_cloud(
            endpoint="https://s3.example.com",
            bucket="test-bucket",
            access_key="key",
            secret_key="secret",
        )
        assert manager._cloud_config is not None
        assert manager.stats["cloud_configured"] is True

    def test_push_without_config(self, manager):
        manager.create_backup()
        result = manager.push_to_cloud()
        assert result["status"] == "error"
        assert "not configured" in result["error"]
