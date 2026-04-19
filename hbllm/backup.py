"""
HBLLM Core Backup System — Snapshot, compress, and restore all knowledge bases.

Backs up all SQLite databases in the data directory to:
- Local compressed archives (.tar.gz)
- Cloud storage (S3-compatible) for disaster recovery

Usage:
    from hbllm.backup import BackupManager

    manager = BackupManager(data_dir="./data")

    # Local backup
    path = manager.create_backup()

    # Restore from backup
    manager.restore(path)

    # Cloud backup (S3-compatible)
    manager.configure_cloud(
        endpoint="https://s3.amazonaws.com",
        bucket="hbllm-backups",
        access_key="...",
        secret_key="...",
    )
    manager.push_to_cloud()

CLI:
    hbllm backup create
    hbllm backup restore <path>
    hbllm backup list
    hbllm backup push --bucket hbllm-backups
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default data directory relative to core
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
_BACKUP_DIR_NAME = "backups"

# Files to include in backup (glob patterns)
_DB_PATTERNS = ["*.db"]
_EXTRA_PATTERNS = ["*.json", "*.yaml", "*.yml"]


class BackupManifest:
    """Metadata about a backup snapshot."""

    def __init__(
        self,
        backup_id: str,
        created_at: str,
        data_dir: str,
        files: list[dict[str, Any]],
        total_size: int,
        checksum: str,
    ):
        self.backup_id = backup_id
        self.created_at = created_at
        self.data_dir = data_dir
        self.files = files
        self.total_size = total_size
        self.checksum = checksum

    def to_dict(self) -> dict:
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at,
            "data_dir": self.data_dir,
            "files": self.files,
            "total_size": self.total_size,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BackupManifest:
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BackupManager:
    """
    Manages backup and restore of HBLLM knowledge bases.

    Supports:
    - Full snapshot backups (all .db files)
    - Compressed archives (.tar.gz)
    - Manifest tracking with checksums
    - Cloud push/pull (S3-compatible)
    - Point-in-time restore
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        backup_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.backup_dir = (
            Path(backup_dir) if backup_dir else self.data_dir.parent / _BACKUP_DIR_NAME
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Cloud config
        self._cloud_config: dict[str, str] | None = None

    # ── Local Backup ──────────────────────────────────────────────────────

    def create_backup(self, label: str = "") -> Path:
        """
        Create a full backup of all data files.

        Args:
            label: Optional human-readable label for the backup

        Returns:
            Path to the created backup archive
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        if label:
            backup_id = f"{backup_id}_{label}"

        archive_path = self.backup_dir / f"{backup_id}.tar.gz"

        # Collect files to backup
        files_info = []
        file_paths = []

        for pattern in _DB_PATTERNS + _EXTRA_PATTERNS:
            for path in self.data_dir.glob(pattern):
                if path.is_file():
                    file_paths.append(path)
                    files_info.append(
                        {
                            "name": path.name,
                            "size": path.stat().st_size,
                            "checksum": self._file_checksum(path),
                        }
                    )

        # Also backup training subdirectory if it exists
        training_dir = self.data_dir / "training"
        if training_dir.is_dir():
            for path in training_dir.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(self.data_dir)
                    file_paths.append(path)
                    files_info.append(
                        {
                            "name": str(rel),
                            "size": path.stat().st_size,
                            "checksum": self._file_checksum(path),
                        }
                    )

        if not file_paths:
            logger.warning("No files found to backup in %s", self.data_dir)
            return archive_path

        # Checkpoint all SQLite databases before backup
        self._checkpoint_databases(file_paths)

        # Create compressed archive
        total_size = sum(f["size"] for f in files_info)
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in file_paths:
                arcname = path.relative_to(self.data_dir)
                tar.add(str(path), arcname=str(arcname))

            # Write manifest into archive
            manifest = BackupManifest(
                backup_id=backup_id,
                created_at=datetime.now().isoformat(),
                data_dir=str(self.data_dir),
                files=files_info,
                total_size=total_size,
                checksum=self._file_checksum(archive_path) if archive_path.exists() else "",
            )
            manifest_data = manifest.to_json().encode()
            import io

            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest_data)
            tar.addfile(info, io.BytesIO(manifest_data))

        compressed_size = archive_path.stat().st_size
        ratio = (1 - compressed_size / total_size) * 100 if total_size > 0 else 0

        logger.info(
            "Backup created: %s (%d files, %.1f KB → %.1f KB, %.0f%% compression)",
            archive_path.name,
            len(file_paths),
            total_size / 1024,
            compressed_size / 1024,
            ratio,
        )

        return archive_path

    def restore(
        self, backup_path: str | Path, target_dir: str | Path | None = None
    ) -> BackupManifest:
        """
        Restore from a backup archive.

        Args:
            backup_path: Path to .tar.gz backup
            target_dir: Where to restore (defaults to original data_dir)

        Returns:
            The backup manifest
        """
        backup_path = Path(backup_path)
        restore_dir = Path(target_dir) if target_dir else self.data_dir

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Extract to temp dir first for safety
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            # Read manifest
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = None
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = BackupManifest.from_dict(json.load(f))

            # Copy files to restore directory
            restore_dir.mkdir(parents=True, exist_ok=True)
            restored = 0
            for item in Path(tmpdir).rglob("*"):
                if item.is_file() and item.name != "manifest.json":
                    rel = item.relative_to(tmpdir)
                    dest = restore_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(item), str(dest))
                    restored += 1

        logger.info("Restored %d files from %s → %s", restored, backup_path.name, restore_dir)
        return manifest or BackupManifest(
            backup_id="unknown",
            created_at=datetime.now().isoformat(),
            data_dir=str(restore_dir),
            files=[],
            total_size=0,
            checksum="",
        )

    def list_backups(self) -> list[dict[str, Any]]:
        """List all available backups."""
        backups = []
        for path in sorted(self.backup_dir.glob("backup_*.tar.gz")):
            try:
                stat = path.stat()
                # Try to read manifest from archive
                manifest = self._read_manifest(path)
                backups.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "files": len(manifest.files) if manifest else "?",
                    }
                )
            except Exception:
                backups.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size": path.stat().st_size,
                    }
                )
        return backups

    def verify_backup(self, backup_path: str | Path) -> dict[str, Any]:
        """Verify backup integrity by checking checksums."""
        backup_path = Path(backup_path)
        manifest = self._read_manifest(backup_path)
        if not manifest:
            return {"valid": False, "error": "No manifest found"}

        errors = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            for file_info in manifest.files:
                file_path = Path(tmpdir) / file_info["name"]
                if not file_path.exists():
                    errors.append(f"Missing: {file_info['name']}")
                elif self._file_checksum(file_path) != file_info["checksum"]:
                    errors.append(f"Checksum mismatch: {file_info['name']}")

        return {
            "valid": len(errors) == 0,
            "files_checked": len(manifest.files),
            "errors": errors,
        }

    def cleanup(self, keep: int = 5) -> int:
        """Remove old backups, keeping the N most recent."""
        backups = sorted(self.backup_dir.glob("backup_*.tar.gz"), reverse=True)
        removed = 0
        for path in backups[keep:]:
            path.unlink()
            removed += 1
            logger.info("Removed old backup: %s", path.name)
        return removed

    # ── Cloud Backup ──────────────────────────────────────────────────────

    def configure_cloud(
        self,
        endpoint: str,
        bucket: str,
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1",
    ) -> None:
        """Configure S3-compatible cloud storage."""
        self._cloud_config = {
            "endpoint": endpoint,
            "bucket": bucket,
            "access_key": access_key,
            "secret_key": secret_key,
            "region": region,
        }
        logger.info("Cloud backup configured: %s/%s", endpoint, bucket)

    def push_to_cloud(self, backup_path: str | Path | None = None) -> dict[str, Any]:
        """
        Push a backup to cloud storage.

        Args:
            backup_path: Specific backup to push, or latest if None

        Returns:
            Upload result dict
        """
        if not self._cloud_config:
            return {
                "status": "error",
                "error": "Cloud not configured. Call configure_cloud() first.",
            }

        if backup_path is None:
            backups = sorted(self.backup_dir.glob("backup_*.tar.gz"), reverse=True)
            if not backups:
                return {"status": "error", "error": "No backups found"}
            backup_path = backups[0]
        else:
            backup_path = Path(backup_path)

        try:
            import boto3
            from botocore.config import Config as BotoConfig

            s3 = boto3.client(
                "s3",
                endpoint_url=self._cloud_config["endpoint"],
                aws_access_key_id=self._cloud_config["access_key"],
                aws_secret_access_key=self._cloud_config["secret_key"],
                region_name=self._cloud_config["region"],
                config=BotoConfig(signature_version="s3v4"),
            )

            key = f"hbllm-backups/{backup_path.name}"
            s3.upload_file(str(backup_path), self._cloud_config["bucket"], key)

            logger.info(
                "Pushed %s → s3://%s/%s", backup_path.name, self._cloud_config["bucket"], key
            )
            return {
                "status": "success",
                "bucket": self._cloud_config["bucket"],
                "key": key,
                "size": backup_path.stat().st_size,
            }

        except ImportError:
            # Fallback: use httpx for S3-compatible PUT
            return self._push_via_http(backup_path)

    def pull_from_cloud(self, backup_name: str) -> Path:
        """Pull a backup from cloud storage."""
        if not self._cloud_config:
            raise RuntimeError("Cloud not configured")

        local_path = self.backup_dir / backup_name

        try:
            import boto3
            from botocore.config import Config as BotoConfig

            s3 = boto3.client(
                "s3",
                endpoint_url=self._cloud_config["endpoint"],
                aws_access_key_id=self._cloud_config["access_key"],
                aws_secret_access_key=self._cloud_config["secret_key"],
                region_name=self._cloud_config["region"],
                config=BotoConfig(signature_version="s3v4"),
            )

            key = f"hbllm-backups/{backup_name}"
            s3.download_file(self._cloud_config["bucket"], key, str(local_path))
            logger.info("Pulled %s from cloud", backup_name)

        except ImportError:
            raise RuntimeError("boto3 required for cloud backup. Install with: pip install boto3")

        return local_path

    def _push_via_http(self, backup_path: Path) -> dict[str, Any]:
        """Fallback HTTP upload for environments without boto3."""
        try:
            import httpx

            endpoint = self._cloud_config["endpoint"]  # type: ignore
            bucket = self._cloud_config["bucket"]  # type: ignore
            key = f"hbllm-backups/{backup_path.name}"

            with open(backup_path, "rb") as f:
                with httpx.Client(timeout=120) as client:
                    resp = client.put(
                        f"{endpoint}/{bucket}/{key}",
                        content=f.read(),
                        headers={"Content-Type": "application/gzip"},
                    )

            return {
                "status": "success" if resp.status_code in (200, 201) else "error",
                "http_status": resp.status_code,
                "key": key,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _file_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _checkpoint_databases(self, file_paths: list[Path]) -> None:
        """Run WAL checkpoint on all SQLite databases for consistent backup."""
        for path in file_paths:
            if path.suffix == ".db":
                try:
                    conn = sqlite3.connect(str(path))
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.close()
                except Exception:
                    pass  # Not all .db files may be SQLite

    def _read_manifest(self, archive_path: Path) -> BackupManifest | None:
        """Read manifest from a backup archive."""
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                member = tar.getmember("manifest.json")
                f = tar.extractfile(member)
                if f:
                    data = json.loads(f.read().decode())
                    return BackupManifest.from_dict(data)
        except (KeyError, json.JSONDecodeError):
            pass
        return None

    @property
    def stats(self) -> dict[str, Any]:
        """Get backup system statistics."""
        backups = list(self.backup_dir.glob("backup_*.tar.gz"))
        total_size = sum(b.stat().st_size for b in backups)
        db_files = list(self.data_dir.glob("*.db"))
        db_size = sum(f.stat().st_size for f in db_files)
        return {
            "data_dir": str(self.data_dir),
            "backup_dir": str(self.backup_dir),
            "database_count": len(db_files),
            "database_size_kb": db_size / 1024,
            "backup_count": len(backups),
            "backup_total_size_kb": total_size / 1024,
            "cloud_configured": self._cloud_config is not None,
        }
