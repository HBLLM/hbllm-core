"""Unit tests validating developmental_adapter plugin.json manifest v2."""

from __future__ import annotations

import json
from pathlib import Path


def test_plugin_manifest_v2():
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"

    data = json.loads(manifest_path.read_text())
    assert data["name"] == "developmental-adapter"
    assert data["version"] == "1.0.0"
    assert data["manifest_version"] == 2
    assert "developmental_discover_causality" in data["capabilities"]
    assert "developmental_observe" in data["capabilities"]
    assert data["entry_point"] == "__init__.py"

    entry_file = manifest_path.parent / data["entry_point"]
    assert entry_file.exists(), f"Entry point {entry_file} must exist"
