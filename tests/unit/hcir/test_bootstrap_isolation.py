"""Tests guaranteeing CognitiveSystem boot isolation and explicit seed wiring (H-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hbllm.hcir.stores.ontology_store import LexicalOntologyStore, StoredLexicalEntry
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_default_boot_does_not_autobind_t4_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default memory_dir resolves to ~/.hbllm/worlds/<world_id>, never checkpoints/t4_reading."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = CognitiveSystemConfig(world_id="iso_test_world", enable_logging=False)
    assert cfg.enable_grounding is True

    system = build_cognitive_system(cfg)
    expected_db = tmp_path / ".hbllm" / "worlds" / "iso_test_world" / "persistent.db"
    assert expected_db.exists()
    assert system.hydrated_count == 0
    assert system.tiered_workspace.persistent.graph.node_count == 0


def test_different_worlds_have_isolated_persistent_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two distinct world_id configs never share persistent state by default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    sys_a = build_cognitive_system(CognitiveSystemConfig(world_id="world_a", enable_logging=False))
    sys_b = build_cognitive_system(CognitiveSystemConfig(world_id="world_b", enable_logging=False))

    assert sys_a.tiered_workspace.persistent is not sys_b.tiered_workspace.persistent
    assert (tmp_path / ".hbllm" / "worlds" / "world_a" / "persistent.db").exists()
    assert (tmp_path / ".hbllm" / "worlds" / "world_b" / "persistent.db").exists()


def test_explicit_seed_ontology_path_wires_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing seed_ontology_path attaches the explicit SQLite ontology store."""
    monkeypatch.setenv("HOME", str(tmp_path))
    seed_db = tmp_path / "seed_ont.db"
    store = LexicalOntologyStore(seed_db)
    store.upsert_many(
        [
            StoredLexicalEntry(
                lemma="telemetry",
                pos="noun",
                language="en",
                definition="Automated measurement and transmission of data.",
                hypernyms=("measurement",),
                domain="science",
            )
        ],
        source="test_seed",
    )
    store.close()

    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="seeded_world",
            seed_ontology_path=seed_db,
            enable_logging=False,
        )
    )
    attached = system.services.extensions.get("ontology_store")
    assert attached is not None
    entries = attached.lookup("telemetry", language="en")
    assert len(entries) == 1
    assert "measurement" in entries[0].definition.lower()
    attached.close()


def test_missing_seed_ontology_path_raises(tmp_path: Path) -> None:
    """Providing a nonexistent seed_ontology_path fails fast."""
    with pytest.raises(FileNotFoundError):
        build_cognitive_system(
            CognitiveSystemConfig(
                world_id="bad_seed",
                memory_dir=None,
                seed_ontology_path=tmp_path / "missing.db",
                enable_logging=False,
            )
        )
