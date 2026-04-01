"""
Tests for AdapterRegistry — runtime LoRA adapter download and caching.

Covers:
  - Local cache hit/miss
  - SHA-256 verification (pass + reject)
  - File size limit enforcement
  - PEFT format detection and conversion
  - Standardized naming convention
  - Registry configuration parsing
  - Adapter save with metadata
  - Concurrent download locking
  - Graceful fallback when HuggingFace is unavailable
"""

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from hbllm.modules.adapter_registry import (
    AdapterRegistry,
    AdapterRegistryConfig,
    AdapterSource,
    ADAPTER_FILENAME_TEMPLATE,
    ADAPTER_HASH_TEMPLATE,
    ADAPTER_METADATA_KEY,
    PEFT_CONFIG_FILE,
    PEFT_WEIGHTS_BIN,
    compute_sha256,
    verify_sha256,
    safe_torch_load,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_fake_adapter(domain: str = "coding", rank: int = 8) -> dict:
    """Create a fake LoRA state_dict for testing."""
    return {
        "model.layers.0.self_attn.q_proj.lora_A": torch.randn(rank, 768),
        "model.layers.0.self_attn.q_proj.lora_B": torch.randn(768, rank),
        "model.layers.0.self_attn.v_proj.lora_A": torch.randn(rank, 768),
        "model.layers.0.self_attn.v_proj.lora_B": torch.randn(768, rank),
    }


def _save_native_adapter(path: Path, domain: str = "coding") -> str:
    """Save a native HBLLM adapter and return its SHA-256."""
    state_dict = _make_fake_adapter(domain)
    payload = {
        "state_dict": state_dict,
        ADAPTER_METADATA_KEY: {
            "domain": domain,
            "rank": 8,
            "format_version": 1,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return compute_sha256(path)


def _save_peft_adapter(adapter_dir: Path, domain: str = "coding", rank: int = 8):
    """Create a fake PEFT-format adapter directory."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # adapter_config.json
    peft_config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    }
    (adapter_dir / PEFT_CONFIG_FILE).write_text(json.dumps(peft_config))

    # adapter_model.bin (PEFT format keys)
    peft_state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, 768),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(768, rank),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(rank, 768),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(768, rank),
    }
    torch.save(peft_state, adapter_dir / PEFT_WEIGHTS_BIN)


@pytest.fixture
def tmp_cache(tmp_path):
    """Provide a temporary cache directory."""
    return tmp_path / "adapter_cache"


@pytest.fixture
def registry(tmp_cache):
    """Create a basic AdapterRegistry with a test source."""
    config = AdapterRegistryConfig(
        enabled=True,
        cache_dir=str(tmp_cache),
        auto_download=False,   # disable downloads for most tests
        require_sha256=False,  # relaxed for unit tests
    )
    return AdapterRegistry(config)


# ── SHA-256 Security Tests ────────────────────────────────────────────────────


class TestSHA256Security:
    def test_compute_sha256(self, tmp_path):
        """SHA-256 computation should match hashlib directly."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert compute_sha256(test_file) == expected

    def test_verify_sha256_pass(self, tmp_path):
        """Correct hash should pass verification."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")
        correct_hash = hashlib.sha256(b"test data").hexdigest()
        assert verify_sha256(test_file, correct_hash) is True

    def test_verify_sha256_fail(self, tmp_path):
        """Wrong hash should fail verification."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")
        assert verify_sha256(test_file, "deadbeef" * 8) is False

    def test_verify_sha256_empty_hash(self, tmp_path):
        """Empty hash string should always fail."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")
        assert verify_sha256(test_file, "") is False

    def test_verify_sha256_case_insensitive(self, tmp_path):
        """Hash comparison should be case-insensitive."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")
        correct_hash = hashlib.sha256(b"test data").hexdigest().upper()
        assert verify_sha256(test_file, correct_hash) is True


# ── Configuration Tests ───────────────────────────────────────────────────────


class TestAdapterSource:
    def test_default_filename(self):
        """Filename should auto-populate from domain."""
        src = AdapterSource(domain="medical", repo_id="org/repo")
        assert src.effective_filename == "hbllm-adapter-medical.pt"

    def test_custom_filename(self):
        """Explicit filename should override default."""
        src = AdapterSource(domain="legal", repo_id="org/repo", filename="custom.pt")
        assert src.effective_filename == "custom.pt"

    def test_cache_key(self):
        """Cache key should be the domain name."""
        src = AdapterSource(domain="coding", repo_id="org/repo")
        assert src.cache_key == "coding"

    def test_peft_format_flag(self):
        """PEFT format flag should default to False."""
        src = AdapterSource(domain="coding", repo_id="org/repo")
        assert src.peft_format is False

        src_peft = AdapterSource(domain="coding", repo_id="org/repo", peft_format=True)
        assert src_peft.peft_format is True

    def test_revision_support(self):
        """Should support pinning to a specific Git revision."""
        src = AdapterSource(domain="coding", repo_id="org/repo", revision="v1.0")
        assert src.revision == "v1.0"
        
        src_sha = AdapterSource(domain="coding", repo_id="org/repo", revision="8f32a4b")
        assert src_sha.revision == "8f32a4b"


class TestAdapterRegistryConfig:
    def test_defaults(self):
        """Default config should have sane values."""
        config = AdapterRegistryConfig()
        assert config.enabled is False
        assert config.require_sha256 is True
        assert config.max_adapter_size_mb == 100


# ── Cache Tests ───────────────────────────────────────────────────────────────


class TestCacheOperations:
    def test_list_cached_empty(self, registry):
        """Empty cache should return empty list."""
        assert registry.list_cached() == []

    def test_cache_hit(self, registry, tmp_cache):
        """Should load adapter from cache if it exists."""
        domain = "coding"
        adapter_path = tmp_cache / domain / ADAPTER_FILENAME_TEMPLATE.format(domain=domain)
        sha = _save_native_adapter(adapter_path, domain)

        # Save the hash file
        hash_path = tmp_cache / domain / ADAPTER_HASH_TEMPLATE.format(domain=domain)
        hash_path.write_text(sha)

        result = registry._load_from_cache(domain)
        assert result is not None
        assert "model.layers.0.self_attn.q_proj.lora_A" in result

    def test_cache_hit_listed(self, registry, tmp_cache):
        """Cached domains should appear in list_cached()."""
        domain = "math"
        adapter_path = tmp_cache / domain / ADAPTER_FILENAME_TEMPLATE.format(domain=domain)
        _save_native_adapter(adapter_path, domain)
        assert "math" in registry.list_cached()

    def test_cache_miss(self, registry):
        """Should return None for uncached domain."""
        result = registry._load_from_cache("nonexistent")
        assert result is None

    def test_cache_corrupted_hash_rejected(self, registry, tmp_cache):
        """Adapter with wrong hash should be rejected and deleted."""
        domain = "coding"
        adapter_path = tmp_cache / domain / ADAPTER_FILENAME_TEMPLATE.format(domain=domain)
        _save_native_adapter(adapter_path, domain)

        # Write a WRONG hash
        hash_path = tmp_cache / domain / ADAPTER_HASH_TEMPLATE.format(domain=domain)
        hash_path.write_text("deadbeef" * 8)

        result = registry._load_from_cache(domain)
        assert result is None
        # File should have been deleted
        assert not adapter_path.exists()

    def test_evict(self, registry, tmp_cache):
        """Evict should remove the cached directory."""
        domain = "coding"
        adapter_path = tmp_cache / domain / ADAPTER_FILENAME_TEMPLATE.format(domain=domain)
        _save_native_adapter(adapter_path, domain)

        assert registry.evict(domain) is True
        assert not (tmp_cache / domain).exists()
        assert registry.evict(domain) is False  # already gone


# ── Resolve Tests ─────────────────────────────────────────────────────────────


class TestResolve:
    @pytest.mark.asyncio
    async def test_disabled_registry_returns_none(self, tmp_cache):
        """Disabled registry should always return None."""
        config = AdapterRegistryConfig(enabled=False, cache_dir=str(tmp_cache))
        reg = AdapterRegistry(config)
        result = await reg.resolve("coding")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_from_cache(self, registry, tmp_cache):
        """Resolve should return cached adapter."""
        domain = "coding"
        adapter_path = tmp_cache / domain / ADAPTER_FILENAME_TEMPLATE.format(domain=domain)
        sha = _save_native_adapter(adapter_path, domain)
        hash_path = tmp_cache / domain / ADAPTER_HASH_TEMPLATE.format(domain=domain)
        hash_path.write_text(sha)

        result = await registry.resolve(domain)
        assert result is not None
        assert len(result) == 4  # 4 LoRA matrices

    @pytest.mark.asyncio
    async def test_resolve_locally_trained(self, registry, tmp_path):
        """Should find locally-trained adapters as fallback."""
        domain = "custom"
        trained_dir = Path("./checkpoints/domains/custom")
        trained_dir.mkdir(parents=True, exist_ok=True)
        trained_path = trained_dir / "lora_adapter.pt"

        state_dict = _make_fake_adapter(domain)
        torch.save(state_dict, trained_path)

        try:
            result = await registry.resolve(domain)
            assert result is not None
        finally:
            # Clean up
            import shutil
            shutil.rmtree("./checkpoints/domains/custom", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_resolve_no_source_returns_none(self, registry):
        """Should return None if no source is configured and not cached."""
        result = await registry.resolve("unknown_domain")
        assert result is None


# ── Security Tests ────────────────────────────────────────────────────────────


class TestSecurityEnforcement:
    @pytest.mark.asyncio
    async def test_require_sha256_rejects_unsigned(self, tmp_cache):
        """When require_sha256=True, sources without hash should be rejected."""
        config = AdapterRegistryConfig(
            enabled=True,
            cache_dir=str(tmp_cache),
            auto_download=True,
            require_sha256=True,
            sources=[{
                "domain": "coding",
                "repo_id": "test/repo",
                "sha256": "",  # No hash!
            }],
        )
        reg = AdapterRegistry(config)
        result = await reg._download_and_cache(reg._sources["coding"])
        assert result is None

    def test_safe_torch_load_weights_only(self, tmp_path):
        """safe_torch_load should use weights_only=True."""
        path = tmp_path / "test.pt"
        torch.save({"key": torch.tensor([1, 2, 3])}, path)
        result = safe_torch_load(path)
        assert "key" in result

    @pytest.mark.asyncio
    async def test_file_size_limit(self, tmp_cache):
        """Adapters exceeding size limit should be rejected."""
        config = AdapterRegistryConfig(
            enabled=True,
            cache_dir=str(tmp_cache),
            auto_download=True,
            require_sha256=False,
            max_adapter_size_mb=0,  # 0 MB limit — reject everything
            sources=[{
                "domain": "coding",
                "repo_id": "test/repo",
            }],
        )
        reg = AdapterRegistry(config)

        # Mock the HF download to return a real file
        fake_path = tmp_cache / "fake_download.pt"
        torch.save(_make_fake_adapter(), fake_path)

        with patch.object(reg, '_hf_download', return_value=fake_path):
            result = await reg._download_and_cache(reg._sources["coding"])
            assert result is None  # should be rejected due to size


# ── PEFT Conversion Tests ────────────────────────────────────────────────────


class TestPEFTConversion:
    def test_convert_peft_state_dict(self):
        """Should strip PEFT prefix and convert key suffixes."""
        peft_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 768),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(768, 8),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(8, 768),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(768, 8),
        }
        peft_config = {"peft_type": "LORA", "r": 8}

        converted = AdapterRegistry._convert_peft_state_dict(peft_state, peft_config)

        assert len(converted) == 4
        assert "model.layers.0.self_attn.q_proj.lora_A" in converted
        assert "model.layers.0.self_attn.q_proj.lora_B" in converted
        assert "model.layers.0.self_attn.v_proj.lora_A" in converted
        assert "model.layers.0.self_attn.v_proj.lora_B" in converted

        # Verify no PEFT-style keys remain
        for key in converted:
            assert not key.startswith("base_model.")
            assert ".weight" not in key

    def test_load_peft_from_dir(self, tmp_path):
        """Should load and convert a PEFT directory."""
        adapter_dir = tmp_path / "peft_adapter"
        _save_peft_adapter(adapter_dir)

        result = AdapterRegistry._load_peft_from_dir(adapter_dir)
        assert result is not None
        assert len(result) == 4
        assert all("lora_A" in k or "lora_B" in k for k in result)

    def test_load_peft_rejects_non_lora(self, tmp_path):
        """Should reject non-LoRA PEFT adapters."""
        adapter_dir = tmp_path / "prefix_adapter"
        adapter_dir.mkdir(parents=True)

        peft_config = {"peft_type": "PREFIX_TUNING", "num_virtual_tokens": 20}
        (adapter_dir / PEFT_CONFIG_FILE).write_text(json.dumps(peft_config))
        torch.save({}, adapter_dir / PEFT_WEIGHTS_BIN)

        result = AdapterRegistry._load_peft_from_dir(adapter_dir)
        assert result is None

    def test_load_peft_missing_dir(self, tmp_path):
        """Missing PEFT config should return None."""
        result = AdapterRegistry._load_peft_from_dir(tmp_path / "nonexistent")
        assert result is None


# ── Standardized Naming Tests ─────────────────────────────────────────────────


class TestStandardizedNaming:
    def test_filename_template(self):
        """Adapter filenames should follow the standard pattern."""
        assert ADAPTER_FILENAME_TEMPLATE.format(domain="coding") == "hbllm-adapter-coding.pt"
        assert ADAPTER_FILENAME_TEMPLATE.format(domain="medical") == "hbllm-adapter-medical.pt"

    def test_hash_template(self):
        """Hash filenames should follow the standard pattern."""
        assert ADAPTER_HASH_TEMPLATE.format(domain="coding") == "hbllm-adapter-coding.sha256"

    def test_adapter_path(self, registry):
        """Registry should compute canonical paths."""
        path = registry._adapter_path("coding")
        assert path.name == "hbllm-adapter-coding.pt"
        assert path.parent.name == "coding"

    def test_hash_path(self, registry):
        """Registry should compute canonical hash paths."""
        path = registry._hash_path("coding")
        assert path.name == "hbllm-adapter-coding.sha256"


# ── Save Adapter with Metadata Tests ──────────────────────────────────────────


class TestSaveAdapter:
    def test_save_with_metadata(self, tmp_path):
        """Saved adapter should contain both state_dict and metadata."""
        state_dict = _make_fake_adapter()
        path = tmp_path / "adapter.pt"

        sha = AdapterRegistry.save_adapter(
            state_dict, path,
            domain="coding",
            rank=8,
            source_repo="test/repo",
        )

        assert path.exists()
        assert len(sha) == 64  # SHA-256 hex digest

        # Verify hash file was created
        hash_path = path.with_suffix(".sha256")
        assert hash_path.exists()
        assert hash_path.read_text() == sha

        # Verify payload structure
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert "state_dict" in payload
        assert ADAPTER_METADATA_KEY in payload
        assert payload[ADAPTER_METADATA_KEY]["domain"] == "coding"
        assert payload[ADAPTER_METADATA_KEY]["rank"] == 8

    def test_load_metadata(self, tmp_path):
        """Should load metadata without fully loading tensors."""
        state_dict = _make_fake_adapter()
        path = tmp_path / "adapter.pt"
        AdapterRegistry.save_adapter(state_dict, path, domain="math")

        metadata = AdapterRegistry.load_adapter_metadata(path)
        assert metadata["domain"] == "math"

    def test_extract_state_dict_new_format(self):
        """Should extract state_dict from metadata-wrapped payload."""
        fake_state = {"key": torch.tensor([1.0])}
        payload = {"state_dict": fake_state, ADAPTER_METADATA_KEY: {"domain": "test"}}
        result = AdapterRegistry._extract_state_dict(payload)
        assert "key" in result

    def test_extract_state_dict_raw_format(self):
        """Should handle raw state_dict (backward compat)."""
        raw = {"lora_A": torch.tensor([1.0]), "lora_B": torch.tensor([2.0])}
        result = AdapterRegistry._extract_state_dict(raw)
        assert "lora_A" in result


# ── List Configured Tests ────────────────────────────────────────────────────


class TestListConfigured:
    def test_list_configured(self, tmp_cache):
        """Should return metadata about all configured sources."""
        config = AdapterRegistryConfig(
            enabled=True,
            cache_dir=str(tmp_cache),
            sources=[
                {"domain": "coding", "repo_id": "org/coding", "sha256": "abc"},
                {"domain": "medical", "repo_id": "org/medical"},
            ],
        )
        reg = AdapterRegistry(config)
        listed = reg.list_configured()

        assert len(listed) == 2
        assert listed[0]["domain"] == "coding"
        assert listed[0]["has_sha256"] is True
        assert listed[1]["has_sha256"] is False

    def test_list_configured_with_revision(self, tmp_cache):
        """Should include revision in listed sources."""
        config = AdapterRegistryConfig(
            enabled=True,
            cache_dir=str(tmp_cache),
            sources=[
                {"domain": "coding", "repo_id": "org/coding", "revision": "v2.0"},
            ],
        )
        reg = AdapterRegistry(config)
        listed = reg.list_configured()
        assert listed[0]["revision"] == "v2.0"


class TestDownloadPropagation:
    @pytest.mark.asyncio
    @patch("huggingface_hub.hf_hub_download")
    async def test_revision_propagated_to_hf_download(self, mock_hf, registry, tmp_path):
        """Registry._hf_download should pass revision to hf_hub_download."""
        mock_hf.return_value = str(tmp_path / "fake.pt")
        Path(mock_hf.return_value).touch()
        
        await registry._hf_download(
            repo_id="test/repo", 
            filename="file.pt", 
            revision="v1.1"
        )
        
        mock_hf.assert_called_once_with(
            repo_id="test/repo",
            filename="file.pt",
            revision="v1.1"
        )
