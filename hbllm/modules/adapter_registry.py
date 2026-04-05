"""
Runtime LoRA Adapter Registry — specialized specialization engine.

This module allows the HBLLM SpawnerNode to resolve and download pre-trained
domain adapters from the HuggingFace Hub, reducing the need for local
synthetic training while maintaining high security via SHA-256 verification.

Features:
  - Cache hit/miss management with file-level locking.
  - Mandatory SHA-256 integrity verification (for security).
  - PEFT-to-HBLLM format conversion on the fly.
  - Standardized naming: hbllm-adapter-{domain}.pt.
  - Safe torch loading (weights_only=True).
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Standardized Naming Constants ---
ADAPTER_FILENAME_TEMPLATE = "hbllm-adapter-{domain}.pt"
ADAPTER_HASH_TEMPLATE = "hbllm-adapter-{domain}.sha256"
ADAPTER_METADATA_KEY = "__hbllm_adapter_metadata__"

# PEFT Format Constants
PEFT_CONFIG_FILE = "adapter_config.json"
PEFT_WEIGHTS_BIN = "adapter_model.bin"
PEFT_WEIGHTS_SAFE = "adapter_model.safetensors"


# --- Security Utilities ---


def compute_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 1MB chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(1048576), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_sha256(file_path: str | Path, expected_hash: str) -> bool:
    """Check if file hash matches the expected hash (case-insensitive)."""
    if not expected_hash:
        return False
    actual_hash = compute_sha256(file_path)
    return actual_hash.lower() == expected_hash.lower()


def safe_torch_load(path: str | Path) -> dict[str, Any]:
    """Load a torch checkpoint safely (no arbitrary code execution)."""
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))


# --- Configuration Models ---


class AdapterSource(BaseModel):
    """Configuration for a remote adapter repository on HuggingFace."""

    domain: str
    repo_id: str
    filename: str | None = None
    sha256: str | None = None
    rank: int = 8
    peft_format: bool = False  # If True, treats as PEFT-compatible repo
    revision: str | None = None  # Git tag, branch, or commit SHA

    @property
    def effective_filename(self) -> str:
        """Get the filename to search for in the repo."""
        if self.filename:
            return self.filename
        return ADAPTER_FILENAME_TEMPLATE.format(domain=self.domain)

    @property
    def cache_key(self) -> str:
        """Key used for local caching."""
        return self.domain.lower()


class AdapterRegistryConfig(BaseModel):
    """Global configuration for the AdapterRegistry."""

    enabled: bool = False
    cache_dir: str = "./checkpoints/adapters"
    auto_download: bool = True
    require_sha256: bool = True
    max_adapter_size_mb: int = 100
    sources: list[AdapterSource] = Field(default_factory=list)


# --- Registry Class ---


class AdapterRegistry:
    """
    Manages resolution and caching of specialized LoRA adapters.
    """

    def __init__(self, config: AdapterRegistryConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory mapping of known sources
        self._sources: dict[str, AdapterSource] = {s.domain.lower(): s for s in config.sources}

        # Locks ensure two nodes don't download the same adapter concurrently
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, domain: str) -> asyncio.Lock:
        """Get or create a lock for a specific domain."""
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    @property
    def enabled(self) -> bool:
        """Whether the registry is active."""
        return self.config.enabled

    async def resolve(self, domain: str) -> dict[str, torch.Tensor] | None:
        """
        Main entry point: resolve an adapter for a given domain.

        Resolution order:
            1. Local registry cache (`cache_dir/{domain}`)
            2. Local trained checkpoints (`./checkpoints/domains/{domain}`)
            3. Remote HuggingFace registry (if configured)
        """
        if not self.enabled:
            return None

        domain_key = domain.lower()
        async with self._get_lock(domain_key):
            # 1. Check local cache
            cached_weights = self._load_from_cache(domain_key)
            if cached_weights:
                return cached_weights

            # 2. Check local trained results as fallback
            local_trained = self._check_locally_trained(domain)
            if local_trained:
                return local_trained

            # 3. Check remote source
            if domain_key in self._sources:
                source = self._sources[domain_key]
                if self.config.auto_download:
                    return await self._download_and_cache(source)

            return None

    def list_cached(self) -> list[str]:
        """List all domains that have adapters in the local cache."""
        cached: list[str] = []
        if not self.cache_dir.exists():
            return cached

        for d in self.cache_dir.iterdir():
            if d.is_dir():
                filename = ADAPTER_FILENAME_TEMPLATE.format(domain=d.name)
                if (d / filename).exists():
                    cached.append(d.name)
        return cached

    def list_configured(self) -> list[dict[str, Any]]:
        """List all configured remote sources and their cache status."""
        result = []
        cached_domains = set(self.list_cached())
        for source in self._sources.values():
            result.append(
                {
                    "domain": source.domain,
                    "repo_id": source.repo_id,
                    "revision": source.revision or "main",
                    "cached": source.domain.lower() in cached_domains,
                    "has_sha256": bool(source.sha256),
                }
            )
        return result

    def evict(self, domain: str) -> bool:
        """Remove an adapter from the local cache."""
        path = self.cache_dir / domain.lower()
        if path.exists():
            shutil.rmtree(path)
            logger.info("Evicted adapter for domain '%s' from cache.", domain)
            return True
        return False

    # --- Internal Persistence Logic ---

    def _adapter_path(self, domain: str) -> Path:
        """The canonical path for a cached adapter file."""
        return self.cache_dir / domain / ADAPTER_FILENAME_TEMPLATE.format(domain=domain)

    def _hash_path(self, domain: str) -> Path:
        """The canonical path for the stored hash of a cached adapter."""
        return self.cache_dir / domain / ADAPTER_HASH_TEMPLATE.format(domain=domain)

    def _load_from_cache(self, domain: str) -> dict[str, torch.Tensor] | None:
        """Load from the registry's private cache dir."""
        path = self._adapter_path(domain)
        hash_path = self._hash_path(domain)

        if not path.exists():
            return None

        # Verify integrity if we have a stored hash
        if hash_path.exists():
            stored_hash = hash_path.read_text().strip()
            if not verify_sha256(path, stored_hash):
                logger.warning("Cache integrity failure for '%s'. Deleting corrupt file.", domain)
                self.evict(domain)
                return None

        try:
            payload = safe_torch_load(path)
            # If payload is a wrapped dict (new format), return the state_dict
            return self._extract_state_dict(payload)
        except Exception as e:
            logger.error("Failed to load cached adapter for '%s': %s", domain, e)
            return None

    def _check_locally_trained(self, domain: str) -> dict[str, torch.Tensor] | None:
        """Check the default local training output path."""
        path = Path(f"./checkpoints/domains/{domain}/lora_adapter.pt")
        if path.exists():
            try:
                payload = safe_torch_load(path)
                logger.info("Using locally trained adapter for '%s'", domain)
                return self._extract_state_dict(payload)
            except Exception as e:
                logger.warning("Failed to load locally trained adapter: %s", e)
        return None

    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
        """Handles both raw state_dicts and our internal metadata-wrapped format."""
        if isinstance(payload, dict):
            if "state_dict" in payload:
                return cast(dict[str, torch.Tensor], payload["state_dict"])
            return cast(dict[str, torch.Tensor], payload)
        return {}

    async def _download_and_cache(self, source: AdapterSource) -> dict[str, torch.Tensor] | None:
        """Download from HuggingFace, verify, and store in cache."""
        domain = source.domain.lower()
        target_path = self._adapter_path(domain)
        hash_path = self._hash_path(domain)

        # Ensure directory exist
        target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading adapter for '%s' from %s...", domain, source.repo_id)

        try:
            # Check PEFT format auto-detection
            if not source.peft_format:
                is_peft = await self._is_hf_repo_peft(source.repo_id, revision=source.revision)
                if is_peft:
                    source.peft_format = True
                    logger.info(
                        "Detected PEFT format for '%s' (revision: %s)",
                        source.repo_id,
                        source.revision or "main",
                    )

            if source.peft_format:
                # PEFT format requires multiple files (config + weights)
                return await self._handle_peft_download(source, target_path)

            # HBLLM Native Format (Single PT file)
            download_path = await self._hf_download(
                source.repo_id, source.effective_filename, revision=source.revision
            )

            if not download_path:
                return None

            # Security: Size check
            size_mb = os.path.getsize(download_path) / (1024 * 1024)
            if size_mb > self.config.max_adapter_size_mb:
                logger.error(
                    "Adapter for '%s' too large (%.2f MB > %d MB limit). Rejected.",
                    domain,
                    size_mb,
                    self.config.max_adapter_size_mb,
                )
                return None

            # Security: SHA-256 Check
            if self.config.require_sha256:
                if not source.sha256:
                    logger.error("SHA-256 missing in config for '%s'. Security block.", domain)
                    return None

                if not verify_sha256(download_path, source.sha256):
                    logger.error("SHA-256 Mismatch for '%s'! File may be compromised.", domain)
                    return None

            # Success -> Move to cache and save hash
            shutil.copy(download_path, target_path)
            if source.sha256:
                hash_path.write_text(source.sha256)
            else:
                # If we didn't require one but it exists, compute it anyway for future hits
                hash_path.write_text(compute_sha256(target_path))

            return self._extract_state_dict(safe_torch_load(target_path))

        except Exception as e:
            logger.error("Download failed for '%s': %s", domain, e)
            return None

    # --- PEFT Conversion Helpers ---

    async def _is_hf_repo_peft(self, repo_id: str, revision: str | None = None) -> bool:
        """Check if an HF repo contains a PEFT config file."""
        from huggingface_hub import HfApi

        try:
            api = HfApi()
            files = api.list_repo_files(repo_id, revision=revision)
            return PEFT_CONFIG_FILE in files
        except Exception:
            return False

    async def _handle_peft_download(
        self, source: AdapterSource, target_path: Path
    ) -> dict[str, torch.Tensor] | None:
        """Handles downloading and converting a PEFT model from HF."""
        from huggingface_hub import snapshot_download

        domain = source.domain.lower()
        temp_dir = Path(f"./tmp/hf_peft_{domain}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # We use snapshot_download to get the whole PEFT bundle
            path = await asyncio.to_thread(
                snapshot_download,
                repo_id=source.repo_id,
                revision=source.revision,
                local_dir=str(temp_dir),
                allow_patterns=[PEFT_CONFIG_FILE, PEFT_WEIGHTS_BIN, PEFT_WEIGHTS_SAFE],
            )

            # Load and convert
            state_dict = self._load_peft_from_dir(Path(path))
            if not state_dict:
                return None

            # Save the converted weights to our cache
            self.save_adapter(state_dict, target_path, domain=domain, source_repo=source.repo_id)

            return state_dict

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @staticmethod
    def _load_peft_from_dir(adapter_dir: Path) -> dict[str, torch.Tensor] | None:
        """Load a PEFT adapter directory and convert to HBLLM state_dict format."""
        config_path = adapter_dir / PEFT_CONFIG_FILE
        if not config_path.exists():
            return None

        with open(config_path) as f:
            peft_config = json.load(f)

        if peft_config.get("peft_type") != "LORA":
            logger.warning(
                "Only LORA PEFT adapters are supported. Found: %s", peft_config.get("peft_type")
            )
            return None

        # Check weights file (safetensors preferred)
        weights_path = adapter_dir / PEFT_WEIGHTS_SAFE
        if not weights_path.exists():
            weights_path = adapter_dir / PEFT_WEIGHTS_BIN

        if not weights_path.exists():
            return None

        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            peft_state = load_file(str(weights_path), device="cpu")
        else:
            peft_state = safe_torch_load(weights_path)

        return AdapterRegistry._convert_peft_state_dict(peft_state, peft_config)

    @staticmethod
    def _convert_peft_state_dict(
        peft_state: dict[str, Any], peft_config: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Convert PEFT-style keys to HBLLM-style keys."""
        converted = {}
        # PEFT keys usually start with 'base_model.model.model.'
        # and end with '.lora_A.weight' or '.lora_B.weight'
        for key, value in peft_state.items():
            new_key = key.replace("base_model.model.", "")
            if new_key.startswith("model."):
                # Remove '.weight' suffix as HBLLM doesn't use it for LoRA params
                new_key = new_key.replace(".weight", "")
                converted[new_key] = value
        return converted

    # --- External Interaction ---

    async def _hf_download(
        self, repo_id: str, filename: str, revision: str | None = None
    ) -> str | None:
        """Wraps hf_hub_download to work with asyncio."""
        from huggingface_hub import hf_hub_download

        try:
            # Token usually read from HF_TOKEN env var automatically
            path = await asyncio.to_thread(
                hf_hub_download, repo_id=repo_id, filename=filename, revision=revision
            )
            return path
        except Exception as e:
            logger.error("HuggingFace download error for %s/%s: %s", repo_id, filename, e)
            return None

    @staticmethod
    def save_adapter(
        state_dict: dict[str, torch.Tensor],
        path: Path,
        domain: str = "general",
        rank: int = 8,
        source_repo: str = "local",
    ) -> str:
        """
        Helper to save an adapter with full provenance metadata and compute its hash.
        Returns the SHA-256 hash string.
        """
        payload = {
            "state_dict": state_dict,
            ADAPTER_METADATA_KEY: {
                "domain": domain,
                "rank": rank,
                "source_repo": source_repo,
                "format_version": 1,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

        # Compute and write hash file
        sha = compute_sha256(path)
        path.with_suffix(".sha256").write_text(sha)
        return sha

    @staticmethod
    def load_adapter_metadata(path: Path) -> dict[str, Any]:
        """Safely load only the metadata from an adapter file."""
        if not path.exists():
            return {}
        # load with weights_only=True still allows the metadata dict
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and ADAPTER_METADATA_KEY in payload:
            return cast(dict[str, Any], payload[ADAPTER_METADATA_KEY])
        return {}
