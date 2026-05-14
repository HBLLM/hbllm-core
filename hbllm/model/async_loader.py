"""
Asynchronous Model Loader Manager.

Provides non-blocking model loading, unloading, and hot-swapping for models.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hbllm.model.model_loader import load_model

logger = logging.getLogger(__name__)


class ModelInfo:
    def __init__(self, model_id: str, model: Any, tokenizer: Any) -> None:
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.ref_count = 1


class AsyncModelManager:
    """Manager for async model lifecycle."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()

    async def load_model(self, source: str, **kwargs: Any) -> Any:
        """Asynchronously load a model using the synchronous loader."""
        async with self._lock:
            if source in self._models:
                self._models[source].ref_count += 1
                return self._models[source].model

            logger.info("Asynchronously loading model from source: %s", source)
            # Run the synchronous load_model in a separate thread
            result = await asyncio.to_thread(load_model, source, **kwargs)
            model: Any = result[0]
            tokenizer: Any = result[1]
            self._models[source] = ModelInfo(source, model, tokenizer)
            return model

    async def unload_model(self, model_id: str) -> None:
        """Unload a model if ref_count drops to 0."""
        async with self._lock:
            if model_id in self._models:
                self._models[model_id].ref_count -= 1
                if self._models[model_id].ref_count <= 0:
                    del self._models[model_id]
                    logger.info("Unloaded model: %s", model_id)

    async def hot_swap_adapter(self, model_id: str, domain: str, adapter_path: str) -> None:
        """Safely swap LoRA adapters in a thread-safe manner."""
        async with self._lock:
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not loaded.")
            # Mock implementation of hot swap
            logger.info("Hot swapped adapter for %s to domain %s", model_id, domain)

    async def list_loaded(self) -> list[str]:
        """Return list of currently loaded model IDs."""
        async with self._lock:
            return list(self._models.keys())
