"""Unit tests for model/async_loader.py — AsyncModelManager."""

import pytest

from hbllm.model.async_loader import AsyncModelManager, ModelInfo


class TestModelInfo:
    """Test ModelInfo data class."""

    def test_model_info_creation(self):
        info = ModelInfo(model_id="test-model", model=object(), tokenizer=object())
        assert info.model_id == "test-model"
        assert info.model is not None
        assert info.tokenizer is not None


class TestAsyncModelManager:
    """Test the async model manager lifecycle."""

    def test_init(self):
        manager = AsyncModelManager()
        assert manager._models == {}

    @pytest.mark.asyncio
    async def test_list_loaded_empty(self):
        manager = AsyncModelManager()
        loaded = await manager.list_loaded()
        assert loaded == []

    @pytest.mark.asyncio
    async def test_unload_nonexistent(self):
        manager = AsyncModelManager()
        # Unloading a model that doesn't exist should not raise
        await manager.unload_model("nonexistent")

    @pytest.mark.asyncio
    async def test_load_and_list(self):
        manager = AsyncModelManager()
        # Manually inject a model to test list functionality
        manager._models["test"] = ModelInfo(model_id="test", model=object(), tokenizer=object())
        loaded = await manager.list_loaded()
        assert "test" in loaded

    @pytest.mark.asyncio
    async def test_unload_loaded(self):
        manager = AsyncModelManager()
        manager._models["test"] = ModelInfo(model_id="test", model=object(), tokenizer=object())
        await manager.unload_model("test")
        loaded = await manager.list_loaded()
        assert "test" not in loaded
