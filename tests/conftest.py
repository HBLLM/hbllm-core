"""
Shared test fixtures and session-level cleanup.

Forces clean exit after all tests by ensuring no orphaned asyncio
tasks or event loops prevent pytest from terminating.

Provides data directory isolation so tests don't pollute each other.
"""

from __future__ import annotations

import glob
import os
import sys

# Ensure llama_venv site-packages are in sys.path so tests can find dependencies
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
llama_venv_libs = glob.glob(os.path.join(base_dir, "llama_venv", "lib", "python*", "site-packages"))
for lib_path in llama_venv_libs:
    if lib_path not in sys.path:
        sys.path.append(lib_path)

# Conditionally mock torch/onnxruntime ONLY when they aren't genuinely installed.
# If real packages exist, let them be imported normally so ML tests can run.
import importlib.machinery
from typing import Any
from unittest.mock import MagicMock

_torch_available = False
try:
    if os.getenv("HBLLM_FORCE_MOCK_TORCH") == "1":
        raise ImportError("Forced mock torch")
    import torch as _torch_probe  # noqa: F401

    _torch_available = True
except ImportError:

    class _MockModule:
        """Mock base class for torch.nn.Module when PyTorch is not installed.

        Prevents MagicMock from being used as a base class metaclass, which turns
        derived classes (e.g. HBLLMForCausalLM) into MagicMock instances with
        exhaustible 1-item side_effect iterators raising StopIteration on subsequent calls.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            if name.startswith("_"):
                raise AttributeError(name)
            val = MagicMock()
            setattr(self, name, val)
            return val

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return MagicMock()

        def to(self, *args: Any, **kwargs: Any) -> _MockModule:
            return self

        def eval(self) -> _MockModule:
            return self

        def train(self, mode: bool = True) -> _MockModule:
            return self

        def parameters(self) -> list[Any]:
            return []

        def named_parameters(self) -> list[tuple[str, Any]]:
            return []

        def modules(self) -> list[Any]:
            return [self]

        def named_modules(self) -> list[tuple[str, Any]]:
            return [("", self)]

        def state_dict(self) -> dict[str, Any]:
            return {}

        def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
            return MagicMock()

        def apply(self, fn: Any) -> _MockModule:
            return self

        def register_buffer(self, name: str, tensor: Any, persistent: bool = True) -> None:
            setattr(self, name, tensor)

        def register_parameter(self, name: str, param: Any) -> None:
            setattr(self, name, param)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return MagicMock()

    class _MockLinear(_MockModule):
        pass

    class _MockEmbedding(_MockModule):
        pass

    class _MockTensor:
        pass

    class _MockDtype:
        pass

    class _MockDevice:
        pass

    spec = importlib.machinery.ModuleSpec("torch", None)

    torch_mock = MagicMock()
    torch_mock.__spec__ = spec
    torch_mock.cuda.is_available.return_value = False
    torch_mock.backends.mps.is_available.return_value = False
    torch_mock.Tensor = _MockTensor
    torch_mock.dtype = _MockDtype
    torch_mock.device = _MockDevice
    torch_mock.float32 = _MockDtype()
    torch_mock.bfloat16 = _MockDtype()
    torch_mock.float16 = _MockDtype()
    torch_mock.int8 = _MockDtype()
    torch_mock.uint8 = _MockDtype()

    nn_mock = MagicMock()
    nn_mock.__spec__ = spec
    nn_mock.Module = _MockModule
    nn_mock.Linear = _MockLinear
    nn_mock.Embedding = _MockEmbedding
    torch_mock.nn = nn_mock

    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = nn_mock
    sys.modules["torch.nn.functional"] = MagicMock()
    torch_mock.nn.functional = sys.modules["torch.nn.functional"]
    sys.modules["torch.utils"] = MagicMock()
    torch_mock.utils = sys.modules["torch.utils"]
    sys.modules["torch.utils.data"] = MagicMock()
    torch_mock.utils.data = sys.modules["torch.utils.data"]
    sys.modules["torch.optim"] = MagicMock()
    torch_mock.optim = sys.modules["torch.optim"]
    sys.modules["torch.optim.lr_scheduler"] = MagicMock()
    torch_mock.optim.lr_scheduler = sys.modules["torch.optim.lr_scheduler"]

_onnx_available = False
try:
    import onnxruntime as _onnx_probe  # noqa: F401

    _onnx_available = True
except ImportError:
    onnx_spec = importlib.machinery.ModuleSpec("onnxruntime", None)
    onnx_mock = MagicMock()
    onnx_mock.__spec__ = onnx_spec
    sys.modules["onnxruntime"] = onnx_mock

import asyncio
import logging
import os

import pytest
import pytest_asyncio

from hbllm.network.bus import InProcessBus

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Isolate each test's data directory to prevent SQLite/file pollution.

    Patches the default data_dir field in BrainConfig so every test that
    creates a BrainConfig without explicit data_dir gets a temp directory.
    Also redirects common path lookups.
    """
    try:
        import dataclasses

        from hbllm.brain.core.factory import BrainConfig

        # Patch the dataclass field default for data_dir
        for f in dataclasses.fields(BrainConfig):
            if f.name == "data_dir":
                monkeypatch.setattr(f, "default", str(tmp_path))
                break
    except Exception:
        pass  # Not all test modules use BrainConfig

    # Ensure deterministic working directory for any relative path access
    monkeypatch.setenv("HBLLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HBLLM_TESTING", "1")
    yield


@pytest.fixture(autouse=True)
def _force_gc_after_test():
    """Force garbage collection after each test to clean up dangling references."""
    yield
    # gc.collect()


@pytest_asyncio.fixture(autouse=True)
async def _force_task_cleanup():
    """Cancel all stray tasks at the end of each async test to prevent hanging."""
    yield

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    if tasks:
        # Use a short timeout to prevent gather itself from hanging indefinitely
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=2.0,
            )
        except (asyncio.TimeoutError, Exception):
            pass  # Tasks that won't cancel in 2s are orphaned — let them die


@pytest_asyncio.fixture
async def bus():
    """A standard InProcessBus that is cleanly shut down after the test."""
    test_bus = InProcessBus()
    await test_bus.start()
    yield test_bus
    await test_bus.stop()


def pytest_collection_modifyitems(config, items):
    """Auto-apply markers and dynamically skip PyTorch-dependent tests."""
    import sys
    from unittest.mock import MagicMock

    # ── Auto-apply markers based on test directory ───────────────────────
    for item in items:
        path_str = str(item.fspath)
        if "/integration/e2e/" in path_str:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.integration)
        elif "/integration/" in path_str:
            item.add_marker(pytest.mark.integration)
        elif "/unit/" in path_str:
            item.add_marker(pytest.mark.unit)

    # ── Skip PyTorch-dependent tests when torch is mocked ────────────────

    # Check if torch is mocked
    torch_mocked = False
    torch_mod = sys.modules.get("torch")
    if (
        torch_mod is None
        or isinstance(torch_mod, MagicMock)
        or getattr(torch_mod, "__name__", None) != "torch"
    ):
        torch_mocked = True

    if torch_mocked:
        skip_marker = pytest.mark.skip(reason="PyTorch is not installed (running in mocked mode)")

        # Specific test file prefixes that ACTUALLY require PyTorch or ONNX.
        # Use exact file stems (without test_ prefix) to avoid false positives.
        # e.g. "model" should match test_model.py but NOT test_self_model.py
        ml_file_stems = {
            # Model / ML core
            "test_model",
            "test_reward_model",
            "test_lora",
            "test_lora_concurrency",
            "test_lora_hotswap",
            "test_lora_loop",
            "test_lora_moe_blending",
            "test_hybrid_quant",
            "test_speculative",
            "test_incremental_grammar",
            "test_fused_gemv",
            "test_hf_adapter",
            "test_onnx",
            "test_projector",
            # Training
            "test_training",
            "test_training_smoke",
            "test_cognitive_training",
            "test_sft_eval_embeddings",
            "test_dpo",
            "test_export_dpo",
            "test_policy_optimizer",
            # Data pipeline (needs torch datasets)
            "test_data",
            "test_downloader",
            "test_benchmarks",
        }

        for item in items:
            mod = getattr(item, "module", None)
            if mod is None:
                continue

            # Get the test file stem: "tests/test_foo.py" -> "test_foo"
            mod_file = getattr(mod, "__file__", "")
            if mod_file:
                import os

                file_stem = os.path.splitext(os.path.basename(mod_file))[0]
                if file_stem in ml_file_stems:
                    item.add_marker(skip_marker)
                    continue

            has_mocked = False
            for val in mod.__dict__.values():
                if (sys.modules.get("torch") is not None and val is sys.modules.get("torch")) or (
                    sys.modules.get("onnxruntime") is not None
                    and val is sys.modules.get("onnxruntime")
                ):
                    has_mocked = True
                    break

                val_mod = getattr(val, "__module__", "")
                if val_mod and (
                    val_mod == "torch"
                    or val_mod.startswith("torch.")
                    or val_mod == "onnxruntime"
                    or val_mod.startswith("onnxruntime.")
                ):
                    has_mocked = True
                    break

            if has_mocked:
                item.add_marker(skip_marker)
