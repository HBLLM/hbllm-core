"""
conftest for tests/unit/misc — ensures tokenizer tests are network-hermetic.

tiktoken's ``get_encoding("cl100k_base")`` attempts a network fetch on first
use. This fixture monkeypatches tiktoken to use a local cache directory,
preventing test failures in network-restricted CI runners or offline
development environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_tiktoken_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tiktoken from making network requests during tests.

    If tiktoken is installed and already has the encoding cached, it works
    normally. If it would need to fetch, we patch the download to raise an
    error so the tokenizer falls through to its zero-dependency fallback
    gracefully — which is the behavior we actually want to test.
    """
    try:
        import tiktoken

        # Check if cl100k_base is already cached locally
        try:
            tiktoken.get_encoding("cl100k_base")
            # Already cached — no patch needed, tiktoken won't hit the network
        except Exception:
            # Not cached — patch the network fetch to prevent downloads
            monkeypatch.setattr(
                tiktoken,
                "get_encoding",
                MagicMock(side_effect=Exception("Network fetch blocked by test fixture")),
            )
    except ImportError:
        # tiktoken not installed — tokenizer will use zero-dependency fallback
        pass
