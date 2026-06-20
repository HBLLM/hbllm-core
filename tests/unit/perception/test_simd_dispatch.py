"""
Test for Rust UniversalEngine SIMD and CPU Dynamic Dispatching.
"""

import pytest


def test_simd_dispatch():
    try:
        from hbllm_compute import UniversalEngine

        engine = UniversalEngine()
        assert hasattr(engine, "arch")
        assert engine.arch in ["aarch64", "x86_64", "unknown"]
        print(f"UniversalEngine dynamic architecture: {engine.arch}")
    except ImportError:
        pytest.skip("hbllm_compute extension not installed")
