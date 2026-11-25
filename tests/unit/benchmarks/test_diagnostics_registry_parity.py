"""Verify diagnostics reports all crates registered in NativeAccelerationRegistry (M-10)."""

from __future__ import annotations

from hbllm.cli.diagnostics import check_rust_crates
from hbllm.native.registry import NativeAccelerationRegistry


def test_diagnostics_reports_all_registered_rust_crates() -> None:
    registry = NativeAccelerationRegistry.get_instance()
    reported = check_rust_crates()
    expected_modules = {mod_name for _, mod_name, _ in registry.REGISTERED_CRATES}

    assert len(expected_modules) == 13
    assert set(reported.keys()) == expected_modules
