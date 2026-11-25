"""Compatibility re-export — canonical implementation lives in ``hbllm.hcir.kernel.subprocess_sandbox``."""

from __future__ import annotations

from hbllm.hcir.kernel.subprocess_sandbox import SandboxResult, _check_unshare, run_sandboxed_python

__all__ = ["SandboxResult", "_check_unshare", "run_sandboxed_python"]
