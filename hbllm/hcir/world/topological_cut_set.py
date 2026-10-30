"""Deprecated re-export shim — import from ``hbllm.hcir.topological_cut_set`` instead.

Kept only for backward compatibility; emits a ``DeprecationWarning`` on import and
will be removed once no external callers remain.
"""

from __future__ import annotations

import warnings

from hbllm.hcir.topological_cut_set import CutSetResult, TopologicalCutSetAnalyzer

warnings.warn(
    "hbllm.hcir.world.topological_cut_set is deprecated; "
    "import from hbllm.hcir.topological_cut_set",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CutSetResult", "TopologicalCutSetAnalyzer"]
