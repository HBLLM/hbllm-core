"""
Hierarchical Domain Registry.

Manages a tree of domain specializations (e.g. ``coding`` → ``coding.python``
→ ``coding.python.django``).  This is a **cognitive** component — it describes
domains as knowledge structures, not as execution targets.

Execution concerns (LoRA adapters, provider selection) are handled by the
Execution OS in ``hbllm.execution``.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DomainSpec:
    """Describes a single domain node in the hierarchy.

    This is a **cognitive** structure — it describes what a domain *knows*,
    not how it's *executed*.  Execution concerns (LoRA adapters, provider
    selection, modifier pipelines) belong in the Execution OS.
    """

    name: str  # e.g. "coding.python"
    centroid_text: str = ""  # Text used to bootstrap vector embedding
    weight_multiplier: float = 1.0  # Priority boost for this domain
    ontology: list[str] = field(default_factory=list)  # Domain concepts/taxonomy
    reasoning_rules: list[str] = field(default_factory=list)  # Domain reasoning hints
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Deprecated: adapter_name ──────────────────────────────────────────
    # Preserved for backward compatibility during Execution OS migration.
    # Will be removed in a future release.
    adapter_name: str = ""

    @property
    def parent(self) -> str | None:
        """Derive parent from dot-notation.  ``coding.python`` → ``coding``."""
        parts = self.name.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else None

    @property
    def depth(self) -> int:
        """Nesting depth.  ``coding`` = 1, ``coding.python`` = 2."""
        return self.name.count(".") + 1

    def __post_init__(self) -> None:
        if not self.adapter_name:
            self.adapter_name = self.name


# ── Default domain definitions ────────────────────────────────────────────────

_DEFAULT_DOMAINS: list[DomainSpec] = [
    DomainSpec(
        name="general",
        centroid_text=(
            "Hello, can you help me with a general question about daily life, "
            "chatting, or common facts?"
        ),
    ),
    DomainSpec(
        name="coding",
        centroid_text=(
            "Write a python script, fix this bug, create an HTML React component, "
            "explain this logic error."
        ),
    ),
    DomainSpec(
        name="math",
        centroid_text=(
            "Calculate the integral, solve this equation, what is the square root, number theory."
        ),
    ),
    DomainSpec(
        name="planner",
        centroid_text=(
            "Can you design a multi-step plan, architect a system layout, or outline the workflow?"
        ),
    ),
    DomainSpec(
        name="api_synth",
        centroid_text=(
            "Make a POST request, fetch data from the REST backend, build an endpoint payload."
        ),
    ),
    DomainSpec(
        name="fuzzy",
        centroid_text=("I have a vague question that needs approximate or uncertain reasoning."),
    ),
]


class DomainRegistry:
    """
    Hierarchical domain management.

    This is a **cognitive** registry — it manages domain knowledge structures,
    not execution targets.  Adapter resolution is deprecated; use the
    Execution OS (``hbllm.execution``) for execution concerns.

    Domains use **dot-notation** to encode parent–child relationships::

        coding            (root)
        coding.python     (child of coding)
        coding.python.django  (child of coding.python)
    """

    def __init__(self, *, load_defaults: bool = True) -> None:
        self._domains: dict[str, DomainSpec] = {}
        self._children: dict[str, list[str]] = defaultdict(list)

        # Deferred to avoid circular import issues at class-body time
        if load_defaults:
            for spec in _DEFAULT_DOMAINS:
                self.register(spec)

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, spec: DomainSpec) -> None:
        """Register a domain specification.  Automatically wires parent links."""
        self._domains[spec.name] = spec

        parent = spec.parent
        if parent is not None:
            if spec.name not in self._children[parent]:
                self._children[parent].append(spec.name)
            # Auto-register stub parent if it doesn't exist
            if parent not in self._domains:
                self.register(
                    DomainSpec(
                        name=parent,
                        centroid_text=f"Topics relating to {parent}",
                    )
                )

        logger.debug("Registered domain: %s (adapter=%s)", spec.name, spec.adapter_name)

    def unregister(self, name: str) -> None:
        """Remove a domain and its children."""
        # Remove children recursively
        for child in list(self._children.get(name, [])):
            self.unregister(child)
        self._domains.pop(name, None)
        self._children.pop(name, None)
        # Remove from parent's children list
        parent = name.rsplit(".", 1)[0] if "." in name else None
        if parent and parent in self._children:
            self._children[parent] = [c for c in self._children[parent] if c != name]

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, name: str) -> DomainSpec | None:
        """Get a domain spec by exact name."""
        return self._domains.get(name)

    def exists(self, name: str) -> bool:
        return name in self._domains

    @property
    def all_domains(self) -> list[str]:
        """All registered domain names (sorted)."""
        return sorted(self._domains.keys())

    @property
    def root_domains(self) -> list[str]:
        """Only top-level domains (no dots)."""
        return sorted(n for n in self._domains if "." not in n)

    def children(self, parent: str) -> list[str]:
        """Direct children of a domain."""
        return list(self._children.get(parent, []))

    def leaf_domains(self) -> list[str]:
        """Domains with no children."""
        return sorted(n for n in self._domains if not self._children.get(n))

    # ── Adapter Resolution (DEPRECATED) ────────────────────────────────────
    # These methods are preserved for backward compatibility during the
    # Execution OS migration.  Use hbllm.execution for new code.

    def resolve_adapter(self, domain: str) -> str:
        """
        Walk up the hierarchy to find the nearest registered LoRA adapter.

        .. deprecated::
            Adapter resolution is now handled by the Execution OS.
            Use ``hbllm.execution.orchestrator.ExecutionOrchestrator`` instead.
        """
        warnings.warn(
            "DomainRegistry.resolve_adapter() is deprecated. "
            "Use hbllm.execution.ExecutionOrchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        parts = domain.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            spec = self._domains.get(candidate)
            if spec is not None:
                return spec.adapter_name
        return "default"

    def resolve_weighted(self, domain_weights: dict[str, float]) -> dict[str, float]:
        """
        Resolve a weighted domain dict to **adapter** weights.

        .. deprecated::
            Adapter resolution is now handled by the Execution OS.
        """
        warnings.warn(
            "DomainRegistry.resolve_weighted() is deprecated. "
            "Use hbllm.execution.ExecutionOrchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        adapter_weights: dict[str, float] = {}
        for domain, weight in domain_weights.items():
            adapter = self.resolve_adapter(domain)
            adapter_weights[adapter] = adapter_weights.get(adapter, 0.0) + weight
        # Re-normalise so weights sum to 1.0
        total = sum(adapter_weights.values())
        if total > 0:
            adapter_weights = {k: round(v / total, 4) for k, v in adapter_weights.items()}
        return adapter_weights

    # ── Hierarchy Matching ────────────────────────────────────────────────

    def is_ancestor(self, parent: str, child: str) -> bool:
        """Check if ``parent`` is an ancestor of ``child`` (or equal)."""
        return child == parent or child.startswith(parent + ".")

    def matches_hint(self, domain_name: str, hint: str | dict[str, float]) -> bool:
        """
        Check if a module's domain matches a routing hint.

        Supports both exact string hints and weighted dicts.
        Hierarchical: ``coding`` matches hint ``coding.python``.
        """
        if isinstance(hint, str):
            return self.is_ancestor(domain_name, hint) or self.is_ancestor(hint, domain_name)
        elif isinstance(hint, dict):
            return any(
                self.is_ancestor(domain_name, h) or self.is_ancestor(h, domain_name) for h in hint
            )
        return False

    # ── Centroid Texts ────────────────────────────────────────────────────

    def centroid_texts(self) -> dict[str, str]:
        """Return {domain_name: centroid_text} for vector bootstrapping."""
        return {
            name: spec.centroid_text for name, spec in self._domains.items() if spec.centroid_text
        }

    def __repr__(self) -> str:
        return f"DomainRegistry({len(self._domains)} domains)"
