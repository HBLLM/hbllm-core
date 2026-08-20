"""Provider Provenance — tracks which ML model produced evidence.

Every evidence object carries a ProviderProvenance so the epistemic
layer can reason about source reliability, compare conflicting
providers, and detect model drift.

Example:
    SpeechEvidence(
        ...
        provider_provenance=ProviderProvenance(
            provider="moonshine",
            model="moonshine-base",
            version="1.2.0",
        ),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderProvenance:
    """Immutable provenance for a perception provider result.

    Attributes:
        provider: Provider identifier (e.g. "moonshine", "yamnet", "resemblyzer").
        model: Specific model variant (e.g. "moonshine-base", "yamnet-v2").
        version: Model or library version string.
        device: Compute device used (e.g. "cpu", "cuda:0", "mps").
        extra: Additional provider-specific metadata.

    """

    provider: str = "unknown"
    model: str = ""
    version: str = ""
    device: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        """Short unique identifier for this provider configuration."""
        parts = [self.provider]
        if self.model:
            parts.append(self.model)
        if self.version:
            parts.append(f"v{self.version}")
        return "/".join(parts)
