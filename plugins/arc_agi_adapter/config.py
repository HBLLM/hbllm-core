"""Configuration settings for ARC-AGI and ARC-AGI-3 benchmark evaluation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ARC3BenchmarkConfig:
    """Benchmark execution configuration for ARC-AGI-3 interactive environments."""

    api_url: str = "https://three.arcprize.org"
    max_steps_per_level: int = 120
    default_games: list[str] = field(
        default_factory=lambda: [
            "ls20",
            "wa30",
        ]
    )
    reports_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "reports")
    retain_dynamics_across_levels: bool = True
    confidence_threshold: float = 0.80
