"""Isolated LLM teacher scaffold plugin (outside hbllm/hcir/).

Provides temporary fallback & verified parallel distillation into HCIR
until Pure-HCIR autonomy reaches the retirement threshold (>= 98%).
"""

from __future__ import annotations

from plugins.llm_scaffold.local_teacher import LocalHFTeacher, parse_teacher_json
from plugins.llm_scaffold.nvidia_teacher import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    NvidiaNIMTeacher,
)
from plugins.llm_scaffold.teacher import (
    DEFAULT_LOCAL_CHECKPOINT,
    AutonomyTracker,
    LLMTeacherScaffold,
)

__all__ = [
    "DEFAULT_LOCAL_CHECKPOINT",
    "DEFAULT_NVIDIA_BASE_URL",
    "DEFAULT_NVIDIA_MODEL",
    "AutonomyTracker",
    "LLMTeacherScaffold",
    "LocalHFTeacher",
    "NvidiaNIMTeacher",
    "parse_teacher_json",
]
