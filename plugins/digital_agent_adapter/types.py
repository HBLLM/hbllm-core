"""
Digital Agent Adapter Types.

Defines strongly-typed representations of shell operations, virtual filesystem
state, DOM tree structures, safety violation classifications, and digital tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DigitalActionType(StrEnum):
    """Primitive digital agent operations."""

    EXEC_SHELL = "exec_shell"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    LIST_DIR = "list_dir"
    CLICK_DOM = "click_dom"
    TYPE_DOM = "type_dom"
    SUBMIT_DOM = "submit_dom"


class DigitalTier(StrEnum):
    """Canonical digital embodiment benchmark evaluation tiers."""

    TIER_1_FILE_INSPECTION = "tier_1_file_inspection"
    TIER_2_WORKSPACE_MUTATION = "tier_2_workspace_mutation"
    TIER_3_DESTRUCTIVE_GUARDRAILS = "tier_3_destructive_guardrails"
    TIER_4_BUILD_TEST_PIPELINE = "tier_4_build_test_pipeline"
    TIER_5_DOM_AUTOMATION = "tier_5_dom_automation"


@dataclass
class DOMNode:
    """Node in simulated web DOM tree."""

    tag: str
    node_id: str
    text: str = ""
    value: str = ""
    clickable: bool = False
    attributes: dict[str, str] = field(default_factory=dict)
    children: list[DOMNode] = field(default_factory=list)


@dataclass
class DigitalAction:
    """Action sent to digital environment."""

    action_type: DigitalActionType
    target: str = ""  # Command string, file path, or DOM node_id
    payload: str = ""  # File content or input value


@dataclass
class DigitalObservation:
    """Observation returned by digital environment sandbox."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    filesystem: dict[str, str] = field(default_factory=dict)  # path -> content
    active_dom: DOMNode | None = None
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    won: bool = False
    safety_violations: int = 0
    info: dict[str, Any] = field(default_factory=dict)
