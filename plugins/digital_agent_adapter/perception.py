"""
Digital Agent Perception Adapter.

Parses stdout/stderr streams, virtual filesystem entries, and DOM trees into
structured epistemic states and diagnostic feature representations.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import DigitalObservation, DOMNode

logger = logging.getLogger(__name__)


class DigitalPerceptionAdapter:
    """Extracts digital environment features, errors, and UI affordances."""

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        """Reset internal states."""
        pass

    def process_observation(self, obs: DigitalObservation) -> dict[str, Any]:
        """Convert observation into structured perceptual state."""
        has_error = obs.exit_code != 0 or bool(obs.stderr)
        is_assertion_error = "AssertionError" in obs.stderr

        dom_inputs = []
        dom_buttons = []
        if obs.active_dom:
            self._extract_dom_affordances(obs.active_dom, dom_inputs, dom_buttons)

        return {
            "stdout": obs.stdout,
            "stderr": obs.stderr,
            "exit_code": obs.exit_code,
            "has_error": has_error,
            "is_assertion_error": is_assertion_error,
            "files": list(obs.filesystem.keys()),
            "dom_inputs": dom_inputs,
            "dom_buttons": dom_buttons,
            "step_count": obs.step_count,
        }

    def _extract_dom_affordances(
        self,
        node: DOMNode,
        inputs: list[str],
        buttons: list[str],
    ) -> None:
        """Recursively collect interactive form elements."""
        if node.tag == "input":
            inputs.append(node.node_id)
        elif node.tag == "button" or node.clickable:
            buttons.append(node.node_id)
        for child in node.children:
            self._extract_dom_affordances(child, inputs, buttons)
