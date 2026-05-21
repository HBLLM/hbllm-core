"""Tests for Dynamic Value Arbitration."""

from __future__ import annotations

import pytest

from hbllm.brain.value_system import (
    DynamicValueArbitrator,
    InterruptionPenaltyPolicy,
    ResourceConservationPolicy,
)


def test_dynamic_utility_battery_override():
    """Ensure resource policy spikes utility of conservation when battery is low."""
    arbitrator = DynamicValueArbitrator(policies=[ResourceConservationPolicy()])

    # Normal battery
    context = {"action_type": "conserve_resource", "system_battery": 80}
    utility = arbitrator.compute_utility(1.0, context)
    assert utility == 1.0

    # Critical battery
    context = {"action_type": "conserve_resource", "system_battery": 10}
    utility = arbitrator.compute_utility(1.0, context)
    assert utility > 3.0  # Spike multiplier


def test_interruption_penalty():
    """Ensure interruption penalty divides utility if user is focused."""
    arbitrator = DynamicValueArbitrator(policies=[InterruptionPenaltyPolicy()])

    context = {"user_is_focused": True, "action_interrupts_user": True}

    utility = arbitrator.compute_utility(1.0, context)
    assert utility == 0.125  # 1.0 / 8.0
