"""Tests for Constitutional Principles.

Validates parsing and formatting of the constitutional rules.
"""

from hbllm.brain.constitutional_principles import (
    CONSTITUTION,
    Principle,
    format_principles_for_prompt,
    get_principles,
)


def test_constitution_has_core_principles():
    assert "harmless" in CONSTITUTION
    assert "helpful" in CONSTITUTION
    assert "honest" in CONSTITUTION
    assert "accurate" in CONSTITUTION


def test_get_principles_all():
    principles = get_principles()
    assert len(principles) == len(CONSTITUTION)
    for p in principles:
        assert isinstance(p, Principle)


def test_get_principles_subset():
    principles = get_principles(["harmless", "honest"])
    assert len(principles) == 2
    names = [p.name for p in principles]
    assert "Harmlessness" in names
    assert "Honesty" in names


def test_get_principles_ignores_unknown():
    principles = get_principles(["helpful", "unknown_rule"])
    assert len(principles) == 1
    assert principles[0].name == "Helpfulness"


def test_format_principles_for_prompt():
    principles = get_principles(["accurate"])
    prompt = format_principles_for_prompt(principles)
    assert "Constitutional Principles to Evaluate" in prompt
    assert "**Accuracy & Logic**" in prompt
    assert "Violation Condition" in prompt
