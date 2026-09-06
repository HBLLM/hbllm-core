"""Unit tests for the consolidated semantic contradiction utility."""

from hbllm.brain.reasoning.contradiction_utils import (
    are_antonyms,
    detect_structural_contradiction,
)


def test_are_antonyms_canonical():
    assert are_antonyms("locked", "unlocked")
    assert are_antonyms("unlocked", "locked")
    assert are_antonyms("open", "closed")
    assert are_antonyms("increase", "decrease")
    assert are_antonyms("true", "false")
    assert are_antonyms("safe", "unsafe")
    assert are_antonyms("present", "absent")


def test_are_antonyms_morphological_affixes():
    assert are_antonyms("connected", "disconnected")
    assert are_antonyms("effective", "ineffective")
    assert are_antonyms("possible", "impossible")
    assert are_antonyms("linear", "nonlinear")
    assert are_antonyms("functional", "non-functional")


def test_are_antonyms_non_antonyms():
    assert not are_antonyms("dog", "cat")
    assert not are_antonyms("blue", "green")
    assert not are_antonyms("run", "walk")
    assert not are_antonyms("door", "window")


def test_detect_structural_contradiction_modal():
    # Direct modal negation on same subject
    conflict, exp, conf = detect_structural_contradiction(
        "the door is locked", "the door is not locked"
    )
    assert conflict
    assert conf >= 0.8
    assert "not locked" in exp or "Negated predicate" in exp


def test_detect_structural_contradiction_antonym():
    # Antonym conflict on same subject
    conflict, exp, conf = detect_structural_contradiction(
        "the front door is locked", "the front door is unlocked"
    )
    assert conflict
    assert conf >= 0.8
    assert "locked" in exp and "unlocked" in exp


def test_detect_structural_contradiction_auxiliary():
    conflict, exp, conf = detect_structural_contradiction(
        "the agent will execute the command", "the agent will not execute the command"
    )
    assert conflict
    assert conf >= 0.8


def test_detect_structural_contradiction_non_conflicting():
    conflict, _, _ = detect_structural_contradiction(
        "the front door is locked", "the back door is unlocked"
    )
    assert not conflict

    conflict, _, _ = detect_structural_contradiction("temperature is high", "pressure is high")
    assert not conflict
