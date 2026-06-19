"""Tests for Intent Integrity Engine and Explanation-First Guard."""

from hbllm.brain.control.guard import IntentEnvelope, IntentIntegrityEngine


def test_intent_integrity_hashing():
    engine = IntentIntegrityEngine()

    envelope = IntentEnvelope(
        planned_actions=[{"tool_name": "email.send", "scope": "finance"}],
        allowed_scopes=["email.send:finance"],
        execution_window_s=900.0,
    )

    # Compute initial hash and record approval
    engine.record_approval(envelope)
    assert envelope.immutable_hash != ""

    # Validate with exact same params
    is_valid = engine.verify_integrity(
        envelope.envelope_id,
        current_actions=[{"tool_name": "email.send", "scope": "finance"}],
        current_scopes=["email.send:finance"],
        current_window=900.0,
    )
    assert is_valid is True

    # Validate with mutated params (planner added an action)
    is_valid_mutated = engine.verify_integrity(
        envelope.envelope_id,
        current_actions=[
            {"tool_name": "email.send", "scope": "finance"},
            {"tool_name": "file.delete", "scope": "temp"},
        ],
        current_scopes=["email.send:finance"],
        current_window=900.0,
    )
    assert is_valid_mutated is False
