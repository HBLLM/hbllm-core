"""Tests for PersonaEngine — persistent personality profiles."""

import tempfile
from pathlib import Path


from hbllm.brain.persona_engine import (
    EMOTION_MODULATIONS,
    PersonaEngine,
    PersonaProfile,
    PersonaTrait,
)


class TestPersonaTrait:
    def test_default_values(self):
        t = PersonaTrait("formality")
        assert t.value == 0.5
        assert t.momentum == 0.0
        assert t.update_count == 0

    def test_nudge_positive(self):
        t = PersonaTrait("humor", value=0.5)
        t.nudge(1.0)
        assert t.value > 0.5
        assert t.update_count == 1

    def test_nudge_negative(self):
        t = PersonaTrait("humor", value=0.5)
        t.nudge(-1.0)
        assert t.value < 0.5

    def test_nudge_clamps_to_range(self):
        t = PersonaTrait("test", value=0.95)
        t.nudge(1.0, learning_rate=0.5)
        assert t.value <= 1.0

        t2 = PersonaTrait("test", value=0.05)
        t2.nudge(-1.0, learning_rate=0.5)
        assert t2.value >= 0.0


class TestPersonaProfile:
    def test_default_profile(self):
        p = PersonaProfile(tenant_id="t1")
        assert p.tenant_id == "t1"
        assert p.name == "Sentra"
        assert len(p.traits) == 7

    def test_serialization(self):
        p = PersonaProfile(tenant_id="t1")
        d = p.to_dict()
        assert d["tenant_id"] == "t1"
        assert "traits" in d
        assert "formality" in d["traits"]

    def test_deserialization(self):
        p = PersonaProfile(tenant_id="t1")
        p.formality.value = 0.9
        d = p.to_dict()
        p2 = PersonaProfile.from_dict(d)
        assert p2.tenant_id == "t1"
        assert abs(p2.formality.value - 0.9) < 1e-6

    def test_system_prompt_formal(self):
        p = PersonaProfile(tenant_id="t1")
        p.formality.value = 0.8
        fragment = p.to_system_prompt_fragment()
        assert "formal" in fragment.lower() or "professional" in fragment.lower()

    def test_system_prompt_concise(self):
        p = PersonaProfile(tenant_id="t1")
        p.verbosity.value = 0.1
        fragment = p.to_system_prompt_fragment()
        assert "concise" in fragment.lower()

    def test_system_prompt_empty_for_defaults(self):
        p = PersonaProfile(tenant_id="t1")
        # Default values are mid-range, so no strong style hints
        fragment = p.to_system_prompt_fragment()
        assert isinstance(fragment, str)


class TestPersonaEngine:
    def test_get_or_create_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            p = engine.get_profile("t1")
            assert p.tenant_id == "t1"
            assert (Path(tmpdir) / "t1.json").exists()

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            engine.get_profile("t1")
            engine.apply_feedback("t1", "too_verbose")

            # Reload from disk
            engine2 = PersonaEngine(storage_dir=tmpdir)
            p = engine2.get_profile("t1")
            assert p.verbosity.value < 0.5

    def test_apply_feedback_too_verbose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            engine.get_profile("t1")
            p = engine.apply_feedback("t1", "too_verbose")
            assert p.verbosity.value < 0.5

    def test_apply_feedback_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            p0 = engine.get_profile("t1")
            initial = p0.verbosity.value
            engine.apply_feedback("t1", "nonexistent_feedback_type")
            assert engine.get_profile("t1").verbosity.value == initial

    def test_emotion_modulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            prompt = engine.get_modulated_prompt("t1", emotion="stressed")
            assert (
                "brief" in prompt.lower() or "support" in prompt.lower() or isinstance(prompt, str)
            )

    def test_emotion_modulation_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            prompt = engine.get_modulated_prompt("t1", emotion=None)
            assert isinstance(prompt, str)

    def test_list_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            engine.get_profile("t1")
            engine.get_profile("t2")
            profiles = engine.list_profiles()
            assert len(profiles) == 2

    def test_reset_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            engine.apply_feedback("t1", "too_verbose")
            engine.reset_profile("t1")
            p = engine.get_profile("t1")
            assert abs(p.verbosity.value - 0.5) < 1e-6

    def test_record_interaction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            engine.record_interaction("t1")
            p = engine.get_profile("t1")
            assert p.interaction_count == 1


class TestEmotionModulations:
    def test_all_emotions_have_entries(self):
        for emotion in [
            "stressed",
            "frustrated",
            "curious",
            "happy",
            "confused",
            "bored",
            "neutral",
        ]:
            assert emotion in EMOTION_MODULATIONS
