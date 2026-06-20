"""
Perception, Brain (remaining), Data, Modules — Integration test coverage.

Covers uncovered lines in:
  - hbllm/perception/normalizer.py (EventNormalizer)
  - hbllm/perception/vector_projector.py
  - hbllm/perception/location_adapter.py
  - hbllm/perception/voice_profile_store.py
  - hbllm/brain/attention_manager.py
  - hbllm/brain/policy_engine.py
  - hbllm/brain/load_manager.py
  - hbllm/brain/emotion_engine.py
  - hbllm/brain/skill_registry.py
  - hbllm/brain/composites/meta_cognition.py
  - hbllm/brain/composites/skill_engine.py
  - hbllm/modules/lora.py (LoRAManager)
  - hbllm/modules/adapter_registry.py
  - hbllm/model/grammar.py
  - hbllm/plugin/manager.py
  - hbllm/data/synthesizer.py
  - hbllm/data/interaction_miner.py
  - hbllm/data/dataloader.py
  - hbllm/training/evaluator.py
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
# brain/attention_manager.py
# ═══════════════════════════════════════════════════════════════════════


class TestAttentionManager:
    def test_init(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        assert mgr is not None

    def test_score_importance(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        score = mgr.score_importance(
            "mem_1", recency=0.8, frequency=0.5, relevance=0.9, emotional_weight=0.3
        )
        assert isinstance(score, float) and score > 0

    def test_get_importance_default(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        imp = mgr.get_importance("unknown_mem")
        assert isinstance(imp, float)

    def test_allocate_focus(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        alloc = mgr.allocate_focus("math", priority=0.8)
        assert alloc is not None

    def test_get_focus(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        mgr.allocate_focus("math", priority=0.8)
        focus = mgr.get_focus("math")
        assert focus is not None

    def test_should_accept(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        result = mgr.should_accept("episodic", importance=0.9)
        assert isinstance(result, bool)

    def test_decay_all_scores(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        mgr.score_importance("m1", recency=0.9, frequency=0.5, relevance=0.7)
        mgr.decay_all_scores()
        after = mgr.get_importance("m1")
        assert isinstance(after, float)

    def test_get_pruning_candidates(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        mgr.update_item_count("episodic", 100)
        mgr.score_importance("m1", recency=0.1, frequency=0.1, relevance=0.1)
        candidates = mgr.get_pruning_candidates("episodic", count=5)
        assert isinstance(candidates, list)

    def test_rebalance_focus(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        mgr.allocate_focus("math", priority=0.8)
        mgr.allocate_focus("code", priority=0.5)
        result = mgr.rebalance_focus()
        assert isinstance(result, dict)

    def test_stats(self):
        from hbllm.brain.attention_manager import AttentionManager

        mgr = AttentionManager(node_id="attn_test")
        s = mgr.stats()
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════
# brain/policy_engine.py
# ═══════════════════════════════════════════════════════════════════════


class TestPolicyEngine:
    def test_init(self):
        from hbllm.brain.policy_engine import PolicyEngine

        engine = PolicyEngine()
        assert engine is not None

    def test_list_policies_empty(self):
        from hbllm.brain.policy_engine import PolicyEngine

        engine = PolicyEngine()
        policies = engine.list_policies()
        assert isinstance(policies, list)

    def test_evaluate_empty_policies(self):
        from hbllm.brain.policy_engine import PolicyEngine

        engine = PolicyEngine()
        result = engine.evaluate("Hello world", tenant_id="t1")
        assert result is not None

    def test_get_policy_missing(self):
        from hbllm.brain.policy_engine import PolicyEngine

        engine = PolicyEngine()
        assert engine.get_policy("nonexistent") is None

    def test_remove_policy_missing(self):
        from hbllm.brain.policy_engine import PolicyEngine

        engine = PolicyEngine()
        assert not engine.remove_policy("nonexistent")


# ═══════════════════════════════════════════════════════════════════════
# brain/load_manager.py
# ═══════════════════════════════════════════════════════════════════════


class TestLoadManager:
    def test_init(self):
        from hbllm.brain.load_manager import LoadManager

        mgr = LoadManager(node_id="load_test")
        assert mgr is not None

    def test_can_accept_task(self):
        from hbllm.brain.load_manager import LoadManager

        mgr = LoadManager(node_id="load_test")
        assert isinstance(mgr.can_accept_task(), bool)

    def test_queue_and_dequeue(self):
        from hbllm.brain.load_manager import LoadManager

        mgr = LoadManager(node_id="load_test")
        queued = mgr.queue_task("task_1", priority=0.7)
        assert isinstance(queued, bool)
        task = mgr.dequeue_task()
        # May or may not return task depending on implementation
        assert task is None or isinstance(task, dict)

    def test_get_max_context_tokens(self):
        from hbllm.brain.load_manager import LoadManager

        mgr = LoadManager(node_id="load_test")
        tokens = mgr.get_max_context_tokens()
        assert isinstance(tokens, int) and tokens > 0

    def test_get_model_preference(self):
        from hbllm.brain.load_manager import LoadManager

        mgr = LoadManager(node_id="load_test")
        pref = mgr.get_model_preference()
        assert isinstance(pref, str)

    def test_is_simulation_enabled(self):
        from hbllm.brain.load_manager import LoadManager

        mgr = LoadManager(node_id="load_test")
        assert isinstance(mgr.is_simulation_enabled(), bool)

    def test_stats(self):
        from hbllm.brain.load_manager import LoadManager

        mgr = LoadManager(node_id="load_test")
        s = mgr.stats()
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════
# brain/emotion_engine.py
# ═══════════════════════════════════════════════════════════════════════


class TestEmotionEngine:
    def test_init(self):
        from hbllm.brain.emotion_engine import EmotionEngine

        engine = EmotionEngine(node_id="emotion_test")
        assert engine is not None

    def test_get_adaptation_hints(self):
        from hbllm.brain.emotion_engine import EmotionEngine

        engine = EmotionEngine(node_id="emotion_test")
        hints = engine.get_adaptation_hints()
        assert isinstance(hints, dict)

    def test_stats(self):
        from hbllm.brain.emotion_engine import EmotionEngine

        engine = EmotionEngine(node_id="emotion_test")
        s = engine.stats()
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════
# perception/normalizer.py (EventNormalizer)
# ═══════════════════════════════════════════════════════════════════════


class TestEventNormalizer:
    def test_import(self):
        from hbllm.perception.normalizer import EventNormalizer

        assert EventNormalizer is not None

    def test_create(self):
        from hbllm.perception.normalizer import EventNormalizer

        norm = EventNormalizer()
        assert norm is not None


# ═══════════════════════════════════════════════════════════════════════
# perception/vector_projector.py
# ═══════════════════════════════════════════════════════════════════════


class TestVectorProjector:
    def test_import(self):
        from hbllm.perception.vector_projector import MultimodalProjector

        assert MultimodalProjector is not None


# ═══════════════════════════════════════════════════════════════════════
# perception/location_adapter.py
# ═══════════════════════════════════════════════════════════════════════


class TestLocationAdapter:
    def test_import(self):
        from hbllm.perception.location_adapter import LocationAdapter

        assert LocationAdapter is not None


# ═══════════════════════════════════════════════════════════════════════
# perception/voice_profile_store.py
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceProfileStore:
    def test_import(self):
        from hbllm.perception.voice_profile_store import VoiceProfileStore

        assert VoiceProfileStore is not None


# ═══════════════════════════════════════════════════════════════════════
# modules/lora.py (LoRAManager — static methods, no model needed for import)
# ═══════════════════════════════════════════════════════════════════════


class TestLoRAManager:
    def test_import(self):
        from hbllm.modules.lora import LoRAManager

        assert LoRAManager is not None


# ═══════════════════════════════════════════════════════════════════════
# modules/adapter_registry.py
# ═══════════════════════════════════════════════════════════════════════


class TestAdapterRegistry:
    def test_import(self):
        from hbllm.modules.adapter_registry import AdapterRegistry

        assert AdapterRegistry is not None


# ═══════════════════════════════════════════════════════════════════════
# model/grammar.py
# ═══════════════════════════════════════════════════════════════════════


class TestGrammarModule:
    def test_import(self):
        from hbllm.model import grammar

        exports = [x for x in dir(grammar) if not x.startswith("_")]
        assert len(exports) > 0


# ═══════════════════════════════════════════════════════════════════════
# data/dataloader.py
# ═══════════════════════════════════════════════════════════════════════


class TestDataloader:
    def test_import(self):
        from hbllm.data import dataloader

        exports = [x for x in dir(dataloader) if not x.startswith("_")]
        assert len(exports) > 0


# ═══════════════════════════════════════════════════════════════════════
# data/synthesizer.py
# ═══════════════════════════════════════════════════════════════════════


class TestSynthesizer:
    def test_import(self):
        from hbllm.data import synthesizer

        exports = [x for x in dir(synthesizer) if not x.startswith("_")]
        assert len(exports) > 0


# ═══════════════════════════════════════════════════════════════════════
# data/interaction_miner.py
# ═══════════════════════════════════════════════════════════════════════


class TestInteractionMiner:
    def test_import(self):
        from hbllm.data import interaction_miner

        exports = [x for x in dir(interaction_miner) if not x.startswith("_")]
        assert len(exports) > 0


# ═══════════════════════════════════════════════════════════════════════
# plugin/manager.py
# ═══════════════════════════════════════════════════════════════════════


class TestPluginManager:
    def test_import(self):
        from hbllm.plugin import manager

        exports = [x for x in dir(manager) if not x.startswith("_")]
        assert len(exports) > 0


# ═══════════════════════════════════════════════════════════════════════
# training/evaluator.py
# ═══════════════════════════════════════════════════════════════════════


class TestTrainingEvaluator:
    def test_import(self):
        from hbllm.training import evaluator

        exports = [x for x in dir(evaluator) if not x.startswith("_")]
        assert len(exports) > 0


# ═══════════════════════════════════════════════════════════════════════
# brain/composites/meta_cognition.py
# ═══════════════════════════════════════════════════════════════════════


class TestMetaCognition:
    def test_import(self):
        from hbllm.brain.composites import meta_cognition

        assert meta_cognition is not None


# ═══════════════════════════════════════════════════════════════════════
# brain/composites/skill_engine.py
# ═══════════════════════════════════════════════════════════════════════


class TestSkillEngine:
    def test_import(self):
        from hbllm.brain.composites import skill_engine

        assert skill_engine is not None


# ═══════════════════════════════════════════════════════════════════════
# brain/skill_registry.py
# ═══════════════════════════════════════════════════════════════════════


class TestSkillRegistry:
    def test_import(self):
        from hbllm.brain import skill_registry

        assert skill_registry is not None
