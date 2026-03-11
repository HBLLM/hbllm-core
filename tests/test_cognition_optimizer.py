"""Tests for CognitionRouter and TokenOptimizer."""

import pytest
from hbllm.network.cognition_router import CognitionRouter, CognitionTask
from hbllm.serving.token_optimizer import TokenOptimizer


# ─── CognitionRouter ─────────────────────────────────────────────────────

class TestCognitionRouter:
    @pytest.fixture
    def router(self):
        r = CognitionRouter()
        r.register_worker("brain_a", ["reasoning", "math"], capacity=5)
        r.register_worker("brain_b", ["research", "summarization"], capacity=5)
        r.register_worker("brain_c", ["creative", "writing"], capacity=5)
        return r

    def test_routes_to_specialist(self, router):
        task = CognitionTask(task_id="t1", domain="reasoning")
        worker = router.route(task)
        assert worker is not None
        assert worker.worker_id == "brain_a"

    def test_routes_research_to_brain_b(self, router):
        task = CognitionTask(task_id="t2", domain="research")
        worker = router.route(task)
        assert worker.worker_id == "brain_b"

    def test_fallback_to_least_loaded(self, router):
        task = CognitionTask(task_id="t3", domain="unknown_domain")
        worker = router.route(task)
        assert worker is not None  # fallback works

    def test_capacity_limit(self, router):
        for i in range(5):
            router.route(CognitionTask(task_id=f"t{i}", domain="reasoning"))
        # Brain A now full
        task = CognitionTask(task_id="overflow", domain="reasoning")
        worker = router.route(task)
        assert worker.worker_id != "brain_a"  # routed elsewhere

    def test_parallel_split(self, router):
        subtasks = [
            CognitionTask(task_id="s1", domain="reasoning"),
            CognitionTask(task_id="s2", domain="research"),
            CognitionTask(task_id="s3", domain="creative"),
        ]
        assignments = router.split_and_route(subtasks)
        assert len(assignments) == 3  # each to different worker

    def test_release_frees_capacity(self, router):
        task = CognitionTask(task_id="t1", domain="reasoning")
        worker = router.route(task)
        assert worker.current_load == 1
        router.release(worker.worker_id, "t1")
        assert router._workers[worker.worker_id].current_load == 0

    def test_cluster_status(self, router):
        status = router.get_cluster_status()
        assert status["total_workers"] == 3


# ─── TokenOptimizer ──────────────────────────────────────────────────────

class TestTokenOptimizer:
    @pytest.fixture
    def optimizer(self):
        return TokenOptimizer(max_context_tokens=500)

    def test_optimize_simple_query(self, optimizer):
        result = optimizer.optimize("Hello there")
        assert result.recommended_model == "small"
        assert result.optimized_tokens > 0

    def test_optimize_complex_query(self, optimizer):
        result = optimizer.optimize("Explain the theory of relativity and derive E=mc^2")
        assert result.recommended_model == "large"

    def test_optimize_domain_query(self, optimizer):
        result = optimizer.optimize("What are the clinical diagnosis criteria for diabetes?")
        assert result.recommended_model == "specialist"

    def test_context_summarization(self, optimizer):
        messages = [
            {"role": "user", "content": f"Message {i} with some content here " * 20}
            for i in range(10)
        ]
        summarized = optimizer.summarize_context(messages, 500)
        assert len(summarized) < len(messages)

    def test_prompt_pruning_removes_fillers(self, optimizer):
        text = "Can you basically just really explain what this actually does?"
        pruned = optimizer.prune_prompt(text)
        assert len(pruned) < len(text)
        assert "basically" not in pruned
        assert "actually" not in pruned

    def test_cache_hit(self, optimizer):
        optimizer.cache_store("What is Python?", "A programming language")
        hit = optimizer.cache_check("What is Python?")
        assert hit == "A programming language"

    def test_cache_miss(self, optimizer):
        hit = optimizer.cache_check("Never asked before")
        assert hit is None

    def test_cost_estimation(self, optimizer):
        cost_small = optimizer.estimate_cost("small", 1000)
        cost_large = optimizer.estimate_cost("large", 1000)
        assert cost_small < cost_large

    def test_savings_tracking(self, optimizer):
        optimizer.optimize("hi", context=[
            {"role": "user", "content": "long message " * 200}
            for _ in range(5)
        ])
        stats = optimizer.stats()
        assert stats["total_optimized"] == 1
        assert stats["tokens_saved"] >= 0
