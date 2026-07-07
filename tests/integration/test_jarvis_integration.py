"""Integration tests for core subsystem interactions.

Tests cross-component workflows:
    1. Perception → Memory: Temporal fuser → pattern storage
    2. Perception → Autonomy: World state → reflex evaluation
    3. Memory → Autonomy: Temporal patterns → proactive insights
    4. Autonomy → Safety: Risk assessment → confirmation → audit
    5. Learning → Autonomy: Online learner → prompt injection
    6. Full Pipeline: Event → fuse → insight → timing → delivery
"""

import time

import pytest

# ─── 1. Perception → Memory Integration ──────────────────────────────────


@pytest.mark.asyncio
async def test_fuser_to_world_state():
    """Temporal fuser events flow into world state snapshot."""
    from hbllm.perception.temporal_fuser import PerceptionSnapshot, TemporalFuser
    from hbllm.perception.world_state import WorldStateEngine

    fuser = TemporalFuser(window_s=60.0)
    world = WorldStateEngine(state_ttl_s=300.0)

    # Simulate door + motion events
    fuser.ingest(PerceptionSnapshot(event_type="iot.door", sub_type="opened", room="front"))
    sequences = fuser.ingest(
        PerceptionSnapshot(event_type="iot.motion", sub_type="detected", room="hallway")
    )

    # Feed fused events into world state
    for seq in sequences:
        world._fused_events.append(
            {
                "narrative": seq.narrative,
                "timestamp": seq.timestamp,
            }
        )
        world._timestamps["fused"] = time.time()

    # World state should reflect the fused events
    state = world.get_state()
    assert len(state["recent_events"]) >= 1
    summary = world.get_summary()
    assert len(summary) > 0


# ─── 2. Memory Pipeline Integration ──────────────────────────────────────


@pytest.mark.asyncio
async def test_temporal_to_spatial_correlation(tmp_path):
    """Temporal patterns + spatial memory can be queried together."""
    from hbllm.memory.spatial_memory import SpatialMemory
    from hbllm.memory.temporal_patterns import TemporalPatternDetector

    temporal = TemporalPatternDetector(db_path=tmp_path / "temporal.db")
    spatial = SpatialMemory(db_path=tmp_path / "spatial.db")
    await temporal.init_db()
    await spatial.init_db()

    # Record coding at office
    spatial.register_location("office", identifiers={"wifi_ssid": "CorpWiFi"})
    for i in range(15):
        temporal.record_interaction("t1", "coding", timestamp=time.time() - i * 1800)
        spatial.record_interaction("t1", "office", "coding")

    # Both subsystems have data
    _patterns = temporal.detect_patterns("t1")  # noqa: F841
    location_ctx = spatial.get_location_context("t1", "office")
    assert len(location_ctx) >= 1
    assert location_ctx[0].domain == "coding"


@pytest.mark.asyncio
async def test_importance_scorer_with_real_data():
    """ImportanceScorer works with realistic memory attributes."""
    from hbllm.memory.importance_scorer import ImportanceScorer, ScoredMemory

    scorer = ImportanceScorer()
    now = time.time()

    memories = [
        ScoredMemory(
            memory_id="recent_important",
            raw_importance=0.9,
            access_count=5,
            created_at=now - 3600,
            last_accessed=now - 60,
            emotional_weight=0.7,
        ),
        ScoredMemory(
            memory_id="old_forgotten",
            raw_importance=0.3,
            access_count=0,
            created_at=now - 86400 * 60,
            memory_type="episodic",
        ),
        ScoredMemory(
            memory_id="procedural_skill",
            raw_importance=0.5,
            access_count=20,
            created_at=now - 86400 * 30,
            memory_type="procedural",
        ),
    ]

    keep, archive = scorer.consolidate(memories, now)
    keep_ids = {m.memory_id for m in keep}
    assert "recent_important" in keep_ids
    # Procedural with high access should be kept
    assert "procedural_skill" in keep_ids


# ─── 3. Autonomy → Safety Integration ────────────────────────────────────


@pytest.mark.asyncio
async def test_risk_classification_to_audit(tmp_path):
    """Risk assessment flows into audit trail logging."""
    from hbllm.actions.confirmation import ActionRiskClassifier
    from hbllm.security.audit_trail import AuditTrail

    classifier = ActionRiskClassifier()
    audit = AuditTrail(db_path=tmp_path / "audit.db")
    await audit.init_db()

    # Classify an action
    assessment = classifier.classify("lock.unlock")
    assert assessment.tier == 3

    # Log to audit trail
    audit.log(
        "t1",
        "lock.unlock",
        "iot",
        risk_tier=assessment.tier,
        source="user",
        target="front_door",
        result="success",
    )

    # Verify in audit
    entries = audit.query(tenant_id="t1", min_risk_tier=3, hours=1)
    assert len(entries) == 1
    assert entries[0].risk_tier == 3


@pytest.mark.asyncio
async def test_interrupt_to_social_timing():
    """Interrupt detector state influences social timing decisions."""
    from hbllm.brain.autonomy.interrupt_detector import InterruptDetector, UserState
    from hbllm.brain.social_timing import SocialContext, SocialTimingEngine

    detector = InterruptDetector()
    timing = SocialTimingEngine()

    # Detector starts with no user input → deep_idle
    current_state = detector.state

    # Map engagement to social context
    ctx = SocialContext(
        in_meeting=False,
        is_sleeping=current_state == UserState.DEEP_IDLE,
    )
    timing.update_context(ctx)

    # Evaluate delivery
    decision = timing.evaluate(priority="suggestion")
    # When deep_idle → might be sleeping → delivery may be held
    assert isinstance(decision.deliver_now, bool)


# ─── 4. Learning → Autonomy Integration ──────────────────────────────────


@pytest.mark.asyncio
async def test_online_learner_to_cognitive_load(tmp_path):
    """Online learner context + cognitive load together shape response."""
    from hbllm.brain.cognitive_load_estimator import CognitiveLoadEstimator
    from hbllm.training.online_learner import OnlineLearner

    learner = OnlineLearner(db_path=tmp_path / "learning.db")
    await learner.init_db()
    estimator = CognitiveLoadEstimator()

    # User has established preferences
    learner.record_preference("t1", "response_style", "concise")
    learner.record_correction("t1", "Always use metric units")

    # User is sending rapid messages (high load)
    for _ in range(10):
        estimator.record_message("fix this")

    # Both systems produce context
    learnings = learner.get_learnings_context("t1")
    _load = estimator.estimate()  # noqa: F841
    prompt_mod = estimator.get_system_prompt_modifier()

    assert "concise" in learnings
    assert "metric" in learnings
    assert isinstance(prompt_mod, str)


@pytest.mark.asyncio
async def test_reflex_learner_to_sandbox(tmp_path):
    """Learned reflexes can be tested in the sandbox."""
    from hbllm.brain.autonomy.reflex_learner import ReflexLearner, ReflexStore
    from hbllm.training.exploration_sandbox import ExplorationSandbox

    store = ReflexStore(db_path=tmp_path / "reflexes.db")
    store.init_db()
    learner = ReflexLearner(store=store)
    sandbox = ExplorationSandbox()

    # Learn a reflex
    reflex = await learner.learn_from_instruction(
        "t1", "When humidity above 70% then turn on dehumidifier"
    )

    # Test the action in sandbox
    result = sandbox.simulate_action(reflex.action, reflex.action_params)
    assert result["success"]


# ─── 5. Multi-Agent + Goal Decomposition ─────────────────────────────────


@pytest.mark.asyncio
async def test_goal_decomposition_with_delegation():
    """Goal sub-goals can be delegated to capable agents."""
    from hbllm.brain.autonomy.goal_decomposition import GoalDecompositionEngine
    from hbllm.network.multi_agent import (
        AgentIdentity,
        MultiAgentCoordinator,
    )

    engine = GoalDecompositionEngine()
    coord = MultiAgentCoordinator(identity=AgentIdentity(name="Main"))
    coord.register_peer(AgentIdentity(name="Research", capabilities=["web_search"]))

    # Decompose a goal
    result = await engine.decompose("Research market trends")
    assert len(result.sub_goals) >= 1

    # Find peers for the first sub-goal
    peers = coord.find_capable_peers(["web_search"])
    assert len(peers) >= 1


# ─── 6. Full Pipeline: Event → Insight → Delivery ────────────────────────


@pytest.mark.asyncio
async def test_full_notification_pipeline(tmp_path):
    """Event → fused sequence → world state → timing → delivery decision."""
    from hbllm.brain.autonomy.notification_suppressor import NotificationSuppressor
    from hbllm.brain.social_timing import SocialTimingEngine
    from hbllm.perception.temporal_fuser import PerceptionSnapshot, TemporalFuser
    from hbllm.perception.world_state import WorldStateEngine
    from hbllm.security.audit_trail import AuditTrail
    from hbllm.serving.push_backends import InMemoryBackend, PushNotification

    # Setup pipeline
    fuser = TemporalFuser(window_s=60.0)
    world = WorldStateEngine(state_ttl_s=300.0)
    suppressor = NotificationSuppressor()
    timing = SocialTimingEngine()
    push = InMemoryBackend()
    audit = AuditTrail(db_path=tmp_path / "audit.db")
    await audit.init_db()

    # 1. Perception: door opens + motion detected
    fuser.ingest(PerceptionSnapshot(event_type="iot.door", sub_type="opened", room="front"))
    sequences = fuser.ingest(
        PerceptionSnapshot(event_type="iot.motion", sub_type="detected", room="front")
    )
    assert len(sequences) >= 1

    # 2. World state absorbs the fused event
    for seq in sequences:
        world._fused_events.append({"narrative": seq.narrative, "timestamp": time.time()})
        world._timestamps["fused"] = time.time()

    # 3. Check timing
    decision = timing.evaluate(priority="normal", category="security")
    assert isinstance(decision.deliver_now, bool)

    # 4. If delivering, check suppression
    if decision.deliver_now:
        should_send = suppressor.should_send(category="security", priority="normal", tenant_id="t1")
        if should_send:
            # 5. Send push notification
            result = await push.send(
                PushNotification(
                    title="Person Detected",
                    body=sequences[0].narrative,
                    device_token="test_device",
                )
            )
            assert result.success

    # 6. Audit the event
    audit.log("t1", "person.detected", "security", risk_tier=1, source="perception")
    entries = audit.query(tenant_id="t1", hours=1)
    assert len(entries) >= 1


# ─── 7. Sandbox + Audit Integration ──────────────────────────────────────


@pytest.mark.asyncio
async def test_sandbox_plan_with_audit(tmp_path):
    """Sandbox validates a plan, then actions are audited."""
    from hbllm.security.audit_trail import AuditTrail
    from hbllm.training.exploration_sandbox import ExplorationSandbox

    sandbox = ExplorationSandbox()
    audit = AuditTrail(db_path=tmp_path / "audit.db")
    await audit.init_db()

    # Validate plan in sandbox first
    plan = [
        {"action": "thermostat.set_temp", "params": {"temp": 22}},
        {"action": "light.on", "params": {"device_id": "bedroom"}},
        {"action": "lock.lock", "params": {"device_id": "front_door"}},
    ]
    result = sandbox.validate_plan(plan)
    assert result["plan_valid"]

    # Execute and audit each step
    for step in plan:
        audit.log(
            "t1",
            str(step["action"]),
            "iot",
            risk_tier=1,
            source="autonomy",
            result="success",
        )

    entries = audit.query(tenant_id="t1", hours=1)
    assert len(entries) == 3


# ─── 8. Parallel Workspace + Cognitive Load ──────────────────────────────


@pytest.mark.asyncio
async def test_parallel_workspace_with_load_monitoring():
    """Background workspaces run while cognitive load is tracked."""
    from hbllm.brain.cognitive_load_estimator import CognitiveLoadEstimator
    from hbllm.brain.parallel_workspace import ParallelWorkspaceManager

    manager = ParallelWorkspaceManager(max_concurrent=3)
    estimator = CognitiveLoadEstimator()

    # Create background workspace
    ws_id = manager.create_workspace("Research", "Find best practices", priority=0)
    result = await manager.run_workspace(ws_id)
    assert result["status"] == "no_provider"

    # User continues chatting (main thread)
    estimator.record_message("What else should I consider?")
    load = estimator.estimate()
    assert load.load_level >= 0

    # Both systems report stats
    ws_stats = manager.stats()
    load_stats = estimator.stats()
    assert ws_stats["total_completed"] == 1
    assert load_stats["messages_recorded"] == 1
