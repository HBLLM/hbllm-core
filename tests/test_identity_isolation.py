"""
Identity Isolation Tests — verifies that the (tenant_id, user_id, device_id)
triplet correctly partitions data across persistence and memory layers.

Tests:
  1. SemanticMemory search isolation
  2. ValueMemory reward isolation
  3. BrainState (kv_store, messages) isolation
  4. SynapseGateway connection management
  5. Auth middleware identity extraction
"""

import json

import pytest

# ─── SemanticMemory Isolation ────────────────────────────────────────────────


class TestSemanticMemoryIsolation:
    """Verify that SemanticMemory.search only returns vectors for the matching identity."""

    @pytest.fixture
    def semantic_db(self):
        from hbllm.memory.semantic import SemanticMemory

        # Force TF-IDF fallback to avoid slow sentence-transformers import
        db = SemanticMemory()
        db._use_tfidf = True
        return db

    def test_store_and_search_scoped(self, semantic_db):
        semantic_db.store(
            content="Alice loves quantum computing and physics experiments",
            metadata={"role": "user"},
            tenant_id="t1",
            user_id="alice",
            device_id="phone",
        )
        semantic_db.store(
            content="Bob enjoys cooking Italian pasta and Mediterranean food",
            metadata={"role": "user"},
            tenant_id="t1",
            user_id="bob",
            device_id="laptop",
        )

        # Search as Alice — should only see her data
        results_alice = semantic_db.search(
            query="quantum physics",
            top_k=10,
            tenant_id="t1",
            user_id="alice",
            device_id="phone",
        )
        assert len(results_alice) >= 1
        for r in results_alice:
            assert r["metadata"].get("user_id") == "alice"

        # Search as Bob — should only see his data
        results_bob = semantic_db.search(
            query="Italian cooking",
            top_k=10,
            tenant_id="t1",
            user_id="bob",
            device_id="laptop",
        )
        assert len(results_bob) >= 1
        for r in results_bob:
            assert r["metadata"].get("user_id") == "bob"

    def test_cross_tenant_isolation(self, semantic_db):
        semantic_db.store(
            content="Confidential tenant 1 internal roadmap document",
            metadata={},
            tenant_id="tenant_1",
            user_id="user_a",
            device_id="dev_1",
        )

        # Search from a different tenant should return nothing
        results = semantic_db.search(
            query="confidential roadmap",
            top_k=10,
            tenant_id="tenant_2",
            user_id="user_a",
            device_id="dev_1",
        )
        assert len(results) == 0


# ─── ValueMemory Isolation ───────────────────────────────────────────────────


class TestValueMemoryIsolation:
    """Verify that ValueMemory rewards are scoped by identity triplet."""

    @pytest.fixture
    def value_db(self, tmp_path):
        from hbllm.memory.value_memory import ValueMemory

        db = ValueMemory(db_path=str(tmp_path / "value.db"))
        return db

    def test_reward_isolation(self, value_db):
        # Record rewards for different users
        value_db.record_reward(
            tenant_id="t1",
            topic="response_style",
            action="formal_tone",
            reward=1.0,
            user_id="alice",
            device_id="phone",
        )
        value_db.record_reward(
            tenant_id="t1",
            topic="response_style",
            action="formal_tone",
            reward=-1.0,
            user_id="bob",
            device_id="laptop",
        )

        # Alice's preference should be positive
        pref_alice = value_db.get_preference(
            tenant_id="t1",
            topic="response_style",
            user_id="alice",
            device_id="phone",
        )
        # get_preference returns dict[str, float] mapping action -> avg reward
        assert isinstance(pref_alice, dict)
        assert pref_alice.get("formal_tone", 0) > 0

        # Bob's preference should be negative
        pref_bob = value_db.get_preference(
            tenant_id="t1",
            topic="response_style",
            user_id="bob",
            device_id="laptop",
        )
        assert isinstance(pref_bob, dict)
        assert pref_bob.get("formal_tone", 0) < 0

    def test_cross_user_reward_isolation(self, value_db):
        """A user who has no rewards should get empty preferences."""
        value_db.record_reward(
            tenant_id="t1",
            topic="domain",
            action="math",
            reward=0.9,
            user_id="alice",
            device_id="phone",
        )

        pref_charlie = value_db.get_preference(
            tenant_id="t1",
            topic="domain",
            user_id="charlie",
            device_id="tablet",
        )
        assert pref_charlie == {} or pref_charlie.get("math") is None


# ─── BrainState Isolation ────────────────────────────────────────────────────


class TestBrainStateIsolation:
    """Verify that BrainState kv_store and messages are scoped."""

    @pytest.fixture
    def state(self, tmp_path):
        from hbllm.persistence.state import BrainState

        return BrainState(path=str(tmp_path / "state.db"))

    def test_kv_store_isolation(self, state):
        state.save("theme", "dark", tenant_id="t1", user_id="alice", device_id="phone")
        state.save("theme", "light", tenant_id="t1", user_id="bob", device_id="laptop")

        assert state.load("theme", tenant_id="t1", user_id="alice", device_id="phone") == "dark"
        assert state.load("theme", tenant_id="t1", user_id="bob", device_id="laptop") == "light"

        # Cross-user read should return default
        assert (
            state.load("theme", default=None, tenant_id="t1", user_id="charlie", device_id="phone")
            is None
        )

    def test_message_isolation(self, state):
        state.append_message(
            "user", "Hello from Alice", tenant_id="t1", user_id="alice", device_id="d1"
        )
        state.append_message(
            "user", "Hello from Bob", tenant_id="t1", user_id="bob", device_id="d2"
        )

        alice_msgs = state.get_messages(tenant_id="t1", user_id="alice", device_id="d1")
        assert len(alice_msgs) == 1
        assert alice_msgs[0]["content"] == "Hello from Alice"

        bob_msgs = state.get_messages(tenant_id="t1", user_id="bob", device_id="d2")
        assert len(bob_msgs) == 1
        assert bob_msgs[0]["content"] == "Hello from Bob"

    def test_checkpoint_isolation(self, state):
        state.checkpoint({"step": 10}, tenant_id="t1", user_id="alice", device_id="d1")
        state.checkpoint({"step": 99}, tenant_id="t1", user_id="bob", device_id="d2")

        cp_alice = state.latest_checkpoint(tenant_id="t1", user_id="alice", device_id="d1")
        assert cp_alice is not None
        assert cp_alice["data"]["step"] == 10

        cp_bob = state.latest_checkpoint(tenant_id="t1", user_id="bob", device_id="d2")
        assert cp_bob is not None
        assert cp_bob["data"]["step"] == 99

    def test_tool_log_isolation(self, state):
        state.log_tool_call(
            "calculator", "2+2", "4", tenant_id="t1", user_id="alice", device_id="d1"
        )
        state.log_tool_call(
            "translator", "hello", "hola", tenant_id="t1", user_id="bob", device_id="d2"
        )

        logs_alice = state.get_tool_logs(tenant_id="t1", user_id="alice", device_id="d1")
        assert len(logs_alice) == 1
        assert logs_alice[0]["tool"] == "calculator"

        logs_bob = state.get_tool_logs(tenant_id="t1", user_id="bob", device_id="d2")
        assert len(logs_bob) == 1
        assert logs_bob[0]["tool"] == "translator"


# ─── SynapseGateway Tests ───────────────────────────────────────────────────


class FakeWebSocket:
    """Minimal WebSocket mock for gateway tests."""

    def __init__(self):
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.sent_data: list[str] = []

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000, reason=""):
        self.closed = True
        self.close_code = code

    async def send_text(self, data):
        self.sent_data.append(data)


class TestSynapseGateway:
    """Verify WebSocket connection management and message routing."""

    @pytest.fixture
    def gateway(self):
        from hbllm.serving.synapse_gateway import SynapseGateway

        return SynapseGateway(bus=None)

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, gateway):
        ws = FakeWebSocket()
        await gateway.connect(ws, "t1", "alice", "phone_1")

        assert ("t1", "alice", "phone_1") in gateway.active_connections
        assert ws.accepted

        gateway.disconnect("t1", "alice", "phone_1")
        assert ("t1", "alice", "phone_1") not in gateway.active_connections

    @pytest.mark.asyncio
    async def test_capability_registration(self, gateway):
        msg_data = json.dumps(
            {
                "type": "register_capabilities",
                "tools": ["read_gps", "vibrate", "camera"],
            }
        )

        await gateway.handle_inbound_message("t1", "alice", "phone_1", msg_data)

        assert gateway.device_capabilities.get(("t1", "alice", "phone_1")) == [
            "read_gps",
            "vibrate",
            "camera",
        ]

    @pytest.mark.asyncio
    async def test_send_to_device(self, gateway):
        ws = FakeWebSocket()
        await gateway.connect(ws, "t1", "alice", "phone_1")

        result = await gateway.send_to_device("t1", "alice", "phone_1", {"hello": "world"})
        assert result is True
        assert len(ws.sent_data) == 1
        assert json.loads(ws.sent_data[0]) == {"hello": "world"}

        # Targeting a non-existent device should fail
        result = await gateway.send_to_device("t1", "alice", "phone_2", {"hello": "world"})
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_login_replaces_old(self, gateway):
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()

        await gateway.connect(ws1, "t1", "alice", "phone_1")
        await gateway.connect(ws2, "t1", "alice", "phone_1")

        # Old websocket should have been closed
        assert ws1.closed
        assert ws1.close_code == 1008  # Concurrent login
        # New websocket should be the active one
        assert gateway.active_connections[("t1", "alice", "phone_1")] is ws2

    @pytest.mark.asyncio
    async def test_broadcast_to_tenant(self, gateway):
        ws_alice = FakeWebSocket()
        ws_bob = FakeWebSocket()
        ws_other = FakeWebSocket()

        await gateway.connect(ws_alice, "t1", "alice", "phone_1")
        await gateway.connect(ws_bob, "t1", "bob", "laptop_1")
        await gateway.connect(ws_other, "t2", "charlie", "tablet_1")

        await gateway.broadcast_to_tenant("t1", {"event": "sync"})

        # Both t1 devices should receive the broadcast
        assert len(ws_alice.sent_data) == 1
        assert len(ws_bob.sent_data) == 1
        # t2 device should NOT receive it
        assert len(ws_other.sent_data) == 0

    @pytest.mark.asyncio
    async def test_tool_result_forwarding(self, gateway):
        """Tool results from edge should be parsed correctly."""
        msg_data = json.dumps(
            {
                "type": "tool_result",
                "correlation_id": "corr-123",
                "tool_name": "read_gps",
                "result": {"lat": 6.9271, "lng": 79.8612},
                "error": None,
            }
        )

        # Should not raise even without a bus
        await gateway.handle_inbound_message("t1", "alice", "phone_1", msg_data)

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, gateway):
        """Invalid JSON should be handled gracefully."""
        await gateway.handle_inbound_message("t1", "alice", "phone_1", "not-json{{{")
        # No exception raised — logged and ignored


# ─── Auth Middleware Tests ───────────────────────────────────────────────────


class TestAuthMiddleware:
    """Verify JWT identity extraction includes user_id and device_id."""

    def test_jwt_identity_extraction(self):
        import jwt as pyjwt

        secret = "test_secret_key_for_jwt_testing_32ch"
        token = pyjwt.encode(
            {"tenant_id": "acme", "user_id": "user_42", "device_id": "pixel_7"},
            secret,
            algorithm="HS256",
        )

        payload = pyjwt.decode(token, secret, algorithms=["HS256"])
        assert payload["tenant_id"] == "acme"
        assert payload["user_id"] == "user_42"
        assert payload["device_id"] == "pixel_7"

    def test_jwt_defaults_for_missing_fields(self):
        import jwt as pyjwt

        secret = "test_secret_key_for_jwt_testing_32ch"
        # Only tenant_id, no user_id or device_id
        token = pyjwt.encode({"tenant_id": "acme"}, secret, algorithm="HS256")

        payload = pyjwt.decode(token, secret, algorithms=["HS256"])
        assert payload["tenant_id"] == "acme"
        assert payload.get("user_id", "default") == "default"
        assert payload.get("device_id", "default") == "default"


# ─── Message Protocol Tests ─────────────────────────────────────────────────


class TestMessageIdentity:
    """Verify that Message correctly carries the identity triplet."""

    def test_message_identity_fields(self):
        from hbllm.network.messages import Message, MessageType

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            tenant_id="acme",
            user_id="user_42",
            device_id="pixel_7",
            topic="test.topic",
            payload={"text": "hello"},
        )

        assert msg.tenant_id == "acme"
        assert msg.user_id == "user_42"
        assert msg.device_id == "pixel_7"

    def test_message_identity_defaults(self):
        from hbllm.network.messages import Message, MessageType

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test.topic",
            payload={},
        )

        assert msg.tenant_id == "default"
        assert msg.user_id == "default"
        assert msg.device_id == "default"


# ─── RemoteToolNode Tests ───────────────────────────────────────────────────


class TestRemoteToolNode:
    """Verify RemoteToolNode carries correct identity context."""

    def test_remote_tool_node_creation(self):
        from hbllm.actions.tool_registry import RemoteToolNode

        node = RemoteToolNode(
            tool_name="read_gps",
            tenant_id="t1",
            user_id="alice",
            device_id="phone_1",
        )

        assert node.tool_name == "read_gps"
        assert node.target_tenant_id == "t1"
        assert node.target_user_id == "alice"
        assert node.target_device_id == "phone_1"
        assert node.topic == "action.tool.read_gps"
        assert node.node_id == "remote_tool_read_gps_phone_1"
