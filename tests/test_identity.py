"""Tests for the Identity Core — per-tenant persona management."""

import pytest
import asyncio

from hbllm.brain.identity_node import IdentityProfile, IdentityStore, IdentityNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.fixture
def identity_store(tmp_path):
    return IdentityStore(db_path=tmp_path / "test_identity.db")


# ─── IdentityProfile Tests ───────────────────────────────────────────────────

def test_profile_to_dict():
    profile = IdentityProfile(
        tenant_id="t1",
        persona_name="helper",
        system_prompt="You are a helpful assistant.",
        goals=["Be accurate", "Be concise"],
        constraints=["Never use profanity"],
        personality_traits={"formality": "high", "humor": "low"},
    )
    d = profile.to_dict()
    assert d["tenant_id"] == "t1"
    assert d["persona_name"] == "helper"
    assert len(d["goals"]) == 2
    assert len(d["constraints"]) == 1


def test_profile_context_string():
    profile = IdentityProfile(
        tenant_id="t1",
        system_prompt="Be helpful.",
        goals=["accuracy"],
        constraints=["no profanity"],
        personality_traits={"tone": "formal"},
    )
    ctx = profile.to_context_string()
    assert "Be helpful." in ctx
    assert "accuracy" in ctx
    assert "no profanity" in ctx
    assert "tone=formal" in ctx


def test_profile_empty_context_string():
    profile = IdentityProfile(tenant_id="t1")
    assert profile.to_context_string() == ""


# ─── IdentityStore Tests ─────────────────────────────────────────────────────

def test_store_upsert_and_get(identity_store):
    profile = IdentityProfile(
        tenant_id="t1",
        persona_name="coder",
        system_prompt="You write Python.",
        goals=["clean code"],
    )
    identity_store.upsert(profile)
    
    loaded = identity_store.get("t1")
    assert loaded is not None
    assert loaded.persona_name == "coder"
    assert loaded.goals == ["clean code"]


def test_store_update_overwrites(identity_store):
    identity_store.upsert(IdentityProfile(tenant_id="t1", persona_name="v1"))
    identity_store.upsert(IdentityProfile(tenant_id="t1", persona_name="v2"))
    
    loaded = identity_store.get("t1")
    assert loaded.persona_name == "v2"


def test_store_tenant_isolation(identity_store):
    identity_store.upsert(IdentityProfile(tenant_id="t1", persona_name="alice"))
    identity_store.upsert(IdentityProfile(tenant_id="t2", persona_name="bob"))
    
    assert identity_store.get("t1").persona_name == "alice"
    assert identity_store.get("t2").persona_name == "bob"


def test_store_get_nonexistent(identity_store):
    assert identity_store.get("nope") is None


def test_store_delete(identity_store):
    identity_store.upsert(IdentityProfile(tenant_id="t1"))
    assert identity_store.delete("t1")
    assert identity_store.get("t1") is None


# ─── IdentityNode Integration Tests ──────────────────────────────────────────

@pytest.fixture
async def identity_node(tmp_path):
    bus = InProcessBus()
    await bus.start()
    node = IdentityNode(node_id="identity_test", db_path=tmp_path / "node_identity.db")
    await node.start(bus)
    yield node
    await node.stop()
    await bus.stop()


async def test_node_update_and_query(identity_node):
    """Update a profile via bus and query it back."""
    bus = identity_node.bus
    
    # Update
    update_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        tenant_id="tenant_a",
        topic="identity.update",
        payload={
            "persona_name": "researcher",
            "system_prompt": "You are a research assistant.",
            "goals": ["thorough analysis"],
            "constraints": ["cite sources"],
            "personality_traits": {"detail": "high"},
        },
    )
    resp = await bus.request("identity.update", update_msg, timeout=5.0)
    assert resp.payload["status"] == "updated"
    
    # Query
    query_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        tenant_id="tenant_a",
        topic="identity.query",
        payload={},
    )
    resp = await bus.request("identity.query", query_msg, timeout=5.0)
    assert resp.payload["found"] is True
    assert resp.payload["profile"]["persona_name"] == "researcher"
    assert "thorough analysis" in resp.payload["context_string"]


async def test_node_query_nonexistent(identity_node):
    """Querying a nonexistent tenant returns found=False."""
    bus = identity_node.bus
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        tenant_id="unknown_tenant",
        topic="identity.query",
        payload={},
    )
    resp = await bus.request("identity.query", msg, timeout=5.0)
    assert resp.payload["found"] is False
