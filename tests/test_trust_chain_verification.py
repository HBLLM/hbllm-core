import pytest

from hbllm.network.node import NodeInfo, NodeType
from hbllm.network.registry import ServiceRegistry
from hbllm.security.identity import NodeIdentity
from hbllm.security.trust_chain import TrustChain


def test_trust_chain_sign_and_verify():
    # 1. Generate Root of Trust (Owner Node)
    owner_identity = NodeIdentity.generate()
    owner_public = owner_identity.public_key_b64

    # 2. Generate Identity for a New Node
    node_identity = NodeIdentity.generate()
    node_public = node_identity.public_key_b64
    node_id = "agent_007"

    # 3. Owner signs the new node's registration
    tc = TrustChain(owner_identity=owner_identity)
    signature = tc.sign_node_registration(node_id, node_public)

    # 4. Verify the signature
    is_valid = tc.verify_node_registration(node_id, node_public, signature, owner_public)
    assert is_valid is True


@pytest.mark.asyncio
async def test_registry_enforces_trust_chain():
    owner_identity = NodeIdentity.generate()
    owner_public = owner_identity.public_key_b64

    # Initialize Registry with Owner Public Key
    registry = ServiceRegistry()
    registry.set_owner(owner_public)

    # Case A: Valid Signature
    node_id = "authorized_node"
    node_identity = NodeIdentity.generate()
    node_public = node_identity.public_key_b64

    tc = TrustChain(owner_identity=owner_identity)
    signature = tc.sign_node_registration(node_id, node_public)

    valid_node = NodeInfo(
        node_id=node_id,
        node_type=NodeType.ACTION,
        public_key=node_public,
        owner_signature=signature,
    )

    # Should not raise
    await registry.register(valid_node)

    # Case B: Missing/Invalid Signature
    invalid_node = NodeInfo(
        node_id="rogue_node",
        node_type=NodeType.ACTION,
        public_key=node_public,
        owner_signature=None,  # Missing
    )

    with pytest.raises(PermissionError, match="not signed by the Owner"):
        await registry.register(invalid_node)
