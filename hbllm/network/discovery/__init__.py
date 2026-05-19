"""
Discovery Layer — Network awareness for HBLLM nodes.

This package provides peer discovery (mDNS), state synchronization (Gossip),
and the Capability Registry.
"""

from hbllm.network.discovery.registry import CapabilityRegistry

# Lazy imports for optional dependencies (zeroconf, etc.)
__all__ = ["CapabilityRegistry", "MDNSDiscovery", "GossipSync"]


def __getattr__(name: str):  # noqa: N807
    if name == "MDNSDiscovery":
        from hbllm.network.discovery.mdns import MDNSDiscovery
        return MDNSDiscovery
    if name == "GossipSync":
        from hbllm.network.discovery.gossip import GossipSync
        return GossipSync
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

