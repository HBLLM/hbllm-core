"""
Discovery Layer — Network awareness for HBLLM nodes.

This package provides the Capability Registry and (in Phase 3)
mDNS discovery and Gossip state synchronization.
"""

from hbllm.network.discovery.registry import CapabilityRegistry

__all__ = ["CapabilityRegistry"]
