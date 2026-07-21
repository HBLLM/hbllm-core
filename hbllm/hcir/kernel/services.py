"""
Kernel Services Container — dependency injection for the Cognitive OS.

Provides the ``KernelServices`` container that holds references
to all kernel service implementations.  Cognitive nodes receive
this container through the ABI ``execute()`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.stores import IEventStore
from hbllm.hcir.workspace import HCIRWorkspaceState


@dataclass
class KernelServices:
    """Dependency container for all kernel services.

    Passed to cognitive nodes via the ABI ``execute()`` contract.
    Nodes interact with the kernel exclusively through this container.

    Usage::

        services = KernelServices(
            workspace=workspace,
            transaction_manager=tx_mgr,
            capability_resolver=resolver,
            scheduler=scheduler,
        )
        # Nodes receive this in execute()
    """

    workspace: HCIRWorkspaceState
    transaction_manager: TransactionManager
    capability_resolver: CapabilityResolver
    scheduler: CognitiveScheduler
    event_store: IEventStore | None = None

    # Extension point for future services
    extensions: dict[str, Any] = field(default_factory=dict)
