"""Autonomy Watchers — proactive environment monitoring adapters.

Register as ``AutonomyCore.add_proactive_handler()`` callbacks to give
the brain awareness of the host environment.
"""

from hbllm.brain.autonomy.watchers.filesystem_watcher import FilesystemWatcher
from hbllm.brain.autonomy.watchers.idle_detector import IdleDetector
from hbllm.brain.autonomy.watchers.system_health_watcher import SystemHealthWatcher

__all__ = [
    "FilesystemWatcher",
    "IdleDetector",
    "SystemHealthWatcher",
]
