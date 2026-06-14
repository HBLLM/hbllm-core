"""
Brain Wiring Modules — extracted from factory.py for maintainability.

These modules contain the wiring logic for different subsystems
of the cognitive brain. They are called by BrainFactory during
brain construction to wire up specific capabilities.

Modules:
  - snn: SNN comprehension + expression stream wiring
  - nodes: Legacy cognitive node creation (27+ individual nodes)
  - subsystems: Shared subsystem wiring (always-on, optional, late-phase)
"""
