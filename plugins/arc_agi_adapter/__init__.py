"""ARC-AGI and ARC-AGI-3 Cognitive USB Peripheral Driver Adapter."""

from hbllm.hcir.world.percept import PerceptionRegistry

from .arc_driver import ARC3Driver, ArcadeDriver
from .arc_evaluator import ARCBenchmarkRunner, ARCTaskAdapter
from .arc_macro_actions import (
    ArcClickCell,
    ArcPrimitiveAction,
    ToggleSwitch,
    arc_macro_plan_synthesis_hook,
    register_arc_macro_actions,
)
from .arc_perception import ARCPerceptionAdapter, extract_arc_objects
from .solvers import (
    ARC3ArchetypeScorer,
    ARC3DeliberationPolicy,
    install_arc3_solvers,
)

# Register plugin adapter into core registry
PerceptionRegistry.register("arc_agi", ARCPerceptionAdapter(), set_default=True)

# Register ARC macro-actions and synthesis hooks
register_arc_macro_actions()


__all__ = [
    "ARC3Driver",
    "ArcadeDriver",
    "ARCPerceptionAdapter",
    "ARCTaskAdapter",
    "ARCBenchmarkRunner",
    "ARC3ArchetypeScorer",
    "ARC3DeliberationPolicy",
    "install_arc3_solvers",
    "ArcPrimitiveAction",
    "ArcClickCell",
    "ToggleSwitch",
]
