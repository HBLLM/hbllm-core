"""WP2: experimental/ablated faculties — lazily imported per-module.

Faculties removed from engine construction by the measured ablation
(``reports/ablation/<ts>.json``) live here for backward-compat imports.
Do NOT re-export through ``hbllm.hcir.world``.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

# Extend package search path to include experiments/hcir_world_experimental
_exp_path = Path(__file__).resolve().parents[4] / "experiments" / "hcir_world_experimental"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))

_MODULES: dict[str, str] = {}


def register(name: str, module: str) -> None:
    _MODULES[name] = module


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        mod = importlib.import_module(_MODULES[name])
        return getattr(mod, name)
    raise AttributeError(name)


def _neutral(cls: type, name: str) -> Any:
    try:
        ann = inspect.signature(getattr(cls, name)).return_annotation
    except Exception:
        return None
    text = str(ann)
    if "bool" in text:
        return False
    if "int" in text:
        return 0
    if "float" in text:
        return 0.0
    if "list" in text:
        return []
    if "dict" in text:
        return {}
    if "set" in text:
        return set()
    if "tuple" in text:
        return ()
    return None


class AblatedFaculty:
    """Null object for an ablated faculty — methods return annotation-neutral
    values; attribute reads return neutral/None. Exactly the object the WP2
    ablation measured (behaviour == measured ablation)."""

    def __init__(self, name: str, cls: type | str | None = None) -> None:
        object.__setattr__(self, "_name", name)
        if isinstance(cls, str):
            object.__setattr__(self, "_cls_name", cls)
            object.__setattr__(self, "_cls", None)
        else:
            object.__setattr__(self, "_cls_name", cls.__name__ if cls is not None else None)
            object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "_cache", {})

    def _get_cls(self) -> type | None:
        cls = object.__getattribute__(self, "_cls")
        if cls is not None:
            return cls
        cls_name = object.__getattribute__(self, "_cls_name")
        target_name = None
        if cls_name and cls_name in _MODULES:
            target_name = cls_name
        elif cls_name:
            for k in _MODULES:
                if k.lower() == cls_name.lower():
                    target_name = k
                    break
        if not target_name:
            name = object.__getattribute__(self, "_name")
            if name in _MODULES:
                target_name = name
            else:
                name_clean = name.replace("_", "").lower()
                for k in _MODULES:
                    if k.lower() == name_clean or k.lower() == name.lower():
                        target_name = k
                        break
        if target_name:
            try:
                mod = importlib.import_module(_MODULES[target_name])
                cls = getattr(mod, target_name, None)
                if cls is None and target_name == "VentralRelationalGraphMatcher":
                    cls = getattr(mod, "RelationalGraphMatcher", None)
                object.__setattr__(self, "_cls", cls)
            except Exception:
                pass
        return cls

    def __getattr__(self, name: str) -> Any:
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]
        cls = self._get_cls()
        if cls is not None:
            member = getattr(cls, name, None)
            if callable(member):
                neutral = _neutral(cls, name)
                cache[name] = lambda *a, **k: neutral
            else:
                cache[name] = None
            return cache[name]
        cache[name] = lambda *a, **k: None
        return cache[name]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def __setattr__(self, name: str, val: Any) -> None:
        pass  # swallow writes — the faculty is absent

    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())

    def __contains__(self, k: Any) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<AblatedFaculty {object.__getattribute__(self, '_name')}>"


def ablated(name: str, cls: type | str | None = None) -> AblatedFaculty:
    return AblatedFaculty(name, cls)


register("GibsonianAffordanceExtractor", "hbllm.hcir.world.cortex_perception")
register(
    "AmygdalarThreatConditioningEngine",
    "hbllm.hcir.world.experimental.amygdalar_threat_conditioning",
)
register("FalsifiableAnalogyEngine", "hbllm.hcir.world.cortex_episodic")
register("AnteriorInsularTimingEngine", "hbllm.hcir.world.experimental.anterior_insular_timing")
register("BilateralCoordinateIntegrator", "hbllm.hcir.world.experimental.bilateral_coordination")
register("PrefrontalBlameAttributor", "hbllm.hcir.world.experimental.prefrontal_blame_attribution")
register("BoundaryGridPhaseAligner", "hbllm.hcir.world.experimental.boundary_grid_phase_aligner")
register("CerebellarMicrozoneAdaptor", "hbllm.hcir.world.experimental.cerebellar_microzone_adaptor")
register(
    "ClaustrumInterhemisphericIntegrator",
    "hbllm.hcir.world.experimental.claustrum_interhemispheric",
)
register("ConfidenceCalibrator", "hbllm.hcir.world.experimental.confidence_calibrator")
register("CounterfactualRegretEvaluator", "hbllm.hcir.world.counterfactual_simulation")
register(
    "SpatialDistanceTransformEngine", "hbllm.hcir.world.experimental.spatial_distance_transform"
)
register("ExpectedFreeEnergyEvaluator", "hbllm.hcir.world.experimental.active_inference_efe")
register(
    "AnteriorCingulateEffortAllocator",
    "hbllm.hcir.world.experimental.anterior_cingulate_effort_allocator",
)
register("EntorhinalBorderCellEngine", "hbllm.hcir.world.experimental.entorhinal_border_cell")
register(
    "FrontopolarExplorationHorizonEngine",
    "hbllm.hcir.world.experimental.frontopolar_exploration_horizon",
)
register("HierarchicalActiveInferenceEngine", "hbllm.hcir.world.active_inference")
register("HippocampalPrioritizedReplayBuffer", "hbllm.hcir.world.experimental.hippocampal_replay")
register("HippocampalTimeCellEngine", "hbllm.hcir.world.experimental.hippocampal_time_cell")
register(
    "LocusCoeruleusNoradrenalineEngine",
    "hbllm.hcir.world.experimental.locus_coeruleus_noradrenaline",
)
register("MedialPrefrontalSchemaEngine", "hbllm.hcir.world.experimental.medial_prefrontal_schema")
register(
    "OrbitofrontalCreditAssignmentEngine",
    "hbllm.hcir.world.experimental.orbitofrontal_credit_assignment",
)
register("OpticalRayProjector", "hbllm.hcir.world.experimental.optical_ray_projection")
register("ParahippocampalContextEngine", "hbllm.hcir.world.experimental.parahippocampal_context")
register(
    "HippocampalPhasePrecessionEngine", "hbllm.hcir.world.experimental.hippocampal_phase_precession"
)
register("PointActuationAffordanceLearner", "hbllm.hcir.world.autonomous_learning")
register(
    "PedunculopontineLocomotorEngine", "hbllm.hcir.world.experimental.pedunculopontine_locomotor"
)
register(
    "VentralPremotorAffordanceResonator", "hbllm.hcir.world.experimental.ventral_premotor_resonator"
)
register("PulvinarAttentionRouter", "hbllm.hcir.world.experimental.pulvinar_attention_router")
register(
    "RetrosplenialLandmarkAnchorEngine",
    "hbllm.hcir.world.experimental.retrosplenial_landmark_anchor",
)
register("RetrosplenialHeadingEngine", "hbllm.hcir.world.experimental.retrosplenial_heading")
register("RetrosplenialCoordinateTransformer", "hbllm.hcir.world.parietal_coordinate_transform")
register(
    "DorsolateralRuleAbstractionEngine",
    "hbllm.hcir.world.experimental.dorsolateral_rule_abstraction",
)
register("SkillProgramSynthesizer", "hbllm.hcir.world.skill_program_synthesizer")
register(
    "VentromedialSomaticMarkerEngine", "hbllm.hcir.world.experimental.ventromedial_somatic_marker"
)
register("SomatosensoryHapticEngine", "hbllm.hcir.world.experimental.somatosensory_haptic")
register(
    "SubthalamicThresholdEscalator", "hbllm.hcir.world.experimental.subthalamic_threshold_escalator"
)
register("SubicularBoundaryVectorEngine", "hbllm.hcir.world.experimental.subicular_boundary_vector")
register(
    "SuperiorColliculusTargetArbiter", "hbllm.hcir.world.experimental.superior_colliculus_arbiter"
)
register("SuperpositionPlanner", "hbllm.hcir.world.experimental.superposition_planner")
register(
    "LateralPrefrontalTaskSwitcher", "hbllm.hcir.world.experimental.lateral_prefrontal_switcher"
)
register("ThalamicReticularGatingEngine", "hbllm.hcir.world.experimental.thalamic_reticular_gating")
register(
    "VentrolateralMotorInhibitionEngine",
    "hbllm.hcir.world.experimental.ventrolateral_motor_inhibition",
)
register(
    "VentralTegmentalCuriosityEngine", "hbllm.hcir.world.experimental.ventral_tegmental_curiosity"
)
register("GlobalWorkspaceIgnitionEngine", "hbllm.hcir.world.experimental.global_workspace_ignition")
register("GlobalNeuronalWorkspace", "hbllm.hcir.world.experimental.global_neuronal_workspace")
register("ThalamocorticalBindingEngine", "hbllm.hcir.world.experimental.thalamocortical_binding")
register("EntorhinalGridCortex", "hbllm.hcir.world.experimental.entorhinal_grid_cortex")
register("VentralRelationalGraphMatcher", "hbllm.hcir.world.experimental.relational_graph_matcher")
register("RelationalGraphMatcher", "hbllm.hcir.world.experimental.relational_graph_matcher")


def restore(obj, name: str) -> bool:
    """Test helper: replace an ablated faculty attr with a fresh real instance."""
    cur = getattr(obj, name, None)
    if isinstance(cur, AblatedFaculty):
        cls = cur._get_cls()
        if cls is not None:
            setattr(obj, name, cls())
            return True
    return False


def restore_all(engine) -> list[str]:
    """Test helper: restore every ablated faculty on an engine (no-arg ctors)."""
    restored = []
    import inspect as _i

    for name, val in list(vars(engine).items()):
        if isinstance(val, AblatedFaculty):
            cls = val._get_cls()
            if cls is None:
                continue
            try:
                sig = _i.signature(cls)
                if all(
                    p.default is not _i.Parameter.empty
                    or p.kind in (_i.Parameter.VAR_POSITIONAL, _i.Parameter.VAR_KEYWORD)
                    for p in sig.parameters.values()
                ):
                    setattr(engine, name, cls())
                    restored.append(name)
            except (TypeError, ValueError):
                pass
    return restored
