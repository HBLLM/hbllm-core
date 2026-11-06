"""WP2: experimental/ablated faculties — lazily imported per-module.

Faculties removed from engine construction by the measured ablation
(``reports/ablation/<ts>.json``) live here for backward-compat imports.
Do NOT re-export through ``hbllm.hcir.world``.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

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

    def __init__(self, name: str, cls: type | None = None) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str) -> Any:
        cls = object.__getattribute__(self, "_cls")
        if cls is not None:
            cache = object.__getattribute__(self, "_cache")
            if name not in cache:
                member = getattr(cls, name, None)
                if callable(member):
                    neutral = _neutral(cls, name)
                    cache[name] = lambda *a, **k: neutral
                else:
                    cache[name] = None
            return cache[name]
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


def ablated(name: str, cls: type | None = None) -> AblatedFaculty:
    return AblatedFaculty(name, cls)


register("GibsonianAffordanceExtractor", "hbllm.hcir.world.cortex_perception")
register("InteroceptiveAllostasisCortex", "hbllm.hcir.world.insula_allostasis")
register("AmygdalarThreatConditioningEngine", "hbllm.hcir.world.amygdalar_threat_conditioning")
register("FalsifiableAnalogyEngine", "hbllm.hcir.world.cortex_episodic")
register("AnteriorInsularTimingEngine", "hbllm.hcir.world.anterior_insular_timing")
register("BilateralCoordinateIntegrator", "hbllm.hcir.world.bilateral_coordination")
register("PrefrontalBlameAttributor", "hbllm.hcir.world.prefrontal_blame_attribution")
register("BoundaryGridPhaseAligner", "hbllm.hcir.world.boundary_grid_phase_aligner")
register("CerebellarMicrozoneAdaptor", "hbllm.hcir.world.cerebellar_microzone_adaptor")
register("ClaustrumSalienceConductor", "hbllm.hcir.world.claustrum_salience_conductor")
register("ClaustrumInterhemisphericIntegrator", "hbllm.hcir.world.claustrum_interhemispheric")
register("ConfidenceCalibrator", "hbllm.hcir.world.confidence_calibrator")
register("CounterfactualRegretEvaluator", "hbllm.hcir.world.counterfactual_simulation")
register("SpatialDistanceTransformEngine", "hbllm.hcir.world.spatial_distance_transform")
register("SubthalamicDriftDiffusionEngine", "hbllm.hcir.world.subthalamic_drift_diffusion")
register("ExpectedFreeEnergyEvaluator", "hbllm.hcir.world.active_inference_efe")
register("AnteriorCingulateEffortAllocator", "hbllm.hcir.world.anterior_cingulate_effort_allocator")
register("EntorhinalBorderCellEngine", "hbllm.hcir.world.entorhinal_border_cell")
register("FrontopolarExplorationHorizonEngine", "hbllm.hcir.world.frontopolar_exploration_horizon")
register("ParietalFoveatedAttentionEngine", "hbllm.hcir.world.parietal_foveated_attention")
register("HierarchicalActiveInferenceEngine", "hbllm.hcir.world.active_inference")
register("HippocampalPrioritizedReplayBuffer", "hbllm.hcir.world.hippocampal_replay")
register("HippocampalTimeCellEngine", "hbllm.hcir.world.hippocampal_time_cell")
register("LocusCoeruleusNoradrenalineEngine", "hbllm.hcir.world.locus_coeruleus_noradrenaline")
register("MedialPrefrontalSchemaEngine", "hbllm.hcir.world.medial_prefrontal_schema")
register("OrbitofrontalCreditAssignmentEngine", "hbllm.hcir.world.orbitofrontal_credit_assignment")
register("OpticalRayProjector", "hbllm.hcir.world.optical_ray_projection")
register("ParahippocampalContextEngine", "hbllm.hcir.world.parahippocampal_context")
register("HippocampalPhasePrecessionEngine", "hbllm.hcir.world.hippocampal_phase_precession")
register("PointActuationAffordanceLearner", "hbllm.hcir.world.autonomous_learning")
register("ParietalPotentialFieldNavigator", "hbllm.hcir.world.parietal_potential_field")
register("PedunculopontineLocomotorEngine", "hbllm.hcir.world.pedunculopontine_locomotor")
register("VentralPremotorAffordanceResonator", "hbllm.hcir.world.ventral_premotor_resonator")
register("PulvinarAttentionRouter", "hbllm.hcir.world.pulvinar_attention_router")
register("RetrosplenialLandmarkAnchorEngine", "hbllm.hcir.world.retrosplenial_landmark_anchor")
register("RetrosplenialHeadingEngine", "hbllm.hcir.world.retrosplenial_heading")
register("RetrosplenialCoordinateTransformer", "hbllm.hcir.world.parietal_coordinate_transform")
register("DorsolateralRuleAbstractionEngine", "hbllm.hcir.world.dorsolateral_rule_abstraction")
register("InferotemporalShapeHasher", "hbllm.hcir.world.inferotemporal_shape_hasher")
register("SkillProgramSynthesizer", "hbllm.hcir.world.skill_program_synthesizer")
register("VentromedialSomaticMarkerEngine", "hbllm.hcir.world.ventromedial_somatic_marker")
register("SomatosensoryHapticEngine", "hbllm.hcir.world.somatosensory_haptic")
register("SubthalamicThresholdEscalator", "hbllm.hcir.world.subthalamic_threshold_escalator")
register("SubicularBoundaryVectorEngine", "hbllm.hcir.world.subicular_boundary_vector")
register("SuperiorColliculusTargetArbiter", "hbllm.hcir.world.superior_colliculus_arbiter")
register("SuperpositionPlanner", "hbllm.hcir.world.superposition_planner")
register("LateralPrefrontalTaskSwitcher", "hbllm.hcir.world.lateral_prefrontal_switcher")
register("ThalamicReticularGatingEngine", "hbllm.hcir.world.thalamic_reticular_gating")
register("DualValueOpponencyCortex", "hbllm.hcir.world.habenula_striatum_opponency")
register("VentrolateralMotorInhibitionEngine", "hbllm.hcir.world.ventrolateral_motor_inhibition")
register("VentralTegmentalCuriosityEngine", "hbllm.hcir.world.ventral_tegmental_curiosity")
register("GlobalWorkspaceIgnitionEngine", "hbllm.hcir.world.global_workspace_ignition")


def restore(obj, name: str) -> bool:
    """Test helper: replace an ablated faculty attr with a fresh real instance."""
    cur = getattr(obj, name, None)
    if isinstance(cur, AblatedFaculty) and object.__getattribute__(cur, "_cls") is not None:
        setattr(obj, name, object.__getattribute__(cur, "_cls")())
        return True
    return False


def restore_all(engine) -> list[str]:
    """Test helper: restore every ablated faculty on an engine (no-arg ctors)."""
    restored = []
    import inspect as _i

    for name, val in list(vars(engine).items()):
        if isinstance(val, AblatedFaculty):
            cls = object.__getattribute__(val, "_cls")
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
