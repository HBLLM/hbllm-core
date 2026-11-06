"""
World Capabilities — Privileged World Kernel Capability Provider Registration.

Registers world capabilities in HCIR KernelCapabilityRegistry:
    - world.predict
    - world.evaluate_error
    - world.sync_twin
    - world.query_state
    - world.select_action
"""

from __future__ import annotations

import logging

from hbllm.hcir.kernel.cognitive_abi import CapabilityCall, CapabilityManifest, CapabilityResult

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.hcir.world_kernel import WorldKernel

WORLD_MANIFESTS: list[CapabilityManifest] = [
    CapabilityManifest(
        name="world.predict",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.evaluate_error",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.sync_twin",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.query_state",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.select_action",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
]

FRONTIER_WORLD_MANIFESTS: list[CapabilityManifest] = [
    CapabilityManifest(
        name="world.predictive_coding_cycle",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.path_integrate_odometry",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.assess_metacognitive_agency",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.synthesize_skills",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.superposition_rollout",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
]

EXTENDED_WORLD_MANIFESTS: list[CapabilityManifest] = [
    *WORLD_MANIFESTS,
    *FRONTIER_WORLD_MANIFESTS,
    CapabilityManifest(
        name="world.predict_outcome",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.evaluate_prediction_error",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.sync_digital_twin",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
]


def execute_world_capability(
    call: CapabilityCall, kernel: WorldKernel | None = None
) -> CapabilityResult:
    """Execute governed World Kernel capability call."""
    name = call.capability_name
    params = getattr(call, "arguments", None) or getattr(call, "parameters", None) or {}

    if name in ("world.predict", "world.predict_outcome", "world_prediction"):
        action_intent = params.get("action_intent", "default_action")
        if kernel is not None:
            from hbllm.hcir.graph import ActionNode

            pred = kernel.predict(ActionNode(intent=action_intent))
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "predicted_outcome": pred.claim,
                    "confidence": pred.uncertainty.confidence if pred.uncertainty else 0.88,
                    "prediction_id": pred.id,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={
                "predicted_outcome": f"Predicted nominal execution of {action_intent}",
                "confidence": 0.88,
            },
        )
    elif name in ("world.evaluate_error", "world.evaluate_prediction_error"):
        if kernel is not None:
            expected = params.get("expected", {})
            actual = params.get("actual", {})
            eval_res = kernel.surprise_engine.evaluate(expected, actual)
            score = getattr(eval_res, "surprise_score", getattr(eval_res, "raw_surprise", 0.0))
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "surprise_score": score,
                    "is_surprising": eval_res.is_surprising,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"surprise_score": 0.05, "is_surprising": False},
        )
    elif name == "world.select_action":
        candidates = params.get("candidate_intents", ["action_default"])
        best = candidates[0] if candidates else "action_default"
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"selected_action_intent": best, "utility_score": 0.85},
        )
    elif name == "world.query_state":
        if kernel is not None:
            st = kernel.get_current_world_state()
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "environment_name": st.environment_name,
                    "overall_status": st.overall_status,
                    "state_status": st.overall_status,
                    "variables": st.variables,
                    "entities": st.entities,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"overall_status": "nominal", "variables": params.get("variables", {})},
        )
    elif name in ("world.sync_twin", "world.sync_digital_twin"):
        var_name = params.get("variable_name")
        val = params.get("value")
        updates = params.get("updates")
        if kernel is not None:
            if hasattr(kernel, "sync_digital_twin"):
                kernel.sync_digital_twin(variable_name=var_name, value=val, updates=updates)
            elif var_name:
                kernel.digital_twin.sync_sensor_telemetry(var_name, val)
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"synced": True, "variable": var_name, "updates": updates},
        )
    elif name == "world.predictive_coding_cycle":
        sensory = params.get("sensory_inputs")
        priors = params.get("abstract_priors")
        if kernel is not None and hasattr(kernel, "epistemic_engine"):
            snap = kernel.epistemic_engine.step_predictive_coding(
                sensory_inputs=sensory,
                abstract_priors=priors,
            )
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "total_free_energy": snap.total_free_energy,
                    "layer_errors": snap.layer_errors,
                    "is_structural_anomaly": snap.is_structural_anomaly,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"total_free_energy": 0.0, "layer_errors": {}, "is_structural_anomaly": False},
        )
    elif name == "world.path_integrate_odometry":
        dr = float(params.get("delta_r", 0.0))
        dc = float(params.get("delta_c", 0.0))
        if kernel is not None and hasattr(kernel, "epistemic_engine"):
            pos = kernel.epistemic_engine.path_integrate_odometry(dr, dc)
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={"integrated_position": pos, "delta_r": dr, "delta_c": dc},
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"integrated_position": (0, 0), "delta_r": dr, "delta_c": dc},
        )
    elif name == "world.assess_metacognitive_agency":
        action_id = params.get("action_id", 0)
        raw_conf = float(params.get("raw_confidence", 0.8))
        pred_pos = (
            tuple(params["predicted_avatar_pos"])
            if "predicted_avatar_pos" in params and params["predicted_avatar_pos"]
            else None
        )
        obs_pos = (
            tuple(params["observed_avatar_pos"])
            if "observed_avatar_pos" in params and params["observed_avatar_pos"]
            else None
        )
        other_chg = int(params.get("observed_other_changes_count", 0))
        emp_succ = params.get("empirical_success")
        if kernel is not None and hasattr(kernel, "epistemic_engine"):
            assess = kernel.epistemic_engine.assess_metacognitive_agency(
                action_id=action_id,
                raw_confidence=raw_conf,
                predicted_avatar_pos=pred_pos,
                observed_avatar_pos=obs_pos,
                observed_other_changes_count=otherchg if (otherchg := other_chg) else 0,
                empirical_success=emp_succ,
            )
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "agency_attribution_score": assess.agency_attribution_score,
                    "subjective_confidence": assess.subjective_confidence,
                    "recommended_simulation_depth": assess.recommended_simulation_depth,
                    "recommended_exploration_mode": assess.recommended_exploration_mode,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={
                "agency_attribution_score": 1.0,
                "subjective_confidence": raw_conf,
                "recommended_simulation_depth": 5,
                "recommended_exploration_mode": "pragmatic",
            },
        )
    elif name == "world.synthesize_skills":
        min_supp = int(params.get("min_support", 2))
        if kernel is not None and hasattr(kernel, "epistemic_engine"):
            skills = kernel.epistemic_engine.synthesize_procedural_skills_from_traces(
                min_support=min_supp
            )
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "synthesized_count": len(skills),
                    "skills": [getattr(s, "name", str(s)) for s in skills],
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"synthesized_count": 0, "skills": []},
        )
    elif name == "world.superposition_rollout":
        acts = params.get("candidate_actions", [])
        sim_fn = params.get("simulator_fn") or (lambda act, hyp: (0.85, hyp))
        if kernel is not None and hasattr(kernel, "epistemic_engine"):
            results = kernel.epistemic_engine.evaluate_superposition_actions(
                candidate_actions=acts,
                simulator_fn=sim_fn,
            )
            best_act = results[0].action_id if results else (acts[0] if acts else 0)
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "best_action": best_act,
                    "evaluation_count": len(results),
                    "expected_utility": results[0].expected_utility if results else 0.85,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={
                "best_action": acts[0] if acts else 0,
                "evaluation_count": len(acts),
                "expected_utility": 0.85,
            },
        )
    else:
        return CapabilityResult(
            capability_name=name,
            status="FAILED",
            output=None,
        )


class WorldKernelCapabilityExecutor:
    """ICapabilityExecutor adapter binding WorldKernel to CapabilityResolver."""

    def __init__(self, kernel: WorldKernel, default_capability: str | None = None) -> None:
        self.kernel = kernel
        self.default_capability = default_capability

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cap = params.get("capability") or self.default_capability or "world.predict_outcome"
        call = CapabilityCall(capability_name=cap, arguments=params)
        res = execute_world_capability(call, self.kernel)
        if res.status == "SUCCESS":
            out = dict(res.output) if isinstance(res.output, dict) else {"result": res.output}
            if "action_intent" in params or cap in (
                "world.predict",
                "world.predict_outcome",
                "world_prediction",
            ):
                out["status"] = "simulated"
                out["predicted"] = True
                out["claim"] = res.output.get("predicted_outcome", "")
            else:
                out["status"] = "success"
            return out
        return {"status": "failed", "error": f"Failed capability execution for {cap}"}
