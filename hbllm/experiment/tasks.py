"""Standardized Experimental Task Battery (E1 through E7).

Implements the 7 evaluative dimensions without hardcoded cohort checks:
E1: Grounded Concept Acquisition (Sample efficiency N_tau)
E2: Artificial Lexical Acquisition (Fast-mapping novel tokens)
E3: Counterfactual Mental Simulation (Simulation fidelity vs Planning regret)
E4: Epistemic Calibration (Brier score, ECE, Selective risk, Coverage)
E5: Active Epistemic Discovery (Independent Oracle Regret & Info Efficiency)
E6: Relational Generalization (2x2 factorial structural vs surface transfer)
E7: Lifelong Continual Curriculum (Sequential T1..T5, Full 5x5 R_{i,j} Matrix)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from hbllm.experiment.cohorts import BaseCohort
from hbllm.experiment.environments import (
    CanonicalTaskEnvironment,
    EnvironmentObservation,
    PhysicalEnvironmentState,
)
from hbllm.experiment.metrics import ExperimentMetricsCalculator


@dataclass
class TaskEvaluationResult:
    """Standardized output record for a task evaluation."""

    task_id: str
    cohort_id: str
    episodes_to_threshold: int | None = None  # N_tau
    accuracy: float = 1.0
    simulation_error: float = 0.0
    plan_regret: float = 0.0
    brier_score: float = 0.0
    ece: float = 0.0
    coverage: float = 1.0
    selective_risk: float = 0.0
    probing_regret: float = 0.0
    info_efficiency: float = 1.0
    transfer_systematicity: float = 1.0
    structural_accuracy: float = 1.0
    surface_distraction_rate: float = 0.0
    continual_matrix_r: list[list[float]] = field(default_factory=list)
    bwt: float = 0.0
    fwt: float = 0.0
    resource_consumption: dict[str, float] = field(default_factory=dict)


class E1_ConceptAcquisitionTask:  # noqa: N801
    """E1: Grounded Concept Acquisition. Evaluates sample efficiency N_tau to achieve >= 0.80 accuracy."""

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        cohort.reset()
        rng = random.Random(seed)
        all_shapes = ["cylinder", "cube", "sphere", "cone", "pyramid", "torus"]
        all_colors = ["blue", "red", "green", "yellow", "purple", "cyan"]

        target_shape = rng.choice(all_shapes)
        target_color = rng.choice(all_colors)

        # Generate seed-dependent training and testing pools
        train_pool = [
            {
                "id": f"tr_{i}",
                "shape": s,
                "color": c,
                "is_concept": (s == target_shape and c == target_color),
            }
            for i, (s, c) in enumerate(
                [
                    (target_shape, target_color),
                    (rng.choice([s for s in all_shapes if s != target_shape]), target_color),
                    (target_shape, rng.choice([c for c in all_colors if c != target_color])),
                    (
                        rng.choice([s for s in all_shapes if s != target_shape]),
                        rng.choice([c for c in all_colors if c != target_color]),
                    ),
                ]
            )
        ]
        test_pool = [
            {
                "id": f"ts_{i}",
                "shape": s,
                "color": c,
                "is_concept": (s == target_shape and c == target_color),
            }
            for i, (s, c) in enumerate(
                [
                    (target_shape, target_color),
                    (target_shape, target_color),
                    (rng.choice([s for s in all_shapes if s != target_shape]), target_color),
                    (target_shape, rng.choice([c for c in all_colors if c != target_color])),
                    (
                        rng.choice([s for s in all_shapes if s != target_shape]),
                        rng.choice([c for c in all_colors if c != target_color]),
                    ),
                ]
            )
        ]

        train_order = list(train_pool)
        rng.shuffle(train_order)

        accuracies = []
        for ep in range(10):
            train_item = train_order[ep % len(train_order)]
            obs = EnvironmentObservation(
                step_index=ep,
                visible_entities=[
                    {
                        "id": train_item["id"],
                        "type": "concept_item",
                        "properties": {
                            "shape": train_item["shape"],
                            "color": train_item["color"],
                        },
                    }
                ],
                spatial_relations=[],
                goal_description="concept_categorization",
                available_actions=[{"name": "CLASSIFY_POSITIVE"}, {"name": "CLASSIFY_NEGATIVE"}],
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)
            is_pos = out.action.get("name") == "CLASSIFY_POSITIVE"
            reward = 1.0 if (is_pos == train_item["is_concept"]) else -1.0
            cohort.learn_from_feedback(obs, reward)

            # Test evaluation on unseen test set
            correct = 0
            for test_item in test_pool:
                t_obs = EnvironmentObservation(
                    step_index=ep,
                    visible_entities=[
                        {
                            "id": test_item["id"],
                            "type": "concept_item",
                            "properties": {
                                "shape": test_item["shape"],
                                "color": test_item["color"],
                            },
                        }
                    ],
                    spatial_relations=[],
                    goal_description="concept_categorization",
                    available_actions=[
                        {"name": "CLASSIFY_POSITIVE"},
                        {"name": "CLASSIFY_NEGATIVE"},
                    ],
                    interaction_history=[],
                    resource_budget={},
                )
                t_out = cohort.process_observation(t_obs)
                t_pos = t_out.action.get("name") == "CLASSIFY_POSITIVE"
                if t_pos == test_item["is_concept"]:
                    correct += 1
            accuracies.append(correct / len(test_pool))

        n_tau = ExperimentMetricsCalculator.calculate_episodes_to_threshold(
            accuracies, tau=0.80, consecutive_m=3
        ) or len(accuracies)

        return TaskEvaluationResult(
            task_id="E1_ConceptAcquisition",
            cohort_id=cohort.cohort_id,
            episodes_to_threshold=n_tau,
            accuracy=round(accuracies[-1], 4),
            resource_consumption=cohort.get_resource_usage(),
        )


class E2_LexicalAcquisitionTask:  # noqa: N801
    """E2: Artificial Lexical Acquisition. Fast-mapping novel artificial tokens ('mepo', 'dax')."""

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        cohort.reset()
        rng = random.Random(seed)
        token_pool = ["dax", "mepo", "blicket", "koba", "fep", "toma"]
        chosen_tokens = rng.sample(token_pool, 2)
        tok_1, tok_2 = chosen_tokens[0], chosen_tokens[1]

        scenes: list[dict[str, Any]] = [
            {
                "token": tok_1,
                "target": "cylinder",
                "entities": [{"id": "e1", "type": "cylinder"}, {"id": "e2", "type": "cube"}],
            },
            {
                "token": tok_2,
                "target": "cube",
                "entities": [{"id": "e3", "type": "cube"}, {"id": "e4", "type": "sphere"}],
            },
            {
                "token": tok_1,
                "target": "cylinder",
                "entities": [{"id": "e5", "type": "cylinder"}, {"id": "e6", "type": "cone"}],
            },
            {
                "token": tok_2,
                "target": "cube",
                "entities": [{"id": "e7", "type": "cube"}, {"id": "e8", "type": "pyramid"}],
            },
        ]
        test_items: list[dict[str, Any]] = [
            {
                "token": tok_1,
                "expected": "cylinder",
                "entities": [{"id": "t1", "type": "cylinder"}, {"id": "t2", "type": "sphere"}],
            },
            {
                "token": tok_2,
                "expected": "cube",
                "entities": [{"id": "t3", "type": "cube"}, {"id": "t4", "type": "cone"}],
            },
        ]

        scenes_order = rng.sample(scenes, len(scenes))
        accuracies = []
        for idx, scene in enumerate(scenes_order):
            entities = rng.sample(scene["entities"], len(scene["entities"]))
            obs = EnvironmentObservation(
                step_index=idx,
                visible_entities=entities,
                spatial_relations=[],
                goal_description=f"lexical_grounding:{scene['token']}",
                available_actions=[
                    {"name": "SELECT", "parameters": {"target": e["id"]}} for e in entities
                ],
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)
            chosen_id = out.action.get("parameters", {}).get("target", "")
            chosen_type = next((e["type"] for e in entities if e["id"] == chosen_id), "")
            reward = 1.0 if chosen_type == scene["target"] else -1.0
            cohort.learn_from_feedback(obs, reward)

            # Test probe
            correct = 0
            for t_item in test_items:
                t_entities = rng.sample(t_item["entities"], len(t_item["entities"]))
                t_obs = EnvironmentObservation(
                    step_index=idx,
                    visible_entities=t_entities,
                    spatial_relations=[],
                    goal_description=f"lexical_grounding:{t_item['token']}",
                    available_actions=[
                        {"name": "SELECT", "parameters": {"target": e["id"]}} for e in t_entities
                    ],
                    interaction_history=[],
                    resource_budget={},
                )
                t_out = cohort.process_observation(t_obs)
                t_id = t_out.action.get("parameters", {}).get("target", "")
                t_type = next((e["type"] for e in t_entities if e["id"] == t_id), "")
                if t_type == t_item["expected"]:
                    correct += 1
            accuracies.append(correct / len(test_items))

        n_tau = ExperimentMetricsCalculator.calculate_episodes_to_threshold(
            accuracies, tau=1.0, consecutive_m=1
        ) or len(accuracies)

        return TaskEvaluationResult(
            task_id="E2_LexicalAcquisition",
            cohort_id=cohort.cohort_id,
            episodes_to_threshold=n_tau,
            accuracy=round(accuracies[-1], 4),
            resource_consumption=cohort.get_resource_usage(),
        )


class E3_CounterfactualSimulationTask:  # noqa: N801
    """E3: Counterfactual Mental Simulation. Evaluates simulation fidelity vs planning regret."""

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        cohort.reset()
        rng = random.Random(seed)
        env = CanonicalTaskEnvironment(domain="counterfactual_simulation")
        oracle = env.oracle

        scenarios = [
            {"base": "support_flat", "geom": "flat", "rigid": True, "goal": "STACK"},
            {"base": "support_curved", "geom": "curved", "rigid": True, "goal": "STACK"},
            {"base": "support_soft", "geom": "flat", "rigid": False, "goal": "STACK"},
            {"base": "support_stable", "geom": "flat", "rigid": True, "goal": "STACK"},
            {"base": "support_dome", "geom": "curved", "rigid": True, "goal": "STACK"},
        ]
        scenarios_order = rng.sample(scenarios, len(scenarios))
        sim_errors: list[float] = []
        regrets: list[float] = []
        successes: list[bool] = []

        for idx, sc in enumerate(scenarios_order):
            true_state = PhysicalEnvironmentState(
                entities={
                    sc["base"]: {"surface_geometry": sc["geom"], "is_rigid": sc["rigid"]},
                    "placed_block": {"surface_geometry": "flat", "is_rigid": True},
                },
                physics_rules={},
                target_goal={"action": sc["goal"]},
            )
            obs = EnvironmentObservation(
                step_index=idx,
                visible_entities=[
                    {
                        "id": sc["base"],
                        "type": "support_base",
                        "properties": {
                            "surface_geometry": sc["geom"],
                            "is_rigid": sc["rigid"],
                        },
                    },
                    {
                        "id": "placed_block",
                        "type": "block",
                        "properties": {
                            "surface_geometry": "flat",
                            "is_rigid": True,
                        },
                    },
                ],
                spatial_relations=[],
                goal_description="spatial_stacking",
                available_actions=[
                    {
                        "name": "STACK",
                        "parameters": {"base_id": sc["base"], "item_id": "placed_block"},
                        "cost": 1.0,
                    },
                    {"name": "WAIT", "parameters": {}, "cost": 0.1},
                ],
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)

            # Oracle optimal utility
            is_physically_stable = sc["geom"] == "flat" and sc["rigid"]
            opt_act = (
                {"name": "STACK", "parameters": {"base_id": sc["base"], "item_id": "placed_block"}}
                if is_physically_stable
                else {"name": "WAIT", "parameters": {}}
            )
            opt_u = (
                oracle.evaluate_true_action_utility(opt_act, true_state)
                if opt_act["name"] == "STACK"
                else 0.0
            )

            achieved_u = (
                oracle.evaluate_true_action_utility(out.action, true_state)
                if out.action.get("name") == "STACK"
                else 0.0
            )
            regret = max(0.0, opt_u - achieved_u)
            regrets.append(regret)

            # Simulation fidelity
            true_stable = 1.0 if is_physically_stable else 0.0
            if out.prediction in ("stable", "stable_support"):
                pred_stable = 1.0
            elif out.prediction in ("unstable", "abstain_uncertain"):
                pred_stable = 0.0
            elif out.prediction == "unsimulated_action":
                pred_stable = 0.50
            else:
                pred_stable = float(out.confidence) if out.confidence is not None else 0.50

            sim_error = abs(pred_stable - true_stable)
            sim_errors.append(sim_error)

            success = achieved_u >= 0.0 and regret < 0.1
            successes.append(success)
            cohort.learn_from_feedback(obs, achieved_u)

        mean_sim_error = sum(sim_errors) / len(sim_errors)
        mean_regret = sum(regrets) / len(regrets)
        acc = sum(1.0 if s else 0.0 for s in successes) / len(successes)

        return TaskEvaluationResult(
            task_id="E3_CounterfactualSimulation",
            cohort_id=cohort.cohort_id,
            simulation_error=round(mean_sim_error, 4),
            plan_regret=round(mean_regret, 4),
            accuracy=round(acc, 4),
            resource_consumption=cohort.get_resource_usage(),
        )


class E4_EpistemicCalibrationTask:  # noqa: N801
    """E4: Epistemic Calibration. Evaluates Brier score, ECE, Coverage, and Selective Risk."""

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        cohort.reset()
        rng = random.Random(seed)

        trials = [
            {"geom": "flat", "rigid": True, "label": True},
            {"geom": "flat", "rigid": True, "label": True},
            {"geom": "flat", "rigid": True, "label": True},
            {"geom": "flat", "rigid": True, "label": True},
            {"geom": "curved", "rigid": True, "label": False},
            {"geom": "curved", "rigid": True, "label": False},
            {"geom": "flat", "rigid": False, "label": False},
            {"geom": "curved", "rigid": False, "label": False},
            {"geom": "irregular_bevel", "rigid": True, "label": False},
            {"geom": "micro_groove", "rigid": True, "label": False},
            {"geom": "non_euclidean_mesh", "rigid": True, "label": False},
            {"geom": "quantum_foam_base", "rigid": True, "label": False},
        ]
        trials_order = rng.sample(trials, len(trials))

        confidences: list[float] = []
        outcomes: list[bool] = []
        abstentions: list[bool] = []

        for idx, tr in enumerate(trials_order):
            obs = EnvironmentObservation(
                step_index=idx,
                visible_entities=[
                    {
                        "id": f"base_{idx}",
                        "type": "support_base",
                        "properties": {
                            "surface_geometry": tr["geom"],
                            "is_rigid": tr["rigid"],
                        },
                    }
                ],
                spatial_relations=[],
                goal_description="epistemic_calibration",
                available_actions=[{"name": "STACK", "parameters": {"base": f"base_{idx}"}}],
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)
            confidences.append(out.confidence)
            abstentions.append(out.abstain)
            pred_true = out.prediction in ("stable", "stable_support")
            actual_true = tr["label"]
            success = pred_true == actual_true
            outcomes.append(success)
            cohort.learn_from_feedback(obs, 1.0 if success else -1.0)

        brier = ExperimentMetricsCalculator.calculate_brier_score(confidences, outcomes)
        ece = ExperimentMetricsCalculator.calculate_ece(confidences, outcomes, num_bins=4)
        risk, cov = ExperimentMetricsCalculator.calculate_selective_risk_and_coverage(
            confidences, outcomes, abstentions
        )
        acc = sum(1.0 if o else 0.0 for o in outcomes) / len(outcomes)

        return TaskEvaluationResult(
            task_id="E4_EpistemicCalibration",
            cohort_id=cohort.cohort_id,
            brier_score=brier,
            ece=ece,
            coverage=cov,
            selective_risk=risk,
            accuracy=round(acc, 4),
            resource_consumption=cohort.get_resource_usage(),
        )


class E5_ActiveEpistemicDiscoveryTask:  # noqa: N801
    """E5: Active Epistemic Discovery. Evaluates Regret against independent oracle and Information Efficiency."""

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        cohort.reset()
        rng = random.Random(seed)
        env = CanonicalTaskEnvironment(domain="active_discovery")
        oracle = env.oracle

        scenarios: list[dict[str, Any]] = [
            {"name": "hollow_cube", "cost_probe": 0.2, "d_power": 0.85, "is_safe": True},
            {"name": "magnetic_cylinder", "cost_probe": 0.15, "d_power": 0.90, "is_safe": True},
            {"name": "fragile_glass", "cost_probe": 0.25, "d_power": 0.80, "is_safe": False},
            {
                "name": "uneven_weight_block",
                "cost_probe": 0.20,
                "d_power": 0.85,
                "is_safe": False,
            },
            {"name": "solid_granite", "cost_probe": 0.10, "d_power": 0.95, "is_safe": True},
        ]
        scenarios_order = rng.sample(scenarios, len(scenarios))
        probing_regrets: list[float] = []
        efficiencies: list[float] = []
        successes: list[bool] = []

        for idx, sc in enumerate(scenarios_order):
            cost_jitter = round(rng.uniform(-0.01, 0.01), 3)
            actual_cost = max(0.05, float(sc["cost_probe"]) + cost_jitter)
            actions = [
                {
                    "name": "PROBE_GENTLE_TAP",
                    "parameters": {"target": sc["name"]},
                    "cost": actual_cost,
                    "discriminative_power": sc["d_power"],
                },
                {"name": "STACK", "parameters": {"base": sc["name"]}, "cost": 1.0},
            ]
            obs = EnvironmentObservation(
                step_index=idx,
                visible_entities=[
                    {"id": sc["name"], "type": "block", "properties": {"opaque": True}}
                ],
                spatial_relations=[],
                goal_description="active_discovery",
                available_actions=actions,
                interaction_history=[],
                resource_budget={},
            )
            _, opt_eff = oracle.compute_optimal_probe(actions, {"safe": 0.50, "unsafe": 0.50})

            out = cohort.process_observation(obs)
            if "PROBE" in out.action.get("name", ""):
                achieved_eff = float(sc["d_power"]) / float(actual_cost)
                regret = max(0.0, opt_eff - achieved_eff)
                success = True
            else:
                achieved_eff = 0.0
                regret = opt_eff
                success = bool(sc["is_safe"])

            probing_regrets.append(regret)
            efficiencies.append(achieved_eff)
            successes.append(success)
            cohort.learn_from_feedback(obs, 1.0 if success else -1.0)

        mean_regret = sum(probing_regrets) / len(probing_regrets)
        mean_eff = sum(efficiencies) / len(efficiencies)
        norm_eff = (
            mean_eff / max(0.01, (mean_eff + mean_regret)) if (mean_eff + mean_regret) > 0 else 0.50
        )
        acc = sum(1.0 if s else 0.0 for s in successes) / len(successes)

        return TaskEvaluationResult(
            task_id="E5_ActiveDiscovery",
            cohort_id=cohort.cohort_id,
            probing_regret=round(mean_regret, 4),
            info_efficiency=round(norm_eff, 4),
            accuracy=round(acc, 4),
            resource_consumption=cohort.get_resource_usage(),
        )


class E6_RelationalTransferTask:  # noqa: N801
    """E6: Relational Generalization. Evaluates 2x2 factorial structural vs surface transfer."""

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        cohort.reset()
        rng = random.Random(seed)

        trials = [
            {
                "is_structural_match": True,
                "is_surface_distractor": False,
                "target": "nucleus_electron",
            },
            {"is_structural_match": True, "is_surface_distractor": False, "target": "star_comet"},
            {
                "is_structural_match": False,
                "is_surface_distractor": True,
                "target": "yellow_blue_balls",
            },
            {
                "is_structural_match": False,
                "is_surface_distractor": True,
                "target": "yellow_lamp_globe",
            },
            {
                "is_structural_match": True,
                "is_surface_distractor": True,
                "target": "solar_system_orbit",
            },
            {
                "is_structural_match": False,
                "is_surface_distractor": False,
                "target": "unrelated_noise",
            },
        ]
        trials_order = rng.sample(trials, len(trials))

        structural_correct = 0
        surface_distracted = 0
        total_structural = 0
        total_surface = 0

        for idx, tr in enumerate(trials_order):
            if tr["is_structural_match"]:
                entities = [
                    {
                        "id": "core",
                        "type": "central_mass",
                        "properties": {"is_center": True, "color": "grey"},
                    },
                    {
                        "id": "orbiter",
                        "type": "satellite",
                        "properties": {"is_center": False, "color": "white"},
                    },
                ]
                relations = [{"source": "core", "target": "orbiter", "relation": "SUPPORTS"}]
                total_structural += 1
            else:
                entities = [
                    {
                        "id": "ball_yellow",
                        "type": "toy",
                        "properties": {"color": "yellow_like_sun"},
                    },
                    {
                        "id": "ball_blue",
                        "type": "toy",
                        "properties": {"color": "blue_like_earth"},
                    },
                ]
                relations = []
                total_surface += 1

            actions = rng.sample(
                [{"name": "MAP_STRUCTURAL_CENTRAL"}, {"name": "MAP_SURFACE_COLOR"}], 2
            )
            obs = EnvironmentObservation(
                step_index=idx,
                visible_entities=entities,
                spatial_relations=relations,
                goal_description="relational_transfer",
                available_actions=actions,
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)
            chosen_action = out.action.get("name", "")

            if tr["is_structural_match"]:
                if chosen_action == "MAP_STRUCTURAL_CENTRAL":
                    structural_correct += 1
            if tr["is_surface_distractor"]:
                if chosen_action == "MAP_SURFACE_COLOR":
                    surface_distracted += 1

            cohort.learn_from_feedback(obs, 1.0 if tr["is_structural_match"] else 0.0)

        struct_acc = structural_correct / max(1, total_structural)
        distract_rate = surface_distracted / max(1, total_surface)
        systematicity = max(0.0, struct_acc - (0.5 * distract_rate))

        return TaskEvaluationResult(
            task_id="E6_RelationalTransfer",
            cohort_id=cohort.cohort_id,
            structural_accuracy=round(struct_acc, 4),
            surface_distraction_rate=round(distract_rate, 4),
            transfer_systematicity=round(systematicity, 4),
            accuracy=round(struct_acc, 4),
            resource_consumption=cohort.get_resource_usage(),
        )


def _build_curriculum_observation(
    task_name: str, prefix: str, step_index: int = 0
) -> EnvironmentObservation:
    """Construct grounded physical entities, spatial and functional relations for a curriculum stage."""
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, str]] = []

    if task_name == "T1_SpatialStacking":
        entities = [
            {
                "id": f"{prefix}_base",
                "type": "physical_entity",
                "properties": {"rigidity": "rigid", "surface": "flat", "stable": True},
            },
            {
                "id": f"{prefix}_payload",
                "type": "physical_entity",
                "properties": {"stable": True, "mass": 1.0},
            },
        ]
        relations = [
            {"source": f"{prefix}_base", "target": f"{prefix}_payload", "relation": "SUPPORTS"},
            {"source": f"{prefix}_base", "target": f"{prefix}_payload", "relation": "STABLE_FOR"},
            {"source": f"{prefix}_payload", "target": f"{prefix}_base", "relation": "ABOVE"},
        ]
    elif task_name == "T2_ContainerPacking":
        entities = [
            {
                "id": f"{prefix}_container",
                "type": "physical_entity",
                "properties": {"open": True, "has_cavity": True, "rigidity": "rigid"},
            },
            {
                "id": f"{prefix}_item",
                "type": "physical_entity",
                "properties": {"fits_inside": True, "stable": True},
            },
            {
                "id": f"{prefix}_interior",
                "type": "physical_entity",
                "properties": {"is_space": True},
            },
        ]
        relations = [
            {"source": f"{prefix}_container", "target": f"{prefix}_item", "relation": "SUPPORTS"},
            {"source": f"{prefix}_container", "target": f"{prefix}_item", "relation": "CONTAINS"},
            {
                "source": f"{prefix}_container",
                "target": f"{prefix}_interior",
                "relation": "HAS_CAVITY",
            },
            {
                "source": f"{prefix}_item",
                "target": f"{prefix}_container",
                "relation": "FITS_INSIDE",
            },
        ]
    elif task_name == "T3_BalanceBeam":
        entities = [
            {
                "id": f"{prefix}_fulcrum",
                "type": "physical_entity",
                "properties": {"is_pivot": True, "stable": True},
            },
            {
                "id": f"{prefix}_beam",
                "type": "physical_entity",
                "properties": {"is_level": True, "rigidity": "rigid"},
            },
            {
                "id": f"{prefix}_weight_a",
                "type": "physical_entity",
                "properties": {"mass": 1.0, "stable": True},
            },
            {
                "id": f"{prefix}_weight_b",
                "type": "physical_entity",
                "properties": {"mass": 1.0, "stable": True},
            },
            {
                "id": f"{prefix}_pos_a",
                "type": "physical_entity",
                "properties": {"is_anchor": True},
            },
            {
                "id": f"{prefix}_pos_b",
                "type": "physical_entity",
                "properties": {"is_anchor": True},
            },
        ]
        relations = [
            {"source": f"{prefix}_fulcrum", "target": f"{prefix}_beam", "relation": "SUPPORTS"},
            {
                "source": f"{prefix}_beam",
                "target": f"{prefix}_fulcrum",
                "relation": "PIVOTS_AROUND",
            },
            {"source": f"{prefix}_beam", "target": f"{prefix}_weight_a", "relation": "SUPPORTS"},
            {"source": f"{prefix}_beam", "target": f"{prefix}_weight_b", "relation": "SUPPORTS"},
            {"source": f"{prefix}_weight_a", "target": f"{prefix}_pos_a", "relation": "LOCATED_AT"},
            {"source": f"{prefix}_weight_b", "target": f"{prefix}_pos_b", "relation": "LOCATED_AT"},
        ]
    elif task_name == "T4_ObstacleNav":
        entities = [
            {
                "id": f"{prefix}_navigator",
                "type": "physical_entity",
                "properties": {"is_mobile": True},
            },
            {
                "id": f"{prefix}_path",
                "type": "physical_entity",
                "properties": {"is_passable": True},
            },
            {
                "id": f"{prefix}_start",
                "type": "physical_entity",
                "properties": {"is_origin": True},
            },
            {
                "id": f"{prefix}_goal",
                "type": "physical_entity",
                "properties": {"is_destination": True},
            },
            {
                "id": f"{prefix}_obstacle",
                "type": "physical_entity",
                "properties": {"blocks_path": True},
            },
        ]
        relations = [
            {
                "source": f"{prefix}_navigator",
                "target": f"{prefix}_path",
                "relation": "TRAVELS_ALONG",
            },
            {"source": f"{prefix}_path", "target": f"{prefix}_start", "relation": "CONNECTS"},
            {"source": f"{prefix}_path", "target": f"{prefix}_goal", "relation": "CONNECTS"},
            {"source": f"{prefix}_path", "target": f"{prefix}_obstacle", "relation": "BLOCKED_BY"},
            {"source": f"{prefix}_navigator", "target": f"{prefix}_obstacle", "relation": "AVOIDS"},
        ]
    elif task_name == "T5_ToolAffordance":
        entities = [
            {
                "id": f"{prefix}_agent",
                "type": "physical_entity",
                "properties": {"is_actor": True},
            },
            {
                "id": f"{prefix}_tool",
                "type": "physical_entity",
                "properties": {"is_graspable": True, "affords_reach": True},
            },
            {
                "id": f"{prefix}_action",
                "type": "physical_entity",
                "properties": {"is_executable": True},
            },
            {
                "id": f"{prefix}_target",
                "type": "physical_entity",
                "properties": {"is_reachable_with_tool": True},
            },
        ]
        relations = [
            {"source": f"{prefix}_agent", "target": f"{prefix}_tool", "relation": "HOLDS"},
            {"source": f"{prefix}_tool", "target": f"{prefix}_action", "relation": "AFFORDS"},
            {
                "source": f"{prefix}_tool",
                "target": f"{prefix}_target",
                "relation": "TRANSMITS_FORCE_TO",
            },
            {
                "source": f"{prefix}_action",
                "target": f"{prefix}_target",
                "relation": "CHANGES_STATE_OF",
            },
        ]
    else:
        entities = [{"id": f"{prefix}_obj", "type": task_name, "properties": {"stable": True}}]
        relations = []

    return EnvironmentObservation(
        step_index=step_index,
        visible_entities=entities,
        spatial_relations=relations,
        goal_description=task_name,
        available_actions=[{"name": "EXECUTE_SKILL", "parameters": {"task": task_name}}],
        interaction_history=[],
        resource_budget={},
    )


class E7_LifelongCurriculumTask:  # noqa: N801
    """E7: Lifelong Continual Curriculum. Evaluates 5-stage sequential curriculum and full 5x5 R_{i,j} matrix."""

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        cohort.reset()
        rng = random.Random(seed)

        tasks = [
            "T1_SpatialStacking",
            "T2_ContainerPacking",
            "T3_BalanceBeam",
            "T4_ObstacleNav",
            "T5_ToolAffordance",
        ]
        n_tasks = len(tasks)
        r_matrix = [[0.0] * n_tasks for _ in range(n_tasks)]

        for stage_i, t_train in enumerate(tasks):
            exemplars = [0, 1, 2]
            rng.shuffle(exemplars)
            for ep in exemplars:
                obs = _build_curriculum_observation(
                    t_train, prefix=f"train_{stage_i}_{ep}", step_index=ep
                )
                _ = cohort.process_observation(obs)
                cohort.learn_from_feedback(obs, 1.0)

            eval_upper_bound = min(stage_i + 2, n_tasks)
            for stage_j in range(eval_upper_bound):
                t_eval = tasks[stage_j]
                eval_obs = _build_curriculum_observation(
                    t_eval, prefix=f"eval_{stage_j}", step_index=0
                )
                eval_out = cohort.process_observation(eval_obs)
                score = (
                    eval_out.confidence
                    if eval_out.action.get("name") == "EXECUTE_SKILL"
                    else max(0.30, eval_out.confidence * 0.5)
                )
                r_matrix[stage_i][stage_j] = round(score, 4)

        bwt, fwt = ExperimentMetricsCalculator.calculate_continual_learning_bwt_fwt(r_matrix)

        return TaskEvaluationResult(
            task_id="E7_LifelongCurriculum",
            cohort_id=cohort.cohort_id,
            continual_matrix_r=r_matrix,
            bwt=bwt,
            fwt=fwt,
            accuracy=r_matrix[-1][-1],
            resource_consumption=cohort.get_resource_usage(),
        )


class E8_CausalInterventionTask:  # noqa: N801
    """E8: Causal Network Discovery & Counterfactual Interventions (Beyond Blocks-World).

    Evaluates an agent's ability to distinguish true interventional causality from
    spurious observational correlations across complex multi-variable gene regulatory networks.
    """

    def evaluate(self, cohort: BaseCohort, seed: int | None = None) -> TaskEvaluationResult:
        from hbllm.experiment.domains.causal_network import CausalNetworkEnvironment

        cohort.reset()
        env = CausalNetworkEnvironment(seed=seed)
        scenarios = env.generate_gene_network_scenarios(n_scenarios=5)

        correct_causal_inferences = 0
        total_trials = len(scenarios)
        sim_errors: list[float] = []

        for idx, sc in enumerate(scenarios):
            entities = [
                {
                    "id": sc.intervention_var,
                    "type": "causal_node",
                    "properties": {
                        "is_confounded": not sc.is_true_causal_cause,
                        "observational_correlation": sc.observational_correlation,
                    },
                },
                {
                    "id": sc.target_gene,
                    "type": "target_phenotype",
                    "properties": {"target": True},
                },
            ]
            relations = [
                {
                    "source": sc.intervention_var,
                    "target": sc.target_gene,
                    "relation": "CAUSES" if sc.is_true_causal_cause else "CORRELATED_WITH",
                }
            ]
            actions = [
                {
                    "name": "PREDICT_CAUSAL_EFFECT",
                    "parameters": {"variable": sc.intervention_var, "target": sc.target_gene},
                },
                {
                    "name": "REJECT_SPURIOUS_CORRELATION",
                    "parameters": {"variable": sc.intervention_var, "target": sc.target_gene},
                },
            ]
            obs = EnvironmentObservation(
                step_index=idx,
                visible_entities=entities,
                spatial_relations=relations,
                goal_description=f"causal_intervention:{sc.scenario_id}",
                available_actions=actions,
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)
            chosen_action = out.action.get("name", "")

            # If true causal cause, correct action is PREDICT_CAUSAL_EFFECT
            # If confounded/spurious, correct action is REJECT_SPURIOUS_CORRELATION
            is_correct = (chosen_action == "PREDICT_CAUSAL_EFFECT" and sc.is_true_causal_cause) or (
                chosen_action == "REJECT_SPURIOUS_CORRELATION" and not sc.is_true_causal_cause
            )
            if is_correct:
                correct_causal_inferences += 1

            # Simulation error against true interventional effect
            pred_effect = (
                out.confidence
                if chosen_action == "PREDICT_CAUSAL_EFFECT"
                else (1.0 - out.confidence)
            )
            sim_error = abs(pred_effect - sc.true_interventional_effect)
            sim_errors.append(sim_error)

            cohort.learn_from_feedback(obs, 1.0 if is_correct else -1.0)

        causal_acc = correct_causal_inferences / max(1, total_trials)
        mean_sim_error = sum(sim_errors) / max(1, len(sim_errors))

        return TaskEvaluationResult(
            task_id="E8_CausalIntervention",
            cohort_id=cohort.cohort_id,
            accuracy=round(causal_acc, 4),
            simulation_error=round(mean_sim_error, 4),
            structural_accuracy=round(causal_acc, 4),
            resource_consumption=cohort.get_resource_usage(),
        )
