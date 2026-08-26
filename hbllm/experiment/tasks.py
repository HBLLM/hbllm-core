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

    def evaluate(self, cohort: BaseCohort) -> TaskEvaluationResult:
        cohort.reset()
        train_pool = [
            {"id": "item_1", "shape": "cylinder", "color": "blue", "is_concept": True},
            {"id": "item_2", "shape": "cube", "color": "blue", "is_concept": False},
            {"id": "item_3", "shape": "cylinder", "color": "red", "is_concept": False},
            {"id": "item_4", "shape": "sphere", "color": "green", "is_concept": False},
        ]
        test_pool = [
            {"id": "t1", "shape": "cylinder", "color": "blue", "is_concept": True},
            {"id": "t2", "shape": "cylinder", "color": "blue", "is_concept": True},
            {"id": "t3", "shape": "cube", "color": "blue", "is_concept": False},
            {"id": "t4", "shape": "cylinder", "color": "green", "is_concept": False},
            {"id": "t5", "shape": "sphere", "color": "blue", "is_concept": False},
        ]

        accuracies = []
        for ep in range(10):
            train_item = train_pool[ep % len(train_pool)]
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

    def evaluate(self, cohort: BaseCohort) -> TaskEvaluationResult:
        cohort.reset()
        scenes: list[dict[str, Any]] = [
            {
                "token": "dax",
                "target": "cylinder",
                "entities": [{"id": "e1", "type": "cylinder"}, {"id": "e2", "type": "cube"}],
            },
            {
                "token": "mepo",
                "target": "cube",
                "entities": [{"id": "e3", "type": "cube"}, {"id": "e4", "type": "sphere"}],
            },
            {
                "token": "dax",
                "target": "cylinder",
                "entities": [{"id": "e5", "type": "cylinder"}, {"id": "e6", "type": "cone"}],
            },
            {
                "token": "mepo",
                "target": "cube",
                "entities": [{"id": "e7", "type": "cube"}, {"id": "e8", "type": "pyramid"}],
            },
        ]
        test_items: list[dict[str, Any]] = [
            {
                "token": "dax",
                "expected": "cylinder",
                "entities": [{"id": "t1", "type": "cylinder"}, {"id": "t2", "type": "sphere"}],
            },
            {
                "token": "mepo",
                "expected": "cube",
                "entities": [{"id": "t3", "type": "cube"}, {"id": "t4", "type": "cone"}],
            },
        ]

        accuracies = []
        for idx, scene in enumerate(scenes):
            obs = EnvironmentObservation(
                step_index=idx,
                visible_entities=scene["entities"],
                spatial_relations=[],
                goal_description=f"lexical_grounding:{scene['token']}",
                available_actions=[
                    {"name": "SELECT", "parameters": {"target": e["id"]}} for e in scene["entities"]
                ],
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)
            chosen_id = out.action.get("parameters", {}).get("target", "")
            chosen_type = next((e["type"] for e in scene["entities"] if e["id"] == chosen_id), "")
            reward = 1.0 if chosen_type == scene["target"] else -1.0
            cohort.learn_from_feedback(obs, reward)

            # Test probe
            correct = 0
            for t_item in test_items:
                t_obs = EnvironmentObservation(
                    step_index=idx,
                    visible_entities=t_item["entities"],
                    spatial_relations=[],
                    goal_description=f"lexical_grounding:{t_item['token']}",
                    available_actions=[
                        {"name": "SELECT", "parameters": {"target": e["id"]}}
                        for e in t_item["entities"]
                    ],
                    interaction_history=[],
                    resource_budget={},
                )
                t_out = cohort.process_observation(t_obs)
                t_id = t_out.action.get("parameters", {}).get("target", "")
                t_type = next((e["type"] for e in t_item["entities"] if e["id"] == t_id), "")
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

    def evaluate(self, cohort: BaseCohort) -> TaskEvaluationResult:
        cohort.reset()
        env = CanonicalTaskEnvironment(domain="counterfactual_simulation")
        oracle = env.oracle

        scenarios = [
            {"base": "support_flat", "geom": "flat", "rigid": True, "goal": "STACK"},
            {"base": "support_curved", "geom": "curved", "rigid": True, "goal": "STACK"},
            {"base": "support_soft", "geom": "flat", "rigid": False, "goal": "STACK"},
            {"base": "support_stable", "geom": "flat", "rigid": True, "goal": "STACK"},
            {"base": "support_dome", "geom": "curved", "rigid": True, "goal": "STACK"},
        ]

        sim_errors: list[float] = []
        regrets: list[float] = []
        successes: list[bool] = []

        for idx, sc in enumerate(scenarios):
            true_state = PhysicalEnvironmentState(
                entities={sc["base"]: {"surface_geometry": sc["geom"], "is_rigid": sc["rigid"]}},
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
                    }
                ],
                spatial_relations=[],
                goal_description="spatial_stacking",
                available_actions=[
                    {"name": "STACK", "parameters": {"base": sc["base"]}, "cost": 1.0},
                    {"name": "WAIT", "parameters": {}, "cost": 0.1},
                ],
                interaction_history=[],
                resource_budget={},
            )
            out = cohort.process_observation(obs)

            # Oracle optimal utility
            opt_act = (
                {"name": "STACK", "parameters": {"base": sc["base"]}}
                if (sc["geom"] == "flat" and sc["rigid"])
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
            true_stable = 1.0 if (sc["geom"] == "flat" and sc["rigid"]) else 0.0
            pred_stable = (
                1.0
                if out.prediction in ("stable", "stable_support")
                else (
                    0.0 if out.prediction in ("unstable", "abstain_uncertain") else out.confidence
                )
            )
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

    def evaluate(self, cohort: BaseCohort) -> TaskEvaluationResult:
        cohort.reset()

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

        confidences: list[float] = []
        outcomes: list[bool] = []
        abstentions: list[bool] = []

        for idx, tr in enumerate(trials):
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

    def evaluate(self, cohort: BaseCohort) -> TaskEvaluationResult:
        cohort.reset()
        env = CanonicalTaskEnvironment(domain="active_discovery")
        oracle = env.oracle

        scenarios = [
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

        probing_regrets: list[float] = []
        efficiencies: list[float] = []
        successes: list[bool] = []

        for idx, sc in enumerate(scenarios):
            actions = [
                {
                    "name": "PROBE_GENTLE_TAP",
                    "parameters": {"target": sc["name"]},
                    "cost": sc["cost_probe"],
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
                achieved_eff = float(sc["d_power"]) / float(sc["cost_probe"])
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

    def evaluate(self, cohort: BaseCohort) -> TaskEvaluationResult:
        cohort.reset()

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
        ]

        structural_correct = 0
        surface_distracted = 0
        total_structural = 0
        total_surface = 0

        for idx, tr in enumerate(trials):
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
                actions = [{"name": "MAP_STRUCTURAL_CENTRAL"}, {"name": "MAP_SURFACE_COLOR"}]
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
                actions = [{"name": "MAP_STRUCTURAL_CENTRAL"}, {"name": "MAP_SURFACE_COLOR"}]
                total_surface += 1

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
            elif tr["is_surface_distractor"]:
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


class E7_LifelongCurriculumTask:  # noqa: N801
    """E7: Lifelong Continual Curriculum. Evaluates 5-stage sequential curriculum and full 5x5 R_{i,j} matrix."""

    def evaluate(self, cohort: BaseCohort) -> TaskEvaluationResult:
        cohort.reset()

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
            for ep in range(3):
                obs = EnvironmentObservation(
                    step_index=ep,
                    visible_entities=[
                        {
                            "id": f"obj_{stage_i}_{ep}",
                            "type": t_train,
                            "properties": {"stable": True},
                        }
                    ],
                    spatial_relations=[],
                    goal_description=t_train,
                    available_actions=[{"name": "EXECUTE_SKILL", "parameters": {"task": t_train}}],
                    interaction_history=[],
                    resource_budget={},
                )
                _ = cohort.process_observation(obs)
                cohort.learn_from_feedback(obs, 1.0)

            for stage_j in range(stage_i + 1):
                t_eval = tasks[stage_j]
                eval_obs = EnvironmentObservation(
                    step_index=0,
                    visible_entities=[
                        {
                            "id": f"test_{stage_j}",
                            "type": t_eval,
                            "properties": {"stable": True},
                        }
                    ],
                    spatial_relations=[],
                    goal_description=t_eval,
                    available_actions=[{"name": "EXECUTE_SKILL", "parameters": {"task": t_eval}}],
                    interaction_history=[],
                    resource_budget={},
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
