"""Developmental Curriculum Definition (D0 through D16).

Provides the staged Piagetian curriculum specifications for Milestone A23.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CurriculumStageId(str, Enum):
    """The 17 sequential developmental curriculum stages."""

    D0_SENSORIMOTOR = "D0_Sensorimotor"
    D1_OBJECT_PERMANENCE = "D1_ObjectPermanence"
    D2_SPATIAL_RELATIONS = "D2_SpatialRelations"
    D3_CAUSAL_DISCOVERY = "D3_CausalDiscovery"  # A23.5 Target
    D4_AFFORDANCE_DISCOVERY = "D4_AffordanceDiscovery"
    D5_TOOL_USE = "D5_ToolUse"
    D6_GOAL_DIRECTED = "D6_GoalDirectedBehavior"
    D7_MULTI_STEP_PLANNING = "D7_MultiStepPlanning"
    D8_CURIOSITY = "D8_Curiosity"
    D9_CONCEPT_ABSTRACTION = "D9_ConceptAbstraction"
    D10_LANGUAGE_GROUNDING = "D10_LanguageGrounding"
    D11_COMPOSITIONAL_LANGUAGE = "D11_CompositionalLanguage"
    D12_CONTINUAL_DEVELOPMENT = "D12_ContinualDevelopment"
    D13_METACOGNITION = "D13_Metacognition"
    D14_CROSS_WORLD_TRANSFER = "D14_CrossWorldTransfer"
    D15_CROSS_DOMAIN_TRANSFER = "D15_CrossDomainTransfer"
    D16_EMBODIED_ROBOT_TRANSFER = "D16_EmbodiedRobotTransfer"


@dataclass
class CurriculumStageSpec:
    """Specification of a developmental stage."""

    stage_id: CurriculumStageId
    title: str
    description: str
    target_wave: str
    prerequisites: list[CurriculumStageId]
    eval_metrics: list[str]
    is_active_target: bool = False


CURRICULUM_SPECS: dict[CurriculumStageId, CurriculumStageSpec] = {
    CurriculumStageId.D0_SENSORIMOTOR: CurriculumStageSpec(
        stage_id=CurriculumStageId.D0_SENSORIMOTOR,
        title="Sensorimotor Babbling",
        description="Explores motor commands (MOVE, REACH, TOUCH) to ground proprioception and physical limits.",
        target_wave="A23.3",
        prerequisites=[],
        eval_metrics=["reach_success_rate", "grasp_success_rate"],
    ),
    CurriculumStageId.D1_OBJECT_PERMANENCE: CurriculumStageSpec(
        stage_id=CurriculumStageId.D1_OBJECT_PERMANENCE,
        title="Object Permanence",
        description="Tracks occluded entities across sensory dropout without deleting persistent existence beliefs.",
        target_wave="A23.4",
        prerequisites=[CurriculumStageId.D0_SENSORIMOTOR],
        eval_metrics=["permanence_accuracy", "spatial_uncertainty_calibration"],
    ),
    CurriculumStageId.D2_SPATIAL_RELATIONS: CurriculumStageSpec(
        stage_id=CurriculumStageId.D2_SPATIAL_RELATIONS,
        title="Spatial Relations",
        description="Induces relational edges (ON, INSIDE, NEAR, BLOCKING) from continuous spatial coordinates.",
        target_wave="A23.7",
        prerequisites=[CurriculumStageId.D1_OBJECT_PERMANENCE],
        eval_metrics=["relational_precision", "relational_recall"],
    ),
    CurriculumStageId.D3_CAUSAL_DISCOVERY: CurriculumStageSpec(
        stage_id=CurriculumStageId.D3_CAUSAL_DISCOVERY,
        title="Active Interventional Causal Discovery",
        description="Resolves correlation vs causation traps through active contrastive intervention probes.",
        target_wave="A23.5",
        prerequisites=[CurriculumStageId.D0_SENSORIMOTOR],
        eval_metrics=["n_tau_median", "false_hypotheses", "l2_generalization", "l3_generalization"],
        is_active_target=True,
    ),
    CurriculumStageId.D4_AFFORDANCE_DISCOVERY: CurriculumStageSpec(
        stage_id=CurriculumStageId.D4_AFFORDANCE_DISCOVERY,
        title="Affordance Discovery",
        description="Discovers grounded functional affordances (AFFORDS(ball, ROLL), AFFORDS(box, CONTAIN)).",
        target_wave="A23.6",
        prerequisites=[CurriculumStageId.D3_CAUSAL_DISCOVERY],
        eval_metrics=["affordance_precision", "novel_entity_affordance_transfer"],
    ),
    CurriculumStageId.D5_TOOL_USE: CurriculumStageSpec(
        stage_id=CurriculumStageId.D5_TOOL_USE,
        title="Tool Use & Functional Reasoning",
        description="Discovers tools (sticks, levers) to extend reach and manipulate distant targets.",
        target_wave="A23.8",
        prerequisites=[CurriculumStageId.D4_AFFORDANCE_DISCOVERY],
        eval_metrics=["tool_selection_efficiency", "force_transmission_accuracy"],
    ),
    CurriculumStageId.D6_GOAL_DIRECTED: CurriculumStageSpec(
        stage_id=CurriculumStageId.D6_GOAL_DIRECTED,
        title="Goal-Directed Behavior",
        description="Translates desires into backward causal chains and achieves target states.",
        target_wave="A23.9",
        prerequisites=[CurriculumStageId.D3_CAUSAL_DISCOVERY],
        eval_metrics=["goal_completion_rate", "planning_steps"],
    ),
    CurriculumStageId.D7_MULTI_STEP_PLANNING: CurriculumStageSpec(
        stage_id=CurriculumStageId.D7_MULTI_STEP_PLANNING,
        title="Multi-Step Compositional Planning",
        description="Synthesizes subgoals and executes multi-step plans with replanning on failure.",
        target_wave="A23.9",
        prerequisites=[CurriculumStageId.D6_GOAL_DIRECTED],
        eval_metrics=["plan_optimality", "replanning_latency"],
    ),
    CurriculumStageId.D8_CURIOSITY: CurriculumStageSpec(
        stage_id=CurriculumStageId.D8_CURIOSITY,
        title="Autonomous Epistemic Curiosity",
        description="Selects self-generated experiments to maximize information gain without external rewards.",
        target_wave="A23.9",
        prerequisites=[CurriculumStageId.D3_CAUSAL_DISCOVERY],
        eval_metrics=["entropy_reduction_rate", "knowledge_coverage"],
    ),
    CurriculumStageId.D9_CONCEPT_ABSTRACTION: CurriculumStageSpec(
        stage_id=CurriculumStageId.D9_CONCEPT_ABSTRACTION,
        title="Concept Abstraction",
        description="Clusters perceptual properties into robust invariant concept nodes in HCIR.",
        target_wave="A23.10",
        prerequisites=[CurriculumStageId.D4_AFFORDANCE_DISCOVERY],
        eval_metrics=["concept_clustering_f1", "out_of_distribution_generalization"],
    ),
    CurriculumStageId.D10_LANGUAGE_GROUNDING: CurriculumStageSpec(
        stage_id=CurriculumStageId.D10_LANGUAGE_GROUNDING,
        title="Grounded Lexical Acquisition",
        description="Fast-maps lexical tokens ('ball', 'box', 'push') onto pre-existing cognitive structures.",
        target_wave="A23.11",
        prerequisites=[CurriculumStageId.D9_CONCEPT_ABSTRACTION],
        eval_metrics=["fast_mapping_n_tau", "lexicon_accuracy"],
    ),
    CurriculumStageId.D11_COMPOSITIONAL_LANGUAGE: CurriculumStageSpec(
        stage_id=CurriculumStageId.D11_COMPOSITIONAL_LANGUAGE,
        title="Compositional Language Understanding",
        description="Parses novel compound sentences into structured multi-step goals.",
        target_wave="A23.12",
        prerequisites=[CurriculumStageId.D10_LANGUAGE_GROUNDING],
        eval_metrics=["compositional_accuracy", "zero_shot_sentence_execution"],
    ),
    CurriculumStageId.D12_CONTINUAL_DEVELOPMENT: CurriculumStageSpec(
        stage_id=CurriculumStageId.D12_CONTINUAL_DEVELOPMENT,
        title="Continual Lifelong Development & Sleep",
        description="Consolidates episodic experience during sleep cycles with zero catastrophic forgetting.",
        target_wave="A23.13",
        prerequisites=[CurriculumStageId.D3_CAUSAL_DISCOVERY],
        eval_metrics=["bwt_backward_transfer", "fwt_forward_transfer"],
    ),
    CurriculumStageId.D13_METACOGNITION: CurriculumStageSpec(
        stage_id=CurriculumStageId.D13_METACOGNITION,
        title="Metacognitive Calibration",
        description="Accurately self-evaluates confidence vs success probability and abstains when uncertain.",
        target_wave="A23.14",
        prerequisites=[CurriculumStageId.D12_CONTINUAL_DEVELOPMENT],
        eval_metrics=["brier_score", "ece_expected_calibration_error"],
    ),
    CurriculumStageId.D14_CROSS_WORLD_TRANSFER: CurriculumStageSpec(
        stage_id=CurriculumStageId.D14_CROSS_WORLD_TRANSFER,
        title="Cross-World Relational Transfer",
        description="Transfers abstract schemas from World A to World B with novel visual appearances.",
        target_wave="A23.15",
        prerequisites=[CurriculumStageId.D3_CAUSAL_DISCOVERY],
        eval_metrics=["relational_transfer_score", "transfer_systematicity"],
    ),
    CurriculumStageId.D15_CROSS_DOMAIN_TRANSFER: CurriculumStageSpec(
        stage_id=CurriculumStageId.D15_CROSS_DOMAIN_TRANSFER,
        title="Cross-Domain Adapter Transfer",
        description="Transfers schemas from BabyWorld into BabyAI, Sokoban, Crafter, and Overcooked.",
        target_wave="A23.16",
        prerequisites=[CurriculumStageId.D14_CROSS_WORLD_TRANSFER],
        eval_metrics=["cross_domain_fwt", "adapter_schema_reuse_rate"],
    ),
    CurriculumStageId.D16_EMBODIED_ROBOT_TRANSFER: CurriculumStageSpec(
        stage_id=CurriculumStageId.D16_EMBODIED_ROBOT_TRANSFER,
        title="Embodied Physical Robot Transfer",
        description="Transfers cognitive graphs to physical hardware with real sensors and actuators.",
        target_wave="A23.17",
        prerequisites=[CurriculumStageId.D15_CROSS_DOMAIN_TRANSFER],
        eval_metrics=["real_world_grasp_rate", "sensor_noise_robustness"],
    ),
}
