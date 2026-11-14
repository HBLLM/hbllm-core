# HBLLM — Gap Analysis: From "Coherent Cognitive Architecture" to General Intelligence

**Date:** 2026-11-08  
**Companion to:** `reports/audit/2026-11-08_project_audit.md` (WP1–WP14)  
**Purpose:** answer, as honestly as the evidence allows, "after the audit's work packages are done, what is still missing for this system to be generally intelligent?"

> **Epistemic status, stated up front.** Nobody has a validated recipe for general intelligence. This document therefore does *not* claim "do these N things and you have AGI". It does three things that *can* be done rigorously: (1) list the functional properties that every serious account of general intelligence — from Newell's unified theories, through Friston's active inference, Chollet's *On the Measure of Intelligence*, Lake et al.'s *Building Machines That Learn and Think Like People*, to the modern scaling literature — agrees a general agent must have; (2) check each against what the codebase actually does today, with file:line evidence; (3) separate **engineering gaps** (we know how to build it; it's missing) from **open research problems** (nobody knows how; the project would be doing research). Anything in the second category is a risk, not a task.

---

## 0. Scorecard

| # | Property of a general agent | Status in HBLLM today | Gap type |
|---|---|---|---|
| 1 | Learns its own representations from raw sensation | ❌ All features are hand-designed (connected components, Hu moments, bbox topology). The SNN stack exists but is not in the perception path. | **Research** (in this architecture); Engineering (if a trainable encoder is admitted) |
| 2 | Open-ended hypothesis space with learned search | ⚠️ Fixed DSL: 15 `RuleFamily` members, 12 `SpatialActionIntent`s, 5 seeded relations. Search is enumeration + heuristics; no learned proposal distribution. | **Research** |
| 3 | Lifelong memory with consolidation | ❌ Total amnesia between runs (audit WP8). | Engineering |
| 4 | Autonomous, self-directed operation | ❌ Purely `step(obs)`-driven (WP9). | Engineering |
| 5 | Intrinsic motivation & value learning | ⚠️ Curiosity-as-surprise exists; no homeostatic/value arbitration in the GI loop (WP10); value is never *learned* from outcomes. | Engineering (arbitration) + Research (value learning) |
| 6 | Compositional, abstract concepts that transfer | ⚠️ Concepts = feature clusters; relations = 5 seeded + synchronous-motion; no concept *composition* (part-whole, role-filler, analogy) and no cross-context reuse (WP13). | **Research** |
| 7 | Causal models that generalise interventions | ⚠️ `CausalPredicate` / `do(a)` exists per entity-feature; no latent-variable causal discovery, no counterfactual queries over *unobserved* variables. | Research-leaning |
| 8 | Language grounded in and generative of world models | ❌ Two halves never meet (WP11). After WP11: alias binding only — no compositional semantics, no learning *from* language. | Engineering (binding) + **Research** (semantics) |
| 9 | Metacognition that changes behaviour | ⚠️ `metacognitive_agency_cortex`, `precision_controller`, `prediction_budget` modulate search depth; no model of *own competence per domain* that drives what to learn next. | Engineering |
| 10 | Social cognition / theory of mind | ❌ `brain/social` is user-persona modelling for chat; `hcir/world` has no other-agent intention model beyond patroller trajectories. | Research-leaning |
| 11 | Temporal abstraction at multiple scales | ⚠️ Macro-actions + subgoal stack exist; no learned options, no hierarchy discovery. | Research |
| 12 | Scalable substrate | ⚠️ 65k LOC Python per step; Rust covers only the graph (WP14). | Engineering |
| 13 | Sample efficiency comparable to humans | ❓ Unmeasured. Only metric is 10/10 on 10 hand-tuned games. | Needs measurement first |
| 14 | Robustness / OOD generalisation | ❌ No held-out evaluation exists (`benchmarks/` has ablation only; `rg -i "held.?out|ood"` → 0). | Engineering (harness) |
| 15 | Safety, corrigibility, interpretability | ✅/⚠️ Best part of the project: transactional state, constitutional verifier, event-sourced replay — *once WP1 makes them actually run*. Missing: value-uncertainty-aware deferral, interpretable goal provenance to a human. | Engineering |
| 16 | A definition of success | ❌ No stated operational definition of "general" for this project. | Decision |

Legend: ✅ present · ⚠️ partial/nominal · ❌ absent · ❓ unknown.

---

## 1. Representation learning — the central gap

**What the code does.** Perception is `cortex_perception.py` (connected components over an integer lattice → `SpatialEntity`), shape identity via Hu moments (`ShapeArchetypeSignature`, AEE:2389), topology via `TopologyClassifier.classify_2d_boxes`, grounding features = `(row, col, color/10, log1p(size))` (`grounding/features/feature_extractor.py:135-140`). Every feature the system can ever reason about was chosen by a programmer.

**Why it matters.** Generality is, operationally, *the ability to succeed on tasks whose relevant features the designer did not anticipate*. A hand-designed feature set makes the system exactly as general as its designer's foresight, and the 90-faculty sprawl in `hcir/world` is the symptom: each new environment mechanic required a new module (`gravity`, `kronecker`, `tiling`, `deformation`, …). This is the same wall ACT-R, SOAR, Icarus and Cyc hit.

**What exists but isn't used.** `hbllm/brain/snn/` (LIF neurons, STDP `plasticity.py`, WTA, dendrites, `perception/`) is a genuine learning substrate and is already LLM-free and deterministic-at-inference. It is not on the perception path; `hcir/world/snn_predictor.py` uses it only for temporal track prediction (AEE:773, :4285).

**Gap type.** Within the current Invariant 3 reading ("no learned weights in core"), this is a research problem with no known solution — nobody has built a general perceiver from hand-coded symbolic features. If Invariant 3 is re-scoped to "no *external LLM calls*, no *online stochastic training* in the decision path; offline-trained deterministic encoders permitted", it becomes an engineering project:

1. Define `FeatureExtractor` implementations backed by (a) the SNN perception stack and (b) a small offline-trained convolutional/contrastive encoder (trained on *unlabelled* environment frames, not task labels).
2. Run both against the hand-designed extractor on *held-out* environments (§14) with identical downstream faculties.
3. Let the data decide. If learned features never win, that is a publishable negative result; if they do, the ablation methodology from WP2 already tells you which hand-designed faculties to retire.

**Honest risk.** Even with learned perception, bridging learned features to the symbolic `CausalPredicate` layer (the "neuro-symbolic interface") is itself unsolved at scale. Expect this to be the research core of the project for years, not a work package.

---

## 2. Hypothesis space & search

**What the code does.** `rule_induction.py` enumerates a closed set of 15 `RuleFamily` transformation types with parameters; `skill_program_synthesizer.py` composes `SpatialActionIntent`s (12 primitives in `macro_actions.py`); `RelationLearner` matches 5 seeded relations via an if-ladder; `SchemaInducer` induces `action → effect` schemas over observed features. Search over these spaces is exhaustive enumeration with pruning (`dead-end pruning`, `ACC EVC` depth regulation).

**What's missing.**
- **Open-endedness.** A general learner's hypothesis language must be able to express hypotheses the designer never enumerated. Today the DSL is fixed at coding time. The declared target (`DeclarativeNeuroSymbolicSkill` + `SymbolicPredicate`, AGENTS.md Pattern B) is the right grammar, but there is no mechanism that *grows* the predicate vocabulary from experience (`representation_expansion.py` expands representations, not the DSL).
- **Learned search.** Enumeration scales exponentially with DSL size; humans (and every strong program-synthesis system since DreamCoder) use a *learned* proposal distribution — "which hypotheses are worth checking given this situation". Nothing in `hcir/` learns over hypotheses; `BayesianGoalInductionEngine` updates beliefs over *goals*, not over rule structure.
- **Library learning.** Successful sub-programs should be abstracted into new primitives (DreamCoder's wake/sleep). `skill_program_synthesizer.py` output is not persisted (WP8) and never becomes a primitive for later synthesis.

**Gap type.** Research. Concrete first step that is engineering: implement library learning on top of WP8 — persist synthesized `SubgoalSequence`s, measure reuse rate across contexts, and let frequently-reused sequences become `SpatialActionIntent` macros. This gives you a growth mechanism to study even before a learned proposer exists.

---

## 3. Lifelong memory & consolidation
Covered in audit WP8/WP9. Additional non-plumbing gaps:
- **Forgetting.** There is no principled forgetting (decay, interference, importance-weighted pruning). Without it, WP8 persistence leads to unbounded, increasingly noisy memory. `TieredWorkspace` has promotion rules (`workspace_tiers.py:378-380`) but no demotion.
- **Schema consolidation vs episodic detail.** `cortex_episodic.py` stores episodes; `mpfc_schema_engine` (ablated) was meant to abstract them. Complementary-learning-systems theory says the *interaction* (replay → gradual schema update) is where generalisation comes from. That interaction is not implemented; `hippocampal_replay.py` replays for planning, not for consolidation.
- **Gap type:** Engineering, given WP8/WP9.

---

## 4–5. Autonomy, motivation and value *learning*
WP9/WP10 cover autonomy and goal arbitration. The deeper gap:
- **Value is not learned.** Rewards come from `TerminalSignal`; curiosity is prediction error. There is no mechanism by which *outcomes change what the agent values* (no learned reward model, no preference learning, no habit formation beyond `habenular_ior` negative caching). A general agent must discover that keys are valuable because doors are, recursively, without being told — `delayed_effects` / `DelayedEffectModeler` is a start but is credit assignment, not valuation.
- **Competence progress** (Oudeyer/Schmidhuber "learning progress") is the intrinsic signal most associated with open-ended development; the project measures surprise, not *reduction in surprise over time*. `precision_controller` is the natural home.
- **Gap type:** Engineering for learning-progress signal; Research for value learning in general.

---

## 6. Compositional, transferable concepts
**What the code does.** `ConceptLearner` clusters feature vectors (streaming k-means-like with split/merge). `GroundedConcept` has `prototypes`, `exemplar_entity_ids`, `predictive_models`, `linguistic_aliases`.

**What's missing.**
- **Composition.** No part-whole (`a key *has* a bow and a blade`), no role-filler binding (`the thing that opens doors`), no analogy (`relational_graph_matcher.py` matches graphs structurally but is not connected to the concept store). Clusters cannot be combined into new concepts.
- **Abstraction over relations, not just objects.** `LearnedRelation` is induced from kinematic invariants but relations between relations (e.g. *blocking*, *enabling*) are not representable.
- **Transfer.** Covered in WP13; `CrossGameKnowledgeTransfer` ablated.
- **Gap type:** Research (this is the Lake et al. "compositionality + causality + learning-to-learn" triad, and no system has all three). Engineering first step: connect `relational_graph_matcher` to `WorkingConceptStore` so analogical retrieval over concept graphs is at least possible, then measure.

---

## 7. Causal reasoning
`HCIRSymbolicWorldTheory` + `CausalPredicate` + `do(a)` schemas are a sound Pearl-level-2 core over *observed* features. Missing for generality:
- **Latent variables.** Hidden state (a switch is "armed", an enemy "has seen you") must be *posited*, not observed. `causal_discovery.py` works on observed feature co-occurrence only.
- **Counterfactuals over unobserved alternatives** (Pearl level 3) are computed only for *action* alternatives (`counterfactual_simulation.py`, AEE:2238 regret), not over hypothetical world structure.
- **Gap type:** Research-leaning; latent causal discovery is active research. Engineering piece: allow `SchemaInducer` to propose a latent boolean when the same `(precondition, action)` yields inconsistent effects — the "there must be something I can't see" move.

---

## 8. Language
WP11 gives alias binding. General intelligence needs more:
- **Compositional semantics grounded in the world model** — "the block *left of* the key" must compose a relation and two groundings; today `HCIRCompiler` produces `SemanticAST` slots but nothing evaluates them against `HCIRSymbolicWorldTheory`.
- **Learning from language** — being *told* a rule ("red doors need red keys") should create a `CausalPredicate` with testable status, not just a goal. `CausalStatus` enum in `grounding/contracts/enums.py` already has the right states (HYPOTHESISED → TESTED); the path from utterance to hypothesis doesn't exist.
- **Generating explanations** of its own decisions in language (interpretability, §15) — `brain/epistemics/explanation.py` exists for the LLM stack; nothing explains an `AutonomousEpistemicEngine` decision.
- **Gap type:** Engineering for the two bridges above (the LLM is allowed here as meta-compiler per AGENTS.md §3); Research for robust open-domain semantics.

---

## 9. Metacognition that changes behaviour
Exists as parameter modulation (`precision_controller`, `prediction_budget`, `metacognitive_agency_cortex`, `confidence_calibrator` ablated). Missing: a **per-domain competence model** (`brain/self_model/` does this for the LLM stack with SQLite, disconnected from `hcir`) that answers "what am I bad at, and is practising it worth it?" — the signal that should drive WP10's homeostatic *competence* variable and WP9's offline consolidation priorities. **Gap type:** Engineering.

---

## 10. Social cognition
Absent in the GI loop. Patrollers are modelled as trajectories (`dynamic patroller trajectories`, `collision_cones`), not as agents with goals. Human-level generality includes modelling other minds (cooperation, deception, teaching). `brain/social/` (persona, relationship memory, user model) is chat-oriented and SQLite-backed. **Gap type:** Research-leaning; a tractable first step is treating any entity whose motion is *not* predictable by physics as a goal-directed agent and inverting a planner over it (inverse planning, Baker/Tenenbaum) — the counterfactual planner already exists to be inverted.

---

## 11. Temporal abstraction
Macro-actions and `frontopolar_subgoal_stack` exist but the hierarchy is designer-specified. Option discovery (bottleneck states, empowerment, successor features) is absent. **Gap type:** Research; engineering first step is to measure whether `skill_program_synthesizer` outputs ever get *reused* — if not, hierarchy isn't forming.

---

## 12–14. Substrate, efficiency, evaluation
- Substrate: audit WP14.
- **Sample efficiency is unmeasured.** Record steps-to-first-success and steps-to-stable-policy per environment; compare to published human baselines where they exist (ARC-AGI-3 publishes them). Without this number, "brain-like" is unfalsifiable.
- **No OOD evaluation exists.** `benchmarks/` contains only `faculty_ablation.py`, `sge_integration_benchmark.py`, 4 synthetic scenarios. The 10-game local suite has been tuned against repeatedly (git log: `resolve push_block, death_reset, stride2, and click_toggle bottlenecks (10/10 local suite)`). **This is the single most important missing thing in the whole project**, because without it no other claim can be checked. Required: a frozen held-out set never used for development; a development set; a report that prints both; a rule that the held-out set is run at most once per release.

---

## 15. Safety, corrigibility, interpretability
This is where the project is *ahead* of most AGI efforts on paper: every mutation is a transaction, every commit passes a verifier, everything is replayable. The audit's WP1 is what makes it true in practice. Remaining gaps for a general agent: (a) **uncertainty-aware deferral** — `DecisionOutcome.REQUEST_EVIDENCE` exists for beliefs; the analogue for *actions under value uncertainty* (ask the human) is only the word-list in `gateway.py:82`; (b) **goal provenance** — a human should be able to ask "why are you doing this?" and get the `GoalProposal` chain; (c) **shutdown/interrupt semantics** — `TaskFrame` interruption exists in `TieredWorkspace`; whether an interrupted agent *resists* (instrumental convergence) is untested and should be a standing test once WP10 gives it intrinsic goals.

---

## 16. Define success before building more
The project has no operational definition of "general". Recommended: adopt Chollet's framing — **skill-acquisition efficiency over a distribution of tasks with controlled priors** — and instantiate it:

- **Priors declared:** list the innate knowledge the core is allowed (objectness, basic topology, agency detection, counting, elementary physics — the "core knowledge" set). Everything else must be learned in-environment.
- **Task distribution:** ≥3 environment families with different physics/mechanics (grid puzzles, continuous manipulation, text/dialogue tasks), each with dev and frozen held-out splits.
- **Metric:** steps (or bits of experience) to reach threshold performance, averaged over held-out tasks, with the same frozen agent code and *no* per-family modules.
- **Generality score:** performance on a family the agent has never seen, with only its priors and cross-family memory (WP8/WP13).

This turns "is it AGI-like?" from an argument into a number that can go up.

---

## 17. What this adds up to

| Horizon | Work | Outcome you can honestly claim |
|---|---|---|
| Now → audit WP1–7 | Plumbing | "One coherent, governed, testable cognitive architecture" |
| + WP8–14 | Memory, autonomy, drives, language binding, generic perception, transfer, speed | "A lifelong-learning embodied agent with grounded language" — rare and scientifically valuable |
| + §14/§16 | Held-out evaluation, declared priors, generality metric | "Measurable claims about generality" — the first point at which 'AGI progress' is a meaningful phrase |
| + §1, §2, §6 (research) | Learned representations, open-ended hypothesis growth, compositional concepts | Unknown. These are the open problems of the field. Success here would be a research contribution, not a milestone on a roadmap. |

**The uncomfortable summary:** everything that can be scheduled (WP1–WP14, the evaluation harness, the priors declaration) gets you a very good *research platform*. The three things that would actually make it general — learning its own representations, growing its own hypothesis language, composing concepts that transfer — are problems no one has solved in any architecture, symbolic or neural, and the current Invariant 3 as written rules out the only partial solutions the field has found. The most valuable decision available right now is not another faculty; it is (a) building the held-out evaluation so that every future claim is checkable, and (b) deciding, deliberately, how much learned computation the "LLM-free core" is allowed to contain.
