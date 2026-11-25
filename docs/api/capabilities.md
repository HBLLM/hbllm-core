---
title: "API Reference — Governed Capabilities, Curiosity & Command Dispatch"
description: "API reference for CommandDispatcher, ConceptExplainer, EpistemicCuriosityController, BeliefConflictResolver, ToolCapabilityExecutor, WebCapabilityExecutor, MemoryCapabilityExecutor, and AutonomyTracker."
---

# Governed Capabilities & Command Dispatch API

**Packages:** `hbllm.hcir.capabilities`, `hbllm.hcir.kernel.command_dispatcher`, `hbllm.hcir.kernel.autonomy_tracker`

---

## `CommandDispatcher`

**Module:** `hbllm.hcir.kernel.command_dispatcher.CommandDispatcher`

Routes parsed `COMMAND` `SemanticFrame`s through `CapabilityResolver` and `CapabilitySandboxManager`:

```python
from hbllm.hcir.kernel.command_dispatcher import CommandDispatcher
from hbllm.hcir.language.core.semantic_frame import FrameType, SemanticFrame
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

system = build_cognitive_system(CognitiveSystemConfig(memory_dir=None, enable_grounding=False))
dispatcher = CommandDispatcher(
    capability_resolver=system.services.capability_resolver,
    workspace=system.workspace,
)

print("Known verbs:", dispatcher.known_verbs())
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `dispatch()` | `async (frame: SemanticFrame) -> CommandExecutionReceipt` | Resolves capability verb, validates permissions, and returns execution receipt |
| `dispatch_discourse()` | `async (frames: list[SemanticFrame]) -> DiscourseExecutionResult` | Executes a multi-command sequence and aggregates `CommandExecutionReceipt`s |
| `register_verb()` | `(verb: str, capability: str) -> None` | Binds an additional imperative verb to a registered capability name |
| `known_verbs()` | `() -> list[str]` | Lists registered imperative command verbs |

---

## `ConceptExplainer`

**Module:** `hbllm.hcir.capabilities.concept_explainer.ConceptExplainer`

```python
from hbllm.hcir.capabilities.concept_explainer import ConceptExplainer

explainer = ConceptExplainer(
    tiered_workspace=system.tiered_workspace,
    ontology_store=system.services.extensions.get("ontology_store"),
)

snap = explainer.gather_knowledge("photosynthesis")
if snap.has_knowledge():
    print(explainer.synthesize_discourse(snap))
    print(explainer.explain_how("photosynthesis"))
```

### Methods

- `gather_knowledge(concept_query: str, *, _allow_fuzzy: bool = True) -> ConceptKnowledgeSnapshot`
- `synthesize_discourse(snapshot: ConceptKnowledgeSnapshot, style: str = 'paragraph') -> str`
- `explain(concept_query: str) -> str`
- `explain_why(concept_query: str, max_depth: int = 3, snapshot: ConceptKnowledgeSnapshot | None = None) -> str`
- `explain_how(concept_query: str, max_depth: int = 3, snapshot: ConceptKnowledgeSnapshot | None = None) -> str`

---

## `EpistemicCuriosityController`

**Module:** `hbllm.hcir.capabilities.epistemic_curiosity.EpistemicCuriosityController`

```python
from hbllm.hcir.capabilities.epistemic_curiosity import EpistemicCuriosityController

curiosity = EpistemicCuriosityController(
    capability_resolver=system.services.capability_resolver,
    tiered_workspace=system.tiered_workspace,
    uncertainty_threshold=0.75,
)

if curiosity.should_research("thermodynamics", uncertainty=0.9):
    result = await curiosity.research_concept("thermodynamics", query_mode="definition")
    print(result.researched, result.propositions_ingested, result.sources)
```

---

## `BeliefConflictResolver`

**Module:** `hbllm.hcir.capabilities.belief_revision.BeliefConflictResolver`

- `evaluate_belief(new_belief: BeliefNode, graph: CognitiveGraph | None) -> BeliefRevisionReport`
- Returns `BeliefRevisionReport(conflicts=[...], operations=[...], adjusted_confidence=..., contested=...)`.

---

## Capability Executors

- **`ToolCapabilityExecutor`** (`hbllm.hcir.capabilities.tool_capabilities`): Sandboxed `fs.read`, `fs.list`, `fs.search`, `code.python_ast`, `git.status`, `shell.exec`.
- **`WebCapabilityExecutor`** (`hbllm.hcir.capabilities.web_capabilities`): Network-sandboxed `web.search` and `web.fetch` protected by `is_safe_url()` SSRF checks.
- **`MemoryCapabilityExecutor`** (`hbllm.hcir.capabilities.memory_capabilities`): `memory.store`, `memory.recall`, `memory.forget`, `knowledge.ingest`, `knowledge.explain`, `knowledge.explain_why`, `knowledge.explain_how`.
- **`PlanningCapabilityExecutor`** (`hbllm.hcir.capabilities.planning_capabilities`): `planning.generate_candidates`, `planning.validate`, `planning.expand`.
- **`RoutingCapabilityExecutor`** (`hbllm.hcir.capabilities.routing_capabilities`): `routing.resolve_route`, `execution.sandbox_run`.
- **`WorldKernelCapabilityExecutor`** (`hbllm.hcir.capabilities.world_capabilities`): Governed world-kernel inspection and simulation actions.

---

## `AutonomyTracker`

**Module:** `hbllm.hcir.kernel.autonomy_tracker.AutonomyTracker`

```python
from hbllm.hcir.kernel.autonomy_tracker import AutonomyTracker

tracker = AutonomyTracker(retirement_threshold=0.98, min_turns_for_retirement=50)
tracker.record_parse(True)
tracker.record_plan(True)
tracker.record_realize(True)
print(tracker.to_dict())
```

