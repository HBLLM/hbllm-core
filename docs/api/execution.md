---
title: "API Reference — Execution Engine & Modifiers"
description: "API reference for ExecutionOrchestrator, ExecutionBus, ExecutionPlan, TextRuntime, and Prompt/LoRA/Grammar modifiers."
---

# Execution Engine & Modifiers API

The **Execution Engine** provides unified DAG-based orchestration, resource policy enforcement, text modifier pipelines, and sandboxed tool execution.

**Package:** `hbllm.execution`

---

## Subsystem Index

| Class | Module | Purpose |
|---|---|---|
| `ExecutionOrchestrator` | `orchestrator.py` | Topologically executes and monitors DAG execution plans |
| `ExecutionBus` | `bus.py` | Internal execution event stream with subscriber hooks |
| `ExecutionPlan` / `PlanStep` | `plan.py` | Declarative execution graph specification |
| `ExecutionResult` | `result.py` | Output container with step latencies and telemetry receipts |
| `ExecutionPolicy` | `policy.py` | Capability matching and sandboxing enforcement |
| `TextRuntime` | `text/text_runtime.py` | Manages chaining of text modifiers before model generation |
| `PromptModifier` | `text/modifiers/prompt_modifier.py` | Injects system instructions, context, and few-shot examples |
| `LoraModifier` | `text/modifiers/lora_modifier.py` | Dynamically applies domain LoRA adapter weights |
| `GrammarModifier` | `text/modifiers/grammar_modifier.py` | Enforces JSON Schema and EBNF structured output constraints |
| `TrainingRuntime` | `training/training_runtime.py` | Asynchronous training loop dispatcher for online DPO/SFT |

---

## `ExecutionOrchestrator`

**Module:** `hbllm.execution.orchestrator.ExecutionOrchestrator`

```python
from hbllm.execution.orchestrator import ExecutionOrchestrator
from hbllm.execution.plan import ExecutionPlan, PlanStep

orchestrator = ExecutionOrchestrator()

# Define a 2-step plan with dependency
plan = ExecutionPlan(
    plan_id="plan-diagnose-01",
    steps=[
        PlanStep(
            step_id="step_ping",
            action="network.ping",
            parameters={"target": "gateway"},
        ),
        PlanStep(
            step_id="step_analyze",
            action="diagnostics.analyze",
            parameters={"source_step": "step_ping"},
            depends_on=["step_ping"],
        ),
    ],
)

# Execute
result = await orchestrator.execute_plan(plan, tenant_id="tenant-prod")
print(f"Status: {result.status}")
print(f"Output: {result.step_outputs.get('step_analyze')}")
```

---

## `TextRuntime` & Modifiers

**Module:** `hbllm.execution.text.text_runtime.TextRuntime`

```python
from hbllm.execution.text.modifiers.grammar_modifier import GrammarModifier
from hbllm.execution.text.modifiers.lora_modifier import LoraModifier
from hbllm.execution.text.modifiers.prompt_modifier import PromptModifier
from hbllm.execution.text.text_runtime import TextRuntime

runtime = TextRuntime()

# 1. Add Prompt Modifier
runtime.add_modifier(PromptModifier(prefix="[Expert: Database Reliability]\n"))

# 2. Add LoRA Modifier for coding domain
runtime.add_modifier(LoraModifier(domain="coding", adapter_id="lora-v2"))

# 3. Add JSON Schema Grammar Modifier
runtime.add_modifier(
    GrammarModifier(
        schema={
            "type": "object",
            "properties": {
                "diagnosis": {"type": "string"},
                "action": {"type": "string"},
            },
            "required": ["diagnosis", "action"],
        }
    )
)

# Transform text input
transformed_prompt = runtime.apply(
    "Check query slow log for PostgreSQL table users"
)
```
