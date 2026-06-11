# Decision Node & Control Plane

## Overview

The `DecisionNode` acts as the cognitive **Gatekeeper** of HBLLM. Placed at the termination of the consensus blackboard, it enforces safety rules, manages resource budgets, and regulates routing behavior using a mathematically hardened control plane.

By implementing a **3-tier validation path**, it ensures that intermediate plans are safe, cost-effective, and resource-bounded before execution.

```
 Consensus consensus → ┌──────────────────────┐
                       │ Level 1: Safety Gate │ → (refuses unsafe content)
                       └──────────┬───────────┘
                                  │
                       ┌──────────▼───────────┐
                       │ Level 2: Policy      │ → (Hysteresis, Lyapunov,
                       │ Router Control Loop  │    Schmitt Trigger, Stable Lock)
                       └──────────┬───────────┘
                                  │
                       ┌──────────▼───────────┐
                       │ Level 3: Budget      │ → (caps tokens, memory,
                       │ Controller           │    recursion depth)
                       └──────────┬───────────┘
                                  │
                                  ▼
                            Action Execution
```

---

## 1. Level 1: Safety Gate

All proposed `ActionPlan` objects pass through the safety gate first. The checks are tiered based on the plan's **Risk Level**:

* **Low Risk** (`TEXT_RESPONSE`, `AUDIO_OUTPUT`): Verified against lightweight string/regex policy engines.
* **Medium Risk** (`WEB_SEARCH`, `API_CALL`): Verified by policy engine + keyword blocklists.
* **High Risk** (`CODE_EXECUTION`, `SHELL_EXECUTION`, `IOT_COMMAND`, `MCP_TOOL`): Undergoes real-time LLM-backed safety classification. Unsafe plans are rejected back to the planner blackboard.

---

## 2. Level 2: Policy Router Control Loop

The policy router determines the execution mode (e.g. *high-performance*, *medium-optimization*, *low-power exploration*, or *negative replanning*). 

To prevent self-stabilizing feedback loops and routing oscillations, it operates as a **3-variable multi-rate control loop**:

### Timescale Separation (Anti-Aliasing)
To avoid frequency coupling and resonance between fast execution loops and slow adaptation:
* **Fast Loop** (every decision): Evaluates current utility and applies hysteresis switching.
* **Medium Loop** (~7 decisions): Smooths utility percentiles using a $\gamma$-EMA.
* **Slow Loop** (~13 decisions): Evaluates systematic error drift and Lyapunov stability metrics.

### Instability Estimator
The raw diagnostic instability energy $S_{\text{diag}}(t)$ measures prediction consistency over a sliding window of recent outcomes:
$$S_{\text{diag}}(t) = \text{variance}(\text{predicted} - \text{actual}) + 2.0 \cdot \text{bias}^2$$

### Control Signal & Anchor Mixing
To break the self-referential calibration loop, the system mixes live percentiles with a **frozen baseline anchor prior** (defaulting to $[0.7, 0.3, 0.0]$). The blending weight is adjusted dynamically by the control signal $S_{\text{ctrl}}(t)$ (an EMA of $S_{\text{diag}}(t)$):
$$\text{anchor\_weight} = \max(0.3, \min(0.9, 0.3 + S_{\text{ctrl}}(t)))$$

Under high noise or prediction error, the system automatically shifts its weights to rely on the stable anchor prior.

### Schmitt Trigger Hysteresis Gating
If instability spikes ($S_{\text{diag}} > 0.08$), the system enters a **restricted cooling state** where resource-heavy tools are disabled. Exiting the cooling state requires sustained low-energy ticks ($S_{\text{diag}} < 0.04$ for 3 ticks).
* **High-Confidence Exits**: If the exit occurs with sustained very low energy ($S_{\text{diag}} < 0.03$), the long-term stable anchor is updated to match the new smoothed percentiles.
* **Marginal Exits**: If the energy is between $0.03$ and $0.04$, the system exits cooling but preserves the historical anchor prior to prevent learning from noise.

### Bounded Invariance (Stable Lock)
If the system rapidly cycles in and out of the cooling state ($\ge 3$ entries in 60 decisions), the regulator engages `stable_lock`. This:
1. Pins all blending weights to $1.0$ (complete reliance on the anchor).
2. Mandates a strict recovery criteria ($S_{\text{diag}} < 0.02$ sustained for 6 consecutive ticks) before restoring normal threshold adaptation.

---

## 3. Level 3: Budget Controller

The budget controller enforces hard resource limits to prevent runaway tasks:

* **Token Budget**: Enforces token usage bounds. Max tokens are halved under `medium` load, and restricted to `1024` tokens during `low` exploration.
* **Memory Limits**: Introspects host virtual memory percent (via `psutil`) and blocks execution if RAM usage exceeds the configured limit (defaulting to 90%).
* **Quadratic Replanning Penalty**: If a plan falls into negative utility and triggers a decomposition request, a quadratic cost penalty of $0.05 \cdot \text{depth}^2$ is subtracted from the utility score. This strongly suppresses deep recursion loops while allowing shallow explorations.
