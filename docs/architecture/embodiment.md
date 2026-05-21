# Embodiment & Real-World Actuation

HBLLM Core interacts with the physical or operating system environment safely through the **Execution Reality Layer**. This system ensures that when the agent takes an action, it is not merely relying on the theoretical success of a tool, but actually verifying that the physical or digital state was altered as expected.

## Core Components

### 1. Device Reality Integration (`os_adapter.py`)
The `os_adapter` acts as a secure boundary between the cognitive loop and the host OS. 
- It maps semantic intentions (e.g., "turn off the wifi") into safe actuator/sensor interfaces.
- Provides isolated contexts for the agent to probe the OS state without risking unconstrained shell execution.

### 2. Execution Verification Engine (`verifier.py`)
Large Language Models are prone to hallucinating the success of their actions based on API responses. 
- The `verifier` implements **asynchronous non-blocking polling**. 
- After an action is executed, the verifier probes the actual physical or digital state (e.g., checking if the light is actually off, or if the file actually exists) to confirm the state change.
- Re-triggers the agent if reality diverges from the expected simulation.

### 3. Idempotency Tracking (`idempotency.py`)
During complex, multi-step execution, retries or crashes are inevitable. 
- To prevent duplicate mutating actions, the system generates deterministic action hashes based on the parameters and context.
- Maintains a lock-table of executed actions. 
- If a crash occurs and the system reboots, the agent will not repeat a destructive or state-mutating action it had successfully executed immediately prior to the crash.
