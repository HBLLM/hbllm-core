# HBLLM Security & Governance

HBLLM is designed for enterprise environments where data isolation, secure code execution, and model integrity are mission-critical. This document outlines the security architecture implemented in the `core` package.

## 1. Code Sandboxing (`ExecutionNode`)

Allowing an LLM to generate and execute code autonomously is a massive security risk. HBLLM mitigates this via a rigorous, multi-layered sandboxing pipeline found in `core/hbllm/actions/execution_node.py`:

- **Static AST Validation**: Before any Python code is executed, it is parsed by an internal Abstract Syntax Tree (AST) walker. This static analysis strictly rejects dangerous imports (e.g., `os`, `sys`, `subprocess`) and built-in functions (e.g., `exec()`, `eval()`, `open()`).
- **Restricted Subprocess Execution**: Validated code is never run in the main process. It is executed in an isolated, restricted subprocess.
- **Strict Timeouts**: Every execution is bound by a strict timeout (default: 5.0 seconds). If the code enters an infinite loop or attempts to halt the system, the subprocess is ruthlessly killed.

## 2. Multi-Tenant Isolation

HBLLM supports multi-tenancy out of the box, allowing a single deployment cluster to serve multiple users or organizations with zero data bleed.

- **Logical Data Isolation**: Every message on the `CognitiveEventBus` carries a mandatory `tenant_id` and `session_id`.
- **Memory Partitioning**: The `MemoryNode` (Episodic, Semantic, Procedural) partitions all storage and vector databases by `tenant_id`. An agent reasoning on behalf of Tenant A cannot access or "remember" data from Tenant B.
- **Cognitive Isolation**: The `WorkspaceNode` blackboard tracks thought hierarchies individually per session. A thought process spawned by one user is completely invisible to another.
*(Verified via automated tests: `test_integration.py::TestEndToEndPipeline::test_multi_tenant_isolation`)*

## 3. LoRA Adapter Integrity

The HBLLM architecture allows for "In-Flight LoRA Hot-Swapping"—dynamically downloading and loading fine-tuned domain adapters from the Sentra Marketplace at runtime. 

To prevent supply-chain attacks or malicious weight injection:
- **SHA-256 Validation**: The `AdapterRegistry` enforces strict cryptographic checksums. Before a new `.safetensors` or `.bin` adapter is loaded into the base LLM, its SHA-256 hash is verified against a known, trusted ledger. 
- **Read-Only Model Memory**: Adapter weights are loaded into read-only memory buffers to prevent runtime manipulation.

## 4. Input Sanitization & Policy Engine

- **Rate Limiting & Authentication**: The FastAPI Gateway uses token-bucket rate limiting per `tenant_id` and requires SHA-256 hashed API keys.
- **Policy Engine**: A Constitutional AI layer intercepting all outputs. Outputs are evaluated against safety policies (DENY, TRANSFORM, SCOPE) before being delivered to the user.

---
*For reporting vulnerabilities, please do not open a public issue. Email our security team directly.*
