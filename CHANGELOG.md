# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-06-13

### Added — SNN Cognitive Stream (v1→v4 Complete)

- **Synaptic Plasticity** (`snn/plasticity.py`): STDP learning rule for all multi-layer SNN projections. Supports online (per-interaction) and batch (periodic sweeps) training modes.
- **Multi-Layer SNN** (`snn/network.py`): `SpikingNetwork` with `NeuronLayer`, `LayerProjection`, configurable LIF neurons, and STDP-enabled connections.
- **ComprehensionStream** (`snn/comprehension/`): 5-channel LIF ensemble (entity, clause, discourse, surprise, constraint) with lexical signal input. Event-triggered embeddings (3× faster than per-token).
- **AssociationLayer** (`snn/comprehension/association_layer.py`): 4→8→2 SNN discovers concept relationships and association types.
- **CausalReasoner** (`snn/reasoning/`): Multi-hop causal graph with SNN-scored chain evaluation via 4→6→2 ReasoningNetwork.
- **ExpressionStream** (`snn/expression/expression_stream.py`): Orchestrates thought→text pipeline with three rendering tiers (broca → shallow → deep).
- **ThoughtPlanner + ThoughtController**: Symbolic goal decomposition with SNN-gated generation.
- **TrainedPRM** (`snn/expression/trained_prm.py`): 6→8→4→2 SNN reward evaluator with `TrainingCollector` persistence and online STDP learning.
- **ShallowRenderer** (v3, `snn/expression/shallow_renderer.py`): Reduces LLM prompts from ~600 to ~300 tokens. SNN handles reasoning, LLM handles rendering.
- **ContentPlanner** (v4, `snn/expression/content_planner.py`): 8→12→6→3 SNN for content type selection (assertion/explanation/example/transition/caveat). SNN decides what to say.
- **BrocaEncoder** (v4, `snn/expression/broca_encoder.py`): Ultra-minimal ~80-token prompts (TYPE/TONE/SAY/MAX format). 87% token reduction from v1.
- **PRMTrainer** (v4, `snn/expression/prm_trainer.py`): Batch STDP training sweeps with pre/post accuracy measurement and weight delta tracking.
- **SNN Debugger** (`snn/debugger.py`): Live introspection of neuron potentials, spike history, and weight matrices.

### Changed
- **BrainFactory**: Auto-wires all SNN components (ComprehensionStream, ExpressionStream, TrainedPRM, ContentPlanner, BrocaEncoder). Shallow/broca modes are opt-in (default: deep mode for full brain pipeline compatibility).
- **DecisionNode**: Integrates ExpressionStream for structured thought-by-thought generation with SNN gating and PRM evaluation.

### Architecture

Three rendering tiers coexist with graceful fallback:

| Tier | Prompt Tokens | LLM Role | SNN Role |
|------|-------------|----------|----------|
| Deep (v1-v2) | ~600 | Reasoning + generation | Gating only |
| Shallow (v3) | ~300 | Rendering conclusions | Reasoning + associations |
| Broca (v4) | ~80 | Grammar/fluency only | Full content planning |

### Tests
- 239 new SNN tests across 9 test files (comprehension, expression, plasticity, network, reasoning, trained PRM, shallow renderer, broca, router).
- Full suite: 2311 passed, 8 skipped.

---

## [1.0.0] — 2026-06-13

### Added
- **Structured Error Handling**: New `hbllm.serving.errors` module with standardized error codes (`ErrorCode` enum), safe exception messages, and a global unhandled exception handler that prevents leaking internal details.
- **Request Tracing**: `X-Request-ID` middleware injects a unique request ID into every request/response for end-to-end log correlation.
- **Health Probes**: `/health/live` (liveness) and `/health/ready` (readiness with deep checks) endpoints for Kubernetes-compatible deployments.
- **Production Secret Validation**: Server refuses to boot with `HBLLM_ENV=production` if secrets are missing or use insecure `.env.example` defaults.
- **Dependency Splitting**: Heavy ML dependencies (torch, transformers, onnxruntime) moved to optional `[local]` extra. Base install is now lightweight (~50MB) for API-only / provider-mode users.
  - `pip install hbllm` — lightweight, API-only
  - `pip install hbllm[local]` — includes torch, transformers, reasoning engines
  - `pip install hbllm[full]` — everything including GPU, embodiment, observability
- **Project URLs**: PyPI package page now links to homepage, docs, repository, and changelog.

### Changed
- **Version**: Bumped from `0.1.0` to `1.0.0`.
- **CORS Hardening**: Restricted `allow_methods` and `allow_headers` from wildcard `*` to explicit lists.
- **Error Responses**: Tenant isolation errors now return structured `{"error": {"code": "...", "message": "..."}}` format instead of raw `{"detail": "..."}`.
- **Brain Degradation Tracking**: When full brain boot fails and falls back to provider mode, the state is tracked as `brain_degraded=True` and logged at ERROR level.

### Fixed
- **Silent Exception Swallowing**: 6 `except Exception: pass` patterns in `pipeline.py` now log at WARNING level with full traceback.
- **Log Injection Prevention**: All f-string logger calls in `api.py` and `synapse_gateway.py` replaced with `%s` formatting.
- **Python 3.16 Compatibility**: Replaced deprecated `asyncio.iscoroutinefunction()` with `inspect.iscoroutinefunction()` in `tenant_guard.py`.
- **Synapse Gateway**: Invalid JSON now truncated to 200 chars in warning log to prevent log flooding.

### Security
- `BodySizeLimitMiddleware` now wired into the middleware stack (was defined but unused).
- `RequestIDMiddleware` added for audit trail correlation.
- WebRTC error responses no longer leak raw exception details (`str(e)` → generic message).
