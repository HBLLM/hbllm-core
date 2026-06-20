# Changelog

All notable changes to the HBLLM Core project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

#### Cognitive Features

- **PersonaEngine** — `hbllm/brain/persona_engine.py`
  - Persistent personality profiles (formality, humor, verbosity, emoji, empathy)
  - Per-tenant persona storage with adaptive learning from feedback
  - Emotion-aware style modulation (stressed → concise, curious → detailed)

- **NotificationGateway** — `hbllm/serving/notifications.py`
  - Proactive push channel for background insights and alerts
  - Priority-based notification queue (critical, info, suggestion)
  - WebSocket, webhook, and in-memory delivery backends

- **HabitTracker** — `hbllm/brain/habit_tracker.py`
  - Temporal pattern mining on episodic memory
  - Routine detection (daily/weekly patterns) and need prediction
  - Context-aware suggestions based on time-of-day and activity

- **ActivityDigest** — `hbllm/brain/activity_digest.py`
  - Summarizes missed activity during user absence
  - Aggregates events, completed goals, and proactive findings
  - Generates natural-language catch-up briefings

- **ConversationThread** — `hbllm/memory/conversation_thread.py`
  - Named, resumable conversation threads
  - Independent context windows per thread
  - Cross-session thread persistence

- **DelegationChain** — `hbllm/brain/delegation_chain.py`
  - Long-running autonomous task execution with progress tracking
  - User approval gates for sensitive steps
  - Persistent across restarts with state recovery

- **SessionMigration** — `hbllm/network/session_migration.py`
  - Cross-device context handoff ("continue on my phone")
  - Exports active session state (history, context, goals)
  - Cryptographic integrity verification on import

#### Production Hardening (Audit Phase)

- **DualLLMRouter + Circuit Breaker** — `hbllm/brain/dual_llm_router.py`
  - Smart local/external LLM routing based on query complexity
  - Circuit breaker with configurable failure threshold and recovery timeout
  - Automatic fallback to local LLM when external provider fails
  - Wired into ExpressionStream for transparent routing

- **BrainConfig Pydantic Migration** — `hbllm/brain/config.py`
  - Migrated from plain dict to Pydantic `BaseModel` with field validators
  - Added `model_validator` for JWT secret enforcement in production
  - Type-safe configuration with defaults and validation

- **Graceful Shutdown** — `hbllm/serving/api.py`
  - Drain period with configurable timeout (`HBLLM_SHUTDOWN_DRAIN_SEC`)
  - Rejects new requests during shutdown with 503 status
  - In-flight request tracking for clean termination

- **HTTP Rate Limiting** — `hbllm/serving/middleware/rate_limit.py`
  - Per-tenant token bucket rate limiting
  - Configurable via `HBLLM_RATE_LIMIT_RPM` (default: 60 RPM)
  - Returns 429 with `Retry-After` header

- **Prometheus Metrics** — `hbllm/serving/middleware/prometheus.py`
  - Request count, latency histogram, in-flight gauge, error counters
  - `/metrics/prometheus` endpoint for scraping
  - Per-endpoint and per-status-code breakdowns

- **Per-Tenant DB Quotas** — `hbllm/memory/episodic.py`
  - Configurable max turns per tenant (`HBLLM_DB_MAX_PER_TENANT`)
  - Automatic eviction of oldest turns when quota exceeded
  - Enforcement on every `store_turn()` call

- **API Versioning Middleware** — `hbllm/serving/middleware/api_version.py`
  - `X-API-Version` and `X-Supported-Versions` response headers
  - `Accept-Version` request header validation
  - Rejects unsupported versions with 400 + available versions list

- **Kubernetes Manifests** — `deploy/k8s/`
  - Deployment with health probes, resource limits, and rolling updates
  - Service and ConfigMap for environment-based configuration
  - Production-ready pod template with security context

- **Integration Tests** — `tests/integration/test_production_readiness.py`
  - 21 tests covering circuit breaker, rate limiting, DB quotas, metrics,
    graceful shutdown, API versioning, body size limits, and CORS

#### Autonomy & Agency (Core Audit — Cognitive Gaps)

- **CognitiveDaemon** — `hbllm/serving/daemon.py`
  - Long-running daemon process with Brain + AutonomyCore lifecycle
  - Boots Brain via BrainFactory, starts cognitive heartbeat
  - Graceful shutdown with state persistence
  - CLI entry point (`python -m hbllm.serving.daemon`)

- **ProactiveProcessor + SSEChannel** — `hbllm/serving/proactive.py`
  - Routes AutonomyCore cognitive actions to user-facing output
  - LLM enrichment of background insights before delivery
  - Real-time Server-Sent Events per-tenant push channel
  - Notification delivery via NotificationGateway + SSE + bus broadcast

- **Notification API** — `hbllm/serving/routes/notifications.py`
  - REST endpoints for listing and dismissing notifications
  - SSE streaming endpoint for real-time push delivery

- **ReActLoop** — `hbllm/actions/tool_chain.py`
  - Iterative Observe → Think → Act reasoning loop (replaces single-pass)
  - Parallel tool execution with configurable concurrency
  - Scratchpad chain-of-thought reasoning trace
  - Budget limits: max iterations, max tokens, max wall-time

- **ConversationTurnManager** — `hbllm/perception/conversation_turn.py`
  - Full-duplex voice state machine (IDLE → LISTENING → PROCESSING → SPEAKING)
  - Barge-in detection and interrupt handling
  - Silence timeout and continuous listening mode

- **ContextFusionEngine** — `hbllm/brain/context_fusion.py`
  - Token-budgeted context assembly from multiple sources
  - Priority-weighted greedy allocation strategy
  - Pre-built providers for memory, world state, emotion, goals

- **EmotionEngine Upgrade** — `hbllm/brain/emotion_engine.py`
  - LLM-based contextual inference (sarcasm, nuance detection)
  - Behavioral pattern tracking (response latency, message frequency)
  - Per-tenant emotional state cache for context fusion

- **ActionVerificationBridge** — `hbllm/brain/autonomy/verification_bridge.py`
  - Closes the execute → verify → correct feedback loop
  - Periodic verification of VERIFYING tasks against WorldStateEngine
  - Auto-generates verification rules for IoT commands
  - Re-executes tasks that fail verification (with correction limit)

- **DeviceBridge** — `hbllm/serving/device_bridge.py`
  - Cross-device session continuity and presence tracking
  - Device registration with capabilities and push tokens
  - Heartbeat-based presence (5-minute timeout)
  - Session handoff between devices with tenant isolation

#### Infrastructure Fixes (Core Audit)

- **LoadManager ↔ AttentionManager** bidirectional integration
- **AnthropicProvider** connection reuse with `httpx` client
- **Per-topic BusMetrics** tracking (publish/delivery/error counters)
- **Per-tenant RateLimitInterceptor** on message bus
- **DB indexes** on `tenant_id` / `session_id` columns
- **Bus drain** with timeout for graceful shutdown
- **Provider lifecycle** with `close()` and async context manager
- **LocalProvider** `_prepare_input()` factored out
- **MemoryNode** `UnifiedMemoryInterface` compliance
- **Ordered shutdown** sequence in lifespan
- **Metrics thread safety** with `threading.Lock`
- **SNN neuron eviction** on capacity overflow
- **Task dispatch** error surfacing

#### Benchmarks

- **SNN Cognitive Benchmark** — `hbllm/benchmarks/bench_cognitive.py`
- **DualLLMRouter Benchmark** — `hbllm/benchmarks/bench_dual_router.py`
- **HTTP API Load Test** — `hbllm/benchmarks/bench_http.py`

### Changed

- **api.py Split** — Extracted modular route packages:
  - `hbllm/serving/routes/health.py` — health and monitoring endpoints
  - `hbllm/serving/routes/memory.py` — memory, sync, feedback, knowledge endpoints
  - `hbllm/serving/deps.py` — FastAPI `Depends()` injection layer
  - api.py reduced from 2582 to 2177 lines (15% reduction)

- **factory.py Split** — Extracted SNN wiring logic:
  - `hbllm/brain/wiring/snn.py` — ComprehensionStream and ExpressionStream wiring
  - factory.py reduced from 1972 to 1735 lines (12% reduction)

- **orjson Graceful Fallback** — Silent fallback to stdlib json when orjson unavailable

### Fixed

- **Exception handling audit** — 460+ bare/pass-only catches across 68 files
  converted to proper logging. Zero silent exception swallows remain.
- **Memory leaks** — 8 leaks fixed across brain, network, serving, persistence
- **FastAPI 0.137+ route detection** — `test_api_endpoints.py` updated for
  `_IncludedRouter` (use OpenAPI + recursive traversal)
- **Tokenizer decode crash** — `ValueError: bytes must be in range(0, 256)`
  fixed in zero-dependency fallback when token IDs ≥ 256
- **torch NameError** in `adapter_registry.py` — `cast()` used `torch.Tensor` at runtime
  but `torch` was only imported under `TYPE_CHECKING`. Fixed with string annotation.
- **Ruff lint errors** — Missing logger imports, unused conditional imports, type narrowing

---


## [0.2.0] — 2026-05-16

### Added
- ExpressionStream SNN pipeline (broca/shallow/deep rendering tiers)
- ComprehensionStream 5-channel SNN ensemble
- ThoughtPlanner symbolic outline generation
- Process Reward Model (PRM) with STDP training
- Speculative decoding integration
- Studio compatibility API endpoints

---

## [0.1.0] — 2026-03-07

### Added
- Initial cognitive architecture with 25+ nodes
- Multi-tiered memory system (episodic, semantic, procedural)
- RouterNode with ONNX fast-path domain classification
- LoRA-based domain specialization (zoning model)
- InProcessBus and RedisBus message transport
- Multi-tenant isolation with JWT authentication
- Plugin system with hot-reload
- Rust SIMD compute kernels (INT4/INT8)
- MkDocs Material documentation site
