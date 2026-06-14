# Changelog

All notable changes to the HBLLM Core project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

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

#### Benchmarks

- **SNN Cognitive Benchmark** — `hbllm/benchmarks/bench_cognitive.py`
  - ComprehensionEnsemble step() latency and per-token cost
  - ComprehensionStream full comprehend() pipeline timing
  - ThoughtPlanner symbolic outline generation overhead
  - ExpressionStream rendering tier token budgets

- **DualLLMRouter Benchmark** — `hbllm/benchmarks/bench_dual_router.py`
  - Routing decision (classify) latency
  - Circuit breaker state transition timing
  - Fallback overhead measurement

- **HTTP API Load Test** — `hbllm/benchmarks/bench_http.py`
  - Health endpoint p50/p99 via ASGI transport
  - Rate limiter validation under burst load
  - Concurrent multi-tenant throughput

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

- **torch NameError** in `adapter_registry.py` — `cast()` used `torch.Tensor` at runtime
  but `torch` was only imported under `TYPE_CHECKING`. Fixed with string annotation.

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
