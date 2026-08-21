# Changelog

All notable changes to the HBLLM Core project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

#### Epistemic Integration of Perception (Waves A9–A10)

- **A9 — Epistemic Primitives in HCIR**: `PerceptualEpistemicProfile` (multidimensional profiles with derived `.reliability`), `CorrelationCandidate`, `PerceptualContradictionLevel`, `EvidenceAssessment`, `PropositionLikelihood` ($P(E|H)$, $P(E|\neg H)$, $LR$), `BeliefTransition`, `BeliefTransitionNode` (event-sourced history node), `ObservationNode`, `EvidenceNode`, `ContradictionNode`, `BeliefNode` HCIR schemas — `hbllm/hcir/types.py`, `hbllm/hcir/graph.py`
- **A9 — Decoupled Evaluators**: `PerceptualEvidenceEvaluator` (general signal reliability and provider calibration) and `EpistemicLikelihoodEvaluator` (proposition-specific discrimination in likelihood space) — `hbllm/brain/epistemics/`
- **A9 — Odds-Space Bayesian Belief Revision**: `DiscoveryBeliefManager.revise()` computing $O(H|E) = O(H) \times LR$ and emitting event-sourced `BeliefTransitionNode` records into the HCIR graph — `hbllm/brain/epistemics/belief_manager.py`
- **A9 — 3-Tier Contradiction Hierarchy**: `ContradictionEngine.scan_for_perceptual_contradictions()` classifying Level 1 (Classifier/Candidate), Level 2 (Cross-Modal Correlated), and Level 3 (Belief-Perception Conflict) — `hbllm/brain/epistemics/contradiction_engine.py`
- **A10 — Perception-Epistemic Bridge**: `PerceptionEpistemicBridge` structural adapter materializing audio/visual observations, evidence, and `CORRELATES_WITH` hyperedges without directly mutating beliefs — `hbllm/perception/perception_epistemic_bridge.py`
- **A10 — Epistemic Loop Multimodal Orchestration**: `EpistemicLoop` processing perceptual evidence, scanning 3-level contradictions, and triggering curiosity investigations — `hbllm/brain/epistemics/epistemic_loop.py`
- **Test Suite**: 11 perceptual epistemics unit tests + 2 multimodal end-to-end integration tests (13/13 passing); full epistemics and perception suite: 576/576 passing

#### Production Audio Providers & Multimodal Evidence Semantics (Waves A7–A8)

- **A7 — Provider Provenance & Production Providers**: `ProviderProvenance` metadata tracking across all audio evidence types; `MoonshineSpeechProvider`, `AmbientEventProvider`, `ResemblyzerSpeakerProvider`; `AudioInputNode` refactored as adapter with provider delegation; 23-test adversarial suite — `hbllm/perception/`
- **A8 — Multimodal Evidence Semantics**: Modality-neutral `PerceptualObservation` and `PerceptualAssessment` base types; `CorrelationEngine` geometric/temporal correlation without early collapse; `CORRELATES_WITH` and `OBSERVED_AS` edge types; `CorrelationTransaction` HCIR commitment — `hbllm/perception/`, `hbllm/hcir/graph.py`

#### Grounded Audio Perception Runtime (Waves A1–A6)

- **A1 — Audio Perception Contracts**: `SpeechProvider`, `AcousticEventProvider`, `AcousticSceneProvider`, `SpeakerProvider`, `SoundLocalizationProvider` protocols; `TemporalSpan` (separated `observation_id`, `event_id`, `segment_id`, `AudioEventState`); composition evidence model (`AcousticObservation`, `SpeechEvidence`, `SoundEventEvidence`, `SoundSourceEvidence`, `AcousticSceneEvidence`); structured `SpeakerIdentification`; probabilistic `ParalinguisticProfile`; `AudioEpistemicProfile`; `AudioRecognitionPolicy` — `hbllm/perception/providers/`
- **A2 — Extracted Providers**: `MockAudioProvider` deterministic SHA-256 testing provider supporting all audio perception protocols — `hbllm/perception/providers/mock_audio_provider.py`
- **A3 — Audio Perception Runtime & Memory**: `AudioPerceptionRuntime` (evidence-only, normalizes provider outputs, never mutates HCIR), `AudioMemory` (observation-first embedding index with running-average prototype acceleration and exemplar diversity enforcement) — `hbllm/perception/`
- **A4 — HCIR Transaction Layer**: `AudioObservationNode`, `AcousticConceptNode` added to HCIR `CognitiveGraph`; `AudioPerceptionTransaction` (atomic HCIR commitment for speech, events, and cognitive artifact learning); `AudioPerception` unified facade API — `hbllm/hcir/graph.py`, `hbllm/perception/`
- **A5 — SNN Audio Gating**: `AudioSignals` cheap numpy feature extraction (~0.1ms: energy, spectral centroid, spectral flux, zero-crossing rate, speech likelihood); `AudioPerceptionEnsemble` (4-channel LIF SNN ensemble: speech, event, change, transient) driving `PerceptionGateDecision`; `AudioPerceptionStream` (SNN-gated continuous stream processor) — `hbllm/brain/snn/perception/`, `hbllm/perception/`
- **A6 — Temporal & Cross-Modal Integration**: `TemporalFuser` candidate pattern extraction (`TemporalPatternCandidate`) with audio observations; `PerceptionFuser` cross-modal fusion (audio + visual); `WorldStateEngine` dual-source gradual migration (`update_from_audio_assessment`, `update_from_hcir`) — `hbllm/perception/`
- **Test Suite**: 134 unit tests across A1–A6, all passing (48 + 31 + 19 + 16 + 14 + 6); full perception suite: 256/256 passing
- **Documentation**: Updated `docs/api/perception.md` with Grounded Audio Perception Architecture

#### Visual Cognition Runtime (Waves V0–V3)

- **V0 — Grounded Perception Contracts**: `VisionProvider`, `VisionDetector`, `VisionOCR` protocols, `VisualEmbedding`, `VisualEvidence` (immutable), `VisualAssessment` (mutable), `EpistemicEvidenceProfile` (multi-dimensional confidence), `RecognitionPolicy` (configurable thresholds), `PerceptionGateDecision`, `PerceptionProcessingLevel` — `hbllm/perception/providers/`, `hbllm/brain/snn/perception/`
- **V1 — Perceptual Evidence**: `MockVisionProvider` (deterministic testing), `SigLIPVisionProvider` (lazy loading, CPU/CUDA/MPS auto-detect), `VisualObservationNode`, `VisualConceptNode` graph types added to HCIR `CognitiveGraph` — `hbllm/perception/providers/`, `hbllm/hcir/graph.py`
- **V2 — Epistemic Visual Memory**: `VisualMemory` (observation-first vector index with prototype acceleration), `VisualPerceptionRuntime` (evidence-only, never mutates HCIR), `VisualPerceptionTransaction` (atomic HCIR commitment: learn/recognize), `VisualPerception` facade (`learn()` / `recognize()`), `OneShotEvaluator` evaluation harness — `hbllm/perception/`
- **V3 — Temporal Attention**: `VisualSignalExtractor` (cheap ~0.1ms frame features: motion, intensity, edge, color, texture), `PerceptionEnsemble` (5-channel LIF ensemble: scene/entity/motion/novelty/stability), `VisualPerceptionStream` (SNN-gated continuous perception with frame-level stats) — `hbllm/brain/snn/perception/`, `hbllm/perception/`
- **Test Suite**: 105 unit tests across V0–V3, all passing (36 + 25 + 26 + 18)
- **Documentation**: Updated `docs/api/perception.md` and `docs/api/snn.md` with visual cognition API reference

#### Epistemic Runtime — Domain-Neutral Discovery Engine (Waves 1–4)

- **Wave 1 — Epistemic Foundations**: `DiscoveryBeliefManager` (Bayesian revision), `SourceReputationTracker` (source trust), `DiscoveryWorkspace` (research program lifecycle) — `hbllm/brain/epistemics/`
- **Wave 2 — Closed Discovery Loop**: `CuriosityEngine`, `IdeaGenerator`, `HypothesisBuilder`, `PredictionTracker`, `ExperimentPlanner`, `EvidenceEvaluator`, `ContradictionEngine`, `ExplanationEngine`, `ResearchStrategyManager`, `EpistemicLoop` — 10 engines forming a complete autonomous discovery cycle
- **Wave 3 — Meta-Epistemic Cognition**: `EpistemicMemory` (SQLite-backed reasoning history), `EpistemicCalibrationEngine` (ECE curves, bias detection, strategy recommendation), `CounterfactualReasoner` ("What if..." graph analysis with 5 methods)
- **Wave 4 — Deep Integration & Wiring**: Memory-aware IdeaGenerator (filters past failures), auto-strategy switching via calibration, counterfactual-driven experiment planning, `wire_epistemics()` one-function AutonomyCore integration
- **Test Suite**: 126 unit tests + 3 E2E integration tests + 5 cross-component tests = 134 total, all passing
- **Documentation**: Architecture deep-dive at `docs/architecture/epistemics.md`, overview and index updated

#### v3: SNN Cognitive Architecture (Milestones 1–5)

- **M1 — Reactive Brain**: LIF spiking neurons, multi-layer `SpikingNetwork`, `LayerProjection` with STDP plasticity, short-term plasticity (STP), `ComprehensionStream`, `ExpressionStream`, `ReasoningNetwork` — `hbllm/brain/snn/`
- **M2 — Learning Brain**: Event-sourced `MemoryEventStore` with `MemCube` fold, `BeliefGraph` for provenance tracking, `GoalMemory` with hierarchical lifecycle, `PredictiveMemoryLoader` with Markov prefetch — `hbllm/memory/`
- **M3 — Predictive Brain**: `DendriticNeuron` (dual-compartment predictive coding), `PopulationEncoder` / `CognitiveStateEncoder` (Gaussian tuning curves), `CognitivePredictors` (multi-domain Markov anticipation), `IzhikevichNeuron` (biologically realistic model) — `hbllm/brain/snn/`
- **M4 — Deliberative Brain**: `SimulationEngine` (candidate evaluation with `HeuristicCritic`), `DeliberationBudget` (adaptive computation: SKIP/SINGLE/MULTIPLE/BEAM), `EvidencePacket` / `EvidenceBuilder` (structured evidence for reasoning) — `hbllm/brain/`
- **M5 — Autonomous Brain**: `OscillationManager` with 4 bands (γ/β/θ/δ), `BrainTick` heartbeat events, phase-amplitude coupling, frequency modulation, `NeuromodulationEngine` (6-transmitter system) — `hbllm/brain/snn/oscillations.py`, `hbllm/brain/neuromodulation.py`

#### v3: Integration Wiring (Cognitive Operating System)

- **`BrainContext`** — Separates `BrainServices` (stateless infrastructure) from `BrainState` (serializable runtime), coordinated via `BrainContext` — `hbllm/brain/brain_context.py`
- **`BrainContainer`** — Bootstrap factory wiring 7 services with 20 capability tags. Single `BrainContainer.build(config)` entry point — `hbllm/brain/brain_container.py`
- **`CapabilityRegistry`** — Dynamic service discovery: `registry.find("simulation")` returns tagged services — `hbllm/brain/capability_registry.py`
- **`TraceCollector`** — End-to-end cognitive observability with trace IDs following events through the full pipeline — `hbllm/brain/trace.py`
- **`MemoryRepository`** — Abstract base for event-sourced memory repositories + `MemoryProjection` layer — `hbllm/memory/repository.py`
- **`ProjectionType`** — `BASAL`, `APICAL`, `MODULATORY` enum on `LayerProjection` — `hbllm/brain/snn/network.py`
- **Integration Tests** — 34 tests covering bootstrap, capability discovery, tracing, BrainTick, DeliberationBudget, EvidencePacket, happy-path cognitive cycle, and failure recovery — `tests/integration/test_cognitive_loop.py`
- **158 total tests** (124 unit + 34 integration) passing with zero regressions

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

#### Cognitive Subsystems — Human Modeling Layer

- **UserModelEngine** — `hbllm/brain/user_model.py` + `user_model_node.py` (~1160 lines)
  - Predictive model of the human operator: expertise, preferences, beliefs, trust
  - Continuous learning from interactions via 9-domain vocabulary analysis
  - Temporal pattern detection (active hours/days) and next-action prediction
  - Ebbinghaus-curve confidence decay with configurable half-life
  - SQLite persistence with per-tenant isolation
  - ContextFusion provider (priority 0.85)

- **ProjectGraph** — `hbllm/brain/project_graph.py` + `project_node.py` (~860 lines)
  - Graph-based project state: goals, blockers, questions, decisions, milestones
  - Auto-detection of active project from query context
  - Project reactivation with context summary generation
  - Cross-project dependency tracking
  - SQLite persistence with entity/relation tables

- **ExecutiveCortex** — `hbllm/brain/executive_cortex.py` (~494 lines)
  - Unified cognitive control: goal arbitration, focus management, interruption control
  - Cognitive budget allocation (heavy_llm / fast_router / reflex / reserve)
  - Task switching cost calculation with fatigue modeling
  - Reads from UserModel (alignment) and GoalManager (priorities)

- **RelationshipMemory** — `hbllm/brain/relationship_memory.py` + `relationship_node.py` (~826 lines)
  - Social graph of people: roles, sentiment, interaction history, topics
  - Trend detection (improving / stable / declining)
  - Notification prioritization based on importance and recency
  - Regex-based multi-word person name extraction
  - SQLite persistence with people/events/relationships tables

- **RealityGraph** — `hbllm/brain/reality_graph.py` (~531 lines)
  - Unified read-only facade over KnowledgeGraph, BrainWorldState, PerceptionWorldState
  - Cross-backend entity merging by confidence score
  - TTL-based entity expiry via `tick()`
  - ContextFusion provider (priority 0.60)

- **ContextFusion Integration** — 4 new providers in `context_fusion.py`
  - `user_model` (0.85), `active_project` (0.85), `relationships` (0.55), `reality_graph` (0.60)

- **BrainFactory Integration** — 5 new `inject_*` config flags
  - All subsystems auto-wired with bus adapters and ContextFusion providers

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
