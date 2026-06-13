# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
