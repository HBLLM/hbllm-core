---
title: "Deployment Guide — Run HBLLM Anywhere Without Expensive Hardware"
description: "Deploy HBLLM Core to production on any hardware — from Raspberry Pi to Kubernetes. CPU-only, Docker, systemd, distributed bus, and cloud API modes."
---

# Deployment Guide

HBLLM is designed to run on **any hardware** — from a Raspberry Pi 5 to a Kubernetes cluster. No expensive GPU is required for production deployment.

## Deployment Modes

| Mode | Hardware | Use Case |
|---|---|---|
| **CPU-Only Local** | Laptop, Raspberry Pi | Personal assistant, IoT, prototyping |
| **GPU-Accelerated** | Desktop with 6GB+ VRAM | Faster local inference, batch processing |
| **Cloud API** | Any machine (200MB RAM) | Use OpenAI/Anthropic as backend — zero local GPU |
| **Distributed** | Multi-server + Redis | Horizontal scaling for enterprise workloads |

---

## CPU-Only Deployment (No GPU)

The Pure-HCIR Cognitive OS runs deterministically on any machine with Python 3.11+ and zero GPU requirements:

```bash
pip install ".[server]"
HBLLM_ENV=production HBLLM_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  hbllm serve --host 0.0.0.0 --port 8000
```

---

## Docker

The repository ships a 3-stage production `Dockerfile` (`rust-builder` → `python-builder` → non-root `runtime`) that compiles the Rust PyO3 wheels with `maturin build --release`, installs `.[server]`, exposes port `8000`, healthchecks `/health/live`, and launches `uvicorn hbllm.serving.api:app`:

```bash
# Build production image from HBLLM/core/
docker build -t hbllm-core .

# Run in production mode (fail-closed JWT auth requires HBLLM_JWT_SECRET)
docker run -p 8000:8000 \
  -e HBLLM_ENV=production \
  -e HBLLM_JWT_SECRET="replace-with-strong-32byte-secret" \
  hbllm-core
```

For multi-replica deployments behind Nginx with Redis and PostgreSQL, use the production overlay (`docker-compose.prod.yml`):

```bash
export HBLLM_JWT_SECRET="replace-with-strong-32byte-secret"
export POSTGRES_PASSWORD="replace-with-db-password"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Raspberry Pi 5 / Edge Deployment

```bash
# Core-only Pure-HCIR install on ARM64 (NEON SIMD via Rust crates, no GPU required)
pip install -e .
hbllm diagnostics
hbllm kernel boot --world-id edge_node
```

!!! tip "ARM Optimization"
    Building the Rust crates (`maturin develop --release`) enables native ARM64 NEON acceleration for `hbllm_hcir_graph`, `hbllm_compute`, `hbllm_simulation_engine`, and `hbllm_structure_matcher`.

---

## Distributed Deployment with Redis

For multi-server deployments, use `RedisBus`:

```bash
# Server 1: API + Cognitive Core
HBLLM_BUS_TYPE=redis \
HBLLM_REDIS_URL=redis://redis-host:6379 \
HBLLM_REDIS_HMAC_KEY=your-secret-key \
python -m hbllm.serving.api

# Server 2: Background Workers (Learning, Sleep Cycle)
HBLLM_BUS_TYPE=redis \
HBLLM_REDIS_URL=redis://redis-host:6379 \
HBLLM_REDIS_HMAC_KEY=your-secret-key \
python -m hbllm.serving.api --worker-mode
```

---

## Kubernetes Deployment

HBLLM ships production-ready K8s manifests in `deploy/k8s/`:

```bash
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

The deployment includes:

- **Liveness probe** → `GET /health/live` (checks process health)
- **Readiness probe** → `GET /health/ready` (checks Brain + Bus initialization)
- **Resource limits** — 2 CPU / 4Gi RAM (configurable via ConfigMap)
- **Rolling updates** with `maxSurge: 1` / `maxUnavailable: 0`
- **Security context** — non-root, read-only filesystem

```yaml
# deploy/k8s/configmap.yaml (key env vars)
HBLLM_PROVIDER: "local"
HBLLM_ENV: "production"
HBLLM_JWT_SECRET: "<your-secret>"
HBLLM_RATE_LIMIT_RPM: "60"
HBLLM_DB_MAX_PER_TENANT: "50000"
HBLLM_SHUTDOWN_DRAIN_SEC: "15"
```

---

## Monitoring

### Prometheus Metrics

HBLLM exposes a `/metrics/prometheus` endpoint with:

| Metric | Type | Description |
|--------|------|-------------|
| `hbllm_http_requests_total` | Counter | Total requests by endpoint and status |
| `hbllm_http_request_duration_seconds` | Histogram | Latency distribution (p50/p95/p99) |
| `hbllm_http_requests_in_flight` | Gauge | Currently processing requests |
| `hbllm_http_errors_total` | Counter | Server errors (5xx) by endpoint |

**Scrape config** for `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'hbllm'
    scrape_interval: 15s
    static_configs:
      - targets: ['hbllm-service:8000']
    metrics_path: '/metrics/prometheus'
```

### OpenTelemetry

HBLLM also ships with OpenTelemetry instrumentation for distributed tracing:

```bash
HBLLM_OTEL_ENDPOINT=http://otel-collector:4317 \
python -m hbllm.serving.api
```

- Request latency histograms
- Node activation spans
- Memory usage per tenant
- Bus throughput metrics

---

## API Versioning

All API responses include version headers:

```
X-API-Version: v1
X-Supported-Versions: v1
```

Clients can request a specific version via the `Accept-Version` header:

```bash
curl -H "Accept-Version: v1" http://localhost:8000/v1/chat
```

Unsupported versions return `400 Bad Request` with the list of supported versions.

---

## Health Checks

The API exposes three health endpoints:

```bash
# Quick liveness check (always fast, no dependency checks)
curl http://localhost:8000/health/live
# {"status": "alive"}

# Readiness check (verifies Brain and Bus are initialized)
curl http://localhost:8000/health/ready
# {"status": "ready", "brain": true, "bus": true}

# Full health check (includes node count and bus status)
curl http://localhost:8000/health
# {"status": "healthy", "nodes": 23, "bus": "running"}
```

---

## Security Checklist

- [x] Set `HBLLM_JWT_SECRET` for production (`HBLLM_ENV=production` enforces this)
- [x] Do **not** set `HBLLM_UNSAFE_DEV_AUTH=1` outside local loopback development
- [x] Set `HBLLM_TENANT_GUARD_MODE=STRICT` for production deployments
- [x] Set `HBLLM_REDIS_HMAC_KEY` for distributed deployments
- [x] Enable per-tenant rate limiting (`HBLLM_RATE_LIMIT_RPM=60`)
- [x] Enable per-tenant DB quotas (`HBLLM_DB_MAX_PER_TENANT=50000`)
- [x] Enable audit logging (`security.audit_enabled: true`)
- [x] Enable encryption at rest for sensitive fields (`security.encryption_enabled: true`)
- [x] Run behind a reverse proxy (nginx/Caddy) with TLS
- [x] Set `HBLLM_CORS_ORIGINS` to specific domains (wildcard blocked in production)

> 📖 **[Full Security Architecture →](../security.md)**

