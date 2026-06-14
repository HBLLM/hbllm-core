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

The simplest production deployment — runs on any machine with Python 3.11+:

```bash
# Install and serve (CPU-only, ~1GB RAM for 125M model)
pip install -e .
HBLLM_PROVIDER=local hbllm serve --port 8000
```

!!! success "No CUDA, No Problem"
    Rust SIMD kernels (AVX2 on x86, NEON on ARM) handle INT4/INT8 quantized inference on CPU. The 125M model serves responses in under 500MB RAM.

---

## Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "hbllm.serving.api"]
```

```bash
docker build -t hbllm-core .

# CPU-only (no --gpus flag needed)
docker run -p 8000:8000 \
  -e HBLLM_PROVIDER=local \
  hbllm-core

# Or with cloud API backend (even lighter — ~200MB RAM)
docker run -p 8000:8000 \
  -e HBLLM_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-... \
  hbllm-core
```

---

## Raspberry Pi 5 Deployment

HBLLM runs a full cognitive brain on a $80 single-board computer:

```bash
# On Raspberry Pi 5 (8GB model recommended)
pip install -e .

# Use the 125M model with INT4 quantization
HBLLM_PROVIDER=local \
HBLLM_QUANTIZE=int4 \
hbllm serve --port 8000 --workers 1
```

!!! tip "ARM Optimization"
    The Rust compute kernels use NEON SIMD instructions natively on ARM64, giving significant speedups over pure Python inference.

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
- [x] Set `HBLLM_TENANT_GUARD_MODE=STRICT` for production deployments
- [x] Set `HBLLM_REDIS_HMAC_KEY` for distributed deployments
- [x] Use `weights_only=True` (enforced automatically)
- [x] Enable per-tenant rate limiting (`HBLLM_RATE_LIMIT_RPM=60`)
- [x] Enable per-tenant DB quotas (`HBLLM_DB_MAX_PER_TENANT=50000`)
- [x] Enable audit logging (`security.audit_enabled: true`)
- [x] Enable encryption at rest for sensitive fields (`security.encryption_enabled: true`)
- [x] Pin adapter revisions to specific Git tags
- [x] Run behind a reverse proxy (nginx/Caddy) with TLS
- [x] Set `HBLLM_CORS_ORIGINS` to specific domains (wildcard blocked in production)

> 📖 **[Full Security Architecture →](../security.md)**

