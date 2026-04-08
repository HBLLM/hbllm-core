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
```

## Monitoring

HBLLM ships with OpenTelemetry instrumentation:

- Request latency histograms
- Node activation counts
- Memory usage per tenant
- Bus throughput metrics

### Prometheus + Grafana

```bash
HBLLM_OTEL_ENDPOINT=http://otel-collector:4317 \
python -m hbllm.serving.api
```

## Health Checks

The API exposes a health endpoint:

```bash
curl http://localhost:8000/health
# {"status": "healthy", "nodes": 23, "bus": "running"}
```

## Security Checklist

- [x] Set `HBLLM_REDIS_HMAC_KEY` for distributed deployments
- [x] Use `weights_only=True` (enforced automatically)
- [x] Enable per-tenant rate limiting
- [x] Pin adapter revisions to specific Git tags
- [x] Run behind a reverse proxy (nginx/Caddy) with TLS
